#!/usr/bin/env python3
"""
run_experiments.py -- batch experiments for the bounded-curvature enclosure
method (polygon evolution) against the hull-disk baseline.

Usage (put it in the same folder as polygon_simulation.py):

    python run_experiments.py --list                  # only print the plan
    python run_experiments.py --tier quick            # smoke test (minutes)
    python run_experiments.py --tier standard         # the set for the paper
    python run_experiments.py --tier full --jobs 4    # larger sweeps, more seeds

Results go to ./results (change with --out). Interrupted runs resume
automatically (finished run ids are read from runs.jsonl).  At the end the
script writes

    results/runs.csv        one row per run (the raw data)
    results/summary.md      aggregated tables  <-- share this file
    results/plan.json       the exact parameter combinations that were run
    results/figures/*.png   gallery + sensitivity plots
    results/results.zip     everything above in one file  <-- or share this

Every metric is computed here, independently of the method: whether the
final curve really encloses the polygon, how deep it penetrates the polygon,
whether it self-intersects, and how much it violates the curvature bound.

Parameter names (as in the paper):
    r      = 1/kappa_max          prescribed minimum radius of curvature
    l0     = mean edge length of the sampled input polygon
    c      = curvature scale: f = 1 + c*l0/(pi*r)  (c = 1 links the angle band
             to kappa_max; see Section "Main idea" of the paper)
    fs     = termination band factor f_s (clamped to >= f)
    eta    = step size (powerFactor)
    df     = refinement factor d_f (distanceFactor)
    p      = sharp-corner exponent (anglePow)
    alpha  = pull-back strength,  rho = pull-back exponent
    T      = iteration cap
    cw     = polygon given clockwise instead of counter-clockwise
    norm / sym = repair flags for the pull-back move (see polygon_simulation.py)
    k_target / k_eta = curvature-driven evolution ("kevolution"): the flow drives
             |kappa| to k_target*kappa_max with step k_eta*l0^2 (defaults 0.85, 0.05)
    os_d   = offset-and-smooth baseline ("offsmooth", group G14): standoff d = os_d*r,
             then closing and opening by a disk of radius r (os_d = 1 is the
             morphological closing). Run only this group, reusing earlier results:
                 python run_experiments.py --tier full --out results_k --only G14
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.path import Path as MplPath  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polygon_simulation as ps  # noqa: E402

# ----------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------

def _resample(corners, l0):
    """Closed polygon through `corners`, vertices spaced about l0 apart
    (corners are kept).  Counter-clockwise."""
    C = np.asarray(corners, dtype=float)
    out = []
    for a, b in zip(C, np.roll(C, -1, axis=0)):
        n = max(1, int(round(np.linalg.norm(b - a) / l0)))
        t = np.arange(n) / n
        out.append(a + t[:, None] * (b - a))
    P = np.vstack(out)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if area > 0 else P[::-1].copy()


def _star(n, outer, inner):
    pts = []
    for k in range(2 * n):
        rad = outer if k % 2 == 0 else inner
        a = math.pi / 2 + math.pi * k / n
        pts.append((rad * math.cos(a), rad * math.sin(a)))
    return pts


def _random_star_shaped(seed, n=16, rmin=35.0, rmax=100.0):
    """Random simple polygon: star-shaped around the origin, random radii."""
    rng = np.random.default_rng(seed)
    ang = np.sort(rng.uniform(0, 2 * math.pi, n))
    rad = rng.uniform(rmin, rmax, n)
    return [(r * math.cos(a), r * math.sin(a)) for r, a in zip(rad, ang)]


def _segments_cross(p, q, r, s):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (o(p, q, r) * o(p, q, s) < 0) and (o(r, s, p) * o(r, s, q) < 0)


def _random_nonstar(seed, n=14, size=100.0, min_edge=12.0):
    """Random simple polygon that is in general NOT star-shaped: random points in a
    square, visited in random order, untangled by 2-opt moves (reverse the stretch
    between two crossing edges) until no two edges cross."""
    rng = np.random.default_rng(10_000 + seed)
    for _ in range(200):
        P = rng.uniform(-size, size, (n, 2))
        order = list(rng.permutation(n))
        changed = True
        it = 0
        while changed and it < 2000:
            changed = False
            it += 1
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = P[order[i]], P[order[(i + 1) % n]]
                    c, d = P[order[j]], P[order[(j + 1) % n]]
                    if _segments_cross(a, b, c, d):
                        order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                        changed = True
        Q = P[order]
        edges = np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)
        # reject degenerate ones: crossings left, very short edges, or near-collinear spikes
        ok = not changed and edges.min() >= min_edge
        if ok:
            return [tuple(p) for p in Q]
    raise RuntimeError("no simple polygon found")


def _random_orthogonal(seed, grid=7, cells=16, unit=25.0):
    """Random orthogonal ("building-like") polygon: a random connected set of grid cells
    without holes and without cells that touch only at a corner, traced as a polygon."""
    rng = np.random.default_rng(20_000 + seed)
    for _ in range(500):
        occ = np.zeros((grid, grid), bool)
        occ[grid // 2, grid // 2] = True
        while occ.sum() < cells:
            cand = [(i, j) for i in range(grid) for j in range(grid) if not occ[i, j] and (
                (i > 0 and occ[i - 1, j]) or (i < grid - 1 and occ[i + 1, j]) or
                (j > 0 and occ[i, j - 1]) or (j < grid - 1 and occ[i, j + 1]))]
            i, j = cand[rng.integers(len(cand))]
            occ[i, j] = True
        # no holes: flood fill the complement from the border
        free = ~occ
        seen = np.zeros_like(free)
        stack = [(i, j) for i in range(grid) for j in (0, grid - 1) if free[i, j]] + \
                [(i, j) for j in range(grid) for i in (0, grid - 1) if free[i, j]]
        while stack:
            i, j = stack.pop()
            if seen[i, j]:
                continue
            seen[i, j] = True
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < grid and 0 <= b < grid and free[a, b] and not seen[a, b]:
                    stack.append((a, b))
        if (free & ~seen).any():
            continue
        # no pinch vertices (two cells touching only at a corner)
        pinch = False
        for i in range(grid - 1):
            for j in range(grid - 1):
                q = occ[i:i + 2, j:j + 2]
                if (q[0, 0] and q[1, 1] and not q[0, 1] and not q[1, 0]) or \
                   (q[0, 1] and q[1, 0] and not q[0, 0] and not q[1, 1]):
                    pinch = True
        if pinch:
            continue
        # boundary edges of the cells, oriented ccw around each occupied cell
        nxt = {}
        for i in range(grid):
            for j in range(grid):
                if not occ[i, j]:
                    continue
                x0, y0, x1, y1 = j, i, j + 1, i + 1
                for a, b, ni, nj in (((x0, y0), (x1, y0), i - 1, j), ((x1, y0), (x1, y1), i, j + 1),
                                     ((x1, y1), (x0, y1), i + 1, j), ((x0, y1), (x0, y0), i, j - 1)):
                    if not (0 <= ni < grid and 0 <= nj < grid and occ[ni, nj]):
                        nxt[a] = b
        start = min(nxt)
        loop, p = [start], nxt[start]
        while p != start:
            loop.append(p)
            p = nxt[p]
        # keep only the corners
        L = np.array(loop, float)
        prev, nex = np.roll(L, 1, axis=0), np.roll(L, -1, axis=0)
        cross = (L[:, 0] - prev[:, 0]) * (nex[:, 1] - L[:, 1]) - (L[:, 1] - prev[:, 1]) * (nex[:, 0] - L[:, 0])
        corners = L[np.abs(cross) > 0] * unit
        corners -= corners.mean(axis=0)
        return [tuple(p) for p in corners]
    raise RuntimeError("no orthogonal polygon found")


FAMILIES = {
    "square":  lambda s: [(0, 0), (100, 0), (100, 100), (0, 100)],
    "rect":    lambda s: [(0, 0), (200, 0), (200, 50), (0, 50)],
    "fatI":    lambda s: [(0, 0), (200, 0), (200, 50), (150, 50), (150, 150), (200, 150),
                          (200, 200), (0, 200), (0, 150), (50, 150), (50, 50), (0, 50)],
    "L":       lambda s: [(0, 0), (150, 0), (150, 50), (50, 50), (50, 150), (0, 150)],
    "U":       lambda s: [(0, 0), (200, 0), (200, 100), (150, 100), (150, 40), (50, 40),
                          (50, 100), (0, 100)],
    "star5":   lambda s: _star(5, 100, 40),
    "random":  lambda s: _random_star_shaped(s),
    "nonstar": lambda s: _random_nonstar(s),
    "ortho":   lambda s: _random_orthogonal(s),
}


def make_polygon(family, seed, l0):
    return _resample(FAMILIES[family](seed), l0)


# ----------------------------------------------------------------------
# Independent metrics
# ----------------------------------------------------------------------

def _seg_dist(points, A, B, chunk_elems=3_000_000):
    """min distance from each point to the closed polyline A->B (segments)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-300)
    out = np.empty(len(points))
    step = max(1, chunk_elems // max(1, len(A)))
    for s in range(0, len(points), step):
        c = points[s:s + step]
        ap = c[:, None, :] - A[None, :, :]
        t = np.clip((ap * AB[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        out[s:s + step] = np.linalg.norm(c[:, None, :] - proj, axis=2).min(axis=1)
    return out


def _closed_segments(P):
    return P, np.roll(P, -1, axis=0)


def count_self_intersections(P, cap=6000):
    """Number of properly crossing non-adjacent edge pairs; -1 if too large."""
    m = len(P)
    if m > cap:
        return -1
    A, B = _closed_segments(P)
    cnt = 0
    jj = np.arange(m)
    step = max(1, 2_000_000 // m)
    for s in range(0, m, step):
        i = np.arange(s, min(m, s + step))
        Ai, Bi = A[i][:, None, :], B[i][:, None, :]
        Aj, Bj = A[None, :, :], B[None, :, :]

        def orient(p, q, r):
            return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - \
                   (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
        o1, o2 = orient(Ai, Bi, Aj), orient(Ai, Bi, Bj)
        o3, o4 = orient(Aj, Bj, Ai), orient(Aj, Bj, Bi)
        hit = (o1 * o2 < 0) & (o3 * o4 < 0)
        valid = (jj[None, :] > i[:, None] + 1) & ~((i[:, None] == 0) & (jj[None, :] == m - 1))
        cnt += int((hit & valid).sum())
    return cnt


def _resample_arclength(P, h):
    Q = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(8, int(math.ceil(s[-1] / h)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])], axis=1)


def vertex_kappa(P):
    """Discrete curvature |turning angle| / mean adjacent edge length."""
    e_in = P - np.roll(P, 1, axis=0)
    e_out = np.roll(P, -1, axis=0) - P
    l_in = np.linalg.norm(e_in, axis=1)
    l_out = np.linalg.norm(e_out, axis=1)
    cross = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
    dot = (e_in * e_out).sum(axis=1)
    turn = np.arctan2(cross, dot)
    kappa = np.abs(turn) / np.maximum(0.5 * (l_in + l_out), 1e-12)
    return kappa, turn


TOUCH_TOL = 1e-3   # a polygon point closer than this to the curve counts as covered


def evaluate_chain(chain, poly, r, l0, tol=0.05):
    """All metrics of one final curve; `poly` is the (ccw) input polygon."""
    chain = np.asarray(chain, dtype=float)
    kmax = 1.0 / r
    m = {}
    m["n_vertices"] = len(chain)
    m["L"] = ps.chain_length(chain)

    # enclosure: every point of the polygon boundary must lie inside the curve
    pd = ps.densify(poly, 1.0)
    inside = MplPath(np.vstack([chain, chain[:1]])).contains_points(pd)
    unc = pd[~inside]
    if len(unc):                       # points exactly on the curve are not penetrations
        A, B = _closed_segments(chain)
        du = _seg_dist(unc, A, B)
        unc, du = unc[du > TOUCH_TOL], du[du > TOUCH_TOL]
    m["uncovered_pts"] = int(len(unc))
    m["uncovered_frac"] = float(len(unc)) / len(pd)
    m["penetration_depth"] = float(du.max()) if len(unc) else 0.0

    # min-max objective: farthest curve point from the polygon boundary
    cd = ps.densify(chain, 0.5)
    A, B = _closed_segments(poly)
    dist = _seg_dist(cd, A, B)
    m["D"] = float(dist.max())
    m["D_min"] = float(dist.min())             # closest approach (>= 0 if outside)
    # min-sum objective: total deviation  D_sum = integral over the curve of dist(x, P) ds
    # (trapezoidal rule on the densified chain), and the mean deviation D_sum / L
    seg = np.linalg.norm(np.roll(cd, -1, axis=0) - cd, axis=1)
    m["D_sum"] = float((0.5 * (dist + np.roll(dist, -1)) * seg).sum())
    m["D_mean"] = m["D_sum"] / float(seg.sum())

    m["self_intersections"] = count_self_intersections(chain)

    # curvature: native (per vertex) and resampled at the scale of the input
    kap, turn = vertex_kappa(chain)
    m["kappa_ratio_native_max"] = float(kap.max() / kmax)
    m["kappa_viol_native_frac"] = float((kap > kmax * (1 + tol)).mean())
    rs = _resample_arclength(chain, max(l0, 0.5))
    kap2, _ = vertex_kappa(rs)
    m["kappa_ratio_resampled_max"] = float(kap2.max() / kmax)
    m["kappa_viol_resampled_frac"] = float((kap2 > kmax * (1 + tol)).mean())
    m["total_turning_over_2pi"] = float(turn.sum() / (2 * math.pi))

    simple_ok = m["self_intersections"] == 0
    m["enclosing"] = bool(m["uncovered_pts"] == 0)
    m["simple"] = bool(simple_ok) if m["self_intersections"] >= 0 else None
    m["curvature_ok"] = bool(m["kappa_ratio_resampled_max"] <= 1 + tol)
    m["admissible"] = bool(m["enclosing"] and simple_ok and m["curvature_ok"])
    return m


# ----------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------

# ablation variants of the curvature-driven evolution (keyword arguments of
# polygon_simulation.simulate_curvature_evolution); "full" is the method itself
ABLATIONS = {
    "full":            {},
    "gradient":        {"solver": "gradient"},
    "no_selfcheck":    {"selfcheck": False},
    "no_blame":        {"precise_blame": False},
    "no_contact":      {"contact_noslide": False},
    "no_chord_bounds": {"chord_bounds": False},
    "no_hold":         {"hold_iters": 0},
    "no_sharp":        {"lap": 0.0},
    "no_tang":         {"tang": 0.0},
    "no_release":      {"stall_iters": 0},
    "no_inflate":      {"inflate": False},
    "native_stop":     {"resampled_stop": False, "certify": False},
}

DEFAULTS = dict(r=20.0, l0=5.0, c=1.0, fs=1.25, eta=0.08, df=1.15, p=1.5,
                alpha=0.0, rho=2.0, T=3000, cw=False, norm=False, sym=False,
                sw_start=2.5, sw_slack=0.92)


def run_id(spec):
    keys = ["group", "method", "family", "seed", "r", "l0", "c", "fs", "eta", "df",
            "p", "alpha", "rho", "T", "cw", "norm", "sym", "f_raw"] + (["sw_start", "sw_slack"] if spec.get("method") == "shrinkwrap" else [])
    if spec.get("method") == "kevolution":
        keys += ["k_target", "k_eta"] + (["abl"] if spec.get("abl") else [])
    if spec.get("method") == "offsmooth":          # new keys only for the new method: old ids unchanged
        keys += ["os_d"]
    blob = json.dumps([spec.get(k) for k in keys], sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


class _Timeout(Exception):
    pass


def execute(spec):
    """Run one spec and return a flat result dict (never raises)."""
    out = dict(spec)
    out["id"] = run_id(spec)
    t0 = time.time()
    try:
        r, l0 = spec["r"], spec["l0"]
        poly = make_polygon(spec["family"], spec.get("seed", 0), l0)
        out["n_input"] = len(poly)
        out["perimeter_input"] = ps.chain_length(poly)
        H = ps.convex_hull(poly)
        out["hull_perimeter"] = ps.chain_length(H)
        out["L_lower_bound"] = max(out["hull_perimeter"], 2 * math.pi * r)
        method = spec["method"]

        if method == "hull":
            chain = H
            out["status"] = "exact"
            out["iterations"] = 0
        elif method == "hulldisk":
            chain = ps.hull_disk_chain(poly, r, 3.0)
            out["status"] = "exact"
            out["iterations"] = 0
        elif method == "blhull":
            # exact shortest enclosure (Boissonnat and Lazard 1996): the min-sum optimum
            chain, out["L_star"], _ = ps.bl_hull_enclosure(poly, r)
            out["status"] = "exact"
            out["iterations"] = 0
        elif method == "shrinkwrap":
            chain, it = ps.shrink_wrap_enclosure(poly, r, start_factor=spec["sw_start"],
                                                 slack=spec["sw_slack"], max_iter=int(spec["T"]), l0=l0)
            out["status"], out["iterations"] = "shrunk", it
        elif method == "morph":
            chain = ps.morphological_enclosure(poly, r)
            out["status"], out["iterations"] = "exact", 0
        elif method == "offsmooth":
            chain, ncomp = ps.offset_smooth_enclosure(poly, r, spec["os_d"] * r)
            out["components"] = ncomp
            out["iterations"] = 0
            if chain is None:                  # the opening removed everything
                out["status"] = "empty"
                out["seconds"] = time.time() - t0
                return out
            out["status"] = "exact" if ncomp == 1 else "split"
        elif method == "kevolution":
            limit = spec.get("time_limit", 120.0) * 4

            def kcb(it, Q):
                if time.time() - t0 > limit:
                    raise _Timeout()
            abl = dict(ABLATIONS[spec.get("abl") or "full"])
            try:
                res = ps.simulate_curvature_evolution(
                    poly, 1.0 / r, l0=l0, target=spec.get("k_target", 0.85),
                    eta=spec.get("k_eta", 0.05), max_iter=int(spec["T"]),
                    callback=kcb, callback_every=50, **abl)
            except _Timeout:
                out["status"] = "timeout"
                out["seconds"] = time.time() - t0
                return out
            out["status"], out["iterations"] = res["status"], res["iterations"]
            out["releases"] = res.get("releases", 0)
            out["inflations"] = res.get("inflations", 0)
            out["strict_rejections"] = res.get("strict_rejections", 0)
            chain = res["final"]
            if res["status"] == "diverged":
                out["seconds"] = time.time() - t0
                return out
        else:
            isCw = bool(spec["cw"])
            P = poly[::-1].copy() if isCw else poly
            ave = float(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).mean())
            f = spec["f_raw"] if spec.get("f_raw") else 1.0 + spec["c"] * l0 / (math.pi * r)
            fs = max(spec["fs"], f)
            out["f"], out["fs_eff"] = f, fs
            limit = spec.get("time_limit", 120.0)

            def cb(it, Q):
                if time.time() - t0 > limit:
                    raise _Timeout()
            try:
                res = ps.simulate_polygon_evolution_fast(
                    P, isCw, f, fs, spec["eta"], spec["df"], spec["p"], spec["rho"],
                    spec["alpha"], int(spec["T"]), pointAveDis=ave,
                    pullback_normalize=bool(spec["norm"]), pullback_symmetric=bool(spec["sym"]),
                    callback=cb, callback_every=25, max_vertices=20000)
            except _Timeout:
                out["status"] = "timeout"
                out["seconds"] = time.time() - t0
                return out
            out["status"], out["iterations"] = res["status"], res["iterations"]
            chain = res["final"]
            if isCw:                       # evaluate everything on the ccw polygon
                chain = chain[::-1].copy()
            if res["status"] == "diverged":
                out["seconds"] = time.time() - t0
                return out

        out.update(evaluate_chain(chain, poly, r, l0))
        if method == "kevolution":
            # the output of the method is the certified segment-and-arc curve, verified in
            # full (polygon_simulation.certified_curve); if the run did not converge or the
            # certificate fails, the method returns the shortest enclosure as a fallback
            cert = ps.certified_curve(chain, poly, 1.0 / r)
            for k in ("g", "regular", "radii_ok", "closure_C1", "contact_free", "witness", "simple"):
                out["cert_" + k] = cert[k]
            out["cert_kappa_ratio"] = cert["kappa_cert"]
            out["cert_offset"] = cert["g"]
            out["cert_enclosing"] = bool(cert["contact_free"] and cert["witness"] and cert["simple"])
            out["cert_ok"] = bool(cert["ok"])
            out["cert_L"], out["cert_D"], out["cert_D_sum"] = cert["L"], cert["D"], cert["D_sum"]
            if out["status"] == "converged" and cert["ok"]:
                out["fallback"] = False
                out["ret_L"], out["ret_D"], out["ret_D_sum"] = cert["L"], cert["D"], cert["D_sum"]
            else:
                fb, Lfb, _ = ps.bl_hull_enclosure(poly, r)
                mf = evaluate_chain(fb, poly, r, l0)
                out["fallback"] = True
                out["ret_L"], out["ret_D"], out["ret_D_sum"] = Lfb, mf["D"], mf["D_sum"]
            if spec.get("keep_dir"):
                np.save(os.path.join(spec["keep_dir"], out["id"] + "_cert.npy"), cert["samples"])
        elif method in ("shrinkwrap", "morph", "offsmooth"):
            # smooth certificate: arc fillets + outward offset (polygon_simulation.fillet_certificate)
            S, kc, g = ps.fillet_certificate(chain)
            pd = ps.densify(poly, 1.0)
            ins = MplPath(np.vstack([S, S[:1]])).contains_points(pd)
            unc = pd[~ins]
            if len(unc):
                A, B = _closed_segments(S)
                unc = unc[_seg_dist(unc, A, B) > TOUCH_TOL]
            out["cert_kappa_ratio"] = kc * r
            out["cert_offset"] = g
            out["cert_enclosing"] = bool(len(unc) == 0)
            out["cert_ok"] = bool(out["cert_enclosing"] and kc * r <= 1.0 + 1e-9)
        out["lambda_L"] = out["L"] / out["L_lower_bound"]
        out["D_over_r"] = out["D"] / r
        if spec.get("keep_dir"):
            np.save(os.path.join(spec["keep_dir"], out["id"] + ".npy"), chain)
            np.save(os.path.join(spec["keep_dir"], out["id"] + "_poly.npy"), poly)
    except Exception as exc:               # record and carry on
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------

def spec_(group, method="evolution", family="square", seed=0, **kw):
    s = dict(DEFAULTS)
    s.update(kw)
    s.update(group=group, method=method, family=family, seed=seed)
    return s


def build_plan(tier):
    plan = []
    quick = tier == "quick"
    full = tier == "full"

    # ---- G1: main comparison on every family, three radii ---------------
    fams = ["square", "fatI", "U"] if quick else ["square", "rect", "fatI", "L", "U", "star5"]
    radii = [20] if quick else [10, 20, 40]
    for fam, r in itertools.product(fams, radii):
        for method in ("hull", "hulldisk"):
            plan.append(spec_("G1_main", method, fam, 0, r=r, keep_dir="KEEP"))
        plan.append(spec_("G1_main", "evolution", fam, 0, r=r, keep_dir="KEEP"))
        if not quick:
            plan.append(spec_("G1_main", "evolution", fam, 0, r=r, alpha=1.5, keep_dir="KEEP"))

    # ---- G2: random polygons ---------------------------------------------
    nseed = 3 if quick else (40 if full else 20)
    for seed in range(nseed):
        for r in ([20] if quick else [10, 20, 40]):
            for method in ("hull", "hulldisk"):
                plan.append(spec_("G2_random", method, "random", seed, r=r))
            plan.append(spec_("G2_random", "evolution", "random", seed, r=r))

    # ---- G3: one-at-a-time sensitivity (baseline r=20, l0=5) ---------------
    sens_fams = ["square"] if quick else ["square", "fatI", "L", "U"]
    sweeps = {
        "c":     [0.25, 0.5, 1.0, 1.5, 2.0] if not quick else [0.5, 1.0],
        "fs":    [1.0, 1.1, 1.25, 1.5, 2.0] if not quick else [1.25],
        "eta":   [0.02, 0.04, 0.08, 0.16, 0.32] if not quick else [0.08],
        "df":    [1.05, 1.15, 1.5, 2.0] if not quick else [1.15],
        "p":     [0.5, 1.0, 1.5, 2.0, 3.0] if not quick else [1.5],
        "alpha": [0.0, 0.5, 1.5, 3.0] if not quick else [0.0, 1.5],
        "rho":   [0.0, 1.0, 2.0, 4.0] if not quick else [2.0],
        "T":     ([500, 1000, 3000, 10000] if full else [500, 1000, 3000]) if not quick else [500],
        "l0":    [2.5, 5.0, 10.0] if not quick else [5.0],
        "cw":    [False, True] if not quick else [False],
    }
    for fam in sens_fams:
        for name, values in sweeps.items():
            for v in values:
                kw = {name: v}
                if name == "rho":
                    kw["alpha"] = 1.5          # rho is meaningless without pull-back
                plan.append(spec_("G3_sens_" + name, "evolution", fam, 0, **kw))
        # orientation with pull-back (the reference pull-back is orientation asymmetric)
        for cw in ([False, True] if not quick else [False]):
            plan.append(spec_("G3_sens_cw", "evolution", fam, 0, cw=cw, alpha=1.5))

    # ---- G4: repaired pull-back (normalised direction, symmetric condition) -
    for fam in (["square"] if quick else ["square", "fatI", "L", "U"]):
        for cw in ([False] if quick else [False, True]):
            for alpha in ([1.5] if quick else [0.5, 1.5, 3.0]):
                plan.append(spec_("G4_repaired", "evolution", fam, 0, cw=cw, alpha=alpha,
                                  norm=True, sym=True))

    # ---- G5: the parameter values of the original script ------------------
    for fam in (["square"] if quick else ["square", "fatI"]):
        plan.append(spec_("G5_script_defaults", "evolution", fam, 0, r=80.0, f_raw=1.02,
                          fs=1.25, eta=0.08, df=1.15, p=1.5, alpha=0.0, rho=2.0, T=3000))

    # ---- G7: strict termination (f_s = f: stop only when the working band is met) ----
    for fam in (["square"] if quick else ["square", "fatI", "L", "U"]):
        for c in ([1.0] if quick else [0.5, 1.0]):
            for T in ([1000] if quick else ([3000, 10000, 30000] if full else [3000, 10000])):
                plan.append(spec_("G7_strict", "evolution", fam, 0, c=c, fs=1.0, T=T))

    # ---- G6: radius sweep (how the gap grows when the bound tightens) -----
    if not quick:
        for fam in ["square", "fatI", "U"]:
            for r in ([5, 10, 20, 30, 40, 60] if not full else [5, 7.5, 10, 15, 20, 30, 40, 60, 80]):
                for method in ("hulldisk", "evolution"):
                    plan.append(spec_("G6_radius", method, fam, 0, r=r))

    # ---- G8: shrink-wrap variant (curvature-capped curve shortening) -----------
    if not quick:
        for fam in ["square", "rect", "fatI", "L", "U", "star5"]:
            for r in [10, 20, 40]:
                plan.append(spec_("G8_shrinkwrap", "shrinkwrap", fam, 0, r=r, T=8000))
        for fam in ["U", "fatI"]:
            for k in [1.5, 2.0, 3.0, 3.5]:
                plan.append(spec_("G8_shrinkwrap_start", "shrinkwrap", fam, 0, r=20, T=8000, sw_start=k))
        for seed in range(10 if not full else 20):
            plan.append(spec_("G8_shrinkwrap_random", "shrinkwrap", "random", seed, r=20, T=8000))
            plan.append(spec_("G8_shrinkwrap_random", "hulldisk", "random", seed, r=20))

    # ---- G9: curvature-driven evolution (the repair) ---------------------------
    kT = 40000
    fams9 = ["square", "fatI", "U"] if quick else ["square", "rect", "fatI", "L", "U", "star5"]
    for fam, r in itertools.product(fams9, [20] if quick else [10, 20, 40]):
        plan.append(spec_("G1_main", "kevolution", fam, 0, r=r, T=kT, keep_dir="KEEP"))
    for seed in range(nseed):
        for r in ([20] if quick else [10, 20, 40]):
            plan.append(spec_("G2_random", "kevolution", "random", seed, r=r, T=kT))
    if not quick:
        for fam in ["square", "fatI", "U"]:
            for r in ([5, 10, 20, 30, 40, 60] if not full else [5, 7.5, 10, 15, 20, 30, 40, 60, 80]):
                plan.append(spec_("G6_radius", "kevolution", fam, 0, r=r, T=kT))
        for fam in ["square", "fatI", "U"]:
            for tgt in [0.75, 0.85, 0.95]:
                plan.append(spec_("G9_kevolution_target", "kevolution", fam, 0, r=20, T=kT, k_target=tgt))

    # ---- G10: morphological baseline (closing of the r-offset) -----------------
    for fam, r in itertools.product(fams9, [20] if quick else [10, 20, 40]):
        plan.append(spec_("G1_main", "morph", fam, 0, r=r, keep_dir="KEEP"))
    for seed in range(nseed):
        for r in ([20] if quick else [10, 20, 40]):
            plan.append(spec_("G2_random", "morph", "random", seed, r=r))

    # ---- G12: further families (non-star-shaped, orthogonal) -------------------
    nfam = 3 if quick else (40 if full else 20)
    for fam in ("nonstar", "ortho"):
        for seed in range(nfam):
            for r in ([20] if quick else [10, 20, 40]):
                for method in ("hulldisk", "morph", "evolution", "kevolution", "shrinkwrap"):
                    kw = dict(r=r)
                    if method == "kevolution":
                        kw["T"] = kT
                    if method == "shrinkwrap":
                        kw["T"] = 8000
                    plan.append(spec_("G12_families", method, fam, seed, **kw))

    # ---- G14: offset-and-smooth baseline (standoff d = os_d * r, then rounding to r) ----
    # Same instances as the main study: named shapes, random star-shaped, non-star-shaped
    # and orthogonal polygons. os_d = 1 reproduces the morphological closing (check).
    os_values = [0.5] if quick else [0.25, 0.5, 1.0]
    for os_d in os_values:
        for fam, r in itertools.product(fams9, [20] if quick else [10, 20, 40]):
            plan.append(spec_("G14_offsmooth", "offsmooth", fam, 0, r=r, os_d=os_d))
        for seed in range(nseed):
            for r in ([20] if quick else [10, 20, 40]):
                plan.append(spec_("G14_offsmooth", "offsmooth", "random", seed, r=r, os_d=os_d))
        for fam in ("nonstar", "ortho"):
            for seed in range(nfam):
                for r in ([20] if quick else [10, 20, 40]):
                    plan.append(spec_("G14_offsmooth", "offsmooth", fam, seed, r=r, os_d=os_d))

    # ---- G11: ablation of the curvature-driven evolution ----------------------
    abl_inst = ([("square", 0, 20), ("fatI", 0, 20)] if quick else
                [(f, 0, r) for f in ["square", "rect", "fatI", "L", "U", "star5"] for r in (20, 40)] +
                [("random", s, 20) for s in range(20)] + [("nonstar", s, 20) for s in range(10)])
    for name in ABLATIONS:
        for fam, seed, r in abl_inst:
            plan.append(spec_("G11_ablation", "kevolution", fam, seed, r=r, T=10000, abl=name))

    # ---- G13: sampling spacing of the curvature-driven evolution ----------------
    if not quick:
        for l0 in (2.5, 10.0):
            for fam in ["square", "rect", "fatI", "L", "U", "star5"]:
                plan.append(spec_("G13_spacing", "kevolution", fam, 0, r=20, l0=l0, T=kT))
            for seed in range(10):
                plan.append(spec_("G13_spacing", "kevolution", "random", seed, r=20, l0=l0, T=kT))

    # unique by id, keep order
    seen, uniq = set(), []
    for s in plan:
        i = run_id(s)
        if i not in seen:
            seen.add(i)
            uniq.append(s)
    return uniq


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

COLS = ["id", "group", "method", "family", "seed", "r", "l0", "c", "fs", "eta", "df", "p",
        "alpha", "rho", "T", "cw", "norm", "sym", "f_raw", "sw_start", "sw_slack", "k_target", "k_eta", "abl",
        "os_d", "components",
        "f", "fs_eff", "status", "releases", "inflations", "strict_rejections", "cert_kappa_ratio", "cert_offset", "cert_enclosing",
        "cert_ok", "cert_regular", "cert_radii_ok", "cert_closure_C1", "cert_contact_free", "cert_witness", "cert_simple",
        "cert_L", "cert_D", "cert_D_sum", "fallback", "ret_L", "ret_D", "ret_D_sum",
        "iterations", "seconds", "n_input", "n_vertices", "perimeter_input", "hull_perimeter",
        "L_lower_bound", "L_star", "L", "lambda_L", "D", "D_over_r", "D_min", "D_sum", "D_mean", "enclosing", "uncovered_pts",
        "uncovered_frac", "penetration_depth", "simple", "self_intersections", "curvature_ok",
        "kappa_ratio_native_max", "kappa_viol_native_frac", "kappa_ratio_resampled_max",
        "kappa_viol_resampled_frac", "total_turning_over_2pi", "admissible", "error"]


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})


def _fmt(x, nd=2):
    if x is None or x == "":
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, float):
        return "inf" if math.isinf(x) else f"{x:.{nd}f}"
    return str(x)


def _md_table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in body:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _med(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(vals)) if vals else None


def _rate(rows, key):
    ok = [bool(r.get(key)) for r in rows if r.get("status") in ("converged", "max_iter", "exact", "shrunk")]
    return (sum(ok) / len(ok)) if ok else None


def write_summary(rows, path, meta):
    out = []
    out.append("# Experiment summary\n")
    out.append("```\n" + json.dumps(meta, indent=1) + "\n```\n")
    out.append("Legend: lamL = L / max{per(conv P), 2 pi r} (length over the proved lower bound); "
               "D/r = largest curve-to-polygon distance over r; enclosing = every polygon boundary "
               "point lies inside the curve; pen = deepest penetration of the polygon by the curve; "
               "SI = number of crossing edge pairs (-1: curve too long to test); k/kmax = maximum "
               "discrete curvature over 1/r after resampling at the input edge length; admissible = "
               "enclosing, simple and k/kmax <= 1.05.\n")

    by_status = {}
    for r in rows:
        if r["method"] == "evolution":
            by_status[r.get("status")] = by_status.get(r.get("status"), 0) + 1
    out.append("Evolution run outcomes: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())) + "\n")

    # --- G1 -----------------------------------------------------------------
    g1 = [r for r in rows if r["group"] == "G1_main"]
    if g1:
        out.append("## G1. Main comparison\n")
        body = []
        for r in sorted(g1, key=lambda r: (r["family"], r["r"], ["hull", "hulldisk", "morph", "shrinkwrap", "evolution", "kevolution"].index(r["method"]),
                                           r.get("alpha", 0))):
            name = r["method"] + (f" (alpha={r['alpha']})" if r["method"] == "evolution" and r["alpha"] else "")
            body.append([r["family"], r["r"], name, r.get("status"), r.get("iterations"), r.get("L"),
                         r.get("lambda_L"), r.get("D"), r.get("D_over_r"), r.get("enclosing"),
                         r.get("penetration_depth"), r.get("self_intersections"),
                         r.get("kappa_ratio_resampled_max"), r.get("admissible")])
        out.append(_md_table(["shape", "r", "method", "status", "iters", "L", "lamL", "D", "D/r",
                              "enclosing", "pen", "SI", "k/kmax", "admissible"], body) + "\n")

    # --- G2 -----------------------------------------------------------------
    g2 = [r for r in rows if r["group"] == "G2_random"]
    if g2:
        out.append("## G2. Random star-shaped polygons (n=16 vertices before resampling)\n")
        body = []
        for r_ in sorted({r["r"] for r in g2}):
            for method in ("hull", "hulldisk", "morph", "evolution", "kevolution"):
                sub = [r for r in g2 if r["r"] == r_ and r["method"] == method]
                done = [r for r in sub if "L" in r]
                body.append([r_, method, len(sub), len(done), _med([x["lambda_L"] for x in done]),
                             _med([x["D_over_r"] for x in done]), _rate(sub, "enclosing"),
                             _rate(sub, "simple"), _rate(sub, "curvature_ok"), _rate(sub, "admissible")])
        out.append(_md_table(["r", "method", "runs", "finished", "median lamL", "median D/r",
                              "rate enclosing", "rate simple", "rate curv ok", "rate admissible"], body) + "\n")

    # --- G12: further families ---------------------------------------------
    g12 = [r for r in rows if r["group"] == "G12_families"]
    if g12:
        out.append("## G12. Further families (non-star-shaped, orthogonal)\n")
        body = []
        for fam in sorted({r["family"] for r in g12}):
            for r_ in sorted({r["r"] for r in g12}):
                for method in ("hulldisk", "morph", "shrinkwrap", "evolution", "kevolution"):
                    sub = [r for r in g12 if r["family"] == fam and r["r"] == r_ and r["method"] == method]
                    if not sub:
                        continue
                    done = [r for r in sub if "L" in r]
                    body.append([fam, r_, method, len(sub), len(done), _med([x["lambda_L"] for x in done]),
                                 _med([x["D_over_r"] for x in done]), _rate(sub, "enclosing"), _rate(sub, "simple"),
                                 _rate(sub, "curvature_ok"), _rate(sub, "admissible"),
                                 _med([x.get("seconds") for x in sub])])
        out.append(_md_table(["family", "r", "method", "runs", "finished", "median lamL", "median D/r",
                              "rate enclosing", "rate simple", "rate curv ok", "rate admissible",
                              "median s"], body) + "\n")

    # --- G14: offset-and-smooth baseline ----------------------------------------
    g14 = [r for r in rows if r["group"] == "G14_offsmooth"]
    if g14:
        def fset(fam):
            return {"random": "star-shaped", "nonstar": "non-star", "ortho": "orthogonal"}.get(fam, "named")
        ref_groups = ("G1_main", "G2_random", "G12_families")

        def rate_all(sub, key):                # over ALL runs: split/empty/timeout count as failures
            return (sum(1 for r in sub if r.get(key)) / len(sub)) if sub else None
        out.append("## G14. Offset-and-smooth baseline (standoff d = os_d * r, then closing and opening by B_r)\n")
        out.append("Reference rows (morph, kevolution) are the same instances from G1/G2/G12. "
                   "os_d = 1 should reproduce morph. 'split' = the result has several components "
                   "(metrics are of the largest one; never counted as admissible); 'empty' = the "
                   "opening removed everything. Rates are over ALL runs (split, empty and timeouts "
                   "count as failures).\n")
        body = []
        for fs_ in ("named", "star-shaped", "non-star", "orthogonal"):
            for r_ in sorted({r["r"] for r in g14}):
                variants = [(f"offsmooth d={v}r", [r for r in g14 if fset(r["family"]) == fs_ and r["r"] == r_
                                                    and r.get("os_d") == v])
                            for v in sorted({r.get("os_d") for r in g14})]
                for m in ("morph", "kevolution"):
                    variants.append((m, [r for r in rows if r["group"] in ref_groups and r["method"] == m
                                         and fset(r["family"]) == fs_ and r["r"] == r_]))
                for name, sub in variants:
                    if not sub:
                        continue
                    done = [r for r in sub if "L" in r]
                    body.append([fs_, r_, name, len(sub), len(done),
                                 sum(1 for r in sub if r.get("status") in ("split", "empty")),
                                 _med([x["lambda_L"] for x in done]), _med([x["D_over_r"] for x in done]),
                                 rate_all(sub, "enclosing"), _med([x.get("penetration_depth") for x in done]),
                                 rate_all(sub, "curvature_ok"),
                                 # a split result is not one enclosing curve: never admissible
                                 (sum(1 for r in sub if r.get("admissible") and r.get("status") != "split")
                                  / len(sub))])
        out.append(_md_table(["family", "r", "method", "runs", "finished", "split/empty", "median lamL",
                              "median D/r", "rate enclosing", "median pen", "rate curv ok",
                              "rate admissible"], body) + "\n")

    # --- G11: ablation ------------------------------------------------------
    g11 = [r for r in rows if r["group"] == "G11_ablation"]
    if g11:
        out.append("## G11. Ablation of the curvature-driven evolution (T = 10000)\n")
        body = []
        for name in ABLATIONS:
            sub = [r for r in g11 if (r.get("abl") or "full") == name]
            if not sub:
                continue
            done = [r for r in sub if "L" in r]
            body.append([name, len(sub), len(done), sum(1 for r in sub if r.get("status") == "converged"),
                         sum(1 for r in done if r.get("admissible")), sum(1 for r in done if r.get("enclosing")),
                         sum(1 for r in done if r.get("self_intersections") == 0),
                         _med([r.get("iterations") for r in sub if r.get("status") == "converged"]),
                         _med([x["D_over_r"] for x in done]), _med([x["kappa_ratio_resampled_max"] for x in done]),
                         _med([x.get("seconds") for x in sub])])
        out.append(_md_table(["variant", "runs", "finished", "converged", "admissible", "enclosing", "simple",
                              "median iters (conv.)", "median D/r", "median k/kmax", "median s"], body) + "\n")

    # --- G3 / G4 / G5 / G6 --------------------------------------------------
    for g in sorted({r["group"] for r in rows if r["group"].startswith(("G3", "G4", "G5", "G6", "G7", "G8", "G9", "G13"))}):
        sub = [r for r in rows if r["group"] == g and (r["method"] == "evolution" or g.startswith(("G6", "G8", "G9", "G13")))]
        if not sub:
            continue
        out.append(f"## {g}\n")
        pname = g.replace("G3_sens_", "")
        body = []
        key_cols = ["family", "seed", "r", "l0", "c", "fs", "eta", "df", "p", "alpha", "rho", "T", "cw", "norm", "sym", "sw_start", "k_target"]
        varying = [k for k in key_cols if len({str(r.get(k)) for r in sub}) > 1]
        for r in sorted(sub, key=lambda r: tuple(str(r.get(k)) for k in (["family"] + varying + ["method"]))):
            body.append([r["method"]] + [r.get(k) for k in varying] +
                        [r.get("status"), r.get("iterations"), r.get("n_vertices"), r.get("L"), r.get("lambda_L"),
                         r.get("D_over_r"), r.get("enclosing"), r.get("penetration_depth"),
                         r.get("self_intersections"), r.get("kappa_ratio_resampled_max"), r.get("admissible")])
        out.append(_md_table(["method"] + varying + ["status", "iters", "m", "L", "lamL", "D/r", "enclosing",
                                                      "pen", "SI", "k/kmax", "admissible"], body) + "\n")

    # --- overall one-line verdicts ---------------------------------------------
    ev = [r for r in rows if r["method"] == "evolution" and "L" in r]
    if ev:
        out.append("## Overall (all finished evolution runs)\n")
        out.append(f"- runs finished: {len(ev)} of {sum(1 for r in rows if r['method']=='evolution')}\n"
                   f"- enclosing: {_fmt(_rate(ev,'enclosing'))}; simple: {_fmt(_rate(ev,'simple'))}; "
                   f"curvature ok: {_fmt(_rate(ev,'curvature_ok'))}; admissible: {_fmt(_rate(ev,'admissible'))}\n"
                   f"- median lamL: {_fmt(_med([r['lambda_L'] for r in ev]))}\n")
    kv = [r for r in rows if r["method"] == "kevolution" and "L" in r and r["group"] != "G11_ablation"]
    if kv:
        out.append("## Overall (all finished curvature-driven evolution runs, ablation variants excluded)\n")
        out.append(f"- runs finished: {len(kv)} of "
                   f"{sum(1 for r in rows if r['method']=='kevolution' and r['group'] != 'G11_ablation')}; "
                   f"converged: {sum(1 for r in kv if r.get('status') == 'converged')}\n"
                   f"- enclosing: {_fmt(_rate(kv,'enclosing'))}; simple: {_fmt(_rate(kv,'simple'))}; "
                   f"curvature ok: {_fmt(_rate(kv,'curvature_ok'))}; admissible: {_fmt(_rate(kv,'admissible'))}\n"
                   f"- median lamL: {_fmt(_med([r['lambda_L'] for r in kv]))}; "
                   f"median D/r: {_fmt(_med([r['D_over_r'] for r in kv]))}\n"
                   f"- smooth certificate (arc fillets + offset): curvature <= kmax and enclosing in "
                   f"{sum(1 for r in kv if r.get('cert_ok'))} of {len(kv)}; "
                   f"max certified k/kmax {_fmt(max(r.get('cert_kappa_ratio', 0) for r in kv))}\n")
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def make_figures(rows, outdir, keep_dir):
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    # gallery of G1 at the middle radius
    g1 = [r for r in rows if r["group"] == "G1_main" and "L" in r and r["r"] == 20]
    fams = sorted({r["family"] for r in g1})
    if fams:
        fig, axes = plt.subplots(len(fams), 4, figsize=(13, 3.2 * len(fams)), squeeze=False)
        for i, fam in enumerate(fams):
            for j, (title, sel) in enumerate([("hull-disk", lambda r: r["method"] == "hulldisk"),
                                              ("evolution (alpha=0)", lambda r: r["method"] == "evolution" and not r["alpha"]),
                                              ("evolution (alpha=1.5)", lambda r: r["method"] == "evolution" and r["alpha"]),
                                              ("curvature evolution", lambda r: r["method"] == "kevolution")]):
                ax = axes[i][j]
                cand = [r for r in g1 if r["family"] == fam and sel(r)]
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                if not cand:
                    ax.set_visible(False)
                    continue
                r0 = cand[0]
                fc = os.path.join(keep_dir, r0["id"] + ".npy")
                fp = os.path.join(keep_dir, r0["id"] + "_poly.npy")
                if os.path.exists(fc):
                    C, Pp = np.load(fc), np.load(fp)
                    ax.fill(*np.vstack([Pp, Pp[:1]]).T, color="0.8")
                    ax.plot(*np.vstack([C, C[:1]]).T, lw=1, color="C3" if not r0.get("admissible") else "C0")
                ax.set_title(f"{fam}: {title}\nlamL={_fmt(r0.get('lambda_L'))}  admissible={_fmt(r0.get('admissible'))}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "gallery_r20.png"), dpi=130)
        plt.close(fig)

    # sensitivity panels
    params = ["c", "fs", "eta", "df", "p", "alpha", "rho", "T", "l0"]
    sens = {p: [r for r in rows if r["group"] == "G3_sens_" + p and "L" in r] for p in params}
    sens = {p: v for p, v in sens.items() if v}
    if sens:
        fig, axes = plt.subplots(2, len(sens), figsize=(2.6 * len(sens), 5), squeeze=False)
        for k, (p, sub) in enumerate(sens.items()):
            for fam in sorted({r["family"] for r in sub}):
                pts = sorted([(r[p], r["lambda_L"], float(bool(r["admissible"]))) for r in sub if r["family"] == fam])
                axes[0][k].plot([x[0] for x in pts], [x[1] for x in pts], "o-", ms=3, label=fam)
                axes[1][k].plot([x[0] for x in pts], [x[2] for x in pts], "o-", ms=3)
            axes[0][k].set_title(p)
            axes[1][k].set_xlabel(p)
            if k == 0:
                axes[0][k].set_ylabel("lamL")
                axes[1][k].set_ylabel("admissible (0/1)")
                axes[0][k].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "sensitivity.png"), dpi=130)
        plt.close(fig)

    # radius sweep
    g6 = [r for r in rows if r["group"] == "G6_radius" and "L" in r]
    if g6:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for fam in sorted({r["family"] for r in g6}):
            for method, ls in (("hulldisk", "-"), ("evolution", "--"), ("kevolution", ":")):
                pts = sorted([(r["r"], r["lambda_L"], r["D_over_r"]) for r in g6
                              if r["family"] == fam and r["method"] == method])
                ax[0].plot([x[0] for x in pts], [x[1] for x in pts], ls, marker="o", ms=3, label=f"{fam} {method}")
                ax[1].plot([x[0] for x in pts], [x[2] for x in pts], ls, marker="o", ms=3)
        ax[0].set_xlabel("r"); ax[0].set_ylabel("lamL"); ax[1].set_xlabel("r"); ax[1].set_ylabel("D/r")
        ax[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "radius_sweep.png"), dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["quick", "standard", "full"], default="standard")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--time-limit", type=float, default=120.0, help="seconds per evolution run")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results in --out")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--only", default=None,
                    help="run only the groups whose name starts with this prefix (e.g. G14); "
                         "earlier results of all groups are still read and summarised")
    args = ap.parse_args()

    out = args.out
    keep = os.path.join(out, "chains")
    os.makedirs(keep, exist_ok=True)
    plan = build_plan(args.tier)
    for s in plan:
        if s.get("keep_dir"):
            s["keep_dir"] = keep
        s["time_limit"] = args.time_limit

    groups = {}
    for s in plan:
        groups[s["group"]] = groups.get(s["group"], 0) + 1
    print(f"tier={args.tier}: {len(plan)} runs")
    for g, n in groups.items():
        print(f"  {g:24s} {n}")
    if args.list:
        return

    jsonl = os.path.join(out, "runs.jsonl")
    if args.fresh and os.path.exists(jsonl):
        os.remove(jsonl)
    done = {}
    if os.path.exists(jsonl):
        with open(jsonl) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    done[d["id"]] = d
                except Exception:
                    pass
    todo = [s for s in plan if run_id(s) not in done]
    if args.only:
        todo = [s for s in todo if s["group"].startswith(args.only)]
    print(f"already done: {len(plan) - len(todo)}; to run: {len(todo)}; jobs={args.jobs}")

    t0 = time.time()
    if todo:
        with open(jsonl, "a") as fh, Pool(args.jobs) as pool:
            for k, res in enumerate(pool.imap_unordered(execute, todo, chunksize=1), 1):
                done[res["id"]] = res
                fh.write(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
                fh.flush()
                if k % 10 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}  elapsed {time.time() - t0:.0f}s", flush=True)

    rows = [done[run_id(s)] for s in plan if run_id(s) in done]
    write_csv(rows, os.path.join(out, "runs.csv"))
    meta = dict(tier=args.tier, runs=len(rows), python=platform.python_version(),
                numpy=np.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
                wall_seconds=round(time.time() - t0, 1), time_limit=args.time_limit)
    write_summary(rows, os.path.join(out, "summary.md"), meta)
    with open(os.path.join(out, "plan.json"), "w") as fh:
        json.dump(dict(defaults=DEFAULTS, groups=groups, runs=[{k: v for k, v in s.items() if k != "keep_dir"} for s in plan]),
                  fh, indent=1, default=str)
    try:
        make_figures(rows, os.path.join(out, "figures"), keep)
    except Exception as exc:
        print("figures skipped:", exc)

    if not args.no_zip:
        zpath = os.path.join(out, "results.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("runs.csv", "summary.md", "plan.json"):
                z.write(os.path.join(out, name), name)
            fdir = os.path.join(out, "figures")
            if os.path.isdir(fdir):
                for fn in sorted(os.listdir(fdir)):
                    z.write(os.path.join(fdir, fn), "figures/" + fn)
        print("wrote", zpath)
    print("done. Share results/summary.md (and runs.csv or results.zip).")


if __name__ == "__main__":
    main()
