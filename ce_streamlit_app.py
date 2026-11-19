"""
ce_streamlit_app.py -- interactive curvature-driven evolution (CE).

Draw a simple polygon on a canvas (drawing runs in the browser, so every click shows at
once; the polygon is kept simple), or pick a preset / type coordinates; choose the
curvature bound and the CE parameters, and run polygon_simulation.simulate_curvature_evolution.
With 'simulation' checked the evolution is shown on the canvas, live while it runs and then
as an animated replay (canvas component in ce_canvas/index.html), in the style of the GIFs:
left, the chain over the polygon (red dots: vertices above the bound; orange / blue
arrows: outward / inward step of that iteration); right, the length, the maximum
curvature, the number of violating vertices and the inward / outward motion.

Run:
    streamlit run ce_streamlit_app.py
"""
import io
import math
import os
import sys
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import streamlit as st                                # noqa: E402
import streamlit.components.v1 as components          # noqa: E402
from PIL import Image, ImageDraw, ImageFont           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import polygon_simulation as ps                       # noqa: E402
import run_experiments as rx                          # noqa: E402

C_OUT, C_IN, C_CURVE, C_BAD, C_CERT = "#e67e22", "#1f5fbf", "0.15", "crimson", "#1F5FAD"
CANVAS = 200.0                                        # drawing area [0, CANVAS]^2
MAX_FRAMES = 1500                                     # stored frames (thinned beyond this)

st.set_page_config(page_title="Curvature-driven evolution", layout="wide")


# =============================================================================
# Geometry: simplicity tests
# =============================================================================
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, eps):
    """p lies on the closed segment ab (within eps)"""
    ab = np.subtract(b, a); ap = np.subtract(p, a)
    L2 = float(ab @ ab)
    if L2 == 0:
        return float(np.hypot(*ap)) <= eps
    t = float(ap @ ab) / L2
    if t < -1e-12 or t > 1 + 1e-12:
        return False
    return abs(_orient(a, b, p)) / math.sqrt(L2) <= eps


def segments_touch(a, b, c, d, eps=1e-9):
    """closed segments ab and cd intersect or touch (including collinear overlap)"""
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    return (_on_segment(a, b, c, eps) or _on_segment(a, b, d, eps) or
            _on_segment(c, d, a, eps) or _on_segment(c, d, b, eps))


def _adjacent_ok(a, b, c, eps=1e-9):
    """edges ab and bc share b only: they must not fold back onto each other"""
    return not (abs(_orient(a, b, c)) <= eps * max(1.0, np.hypot(*np.subtract(c, a)))
                and float(np.dot(np.subtract(a, b), np.subtract(c, b))) > 0)


def can_add_vertex(pts, p):
    """Is the open polyline pts + [p] still simple?  Returns (ok, reason)."""
    if pts and np.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) < 1e-9:
        return False, "the same point as the previous vertex"
    if any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9 for q in pts):
        return False, "this point is already a vertex"
    n = len(pts)
    if n == 0:
        return True, ""
    a, b = pts[-1], p
    for i in range(n - 1):
        c, d = pts[i], pts[i + 1]
        if i == n - 2:                               # the edge adjacent to the new one
            if not _adjacent_ok(c, d, b):
                return False, "the new edge folds back onto the previous edge"
            continue
        if segments_touch(a, b, c, d):
            return False, f"the new edge would cross or touch edge {i + 1}"
    return True, ""


def is_simple_polygon(pts):
    """closed polygon pts (no repeated last vertex) is simple and non-degenerate"""
    P = [tuple(map(float, p)) for p in pts]
    n = len(P)
    if n < 3:
        return False, "a polygon needs at least 3 vertices"
    if len({p for p in P}) < n:
        return False, "repeated vertices"
    area = 0.5 * sum(P[i][0] * P[(i + 1) % n][1] - P[(i + 1) % n][0] * P[i][1] for i in range(n))
    if abs(area) < 1e-6:
        return False, "the polygon has zero area"
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        if not _adjacent_ok(P[i - 1], a, b):
            return False, f"edges at vertex {i + 1} fold back onto each other"
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if segments_touch(a, b, P[j], P[(j + 1) % n]):
                return False, f"edges {i + 1} and {j + 1} cross or touch"
    return True, ""


def can_close(pts):
    if len(pts) < 3:
        return False, "at least 3 vertices are needed"
    return is_simple_polygon(pts)


# =============================================================================
# Running CE
# =============================================================================
def outward_normal(Q):
    chord = np.roll(Q, -1, axis=0) - np.roll(Q, 1, axis=0)
    n = np.stack([chord[:, 1], -chord[:, 0]], axis=1)
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def turning(Q):
    e_in = Q - np.roll(Q, 1, axis=0)
    e_out = np.roll(Q, -1, axis=0) - Q
    return np.arctan2(e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0], (e_in * e_out).sum(1))


class EarlyStop(Exception):
    """Raised from the callback to end the run early (no violating vertex left)."""


class UserStop(Exception):
    """Raised from the callback when the user pressed Stop."""


class Recorder:
    """Callback for simulate_curvature_evolution: per-iteration statistics and a thinned
    list of frames. finalize() turns what was recorded into a run, also when the run was
    interrupted (stop button) or ended early."""

    def __init__(self, kmax, progress=None, live=None, stop_when_clean=False):
        self.kmax, self.progress, self.live, self.stop_when_clean = kmax, progress, live, stop_when_clean
        self.hist = {k: [] for k in ("L", "K", "viol", "tot", "inw", "outw")}
        self.frames, self.its, self.disp = [], [], []
        self.state = {"stride": 1, "prev": None, "pending": None, "t0": time.time()}
        self.last = None                                 # (iteration, chain) of the last call
        self.prev_of_last = None                         # the chain one iteration before
        self.stop_requested = False                      # set by the Stop button

    def __call__(self, it, Q):
        _record(self, it, Q)

    def finalize(self, status, iterations, final, releases=None, inflations=None):
        frames, fr_its, fr_disp, state = self.frames, self.its, self.disp, self.state
        final = np.asarray(final, float)
        if not frames or not np.array_equal(frames[-1], final):
            if state["pending"] is not None:
                k, Qk = state["pending"]
                fr_disp[k] = (final - Qk) if len(final) == len(Qk) else None
            frames.append(final); fr_its.append(int(iterations)); fr_disp.append(None)
        res = dict(status=status, iterations=int(iterations), final=final,
                   releases=releases, inflations=inflations)
        hist = {k: np.array(v, float) for k, v in self.hist.items()}
        return dict(res=res, frames=frames, its=fr_its, disp=fr_disp, hist=hist,
                    seconds=time.time() - state["t0"])


def _record(rec, it, Q):
    kmax, hist, frames, fr_its, fr_disp, state = rec.kmax, rec.hist, rec.frames, rec.its, rec.disp, rec.state
    if True:
        if rec.stop_requested:
            raise UserStop()
        Q = Q.copy()
        rec.prev_of_last = rec.last[1] if rec.last is not None else None
        rec.last = (it, Q)
        kap = ps.vertex_curvature(Q)
        hist["L"].append(ps.chain_length(Q))
        hist["K"].append(float(kap.max() / kmax))
        hist["viol"].append(int((kap > kmax * 1.01).sum()))
        hist["tot"].append(float(np.abs(turning(Q)).sum()))
        prev = state["prev"]
        if prev is not None and len(prev) == len(Q):
            s = ((Q - prev) * outward_normal(prev)).sum(1)
            hist["inw"].append(float(-s[s < 0].sum())); hist["outw"].append(float(s[s > 0].sum()))
        else:
            hist["inw"].append(np.nan); hist["outw"].append(np.nan)
        # the arrows of a stored frame show the step that leaves it
        if state["pending"] is not None:
            k, Qk = state["pending"]
            fr_disp[k] = (Q - Qk) if len(Q) == len(Qk) else None
            state["pending"] = None
        if it % state["stride"] == 0:
            frames.append(Q); fr_its.append(it); fr_disp.append(None)
            state["pending"] = (len(frames) - 1, Q)
            if len(frames) > MAX_FRAMES:                  # thin: keep every other frame
                keep = list(range(0, len(frames) - 1, 2)) + [len(frames) - 1]
                frames[:] = [frames[i] for i in keep]
                fr_its[:] = [fr_its[i] for i in keep]
                fr_disp[:] = [fr_disp[i] for i in keep]
                state["pending"] = (len(frames) - 1, Q)
                state["stride"] *= 2
        if rec.live is not None:
            rec.live(it, Q, prev, hist)
        state["prev"] = Q
        if rec.progress is not None and it % 50 == 0:
            rec.progress(it, hist)
        if rec.stop_when_clean and hist["viol"][-1] == 0:
            raise EarlyStop()


def run_ce(poly, kmax, p, progress=None, live=None, stop_when_clean=False, rec=None):
    """Run CE. With stop_when_clean the run ends at the first iteration in which no vertex
    violates the bound (the red dots), before the paper's full stop test."""
    rec = rec or Recorder(kmax, progress, live, stop_when_clean)
    try:
        res = ps.simulate_curvature_evolution(
            poly, kmax, l0=p["l0"], target=p["target"], cap=p["cap"], tang=p["tang"],
            split=p["split"], merge=p["merge"], max_iter=int(p["max_iter"]), omega=p["omega"],
            damping=p["damping"], hold_iters=int(p["hold_iters"]), stall_iters=int(p["stall_iters"]),
            resampled_stop=p["resampled_stop"], inflate=p["inflate"], certify=p["certify"],
            callback=rec, callback_every=1)
    except EarlyStop:
        it, Q = rec.last
        return rec.finalize("no violations", it, Q)
    except UserStop:
        it, Q = rec.last
        return rec.finalize("stopped", it, Q)
    return rec.finalize(res["status"], res["iterations"], res["final"],
                        res.get("releases", 0), res.get("inflations", 0))


def summary(run, poly, kmax, l0):
    Q = np.asarray(run["res"]["final"], float)
    L = ps.chain_length(Q)
    hull_per = ps.chain_length(ps.convex_hull(poly))
    kc, g = ps.certificate_curvature(Q)
    D = ps.max_distance_to_polygon(Q, poly, spacing=max(0.5, l0 / 5))
    # mean deviation along the chain (vertex distances weighted by edge length)
    q = ps.densify(Q, max(0.5, l0 / 5))
    A = np.asarray(poly); B = np.roll(A, -1, axis=0); AB = B - A; L2 = (AB ** 2).sum(1)
    ap = q[:, None, :] - A[None]
    t = np.clip((ap * AB[None]).sum(2) / L2[None], 0, 1)
    dist = np.linalg.norm(q[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2).min(1)
    return dict(status=run["res"]["status"], iterations=int(run["res"]["iterations"]),
                L=L, L_hull=hull_per, K=float(ps.vertex_curvature(Q).max() / kmax),
                K_res=ps._resampled_max_curvature(Q, l0) / kmax, K_cert=kc / kmax, h=g, D=D,
                Dmean=float(dist.mean()), releases=run["res"].get("releases"),
                inflations=run["res"].get("inflations"), vertices=len(Q),
                seconds=run["seconds"])


# =============================================================================
# Drawing a frame (same layout as the GIFs)
# =============================================================================
def draw_frame(run, poly, kmax, l0, k, show_cert=False, arrow_frac=0.10, max_arrows=400,
               title=None, dpi=90):
    frames, its, disp, H = run["frames"], run["its"], run["disp"], run["hist"]
    Q, D, it = frames[k], disp[k], its[k]
    n_it = np.arange(len(H["L"]))
    fig = plt.figure(figsize=(12.5, 7.6), dpi=dpi)
    gs = fig.add_gridspec(5, 2, width_ratios=[1.3, 1], hspace=0.75, wspace=0.16,
                          left=0.03, right=0.97, top=0.88, bottom=0.07)
    ax = fig.add_subplot(gs[:, 0])
    axes = [fig.add_subplot(gs[i, 1]) for i in range(5)]

    # ---- left panel -----------------------------------------------------------
    allpts = np.vstack([poly] + [frames[0], frames[-1]])
    lo, hi = allpts.min(0), allpts.max(0)
    w = 1.2 * max(hi - lo); cx, cy = (lo + hi) / 2
    ax.fill(*np.vstack([poly, poly[:1]]).T, color="0.85", zorder=0)
    ax.plot(*np.vstack([poly, poly[:1]]).T, color="0.5", lw=0.8, zorder=1)
    ax.plot(*np.vstack([Q, Q[:1]]).T, color=C_CURVE, lw=1.3, zorder=3)
    ax.plot(Q[:, 0], Q[:, 1], ".", color=C_CURVE, ms=3 if len(Q) < 400 else 1, zorder=4)
    bad = ps.vertex_curvature(Q) > kmax * 1.01
    ax.plot(Q[bad, 0], Q[bad, 1], "o", color=C_BAD, ms=5 if len(Q) < 400 else 2, zorder=5)
    note = ""
    if D is not None:
        mag = np.linalg.norm(D, axis=1)
        mv = np.nonzero(mag > 1e-9)[0]
        if len(mv):
            if len(mv) > max_arrows:
                mv = mv[np.argsort(mag[mv])[-max_arrows:]]
            scale = arrow_frac * w / mag.max()
            sgn = (D[mv] * outward_normal(Q)[mv]).sum(1)
            for sel, col in ((sgn >= 0, C_OUT), (sgn < 0, C_IN)):
                if sel.any():
                    P0, V = Q[mv[sel]], D[mv[sel]] * scale
                    ax.quiver(P0[:, 0], P0[:, 1], V[:, 0], V[:, 1], color=col, angles="xy",
                              scale_units="xy", scale=1, width=0.004, headwidth=3.5, zorder=6)
            note = (f"\nthis step: {int((sgn >= 0).sum())} outward, {int((sgn < 0).sum())} inward"
                    f" (arrows x{scale:.0f})")
    if show_cert:
        try:
            S, _, _ = ps.fillet_certificate(Q)
            ax.plot(*np.vstack([S, S[:1]]).T, color=C_CERT, lw=2.0, zorder=7,
                    label="certified curve (segments and arcs)")
            ax.legend(loc="lower center", fontsize=8, frameon=False)
        except Exception:                               # noqa: BLE001
            pass
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(cy - w / 2, cy + w / 2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"iteration {it}    L = {H['L'][min(it, len(H['L']) - 1)]:.0f}    "
                 f"max $\\kappa/\\kappa_{{max}}$ = {ps.vertex_curvature(Q).max() / kmax:.2f}    "
                 f"vertices = {len(Q)}{note}", fontsize=10)

    # ---- right panels ------------------------------------------------------------
    lb = ps.chain_length(ps.convex_hull(poly))
    specs = [("L", "length $L$", None), ("K", r"max curvature $\kappa/\kappa_{max}$", 1.0),
             ("viol", r"vertices violating the bound (red dots)", None),
             ("tot", r"total curvature $\sum_i|\theta_i|$ (rad)", 2 * np.pi), (None, None, None)]
    for a, (key, ttl, ref) in zip(axes, specs):
        a.tick_params(labelsize=7.5); a.grid(alpha=0.3)
        if key is None:
            a.plot(n_it, H["outw"], color=C_OUT, lw=1.0, label="outward")
            a.plot(n_it, H["inw"], color=C_IN, lw=1.3, label="inward")
            a.set_title("total motion per iteration (normal component)", fontsize=9.5)
            a.legend(fontsize=7, loc="upper right"); a.set_xlabel("iteration", fontsize=8.5)
        else:
            a.plot(n_it, H[key], color=C_BAD if key == "viol" else C_CURVE, lw=1.2)
            a.set_title(ttl, fontsize=9.5)
            if key == "K" and np.nanmax(H["K"]) > 20:
                a.set_yscale("log")
            if ref is not None:
                a.axhline(ref, color=C_BAD if key == "K" else "0.45", ls="--", lw=0.9)
            if key == "L":
                a.axhline(lb, color="0.45", ls="--", lw=0.9)
                a.text(n_it[-1], lb, "per(conv P) ", fontsize=7, color="0.35", va="bottom", ha="right")
            if key in ("L", "K"):
                j = min(it, len(H[key]) - 1)
                a.plot([j], [H[key][j]], "o", color=C_BAD, ms=6)
        a.axvline(it, color=C_BAD, lw=1.0)
    fig.suptitle(title or "Curvature-driven evolution", fontsize=13, fontweight="bold")
    return fig


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_gif(run, poly, kmax, l0, title, n_frames=150, fps=8):
    F = len(run["frames"])
    ids = sorted(set(np.linspace(0, F - 1, min(n_frames, F)).round().astype(int).tolist()))
    imgs = [fig_to_image(draw_frame(run, poly, kmax, l0, k, show_cert=(k == F - 1), title=title, dpi=70))
            for k in ids]
    imgs = [imgs[0]] * 6 + imgs + [imgs[-1]] * 12
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    return buf.getvalue()


# =============================================================================
# Live frame (fast, PIL) shown on the canvas while CE runs
# =============================================================================
class LiveView:
    """Renders the current chain on the canvas area, at most every `min_dt` seconds."""

    def __init__(self, holder, poly, kmax, r, size=600, min_dt=0.08):
        self.h, self.poly, self.kmax, self.size, self.min_dt = holder, np.asarray(poly), kmax, size, min_dt
        lo, hi = self.poly.min(0), self.poly.max(0)
        pad = max(0.08 * max(hi - lo), 1.3 * r)
        self.box = [lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad]
        self.t = 0.0
        try:
            self.font = ImageFont.load_default(size=15)
            self.small = ImageFont.load_default(size=12)
        except TypeError:                               # old Pillow
            self.font = self.small = ImageFont.load_default()

    def _grow(self, Q):
        lo, hi = Q.min(0), Q.max(0)
        b = self.box
        if lo[0] < b[0] or lo[1] < b[1] or hi[0] > b[2] or hi[1] > b[3]:
            w = max(hi[0], b[2]) - min(lo[0], b[0])
            self.box = [min(lo[0], b[0]) - 0.1 * w, min(lo[1], b[1]) - 0.1 * w,
                        max(hi[0], b[2]) + 0.1 * w, max(hi[1], b[3]) + 0.1 * w]

    def __call__(self, it, Q, prev, hist, force=False):
        now = time.time()
        if not force and it > 25 and now - self.t < self.min_dt:
            return
        self.t = now
        self._grow(Q)
        x0, y0, x1, y1 = self.box
        w = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        S = self.size

        def T(P):
            P = np.asarray(P)
            return [((p[0] - (cx - w / 2)) / w * S, S - (p[1] - (cy - w / 2)) / w * S) for p in P]

        img = Image.new("RGB", (S, S), "white")
        d = ImageDraw.Draw(img)
        d.polygon(T(self.poly), fill=(217, 217, 217), outline=(128, 128, 128))
        pts = T(Q)
        d.line(pts + [pts[0]], fill=(38, 38, 38), width=2)
        for x, y in pts:
            d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=(38, 38, 38))
        kap = ps.vertex_curvature(Q)
        for (x, y), b in zip(pts, kap > self.kmax * 1.01):
            if b:
                d.ellipse([x - 3.6, y - 3.6, x + 3.6, y + 3.6], fill=(220, 20, 60))
        note = ""
        if prev is not None and len(prev) == len(Q):
            D = Q - prev
            mag = np.linalg.norm(D, axis=1)
            if mag.max() > 1e-9:
                sc = 0.10 * w / mag.max()
                sgn = (D * outward_normal(prev)).sum(1)
                for i in np.nonzero(mag > 1e-9)[0]:
                    (px, py), = T([Q[i]]); (qx, qy), = T([Q[i] + D[i] * sc])
                    col = (230, 126, 34) if sgn[i] >= 0 else (31, 95, 191)
                    d.line([px, py, qx, qy], fill=col, width=2)
                    a = math.atan2(qy - py, qx - px)
                    d.polygon([(qx, qy), (qx - 6 * math.cos(a - 0.45), qy - 6 * math.sin(a - 0.45)),
                               (qx - 6 * math.cos(a + 0.45), qy - 6 * math.sin(a + 0.45))], fill=col)
                moved = mag > 1e-9
                note = (f"this step: {int((moved & (sgn >= 0)).sum())} outward, "
                        f"{int((moved & (sgn < 0)).sum())} inward")
        d.rectangle([0, 0, S, 46 if note else 26], fill=(255, 255, 255))
        d.text((8, 5), f"running  -  iteration {it}    L = {hist['L'][-1]:.0f}    "
                       f"max k/kmax = {kap.max() / self.kmax:.2f}    vertices = {len(Q)}",
               fill=(27, 42, 65), font=self.font)
        if note:
            d.text((8, 26), note, fill=(95, 107, 122), font=self.small)
        if self.h is not None:
            self.h.image(img, width=S)
        return img


def sim_payload(run, max_frames=400):
    """The run, thinned, as plain lists for the canvas player."""
    F = len(run["frames"])
    if F <= max_frames:
        ids = list(range(F))
    else:
        head = list(range(min(120, F)))
        ids = sorted(set(head + np.linspace(120, F - 1, max_frames - 120).round().astype(int).tolist()))
    kmax, H = run["kmax"], run["hist"]
    frames, disp, its, L, K, V, bad = [], [], [], [], [], [], []
    for k in ids:
        Q = run["frames"][k]
        frames.append(np.round(Q, 2).ravel().tolist())
        D = run["disp"][k]
        disp.append(None if D is None else np.round(D, 4).ravel().tolist())
        its.append(int(run["its"][k]))
        L.append(round(ps.chain_length(Q), 1))
        kap = ps.vertex_curvature(Q)
        K.append(round(float(kap.max() / kmax), 4))
        V.append(int((kap > kmax * 1.01).sum()))
        # red dots from the exact coordinates (the canvas gets them rounded to 0.01, which
        # can push a vertex just under the threshold over it)
        bad.append(np.nonzero(kap > kmax * 1.01)[0].tolist())
    cert = None
    try:
        S, _, _ = ps.fillet_certificate(run["frames"][-1])
        cert = np.round(S, 2).ravel().tolist()
    except Exception:                                   # noqa: BLE001
        pass
    return dict(poly=np.round(run["poly"], 3).ravel().tolist(), frames=frames, disp=disp, its=its,
                L=L, K=K, V=V, bad=bad, kmax=kmax, cert=cert, status=run["res"]["status"])


# =============================================================================
# UI
# =============================================================================
_canvas = components.declare_component("ce_canvas", path=os.path.join(HERE, "ce_canvas"))
_slider = components.declare_component("ce_slider", path=os.path.join(HERE, "ce_slider"))
CANVAS_PX = 600


def slider(label, lo, hi, value, step, key, decimals=2, log=False, help=None, integer=False):
    """Plain left-to-right slider (ce_slider/index.html): grey bar, no coloured fill.
    Streamlit's own slider draws its knob mirrored when the browser language is
    right-to-left (e.g. Hebrew), so it is not used in this app."""
    cur = st.session_state.get(key)
    v = _slider(label=label, min=lo, max=hi, value=value, step=step, decimals=decimals, log=log,
                help=help or "", current=cur, key=key, default=value)
    v = value if v is None else v
    return int(round(v)) if integer else float(v)


class Job:
    """A CE run in a background thread, so that the page stays responsive: the canvas is
    refreshed by a fragment, and the Stop button only sets a flag that the next iteration
    of the run checks."""

    def __init__(self, poly, kmax, l0, r, params, sim_on, stop_clean):
        self.meta = dict(poly=poly, kmax=kmax, l0=l0, r=r, sim_on=sim_on)
        self.rec = Recorder(kmax, stop_when_clean=stop_clean)
        self.view = LiveView(None, poly, kmax, r, size=CANVAS_PX)
        self.T = int(params["max_iter"])
        self.run, self.error, self.done = None, None, False
        self.thread = threading.Thread(target=self._work, args=(poly, kmax, params), daemon=True)
        self.thread.start()

    def _work(self, poly, kmax, params):
        try:
            self.run = run_ce(poly, kmax, params, rec=self.rec)
        except Exception as exc:                        # noqa: BLE001
            self.error = repr(exc)
            if self.rec.last is not None:
                it, Q = self.rec.last
                self.run = self.rec.finalize("error", it, Q)
        finally:
            self.done = True


def store_run(run, meta):
    run.update(poly=meta["poly"], kmax=meta["kmax"], l0=meta["l0"], r=meta["r"])
    run["summary"] = summary(run, meta["poly"], meta["kmax"], meta["l0"])
    ss.run_id += 1
    run["id"] = ss.run_id
    ss.run, ss.view = run, ("sim" if meta["sim_on"] else "draw")
    ss["frame"] = 10 ** 9
    ss.pop("gif", None)


@st.fragment(run_every=0.25)
def live_panel():
    """While a job runs: the live canvas and the progress line; at the end, store the run."""
    job = ss.get("job")
    if job is None:
        return
    if job.done:
        ss.job = None
        if job.run is not None:
            store_run(job.run, job.meta)
        if job.error:
            ss.job_error = job.error
        st.rerun(scope="app")
    last = job.rec.last
    if last is None:
        st.info("starting ...")
        return
    it, Q = last
    H = job.rec.hist
    if job.meta["sim_on"]:
        img = job.view(it, Q, job.rec.prev_of_last, H, force=True)
        st.image(img, width=CANVAS_PX)
    st.progress(min(1.0, it / max(1, job.T)),
                text=f"iteration {it}:  max kappa/kappa_max = {H['K'][-1]:.2f},  violating "
                     f"vertices = {H['viol'][-1]},  L = {H['L'][-1]:.0f}"
                     + ("   (stopping ...)" if job.rec.stop_requested else ""))

PRESETS = {
    "Square (100 x 100)": ("square", 0),
    "Rectangle (200 x 50)": ("rect", 0),
    "Fat-I (200 x 200)": ("fatI", 0),
    "L-shape": ("L", 0),
    "U-shape": ("U", 0),
    "Star (5 points)": ("star5", 0),
    "Random star-shaped (seed 10)": ("random", 10),
    "Random non-star-shaped (seed 2)": ("nonstar", 2),
    "Random orthogonal (seed 1)": ("ortho", 1),
}

ss = st.session_state
ss.setdefault("pts", [])          # the drawn polygon, as last reported by the canvas
ss.setdefault("closed", False)
ss.setdefault("seq", 0)
ss.setdefault("run", None)
ss.setdefault("view", "draw")     # what the canvas shows: "draw" or "sim"
ss.setdefault("run_id", 0)
ss.setdefault("job", None)        # the running CE job, if any
running = ss.job is not None

# --- the drawing canvas reports only close / reopen / clear (no rerun per click) ---
val = ss.get("drawcanvas")
if isinstance(val, dict) and val.get("seq", 0) > ss.seq:
    ss.seq = val["seq"]
    new_pts = [tuple(map(float, p)) for p in val.get("pts", [])]
    if new_pts != ss.pts or bool(val.get("closed")) != ss.closed:
        ss.pts, ss.closed = new_pts, bool(val.get("closed"))
        ss.run, ss.view = None, "draw"

st.title("Curvature-driven evolution of a simple polygon")
st.caption("Draw a simple polygon, set the curvature bound, and run the curvature-driven evolution "
           "(CE) of the paper. With 'simulation' checked, the evolution is shown on the canvas: red "
           "dots are vertices above the bound, orange / blue arrows the outward / inward step.")

with st.sidebar:
    st.header("1  Polygon")
    source = st.radio("Source", ["Draw", "Preset", "Coordinates"], horizontal=True)
    corners, poly_error = None, None
    if source == "Draw":
        snap = st.selectbox("Snap grid", [1.0, 2.0, 2.5, 5.0, 10.0], index=2)
        if ss.closed and len(ss.pts) >= 3:
            ok, why = is_simple_polygon(ss.pts)        # server-side check as well
            if ok:
                corners = np.array(ss.pts, float)
            else:
                poly_error = f"Not a simple polygon: {why}."
    elif source == "Preset":
        snap = 2.5
        name = st.selectbox("Preset polygon", list(PRESETS))
        fam, seed = PRESETS[name]
        corners = np.array(rx.FAMILIES[fam](seed), float)
    else:
        snap = 2.5
        txt = st.text_area("One vertex per line: x, y", height=220,
                           value="0, 0\n200, 0\n200, 100\n150, 100\n150, 40\n50, 40\n50, 100\n0, 100")
        try:
            rows = [ln.replace(";", ",").split(",") for ln in txt.strip().splitlines() if ln.strip()]
            cand = np.array([[float(a), float(b)] for a, b in rows], float)
            ok, why = is_simple_polygon(cand)
            corners = cand if ok else None
            if not ok:
                poly_error = f"Not a simple polygon: {why}."
        except Exception:                               # noqa: BLE001
            poly_error = "Could not parse the coordinates: use one 'x, y' pair per line."
    if poly_error:
        st.error(poly_error)

    st.header("2  Curvature bound")
    r = slider("Minimum turning radius r", 1.0, 200.0, 20.0, 1.0, key="sl_r", decimals=0,
               help="kappa_max = 1/r: no turn of the curve may be sharper than a circle of radius r.")
    l0 = slider("Sampling spacing l_s", 0.5, 20.0, 5.0, 0.5, key="sl_l0", decimals=1,
                help="The polygon boundary is sampled at this spacing; it is also the length "
                     "scale of CE. Keep l_s < r.")
    st.header("3  CE parameters")
    target = slider("Target factor gamma (aim for curvature gamma x kappa_max)", 0.50, 0.99, 0.85, 0.01, key="sl_gamma", help="CE aims at the curvature kappa_t = gamma * kappa_max, a little below the bound, to keep a safety margin for the final arc curve. Paper: 0.85.")
    max_iter = slider("Iteration cap T", 100, 100000, 40000, 100, key="sl_T", decimals=0, log=True,
                      integer=True, help="Logarithmic scale. The paper uses 40,000.")
    with st.expander("Step and solver"):
        omega = slider("Step fraction beta", 0.1, 1.0, 0.5, 0.05, key="sl_beta")
        damping = st.number_input("Damping lambda_0 (lambda = lambda_0 / l_s^2)", 0.001, 10.0, 0.1, 0.01,
                                  format="%.3f")
        cap = slider("Move cap c_max (units of l_s)", 0.05, 1.0, 0.25, 0.05, key="sl_cap")
        tang = slider("Spacing equaliser strength", 0.0, 1.0, 0.25, 0.05, key="sl_tang")
    with st.expander("Refinement, stall, stop test"):
        split = slider("Split edges longer than (x l_s)", 1.05, 3.0, 1.25, 0.05, key="sl_split")
        merge = slider("Merge edges shorter than (x l_s)", 0.1, 0.9, 0.5, 0.05, key="sl_merge")
        stall_iters = st.number_input("Stall window (iterations)", 50, 20000, 1000, 50)
        hold_iters = st.number_input("Hold a rejected vertex for (iterations)", 0, 500, 25, 5)
        resampled_stop = st.checkbox("Stop test on the resampled curve", True)
        certify = st.checkbox("Stop test on the certified curve", True)
        inflate = st.checkbox("Inflate a convex chain on a stall", True)

params = dict(l0=l0, target=target, max_iter=max_iter, omega=omega, damping=damping, cap=cap,
              tang=tang, split=split, merge=merge, stall_iters=stall_iters, hold_iters=hold_iters,
              resampled_stop=resampled_stop, certify=certify, inflate=inflate)
kmax = 1.0 / r
poly = rx._resample(corners, l0) if corners is not None else None


def run_matches(run):
    return (run is not None and poly is not None and run["kmax"] == kmax and
            len(run["poly"]) == len(poly) and np.allclose(run["poly"], poly))


col_draw, col_info = st.columns([1.15, 1])
with col_draw:
    head_ph = st.empty()
    canvas_ph = st.empty()

# ---------------------------------------------------------------- right column
with col_info:
    st.subheader("Polygon and run")
    sim_on = st.checkbox("simulation", value=True,
                         help="Show the evolution on the canvas: live while CE runs, then an "
                              "animated replay (as in the GIF animations).")
    stop_clean = st.checkbox("stop when no vertex violates the bound (no red dots)", value=False,
                             help="End the run at the first iteration in which no vertex has "
                                  "curvature above kappa_max (the red dots), without waiting for "
                                  "the paper's full stop test (resampled and certified curve).")
    if poly is None:
        st.info("Draw a polygon and press 'Close polygon' (or choose a preset) to run CE.")
    else:
        hull_per = ps.chain_length(ps.convex_hull(poly))
        c1, c2, c3 = st.columns(3)
        c1.metric("corners", len(corners)); c2.metric("sampled vertices", len(poly))
        c3.metric("kappa_max", f"{kmax:.4f}")
        c1.metric("per(conv P)", f"{hull_per:.0f}"); c2.metric("2 pi r", f"{2 * math.pi * r:.0f}")
        c3.metric("r / l_s", f"{r / l0:.1f}")
        if l0 >= r:
            st.warning("l_s >= r: the certified curve can fail at sharp corners even when no vertex "
                       "violates the bound. Use a smaller spacing.")
        if 2 * math.pi * r > hull_per:
            st.info("2 pi r > per(conv P): every admissible curve is longer than the hull, so CE "
                    "will have to inflate the curve (slow for large r).")
        b1, b2 = st.columns([2, 1])
        if b1.button("Run curvature-driven evolution", type="primary", width="stretch",
                     disabled=running):
            ss.job = Job(poly, kmax, l0, r, params, sim_on, stop_clean)
            ss.pop("job_error", None)
            st.rerun()
        if b2.button("Stop simulation", width="stretch", disabled=not running,
                     help="Stop now and keep the iterations computed so far."):
            ss.job.rec.stop_requested = True
        if ss.get("job_error"):
            st.error(f"The run ended with an error: {ss.job_error}")

        run = ss.run
        if running:
            st.info("CE is running. The result appears here when it ends or when you press "
                    "'Stop simulation'.")
            run = None
        if run_matches(run):
            S = run["summary"]
            head = f"after {S['iterations']} iterations ({S['seconds']:.1f} s)"
            if S["status"] == "converged":
                st.success(f"**CONVERGED** {head}")
            elif S["status"] == "no violations":
                st.success(f"**STOPPED: no vertex violates the bound** {head}. The paper's full "
                           "stop test (resampled and certified curve) was not applied; see the "
                           "curvatures below.")
            elif S["status"] == "stopped":
                st.warning(f"**STOPPED by the user** {head}. The curve may still violate the bound.")
            else:
                st.error(f"**{S['status'].upper()}** {head} - the bound was not met; the paper's "
                         "method would return the shortest enclosure and report a failure.")
            m1, m2, m3 = st.columns(3)
            m1.metric("length L", f"{S['L']:.0f}", f"{100 * (S['L'] / S['L_hull'] - 1):+.0f}% vs per(conv P)")
            m2.metric("max deviation D", f"{S['D']:.1f}", f"{S['D'] / r:.2f} r", delta_color="off")
            m3.metric("mean deviation", f"{S['Dmean']:.1f}", f"{S['Dmean'] / r:.2f} r", delta_color="off")
            m1.metric("chain kappa/kappa_max", f"{S['K']:.2f}")
            m2.metric("resampled kappa/kappa_max", f"{S['K_res']:.2f}")
            m3.metric("certified kappa/kappa_max", f"{S['K_cert']:.2f}")
            m1.metric("vertices", S["vertices"])
            m2.metric("pocket releases", "-" if S["releases"] is None else S["releases"])
            m3.metric("inflations", "-" if S["inflations"] is None else S["inflations"])
        elif run is not None:
            st.info("The polygon or the bound changed since the last run. Run CE again.")

# ---------------------------------------------------------------- left column: the canvas
show_sim = sim_on and run_matches(ss.run) and ss.view == "sim"
with col_draw:
    if running:
        head_ph.markdown("**Simulation running ...** (press 'Stop simulation' to stop it)"
                         if ss.job.meta["sim_on"] else "**CE is running ...**")
        with canvas_ph.container():
            live_panel()
    elif show_sim:
        if head_ph.button("Back to the polygon", key="back"):
            ss.view = "draw"
            st.rerun()
        with canvas_ph.container():
            _canvas(mode="sim", size=CANVAS_PX, sim=sim_payload(ss.run), sim_key=ss.run["id"],
                    fps=12, key="simcanvas", default=None)
    else:
        if sim_on and run_matches(ss.run):
            if head_ph.button("Show the simulation", key="showsim"):
                ss.view = "sim"
                st.rerun()
        elif source == "Draw":
            head_ph.markdown("**Click to add vertices** (green dot: first vertex; click it or "
                             "press 'Close polygon' to close). Invalid edges show in red and are rejected.")
        else:
            head_ph.markdown(f"**{source} polygon**")
        with canvas_ph.container():
            if source == "Draw":
                _canvas(mode="draw", size=CANVAS_PX, extent=CANVAS, snap=float(snap), readonly=False,
                        init_pts=[list(p) for p in ss.pts], init_closed=ss.closed, init_seq=ss.seq,
                        key="drawcanvas", default=None)
            else:
                _canvas(mode="draw", size=CANVAS_PX, extent=CANVAS, snap=2.5, readonly=True,
                        view_poly=[] if corners is None else corners.tolist(), key="viewcanvas",
                        default=None)

# ---------------------------------------------------------------- detailed replay (GIF style)
run = ss.run
if run_matches(run):
    st.divider()
    title = (f"Curvature-driven evolution  (r = {run['r']:g})  -  "
             f"{run['summary']['status'].upper()} in {run['summary']['iterations']} iterations")
    if not sim_on:
        st.subheader("Replay")
        F = len(run["frames"])
        ss.setdefault("frame", F - 1)
        ss.frame = min(ss.frame, F - 1)
        cA, cB = st.columns([4, 1])
        with cA:
            k = slider("frame", 0, max(1, F - 1), F - 1, 1, key=f"frame_{run['id']}", decimals=0,
                       integer=True,
                       help=f"{F} stored frames; iteration {run['its'][0]} to {run['its'][-1]}")
        k = min(k, F - 1)
        show_cert = cB.checkbox("certified curve", value=True)
        st.image(fig_to_image(draw_frame(run, run["poly"], run["kmax"], run["l0"], k,
                                         show_cert=show_cert and k == F - 1, title=title)),
                 width="stretch")
        st.caption("Red dots: vertices above the bound. Arrows: the step of that iteration, "
                   "magnified (orange outward, blue inward). Certified curve on the last frame.")
    g1, g2 = st.columns([1, 3])
    if g1.button("Make GIF of the run"):
        with st.spinner("rendering frames"):
            ss.gif = make_gif(run, run["poly"], run["kmax"], run["l0"], title)
    if ss.get("gif"):
        g2.download_button("Download GIF", ss.gif, file_name="curvature_driven_evolution.gif",
                           mime="image/gif")
