
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.graph_objs as go   # not used by the code below
except ImportError:
    go = None
import sys
import math
import matplotlib.pyplot as plt
import copy
from matplotlib.animation import FuncAnimation


def is_right_turn(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    return det < 0


def create_square(square_size, points_per_side):
    side1 = np.linspace([0, 0], [square_size, 0], points_per_side, endpoint=False)
    side2 = np.linspace([square_size, 0], [square_size, square_size], points_per_side, endpoint=False)
    side3 = np.linspace([square_size, square_size], [0, square_size], points_per_side, endpoint=False)
    side4 = np.linspace([0, square_size], [0, 0], points_per_side, endpoint=False)

    points = np.vstack((side1, side2, side3, side4))

    return points, False


def create_fat_I_boundary(width, height, thickness):
    numberOgPointsOnEdge1 = 10
    numberOgPointsOnEdge2 = 20
    numberOgPointsOnEdge3 = 50
    numberFactor = 2
    side1 = np.linspace([width, 0], [0, 0], numberOgPointsOnEdge3 * numberFactor, endpoint=False)
    side2 = np.linspace([0, 1], [0, thickness], numberOgPointsOnEdge1 * numberFactor, endpoint=False)
    side3 = np.linspace([1, thickness], [thickness, thickness], numberOgPointsOnEdge1 * numberFactor, endpoint=False)
    side4 = np.linspace([thickness, thickness + 1], [thickness, height - thickness], numberOgPointsOnEdge2 * numberFactor, endpoint=False)
    side5 = np.linspace([thickness - 1, height - thickness], [0, height - thickness], numberOgPointsOnEdge1 * numberFactor, endpoint=False)
    side6 = np.linspace([0, height - thickness + 1], [0, height], numberOgPointsOnEdge1 * numberFactor, endpoint=False)
    side7 = np.linspace([1, height], [width, height], numberOgPointsOnEdge3 * numberFactor, endpoint=False)
    side8 = np.linspace([width, height - 1], [width, height - thickness], numberOgPointsOnEdge1 * numberFactor, endpoint=False)
    side9 = np.linspace([width - 1, height - thickness], [width - thickness, height - thickness], numberOgPointsOnEdge1 * numberFactor, endpoint=False)
    side10 = np.linspace([width - thickness, height - thickness - 1], [width - thickness, thickness], numberOgPointsOnEdge2 * numberFactor, endpoint=False)
    side11 = np.linspace([width - thickness + 1, thickness], [width, thickness], numberOgPointsOnEdge1 * numberFactor, endpoint=False)
    side12 = np.linspace([width, thickness - 1], [width, 0], numberOgPointsOnEdge1 * numberFactor, endpoint=False)

    points = np.vstack((side1, side2, side3, side4, side5, side6, side7, side8, side9, side10, side11, side12))

    return points, True


def vector_length(v):
    return math.sqrt(v[0]**2 + v[1]**2)


def vertex_curvature(points):
    """Absolute discrete curvature at every vertex of a closed chain:
    |turning angle| / mean length of the two adjacent edges.
    It does not depend on the orientation of the chain."""
    points = np.asarray(points, dtype=float)
    prev = points - np.roll(points, 1, axis=0)
    nxt = np.roll(points, -1, axis=0) - points
    cross = prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0]
    dot = (prev * nxt).sum(axis=1)
    turning = np.abs(np.arctan2(cross, dot))
    mean_len = 0.5 * (np.linalg.norm(prev, axis=1) + np.linalg.norm(nxt, axis=1))
    return turning / mean_len


def angle_bisector_point(a, b, c):
    ab = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    ac = math.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)

    ratio = ab / ac

    x_d = (b[0] + ratio * c[0]) / (1 + ratio)
    y_d = (b[1] + ratio * c[1]) / (1 + ratio)

    return [x_d, y_d]


def animate_polygonal_chains(X, isCw, angleFactor, kappaMax=None):
    print("Animating")
    fig, ax = plt.subplots(figsize=(6, 6))

    xMax = max([max([z[0] for z in x]) for x in X])
    yMax = max([max([z[1] for z in x]) for x in X])
    xMin = min([min([z[0] for z in x]) for x in X])
    yMin = min([min([z[1] for z in x]) for x in X])

    ax.set_xlim(xMin - 10, xMax + 10)
    ax.set_ylim(yMin - 10, yMax + 10)
    ax.set_aspect('equal')
    ax.grid(True)
    if kappaMax is not None:
        ax.set_title(f"red: curvature > {kappaMax:g}  (radius < {1 / kappaMax:g})")

    # Draw the original polygon as a fixed blue outline.
    original_points = X[0]
    original_x = np.append(original_points[:, 0], original_points[0, 0])
    original_y = np.append(original_points[:, 1], original_points[0, 1])
    original_line, = ax.plot(
        original_x,
        original_y,
        color='blue',
        linewidth=0.8
    )

    scatter = ax.scatter([], [], s=50)

    # Line plot for connecting points.
    line, = ax.plot([], [], linestyle='-', linewidth=1, color='black')

    def decideColor(points, i):
        if i == 0:
            j = len(points) - 1
        else:
            j = i - 1

        p0 = points[j]
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        angle = calculate_angle(p0, p1, p2, isCw)

        if angle > 3.1415 * angleFactor or angle < 3.1415 / angleFactor:
            return 1, 0, 0
        else:
            return 0, 1, 0

    def update(frame):
        """Update the scatter plot and connecting lines for the current frame."""
        if not hasattr(update, "counter"):
            update.counter = 0

        update.counter += 1
        print(update.counter)

        points = X[frame]
        x, y = points[:, 0], points[:, 1]

        if kappaMax is None:
            # original rule: red if the angle is outside [pi/angleFactor, pi*angleFactor]
            colors = [decideColor(points, i) for i in range(len(points))]
        else:
            # curvature rule: red if the vertex violates the curvature bound
            # (1 % tolerance absorbs the discretisation error of arc samples)
            violates = vertex_curvature(points) > kappaMax * 1.01
            colors = [(1, 0, 0) if v else (0, 1, 0) for v in violates]

        scatter.set_offsets(np.c_[x, y])
        scatter.set_facecolors(colors)

        xList = x.tolist()
        yList = y.tolist()
        xList.append(x[0])
        yList.append(y[0])
        line.set_data(xList, yList)

        return scatter, line, original_line

    ani = FuncAnimation(
        fig,
        update,
        frames=len(X),
        interval=100,
        blit=True
    )

    plt.show()
    return ani


def perpendicular_vector(p0, p1, direction="counterclockwise"):
    v = np.array(p1) - np.array(p0)

    if direction == "counterclockwise":
        perp_vector = np.array([-v[1], v[0]])
    elif direction == "clockwise":
        perp_vector = np.array([v[1], -v[0]])
    else:
        raise ValueError("Direction must be 'counterclockwise' or 'clockwise'")

    return perp_vector


def recursive_copy(lst):
    if not isinstance(lst, list):
        return lst
    return [recursive_copy(item) for item in lst]


def calculate_angle(p1, p2, p3, isCw):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    rt = is_right_turn(p1, p2, p3)
    isWideAngle = rt ^ isCw
    if isWideAngle and angle < 3.1415 or not isWideAngle and angle > 3.1415:
        angle = 2 * 3.1415 - angle

    return angle


def prepare_polygon_simulation(mode, square_size, num_points):
    """Create the initial polygon and prepare the data used by the simulation."""
    points_per_side = num_points // 4

    if mode == 0:
        points, isCw = create_square(square_size, points_per_side)
    elif mode == 1:
        points, isCw = create_fat_I_boundary(200, 200, 50)
    else:
        raise ValueError("mode must be 0 (square) or 1 (I-shape)")

    square_points_sequence = [copy.deepcopy(points)]
    prev_points = recursive_copy(points)
    original_points = recursive_copy(points)

    total_edge_length = 0
    for i in range(len(prev_points)):
        p0 = prev_points[i]
        p1 = prev_points[(i + 1) % len(prev_points)]
        total_edge_length += vector_length(p1 - p0)

    pointAveDis = total_edge_length / len(prev_points)

    return points, isCw, square_points_sequence, prev_points, original_points, pointAveDis


def simulate_polygon_evolution(
    prev_points,
    original_points,
    isCw,
    square_points_sequence,
    pointAveDis,
    angleFactor,
    sufficientAngleFactor,
    powerFactor,
    distanceFactor,
    anglePow,
    distanceFromOriginPower,
    alpha,
    simulation_length
):
    """
    Simulate the evolution of a polygon by gradually moving its vertices.

    The algorithm consists of three main stages in each iteration:

    1. First movement:
       Move vertices according to their interior angles. The goal is
       to reduce deviations from 180 degrees.

    2. Second movement:
       Apply an additional movement based on how far each vertex has
       moved from its original position. Its strength is controlled
       by alpha.

    3. Edge refinement:
       Insert new vertices into edges that have become too long.

    Parameters
    ----------
    prev_points : numpy array
        Coordinates of the polygon vertices at the beginning of the
        current iteration. Shape: (number_of_vertices, 2).

    original_points : numpy array
        Coordinates of the corresponding vertices in the original
        polygon. These points are used to measure displacement.

    isCw : bool
        Indicates the orientation of the polygon:
        True  - clockwise
        False - counterclockwise.

    square_points_sequence : list
        Stores the polygon coordinates at every iteration. This list
        is later used to create the animation.

    pointAveDis : float
        Average edge length of the original polygon. Used as a
        reference for deciding when to insert new vertices.

    angleFactor : float
        Determines the permitted range of interior angles.
        An angle outside the range

            [180 / angleFactor, 180 * angleFactor]

        triggers the first movement mechanism.

    sufficientAngleFactor : float
        Determines whether the polygon still requires movement.
        If all angles are within

            [180 / sufficientAngleFactor,
             180 * sufficientAngleFactor]

        the simulation stops.

    powerFactor : float
        Controls the size of each vertex movement.

    distanceFactor : float
        Determines when an edge is considered too long.
        A new vertex is inserted if its length is greater than or
        equal to distanceFactor * pointAveDis.

    anglePow : float
        Controls how strongly the movement caused by small angles
        depends on the deviation from 180 degrees.

    distanceFromOriginPower : float
        Exponent controlling how strongly the second movement
        mechanism depends on the displacement from the original
        polygon.

    alpha : float
        Controls the strength of the second movement mechanism.
        Setting alpha = 0 disables this mechanism.

    simulation_length : int
        Maximum number of simulation iterations.

    Returns
    -------
    list
        The sequence of polygon states, including the initial
        polygon and the polygon after each iteration.
    """

    # ------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------
    # Each iteration calculates new positions for the polygon
    # vertices based on their current geometry.
    for iteration in range(simulation_length):

        if iteration % 100 == 0:
            print(f"{iteration} out of {simulation_length}")

        # Create an array of movement vectors.
        # Each row corresponds to one polygon vertex.
        moves = np.zeros([len(prev_points), 2])

        # Indicates whether the polygon still requires movement.
        movesOccured = False

        # FIRST MOVEMENT MECHANISM
        # Move vertices according to their interior angles.

        for i in range(len(prev_points)):

            # Get the indices of the previous, current, and next vertices. 
            j = (i - 1) % len(prev_points)
            p0 = prev_points[j]                       # Previous vertex
            p1 = prev_points[i]                       # Current vertex
            p2 = prev_points[(i + 1) % len(prev_points)]  # Next vertex

            # Calculate the interior angle at vertex p1.
            angle = calculate_angle(p0, p1, p2, isCw)

            # Case 1: The angle is too large.
            # Move the current vertex in the direction that tends
            # to reduce its interior angle.
            if angle > 3.1415 * angleFactor:

                # Calculate a vector perpendicular to the edge
                # connecting the previous and next vertices.
                v = perpendicular_vector(p0, p2)

                # Adjust the direction according to the polygon
                # orientation.
                if isCw:
                    v = -v

                # Normalize the vector so that its length is 1.
                # Subtract it from the movement vector of vertex i.
                moves[i] -= v / vector_length(v)

                # At least one vertex requires movement.
                movesOccured = True

            # Case 2: The angle is too small.
            # Instead of moving the current vertex, distribute
            # movement between its two neighboring vertices.
            elif angle < 3.1415 / angleFactor:
                # Calculate the perpendicular direction.
                v = perpendicular_vector(p0, p2)

                # Adjust direction according to polygon orientation.
                if isCw:
                    v = -v

                # Calculate the movement magnitude.
                # Larger deviations from 180 degrees produce
                # stronger movement.
                # anglePow controls how quickly the magnitude grows.
                anglePower = (abs(angle - 3.1415) / 3.1415) ** anglePow

                # Distribute the movement between the previous
                # and next vertices.
                moves[j] -= v / vector_length(v) * anglePower
                moves[(i + 1) % len(prev_points)] -= (
                    v / vector_length(v) * anglePower
                )

            # Check the stopping condition.
            # Even if the angle does not trigger movement under
            # angleFactor, it may still be sufficiently extreme
            # to require another simulation iteration.
            if angle > 3.1415 * sufficientAngleFactor or angle < 3.1415 / sufficientAngleFactor:
                movesOccured = True

        # Calculate displacement from the original polygon.
        # For every vertex, calculate its Euclidean distance from
        # its corresponding original position.
        distanceFromOrigin = np.linalg.norm(
            prev_points - original_points,
            axis=1
        )

        # Find the largest displacement among all vertices.
        # This value is used to normalize the displacement forces
        # in the second movement mechanism.
        maxDistance = np.max(distanceFromOrigin)

        # If no angle requires further movement, stop the simulation.
        if not movesOccured:
            print(f"Stopped after {iteration}")
            break

        # Apply the first movement to all vertices.
        # powerFactor controls the actual step size.
        prev_points = prev_points + moves * powerFactor

        # ========================================================
        # SECOND MOVEMENT MECHANISM
        # Apply an additional movement based on displacement
        # from the original polygon.
        # ========================================================

        if maxDistance > 0:

            # Reset the movement vectors. The second mechanism
            # calculates its own movement independently.
            moves = np.zeros([len(prev_points), 2])

            for i in range(len(prev_points)):

                # Find the previous vertex, taking into account
                # that the polygon is a closed chain.
                j = (i - 1) % len(prev_points)

                p0 = prev_points[j]
                p1 = prev_points[i]
                p2 = prev_points[(i + 1) % len(prev_points)]

                # Recalculate the angle after the first movement.
                angle = calculate_angle(p0, p1, p2, isCw)

                # Determine whether this vertex satisfies the
                # condition for the second movement.
                if (angle > 3.1415 * angleFactor and not isCw) or (
                    angle < 3.1415 * angleFactor and isCw
                ):

                    # Normalize the vertex's displacement.
                    # The resulting value is between 0 and 1.
                    distanceFromOriginForce = (
                        distanceFromOrigin[i] / maxDistance
                    )

                    # Calculate the perpendicular movement direction.
                    v = perpendicular_vector(p0, p2)

                    # Adjust the direction for counterclockwise polygons.
                    if not isCw:
                        v = -v

                    # Calculate the movement magnitude:
                    # - distanceFromOriginForce determines the relative
                    #   displacement of this vertex.
                    # - distanceFromOriginPower controls its influence.
                    # - alpha controls the overall strength of this move.
                    moves[i] -= (
                        v / len(v)
                        * distanceFromOriginForce ** distanceFromOriginPower
                        * alpha
                    )

            # Apply the second movement.
            next_points = prev_points + moves * powerFactor

        else:
            # If no vertex has moved from its original position,
            # there is no displacement-based force to apply.
            next_points = prev_points

        # EDGE REFINEMENT
        # Insert new vertices into edges that have become too long.
        for i in range(len(next_points)):

            # Get the endpoints of the current edge.
            # Modulo ensures that the last vertex is connected
            # to the first vertex.
            p0 = next_points[i]
            p1 = next_points[(i + 1) % len(next_points)]

            # Calculate the current edge length.
            dis = vector_length(p1 - p0)

            # If the edge is too long relative to the original
            # average edge length, split it into two edges.
            if dis >= distanceFactor * pointAveDis:

                # Insert a new vertex at the midpoint of the edge.
                newPoint = (p0 + p1) / 2

                # Also insert the corresponding midpoint in the
                # original polygon. This maintains the correspondence
                # between current and original vertices.
                originalPoint = (
                    original_points[i]
                    + original_points[(i + 1) % len(next_points)]
                ) / 2

                # Insert the new points immediately after vertex i.
                next_points = np.insert(
                    next_points, i + 1, newPoint, axis=0
                )
                original_points = np.insert(
                    original_points, i + 1, originalPoint, axis=0
                )

        # Save the current polygon state.
        # A copy of the current polygon is appended to the sequence
        # so that it can later be displayed in the animation.
        square_points_sequence.append(np.array(next_points))

        # Update prev_points so that the next iteration starts
        # from the polygon generated in this iteration.
        prev_points = next_points

    # Return all polygon states generated during the simulation.
    return square_points_sequence



# ----------------------------------------------------------------------
# Fast engine: vectorised version of simulate_polygon_evolution.
#
# It follows the reference implementation line by line (same constants,
# same movement rules, same stopping rule, same sequential edge
# refinement), but works on whole arrays instead of looping over
# vertices, which makes it one to two orders of magnitude faster. It is
# meant for experiments with many parameter combinations. The reference
# function above is kept for the animation (it stores every frame).
#
# Two options reproduce or repair quirks of the reference second move:
#   pullback_normalize = False : the direction is divided by len(v) = 2
#                                (reference behaviour); True: divided by
#                                its Euclidean norm (intended).
#   pullback_symmetric = False : the vertex condition is
#                                (angle > pi*f and CCW) or (angle < pi*f and CW)
#                                (reference behaviour); True: angle > pi*f
#                                for both orientations (intended).
# The defaults (False, False) reproduce the reference exactly.
# ----------------------------------------------------------------------
_PI = 3.1415   # the reference code uses this rounded constant everywhere


def _interior_angles(Q, isCw):
    """Vectorised calculate_angle for every vertex of the closed chain Q."""
    P0 = np.roll(Q, 1, axis=0)
    P2 = np.roll(Q, -1, axis=0)
    v1 = P0 - Q
    v2 = P2 - Q
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    n1 = np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2)
    n2 = np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)
    ang = np.arccos(np.clip(dot / (n1 * n2), -1.0, 1.0))
    det = (Q[:, 0] - P0[:, 0]) * (P2[:, 1] - P0[:, 1]) - (Q[:, 1] - P0[:, 1]) * (P2[:, 0] - P0[:, 0])
    wide = (det < 0) ^ bool(isCw)
    flip = (wide & (ang < _PI)) | (~wide & (ang > _PI))
    return np.where(flip, 2 * _PI - ang, ang)


def _chord_perp(Q, isCw_sign_flip):
    """perpendicular_vector(p_prev, p_next) for every vertex (counter-clockwise
    rotation of the chord), optionally negated."""
    chord = np.roll(Q, -1, axis=0) - np.roll(Q, 1, axis=0)
    v = np.stack([-chord[:, 1], chord[:, 0]], axis=1)
    return -v if isCw_sign_flip else v


def simulate_polygon_evolution_fast(
    points,
    isCw,
    angleFactor,
    sufficientAngleFactor,
    powerFactor,
    distanceFactor,
    anglePow,
    distanceFromOriginPower,
    alpha,
    simulation_length,
    pointAveDis=None,
    original_points=None,
    pullback_normalize=False,
    pullback_symmetric=False,
    snapshot_iters=(),
    callback=None,
    callback_every=0,
    max_vertices=50000,
):
    """Vectorised evolution. Returns a dict with
        final      - final chain (array)
        iterations - number of iterations performed
        status     - 'converged' | 'max_iter' | 'diverged'
        frames     - {iteration: chain} for the requested snapshot_iters
        original   - the anchor array that matches `final` (same length)
    callback(iteration, chain) is called every `callback_every` iterations.
    """
    Q = np.array(points, dtype=float)
    O = np.array(points if original_points is None else original_points, dtype=float)
    if pointAveDis is None:
        pointAveDis = float(np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1).mean())
    snap = set(snapshot_iters)
    frames = {}
    status = "max_iter"
    iterations = simulation_length
    thr = distanceFactor * pointAveDis

    for iteration in range(simulation_length):
        if iteration in snap:
            frames[iteration] = Q.copy()
        if callback is not None and callback_every and iteration % callback_every == 0:
            callback(iteration, Q)

        angle = _interior_angles(Q, isCw)

        # ---- first movement ------------------------------------------
        big = angle > _PI * angleFactor
        small = (~big) & (angle < _PI / angleFactor)
        v = _chord_perp(Q, isCw)
        vn = v / np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)[:, None]
        moves = np.zeros_like(Q)
        moves[big] -= vn[big]
        if small.any():
            ap = (np.abs(angle - _PI) / _PI) ** anglePow
            contrib = np.where(small[:, None], vn * ap[:, None], 0.0)
            moves -= np.roll(contrib, -1, axis=0)    # vertex j = i-1 receives from i
            moves -= np.roll(contrib, 1, axis=0)     # vertex i+1 receives from i

        moves_occured = bool(big.any()) or bool(
            ((angle > _PI * sufficientAngleFactor) | (angle < _PI / sufficientAngleFactor)).any()
        )
        dist0 = np.linalg.norm(Q - O, axis=1)
        max_dist = dist0.max()

        if not moves_occured:
            status = "converged"
            iterations = iteration
            break

        Q = Q + moves * powerFactor

        # ---- second movement (displacement-weighted pull-back) --------
        if alpha != 0 and max_dist > 0:
            angle2 = _interior_angles(Q, isCw)
            if pullback_symmetric:
                cond = angle2 > _PI * angleFactor
            else:
                cond = ((angle2 > _PI * angleFactor) & (not isCw)) | ((angle2 < _PI * angleFactor) & bool(isCw))
            if cond.any():
                force = dist0 / max_dist
                v2 = _chord_perp(Q, not isCw)
                norm = np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2) if pullback_normalize else 2.0
                step = (v2 / (norm[:, None] if pullback_normalize else norm)) \
                    * ((force ** distanceFromOriginPower) * alpha)[:, None]
                moves2 = np.where(cond[:, None], -step, 0.0)
                Q = Q + moves2 * powerFactor

        # ---- edge refinement (sequential, exactly as in the reference) --
        edge = np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)
        if np.any(edge >= thr):
            for i in range(len(Q)):               # range is fixed at the start
                p0 = Q[i]
                p1 = Q[(i + 1) % len(Q)]
                if math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) >= thr:
                    new_point = (p0 + p1) / 2
                    orig_point = (O[i] + O[(i + 1) % len(Q)]) / 2
                    Q = np.insert(Q, i + 1, new_point, axis=0)
                    O = np.insert(O, i + 1, orig_point, axis=0)

        if len(Q) > max_vertices or not np.isfinite(Q).all():
            status = "diverged"
            iterations = iteration + 1
            break

    return {"final": Q, "iterations": iterations, "status": status,
            "frames": frames, "original": O}

# ----------------------------------------------------------------------
# Hull-plus-disk baseline
#
# The boundary of conv(P) (+) disk(r), with r = 1 / kappa_max, is a valid
# enclosure of P for the curvature bound kappa_max:
#   * a straight segment for every convex-hull edge (curvature 0), and
#   * a circular arc of radius r for every convex-hull vertex (curvature 1/r).
# Its length is exactly  perimeter(conv(P)) + 2*pi*r,  and no enclosure is
# shorter than perimeter(conv(P)), so it is within 2*pi*r of optimal (min-sum).
# ----------------------------------------------------------------------
def convex_hull(points):
    """Andrew's monotone chain. Hull vertices in counter-clockwise order."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        # orientation, minus a relative tolerance: nearly collinear points (e.g. the
        # resampled points of a long polygon edge) are dropped from the hull
        c = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        scale = math.hypot(a[0] - o[0], a[1] - o[1]) * math.hypot(b[0] - o[0], b[1] - o[1])
        return c - 1e-9 * scale

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def densify(chain, spacing):
    """Resample a closed polyline so that consecutive vertices are at most
    `spacing` apart (original vertices are kept)."""
    out = []
    for a, b in zip(chain, np.roll(chain, -1, axis=0)):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / spacing)))
        t = np.arange(n) / n
        out.append(a + t[:, None] * (b - a))
    return np.vstack(out)


def hull_disk_chain(points, r, arc_step_deg=3.0):
    """Boundary of conv(points) (+) disk(r) as a closed counter-clockwise
    polyline. For r = 0 it is the convex hull itself."""
    hull = convex_hull(points)
    if r <= 0:
        return hull
    k = len(hull)
    step = math.radians(arc_step_deg)
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / lengths[:, None]
    ang = np.arctan2(normals[:, 1], normals[:, 0])

    out = []
    for i in range(k):
        j = (i + 1) % k
        out.append(hull[i] + r * normals[i])        # start of the straight piece
        out.append(hull[j] + r * normals[i])        # end of the straight piece
        # exterior angle at hull[j], in [0, pi): a nearly collinear hull vertex can give
        # a tiny NEGATIVE turn by round-off, and "% (2*pi)" would turn it into a full
        # circle; atan2 of the two normals, clamped at 0, avoids that
        turn = math.atan2(normals[i, 0] * normals[j, 1] - normals[i, 1] * normals[j, 0],
                          float(normals[i] @ normals[j]))
        turn = max(turn, 0.0)
        m = max(1, int(math.ceil(turn / step)))
        for s in range(1, m):                       # samples of the arc around hull[j]
            a = ang[i] + turn * s / m
            out.append(hull[j] + r * np.array([math.cos(a), math.sin(a)]))
    return np.array(out)


def bl_hull_enclosure(points, r, arc_step_deg=0.25, tol=1e-10):
    """Shortest enclosure of bounded curvature (Boissonnat and Lazard, CCCG 1996).

    The shortest closed curve with curvature at most 1/r that encloses `points` is
    convex (their Lemma 1) and depends only on the vertices h_1..h_k of the convex hull.
    Its region is Q* (+) B_r, where Q* is a polygon of minimal perimeter that meets
    every disk D(h_j, r) (Proposition 3), and Q* can be taken as the polygon
    M_1..M_k with M_j in D(h_j, r), in hull order (Proposition 7). Hence
        L* = min sum_j |M_{j+1} - M_j|  subject to  |M_j - h_j| <= r,   plus 2*pi*r,
    a convex problem (a sum of norms over disks), solved here with SLSQP.
    If all disks share a point, Q* is a point and the curve is a circle of radius r.

    Returns (chain, L_star, M): the boundary of conv(M) (+) B_r as a ccw polyline
    (arcs sampled every `arc_step_deg` degrees, fine enough that hull vertices lying
    on an arc are within ~1e-5 r of the polyline), the exact length per(conv M) + 2 pi r,
    and the optimal points M (k x 2)."""
    from scipy.optimize import minimize
    H = convex_hull(points)
    k = len(H)
    c = H.mean(axis=0)
    # feasible start: every M_j halfway (r/2) from h_j towards the centroid
    d = c - H
    nd = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    x0 = (H + 0.5 * r * np.minimum(1.0, nd / r)[:, None] * d / nd[:, None]).ravel()
    eps = 1e-9 * max(1.0, r)

    def f(x):
        M = x.reshape(k, 2)
        e = np.roll(M, -1, axis=0) - M
        n = np.sqrt((e ** 2).sum(1) + eps ** 2)
        g = np.zeros_like(M)
        u = e / n[:, None]
        g -= u                           # d|M_{j+1}-M_j| / dM_j
        g += np.roll(u, 1, axis=0)       # d|M_j-M_{j-1}| / dM_j
        return float(n.sum()), g.ravel()

    def cons(x):
        M = x.reshape(k, 2)
        return r * r - ((M - H) ** 2).sum(1)

    def cons_jac(x):
        M = x.reshape(k, 2)
        J = np.zeros((k, 2 * k))
        for j in range(k):
            J[j, 2 * j:2 * j + 2] = -2.0 * (M[j] - H[j])
        return J

    res = minimize(f, x0, jac=True, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   options={"maxiter": 2000, "ftol": tol})
    M = res.x.reshape(k, 2)
    # project back into the disks (SLSQP may end a hair outside)
    dv = M - H
    dn = np.linalg.norm(dv, axis=1)
    over = dn > r
    M[over] = H[over] + dv[over] * (r / dn[over])[:, None]

    Q = convex_hull(M)
    span = np.ptp(M, axis=0).max() if len(M) else 0.0
    if len(Q) <= 1 or span < 1e-9 * max(1.0, r):
        cen = M.mean(axis=0)
        m = int(math.ceil(360.0 / arc_step_deg))
        a = np.linspace(0.0, 2 * math.pi, m, endpoint=False)
        chain = cen + r * np.stack([np.cos(a), np.sin(a)], axis=1)
        return chain, 2 * math.pi * r, M
    perQ = float(np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1).sum()) if len(Q) > 2 \
        else 2.0 * float(np.linalg.norm(Q[1] - Q[0]))
    chain = hull_disk_chain(Q, r, arc_step_deg)
    return chain, perQ + 2 * math.pi * r, M


def morphological_enclosure(points, r, h=None):
    """Boundary of Y = (P (+) B_{2r}) (-) B_r = closing by B_r of the offset P (+) B_r,
    the classical r-regularisation of mathematical morphology (opening/closing by a
    disk). Y contains P (+) B_r, its boundary consists of convex arcs of radius >= r
    (inner parallel of the 2r-offset) and concave arcs of radius >= r (the closing), so
    it is a candidate admissible enclosure with D >= r. Computed on a raster of pixel
    size h (default r/40, at most 0.5) by exact Euclidean distance transforms and a
    sub-pixel contour; returns a counter-clockwise closed polyline."""
    from scipy import ndimage as _nd
    from contourpy import contour_generator as _cg
    poly = np.asarray(points, dtype=float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1].copy()
    if h is None:
        h = min(0.5, r / 40.0)
    lo = poly.min(0) - 2 * r - 4 * h
    hi = poly.max(0) + 2 * r + 4 * h
    nx, ny = int(math.ceil((hi[0] - lo[0]) / h)) + 1, int(math.ceil((hi[1] - lo[1]) / h)) + 1
    xs, ys = lo[0] + h * np.arange(nx), lo[1] + h * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.c_[X.ravel(), Y.ravel()]
    inside = _MplPath(np.vstack([poly, poly[:1]])).contains_points(pts).reshape(X.shape)

    def seg_dist(P, A, B):                                   # exact distance to segments A->B
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-300)
        out = np.empty(len(P))
        step = max(1, 2_000_000 // max(1, len(A)))
        for s in range(0, len(P), step):
            c = P[s:s + step]
            t = np.clip(((c[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
            out[s:s + step] = np.linalg.norm(c[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2).min(1)
        return out

    # distance to P: raster EDT, made exact in a band around the level 2r (so that the
    # contour of the level is sub-pixel accurate and has no staircase)
    d_out = _nd.distance_transform_edt(~inside) * h
    band = np.abs(d_out - 2 * r) < 4 * h
    d_out.ravel()[band.ravel()] = seg_dist(pts[band.ravel()], poly, np.roll(poly, -1, axis=0))
    Z = d_out <= 2 * r                                       # P (+) B_2r
    zl = [l[:-1] if np.allclose(l[0], l[-1]) else l for l in _cg(xs, ys, d_out).lines(2 * r)]
    A_z = np.vstack([l for l in zl])                         # boundary of Z as closed polylines
    B_z = np.vstack([np.roll(l, -1, axis=0) for l in zl])
    # distance to the complement of Z, exact (to the boundary polylines) near the level r
    d_in = _nd.distance_transform_edt(Z) * h
    band = Z & (np.abs(d_in - r) < 4 * h)
    d_in.ravel()[band.ravel()] = seg_dist(pts[band.ravel()], A_z, B_z)
    lines = _cg(xs, ys, d_in).lines(r)                       # boundary of Z (-) B_r
    C = max(lines, key=len)
    if np.allclose(C[0], C[-1]):
        C = C[:-1]
    a = 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])
    return C if a > 0 else C[::-1].copy()


def offset_smooth_enclosure(points, r, d, h=None):
    """Offset-and-smooth baseline (the usual practitioner pipeline): offset P by a
    standoff d, then round the result to radius r on both sides. In exact form this is
    the closing followed by the opening by B_r of the offset,

        Y_d = ((P (+) B_d) . B_r) o B_r = ((P (+) B_{d+r}) (-) B_{2r}) (+) B_r ,

    because the closing of P (+) B_d by B_r is (P (+) B_{d+r}) (-) B_r and erosions by
    disks compose. Y_d is a union of disks of radius r whose complement is open under
    B_r too, so its boundary has curvature at most 1/r on both sides (Walther); unlike
    the morphological closing (the case d = r), the opening may cut into P where the
    d-offset turns tighter than r (sharp convex corners, thin spikes), so enclosure is
    NOT guaranteed for d < r. Computed on a raster of pixel size h (default r/40, at
    most 0.5) with exact distances near every contour level, as in
    morphological_enclosure.

    Returns (C, n_components): C is the boundary of the largest component of Y_d as a
    counter-clockwise closed polyline, or None if Y_d is empty (P too thin for a disk
    of radius 2r inside P (+) B_{d+r}); n_components counts the closed boundary lines of
    Y_d with non-negligible area (outer boundaries and holes), so n_components > 1 means
    that the pipeline did not produce a single simple enclosing curve."""
    from scipy import ndimage as _nd
    from contourpy import contour_generator as _cg
    poly = np.asarray(points, dtype=float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1].copy()
    if h is None:
        h = min(0.5, r / 40.0)
    R1 = d + r
    lo = poly.min(0) - R1 - 4 * h
    hi = poly.max(0) + R1 + 4 * h
    nx, ny = int(math.ceil((hi[0] - lo[0]) / h)) + 1, int(math.ceil((hi[1] - lo[1]) / h)) + 1
    xs, ys = lo[0] + h * np.arange(nx), lo[1] + h * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.c_[X.ravel(), Y.ravel()]
    inside = _MplPath(np.vstack([poly, poly[:1]])).contains_points(pts).reshape(X.shape)

    def seg_dist(P, A, B):                                   # exact distance to segments A->B
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-300)
        out = np.empty(len(P))
        step = max(1, 2_000_000 // max(1, len(A)))
        for s in range(0, len(P), step):
            c = P[s:s + step]
            t = np.clip(((c[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
            out[s:s + step] = np.linalg.norm(c[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2).min(1)
        return out

    def closed_lines(field, level):
        ls = [l[:-1] if np.allclose(l[0], l[-1]) else l for l in _cg(xs, ys, field).lines(level)]
        return [l for l in ls if len(l) >= 3]

    def stack(ls):
        return np.vstack(ls), np.vstack([np.roll(l, -1, axis=0) for l in ls])

    # A = P (+) B_{d+r}: distance to P, exact near the level d + r
    d_out = _nd.distance_transform_edt(~inside) * h
    band = np.abs(d_out - R1) < 4 * h
    d_out.ravel()[band.ravel()] = seg_dist(pts[band.ravel()], poly, np.roll(poly, -1, axis=0))
    A = d_out <= R1
    la = closed_lines(d_out, R1)
    if not la:
        return None, 0
    Aa, Ba = stack(la)
    # W = A (-) B_{2r}: distance to the complement of A, exact near the level 2r
    d_in = _nd.distance_transform_edt(A) * h
    band = A & (np.abs(d_in - 2 * r) < 4 * h)
    d_in.ravel()[band.ravel()] = seg_dist(pts[band.ravel()], Aa, Ba)
    W = d_in >= 2 * r
    lw = closed_lines(d_in, 2 * r)
    if not W.any() or not lw:
        return None, 0
    Aw, Bw = stack(lw)
    # Y = W (+) B_r: distance to W, exact near the level r
    d_w = _nd.distance_transform_edt(~W) * h
    band = (~W) & (np.abs(d_w - r) < 4 * h)
    d_w.ravel()[band.ravel()] = seg_dist(pts[band.ravel()], Aw, Bw)
    lines = closed_lines(d_w, r)
    if not lines:
        return None, 0

    def signed_area(C):
        return 0.5 * np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1])

    areas = [abs(signed_area(l)) for l in lines]
    n_comp = sum(1 for a_ in areas if a_ > 4 * h * h)
    C = lines[int(np.argmax(areas))]
    return (C if signed_area(C) > 0 else C[::-1].copy()), n_comp


def simulate_hull_disk(points, isCw, r, num_frames, pointAveDis, arc_step_deg=3.0):
    """Build the sequence of states used by animate_polygonal_chains.

    Frame 0 is the original polygon; the following frames are the
    hull-plus-disk enclosures with radius growing from 0 (the convex hull)
    to r.  All frames are counter-clockwise, so display them with isCw=False.
    """
    original = np.asarray(points, dtype=float)
    if isCw:                                        # make the sequence counter-clockwise
        original = original[::-1].copy()
    sequence = [original]
    for radius in np.linspace(0.0, r, num_frames):
        chain = hull_disk_chain(original, radius, arc_step_deg)
        sequence.append(densify(chain, pointAveDis))
    return sequence



# ----------------------------------------------------------------------
# Shrink-wrap variant (curvature-capped curve shortening).
#
# Start from the hull-disk curve of a LARGER radius (start_factor * r), which
# is enclosing and has curvature 1/(start_factor*r) < 1/r, and shorten it:
# every vertex moves towards the midpoint of its two neighbours, but only if
# afterwards (a) the vertex and both adjacent chords stay outside the polygon
# and (b) the discrete curvature at the vertex and its two neighbours stays
# <= slack/r.  Vertices are processed in three interleaved classes (i mod 3),
# so the checks of simultaneously moving vertices never interact.  (Superseded:
# the update is now simultaneous (Jacobi): all vertices propose a step at once,
# violated neighbourhoods halve their steps until the whole candidate curve is
# valid, which makes the result independent of the vertex numbering.)  Because a
# move is accepted only if it keeps all constraints, EVERY iterate is
# enclosing, simple (in practice) and curvature-bounded, so the corner lobes
# of the angle-driven evolution cannot appear.  A convex-corner arc is
# already tight at radius r, so the final curve is never shorter than the
# lower bound, and in pockets it is shorter than the hull-disk of radius r.
# ----------------------------------------------------------------------
from matplotlib.path import Path as _MplPath


def shrink_wrap_enclosure(polygon, r, start_factor=2.5, slack=0.92, step=0.5,
                          max_iter=8000, tol=1e-6, margin=1e-3, l0=None, start=None,
                          stages=12, iters_per_stage=400):
    """Returns (chain, iterations). `polygon`: (n,2) array. Result is counter-clockwise."""
    poly = np.asarray(polygon, dtype=float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1].copy()
    if l0 is None:
        l0 = float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).mean())
    kmax_box = [slack / (start_factor * r)]          # current curvature cap (annealed)
    if start is None:
        # uniform arclength resampling: no short edges at the arc/line joins, so the start
        # curve has curvature ~ 1/(start_factor*r) everywhere (densify() would put 3-degree
        # turns on 1-unit edges there, i.e. a discrete curvature far above the cap)
        H = hull_disk_chain(poly, start_factor * r, 3.0)
        Hc = np.vstack([H, H[:1]])
        sl = np.linalg.norm(np.diff(Hc, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(sl)])
        nn = max(16, int(math.ceil(cum[-1] / l0)))
        tt = np.linspace(0.0, cum[-1], nn, endpoint=False)
        Q = np.stack([np.interp(tt, cum, Hc[:, 0]), np.interp(tt, cum, Hc[:, 1])], axis=1)
    else:
        Q = np.array(start, dtype=float)
    A, B = poly, np.roll(poly, -1, axis=0)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(axis=1), 1e-300)
    poly_path = _MplPath(np.vstack([poly, poly[:1]]))

    def orient(p, q, t):
        return (q[..., 0] - p[..., 0]) * (t[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (t[..., 0] - p[..., 0])

    def dist_poly(P):
        ap = P[:, None, :] - A[None]
        t = np.clip((ap * AB[None]).sum(2) / L2[None], 0, 1)
        return np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2).min(1)

    def chords_clear(P1, P2):
        a, b, c, d = P1[:, None, :], P2[:, None, :], A[None], B[None]
        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        return ~(((o1 * o2) < 0) & ((o3 * o4) < 0)).any(1)

    def kappa_ok(Pm, P, Pp):
        e1, e2 = P - Pm, Pp - P
        l1, l2 = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
        turn = np.abs(np.arctan2(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0], (e1 * e2).sum(1)))
        return turn <= kmax_box[0] * 0.5 * (l1 + l2)

    def violations(C):
        """Boolean array: vertex j is in violation (inside / too close to the polygon, an adjacent
        chord crosses the polygon, or the discrete curvature at j exceeds the cap)."""
        m = len(C)
        Cm, Cp = np.roll(C, 1, axis=0), np.roll(C, -1, axis=0)
        bad = poly_path.contains_points(C) | (dist_poly(C) < margin)
        bad |= ~chords_clear(C, Cp)                  # chord j -> j+1
        bad |= ~chords_clear(Cm, C)                  # chord j-1 -> j
        bad |= ~kappa_ok(Cm, C, Cp)
        return bad

    def run(Q, n_iter):
      it = 0
      for it in range(1, n_iter + 1):
          m = len(Q)
          target = 0.5 * (np.roll(Q, 1, axis=0) + np.roll(Q, -1, axis=0))
          scale = np.ones(m)                           # per-vertex fraction of the step
          for _ in range(12):
              C = Q + (step * scale)[:, None] * (target - Q)
              bad = violations(C)
              moving = scale > 0
              near = np.zeros(m, bool)
              for d in (-2, -1, 0, 1, 2):
                  near |= np.roll(moving, d)
              bad &= near      # a contact vertex already on the polygon is a pre-existing state, not a violation
              if not bad.any():
                  break
              # a violation at j is caused by moving j-2 .. j+2: shrink all of them (symmetric rule)
              culprit = np.zeros(m, bool)
              for d in (-2, -1, 0, 1, 2):
                  culprit |= np.roll(bad, d)
              scale = np.where(culprit & (scale > 0), scale * 0.5, scale)
              scale[scale < 1.0 / 64] = 0.0
          else:
              C = Q                                     # could not repair: take no step
          moved = float(np.linalg.norm(C - Q, axis=1).sum())
          Q = C
          if moved < tol:
              break
      return Q, it

    # continuation: tighten the curvature cap from slack/(start_factor*r) to slack/r in `stages`
    # geometric steps; each stage shortens the curve until it stalls, which un-jams the arcs.
    total = 0
    e1, e2 = Q - np.roll(Q, 1, axis=0), np.roll(Q, -1, axis=0) - Q
    k0 = float((np.abs(np.arctan2(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0], (e1 * e2).sum(1)))
                / (0.5 * (np.linalg.norm(e1, axis=1) + np.linalg.norm(e2, axis=1)))).max())
    k0 = max(k0 * 1.001, slack / (start_factor * r))    # the start curve itself must be feasible
    caps = [k0 * ((slack / r / k0) ** (k / max(1, stages))) for k in range(stages + 1)]
    for k, cap in enumerate(caps):
        kmax_box[0] = cap
        last = (k == len(caps) - 1)
        Q, n = run(Q, max_iter - total if last else iters_per_stage)
        total += n
        if total >= max_iter:
            break
    it = total
    return Q, it


# ----------------------------------------------------------------------
# Curvature-driven evolution (repair of the angle-driven evolution).
#
# Why the angle-driven evolution cannot meet the bound
# ----------------------------------------------------
# (1) It terminates on an angle band [pi/f_s, pi*f_s]; that band, not kappa_max,
#     decides the final curvature (about 2.8x the bound in the reference setup).
# (2) Its vertices only ever move OUTWARD. Write theta_i for the turning angle at
#     vertex i. Moving vertex j outward by d changes theta_j by +2d/l and
#     theta_{j+-1} by -d/l. For any outward-only pattern (own step a >= 0,
#     neighbour step b >= 0) the response of the turning angles to a smooth excess
#     with wave number w is (a + 2b) w^2 + O(w^4) > 0: broad convex bumps get
#     SHARPER. Outward-only rules are therefore anti-diffusive (unstable); this
#     is why the corner rule leaves lobes and why local patches do not converge.
#
# The repair
# ----------
#   * termination on discrete curvature: stop when |theta_i| / min(l_i, l0) <= kappa_max
#     at every vertex AND the curve resampled at spacing l0 satisfies the same bound
#     (the scale at which the experiments measure it);
#   * the motion minimises the curvature excess
#         E = sum_i max(|theta_i| - kappa_t * l_i, 0)^2 ,  kappa_t = target * kappa_max,
#     two-sided: convex violators may move INWARD, concave ones OUTWARD, so the
#     flow is diffusive and cannot inflate. Only normal displacements are used (the
#     tangential part of the gradient bunches vertices at corners, the origin of the
#     knobs); a small tangential term keeps the spacing even, except at contact
#     vertices (on P), which must not slide off a corner of P;
#   * default solver "lm": each iteration solves a damped Gauss-Newton problem for
#     the normal displacements u of all vertices coupled to an active excess,
#         min |J u + r|^2 + lambda |u|^2   s.t.   u_j >= lower_j ,
#     by an exact primal active-set method on the sparse (pentadiagonal) normal
#     equations, and takes the fraction omega of it. lower_j is the largest inward
#     move that keeps vertex j AND both its chords clear of P (a short line search);
#     vertices whose step the projection had to drop are held (lower = 0) for a few
#     iterations. A convex corner of P therefore pins the curve and the solver moves
#     the free neighbours outward instead -- this replaces the old sharp-corner rule
#     and builds an arc of radius >= r through the corner;
#     solver "gradient": explicit projected gradient descent, tau = eta * l0^2 with
#     eta < 1/16 (largest Hessian eigenvalue ~ 32 / l^2, zig-zag mode) -- correct but
#     slow, the turning only spreads diffusively;
#   * sharp-vertex term: at a near-reversal (|theta| > sharp_angle, e.g. the tip of a
#     narrow notch) d theta / d p_i vanishes; a Laplacian step towards the
#     neighbours' midpoint is added there (and tried alone if the full step fails);
#   * the polygon is a hard obstacle: every candidate step is checked (vertex
#     outside P, adjacent chords not crossing P, with precise blame of the endpoint
#     that causes a crossing); rejected steps are halved, then dropped, and the loop
#     only accepts a candidate that passes, so every iterate encloses P;
#   * pocket release: if E has not decreased by 2 % in stall_iters iterations (the
#     curve sits in a pocket it cannot leave with bounded curvature), the stretch
#     around the worst vertex is replaced by the curve's own convex-hull edge
#     (an outward-only change); in the worst case the problem becomes convex;
#   * long edges are split, short edges are merged when the replacing chord is clear.
# Simplicity and the curvature bound are checked independently in run_experiments.py
# (self-intersection is not prevented by construction, only measured).
# ----------------------------------------------------------------------
class _PolygonObstacle:
    def __init__(self, poly, margin=1e-6):
        self.A = np.asarray(poly, dtype=float)
        self.B = np.roll(self.A, -1, axis=0)
        self.AB = self.B - self.A
        self.L2 = np.maximum((self.AB ** 2).sum(1), 1e-300)
        self.path = _MplPath(np.vstack([self.A, self.A[:1]]))
        self.margin = margin
        self.strict_rejections = 0     # chords rejected only by the interior test (diagnostics)

    def dist(self, P):
        ap = P[:, None, :] - self.A[None]
        t = np.clip((ap * self.AB[None]).sum(2) / self.L2[None], 0, 1)
        return np.linalg.norm(P[:, None, :] - (self.A[None] + t[..., None] * self.AB[None]), axis=2).min(1)

    def chords_clear(self, P1, P2):
        """True where the segment P1->P2 does not meet the interior of P.

        Two tests. (1) No proper crossing with an edge of P; points within `margin` of a
        line count as on it, so that round-off on vertices that slide along a polygon edge
        is not a crossing. (2) A chord can still enter the interior of P without any
        proper crossing, through contacts: endpoints on the boundary, a vertex of P on the
        chord, or an overlap with an edge (e.g. a chord from (10,9) to (9,10) cuts the
        corner of the square [0,10]^2). For every chord that touches the boundary, all its
        contact parameters with the boundary are collected; between two consecutive
        contacts the chord lies entirely inside or entirely outside P, so the midpoint of
        each piece decides."""
        def sdist(p, q, t):        # signed distance of t from the line p->q
            L = np.maximum(np.hypot(q[..., 0] - p[..., 0], q[..., 1] - p[..., 1]), 1e-300)
            return ((q[..., 0] - p[..., 0]) * (t[..., 1] - p[..., 1])
                    - (q[..., 1] - p[..., 1]) * (t[..., 0] - p[..., 0])) / L
        P1 = np.asarray(P1, dtype=float)
        P2 = np.asarray(P2, dtype=float)
        a, b, c, d = P1[:, None, :], P2[:, None, :], self.A[None], self.B[None]
        tol = self.margin
        o1, o2, o3, o4 = sdist(a, b, c), sdist(a, b, d), sdist(c, d, a), sdist(c, d, b)
        hit = (((o1 < -tol) & (o2 > tol)) | ((o1 > tol) & (o2 < -tol))) & \
              (((o3 < -tol) & (o4 > tol)) | ((o3 > tol) & (o4 < -tol)))
        clear = ~hit.any(1)
        if not clear.any():
            return clear
        # (2) chords that touch the boundary: an endpoint on an edge, or a vertex of P on
        # the chord (generous tolerance for the preselection; the exact test follows)
        tol2 = 10.0 * tol
        seg = P2 - P1
        L2 = np.maximum((seg ** 2).sum(1), 1e-300)
        Lc = np.sqrt(L2)
        tv_raw = ((self.A[None] - P1[:, None]) * seg[:, None]).sum(2) / L2[:, None]
        tv = np.clip(tv_raw, 0.0, 1.0)
        dvv = np.linalg.norm(self.A[None] - (P1[:, None] + tv[..., None] * seg[:, None]), axis=2)
        # a vertex of P on the chord, strictly between its endpoints
        eps = tol2 / Lc[:, None]
        inner = ((dvv <= tol2) & (tv_raw > eps) & (tv_raw < 1 - eps)).any(1)
        ends = (self.dist(P1) <= tol2) | (self.dist(P2) <= tol2)
        # fast case: contacts only at the endpoints, so the open chord lies entirely inside
        # or entirely outside P, and its midpoint decides
        fast = clear & ends & ~inner
        if fast.any():
            k = np.nonzero(fast)[0]
            bad = self.inside(0.5 * (P1[k] + P2[k]))
            clear[k[bad]] = False
            self.strict_rejections += int(bad.sum())
        # general case: a vertex of P inside the chord -> all contacts, exact test
        for k in np.nonzero(clear & inner)[0]:
            if not self._chord_clear_exact(P1[k], P2[k], tol2):
                clear[k] = False
                self.strict_rejections += 1
        return clear

    def _chord_clear_exact(self, p, q, tol):
        """Exact test of one chord p->q (no proper crossing assumed): the midpoints of the
        pieces between consecutive boundary contacts must not lie in the interior of P."""
        dvec = q - p
        Ld = float(np.hypot(*dvec))
        if Ld <= 1e-300:
            return not self.inside(p[None])[0]
        E = self.AB
        Le = np.sqrt(self.L2)
        w = self.A - p
        den = dvec[0] * E[:, 1] - dvec[1] * E[:, 0]
        params = [0.0, 1.0]
        nonpar = np.abs(den) > 1e-12 * Ld * Le
        s = np.full(len(E), np.nan)
        t = np.full(len(E), np.nan)
        s[nonpar] = (w[nonpar, 0] * E[nonpar, 1] - w[nonpar, 1] * E[nonpar, 0]) / den[nonpar]
        t[nonpar] = (w[nonpar, 0] * dvec[1] - w[nonpar, 1] * dvec[0]) / den[nonpar]
        ok = nonpar & (s >= -tol / Ld) & (s <= 1 + tol / Ld) & (t >= -tol / Le) & (t <= 1 + tol / Le)
        params.extend(np.clip(s[ok], 0.0, 1.0).tolist())
        # parallel edges on the chord's line (overlaps): their endpoints are contacts
        par = ~nonpar
        if par.any():
            off = np.abs(w[par, 0] * dvec[1] - w[par, 1] * dvec[0]) / Ld
            on = np.nonzero(par)[0][off <= tol]
            for j in on:
                for e_pt in (self.A[j], self.B[j]):
                    sp = float(np.dot(e_pt - p, dvec)) / Ld ** 2
                    if -tol / Ld <= sp <= 1 + tol / Ld:
                        params.append(min(max(sp, 0.0), 1.0))
        ps = np.unique(np.round(np.array(params), 12))
        mids = 0.5 * (ps[:-1] + ps[1:])
        mids = mids[(ps[1:] - ps[:-1]) * Ld > 1e-9]
        if len(mids) == 0:
            return True
        pts = p[None] + mids[:, None] * dvec[None]
        return not self.inside(pts).any()

    def inside(self, P):
        return self.path.contains_points(P) & (self.dist(P) > self.margin)


def _curvature_excess_gradient(Q, kappa_t, l0):
    """Gradient of E = sum max(|theta_i| - kappa_t * l_i, 0)^2 (l_i = min(mean edge, l0),
    treated as constant). Returns (G, theta, l)."""
    e_in = Q - np.roll(Q, 1, axis=0)
    e_out = np.roll(Q, -1, axis=0) - Q
    li2 = np.maximum((e_in ** 2).sum(1), 1e-300)
    lo2 = np.maximum((e_out ** 2).sum(1), 1e-300)
    theta = np.arctan2(e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0], (e_in * e_out).sum(1))
    # effective length: the SHORTER adjacent edge (capped at l0), which is the length that
    # controls the radius of the certified arc fillet (fillet_certificate)
    ell = np.minimum(np.minimum(np.sqrt(li2), np.sqrt(lo2)), l0)
    g = np.maximum(np.abs(theta) - kappa_t * ell, 0.0)
    w = (2.0 * g * np.sign(theta))[:, None]
    a = np.stack([-e_out[:, 1], e_out[:, 0]], axis=1) / lo2[:, None]   # d theta_i / d p_{i+1}
    b = np.stack([-e_in[:, 1], e_in[:, 0]], axis=1) / li2[:, None]     # d theta_i / d p_{i-1}
    G = w * (-a - b)
    G += np.roll(w * a, 1, axis=0)
    G += np.roll(w * b, -1, axis=0)
    return G, theta, ell


def _lm_normal_step(Q, theta, ell, kappa_t, nrm, pinned_test, damping=0.1, l0=1.0, passes=20):
    """Damped Gauss-Newton (Levenberg-Marquardt) step for the active curvature excesses.

    Unknowns: the normal displacement u_j of every vertex (along the outward normal nrm_j).
    Residuals: r_i = |theta_i| - kappa_t * l_i for the active vertices (r_i > 0), with the
    exact Jacobian of theta_i with respect to p_{i-1}, p_i, p_{i+1}. Solves
        min 1/2 |J u + r|^2 + lambda/2 |u|^2   s.t.  u >= lower ,  lambda = damping / l0^2,
    (`pinned_test` is the array `lower`) with an exact primal active-set method: bounded
    variables are fixed at their bound, violators are added and fixed variables whose
    gradient points into the feasible side are released (at most `passes` rounds).
    The free vertices around a pinned corner thereby take over its share of the correction.
    """
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spl
    m = len(Q)
    e_in = Q - np.roll(Q, 1, axis=0)
    e_out = np.roll(Q, -1, axis=0) - Q
    li2 = np.maximum((e_in ** 2).sum(1), 1e-300)
    lo2 = np.maximum((e_out ** 2).sum(1), 1e-300)
    a = np.stack([-e_out[:, 1], e_out[:, 0]], axis=1) / lo2[:, None]   # d theta_i / d p_{i+1}
    b = np.stack([-e_in[:, 1], e_in[:, 0]], axis=1) / li2[:, None]     # d theta_i / d p_{i-1}
    res = np.abs(theta) - kappa_t * ell
    act = np.nonzero(res > 0)[0]
    if len(act) == 0:
        return np.zeros(m)
    s = np.sign(theta[act])
    ip, im = (act + 1) % m, (act - 1) % m
    rows = np.repeat(np.arange(len(act)), 3)
    cols = np.stack([im, act, ip], axis=1).ravel()
    vals = np.stack([s * (b[act] * nrm[im]).sum(1),
                     s * ((-a[act] - b[act]) * nrm[act]).sum(1),
                     s * (a[act] * nrm[ip]).sum(1)], axis=1).ravel()
    J = _sp.csr_matrix((vals, (rows, cols)), shape=(len(act), m))
    lam = damping / l0 ** 2
    lower = np.minimum(pinned_test, 0.0)   # array: lower bound on u (u_j >= lower_j)
    # only the vertices coupled to an active residual can move
    var = np.unique(cols)
    Jv = J[:, var]
    M = (Jv.T @ Jv + lam * _sp.identity(len(var))).tocsc()
    rhs = -(Jv.T @ res[act])
    lo = lower[var]
    # primal active-set method for  min 1/2 u'Mu - rhs'u  s.t. u >= lo  (M is SPD):
    # variables at their bound are fixed; add the ones that violate, release the ones
    # whose gradient points into the feasible side
    fixed = np.zeros(len(var), bool)
    x = np.zeros(len(var))
    for _ in range(passes):
        x = np.where(fixed, lo, 0.0)
        free = ~fixed
        if free.any():
            Mff = M[free][:, free]
            b = rhs[free] - M[free][:, fixed] @ lo[fixed]
            x[free] = _spl.spsolve(Mff.tocsc(), b) if free.sum() > 1 else b / Mff.toarray().ravel()
        viol = free & (x < lo - 1e-12)
        grad = M @ x - rhs
        release = fixed & (grad < -1e-12)
        if not viol.any() and not release.any():
            break
        fixed = (fixed | viol) & ~release
    u = np.zeros(m)
    u[var] = np.maximum(x, lo)
    return u


def _orient(p, q, t):
    return (q[..., 0] - p[..., 0]) * (t[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (t[..., 0] - p[..., 0])


def _self_cross_blame(C, moved):
    """Self-intersection test of the closed chain C restricted to the chords that have
    a moved endpoint (moved: bool mask). Returns a bool mask of the moved vertices that
    take part in a proper crossing of two non-adjacent chords."""
    m = len(C)
    bad = np.zeros(m, bool)
    act = np.nonzero(moved | np.roll(moved, -1))[0]        # chord k = (k, k+1)
    if len(act) == 0 or m < 4:
        return bad
    A, B = C, np.roll(C, -1, axis=0)
    a, b = A[act][:, None, :], B[act][:, None, :]
    c, d = A[None], B[None]
    hit = ((_orient(a, b, c) * _orient(a, b, d)) < 0) & ((_orient(c, d, a) * _orient(c, d, b)) < 0)
    k = act[:, None]
    j = np.arange(m)[None]
    hit &= ~((j == k) | (j == (k + 1) % m) | (j == (k - 1) % m))
    kk, jj = np.nonzero(hit)
    for v in (act[kk], (act[kk] + 1) % m, jj, (jj + 1) % m):
        bad[v[moved[v]]] = True
    return bad


def _chord_crosses_chain(Q, i0, i1):
    """True if the segment Q[i0]-Q[i1] properly crosses a chord of Q not incident to i0 or i1."""
    m = len(Q)
    a, b = Q[i0][None], Q[i1][None]
    A, B = Q, np.roll(Q, -1, axis=0)
    hit = ((_orient(a, b, A) * _orient(a, b, B)) < 0) & ((_orient(A, B, a) * _orient(A, B, b)) < 0)
    k = np.arange(m)
    inc = (k == i0) | (k == i1) | ((k + 1) % m == i0) | ((k + 1) % m == i1)
    return bool((hit & ~inc).any())


def _interior_point(poly):
    """A point strictly inside the (ccw) polygon: an edge midpoint pushed inward."""
    path = _MplPath(np.vstack([poly, poly[:1]]))
    e = np.roll(poly, -1, axis=0) - poly
    L = np.linalg.norm(e, axis=1)
    for i in np.argsort(-L):
        mid = poly[i] + 0.5 * e[i]
        nin = np.array([-e[i, 1], e[i, 0]]) / L[i]           # left normal = inward (ccw)
        for s in (1e-1, 1e-2, 1e-3):
            p = mid + s * L[i] * nin
            if path.contains_point(p):
                return p
    return poly.mean(axis=0)


def certificate_curvature(Q):
    """Curvature of the certified smooth curve of fillet_certificate (without sampling):
    max over vertices of kappa_i/(1 + g kappa_i) (convex) and kappa_i/(1 - g kappa_i)
    (concave), kappa_i = tan(|theta_i|/2)/t_i, g = largest cut depth at a convex vertex."""
    e_in = Q - np.roll(Q, 1, axis=0)
    e_out = np.roll(Q, -1, axis=0) - Q
    l_in, l_out = np.linalg.norm(e_in, axis=1), np.linalg.norm(e_out, axis=1)
    theta = np.arctan2(e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0], (e_in * e_out).sum(1))
    t = 0.5 * np.minimum(l_in, l_out)
    half = np.abs(theta) / 2
    # no cap on the half-angle: near a reversal (|theta| -> pi) the radius tends to 0 and the
    # certified curvature to infinity, as it should
    k = np.tan(np.minimum(half, 0.5 * math.pi - 1e-12)) / np.maximum(t, 1e-300)
    g = float(np.where(theta > 0, t * np.tan(half / 2), 0.0).max())
    kc = np.where(theta > 0, k / (1 + g * k), k / np.maximum(1 - g * k, 1e-12))
    kc = np.where(half > 1e-12, kc, 0.0)
    return float(kc.max()), g


def fillet_certificate(Q, n_arc=8, offset=True):
    """Smooth (C^1) curve from the closed polyline Q: every vertex is replaced by the
    circular arc tangent to its two edges at distance t_i = min(l_{i-1}, l_i)/2 from the
    vertex. The arcs cut the convex vertices by at most g = max t_i tan(|theta_i|/4); if
    offset, the curve is moved outward by g (convex radii grow by g, concave radii shrink
    by g), so that it contains the region of the polyline. Returns (samples of the curve,
    its certified max curvature, g). The curve is exact (segments and arcs); the samples
    are only used to check enclosure."""
    Q = np.asarray(Q, dtype=float)
    m = len(Q)
    e_in = Q - np.roll(Q, 1, axis=0)
    e_out = np.roll(Q, -1, axis=0) - Q
    l_in, l_out = np.linalg.norm(e_in, axis=1), np.linalg.norm(e_out, axis=1)
    theta = np.arctan2(e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0], (e_in * e_out).sum(1))
    t = 0.5 * np.minimum(l_in, l_out)
    half = np.abs(theta) / 2
    kappa = np.where(half > 1e-12, np.tan(half) / np.maximum(t, 1e-300), 0.0)
    gap = np.where(half > 1e-12, t * np.tan(half / 2), 0.0)   # vertex-to-arc distance R(1/cos-1)
    out = []
    for i in range(m):
        u_in, u_out = e_in[i] / l_in[i], e_out[i] / l_out[i]
        p0, p1 = Q[i] - t[i] * u_in, Q[i] + t[i] * u_out
        if half[i] <= 1e-12:
            out.append(p0)
            continue
        R = 1.0 / kappa[i]
        nl = np.array([-u_in[1], u_in[0]]) * np.sign(theta[i])   # towards the centre
        cen = p0 + R * nl
        a0 = math.atan2(p0[1] - cen[1], p0[0] - cen[0])
        da = theta[i]
        for s in np.linspace(0.0, 1.0, n_arc + 1)[:-1]:
            a = a0 + s * da
            out.append(cen + R * np.array([math.cos(a), math.sin(a)]))
        out.append(p1)
    S = np.array(out)
    if not offset:
        return S, float(kappa.max()), float(gap.max())
    kc, g = certificate_curvature(Q)
    # outward offset by g along the normals of the sampled smooth curve (ccw: right normal)
    ch = np.roll(S, -1, axis=0) - np.roll(S, 1, axis=0)
    nr = np.stack([ch[:, 1], -ch[:, 0]], axis=1)
    nr /= np.maximum(np.linalg.norm(nr, axis=1), 1e-300)[:, None]
    return S + g * nr, kc, g


def certified_curve(Q, poly, kappa_max, sagitta=1e-3, margin=1e-6):
    """Build and verify the smooth certificate of the closed ccw chain Q.

    The curve is represented exactly: every vertex with turning angle theta_i != 0 is
    replaced by the circular arc tangent to both incident edges at distance
    t_i = min(l_{i-1}, l_i)/2, of radius R_i = t_i / tan(|theta_i|/2); the rest of each
    edge is a segment. The whole curve is then offset outward by g = max over convex
    vertices of t_i tan(|theta_i|/4): segments move by g along their outward normal,
    convex arcs get radius R_i + g, concave arcs R_i - g (same centres and angles).

    Checks (all on the exact pieces, except simplicity, see below):
      closure / C^1   consecutive pieces share endpoint and unit tangent
      regular         every concave radius R_i - g > 0
      radii           every arc radius >= 1/kappa_max  (curvature <= kappa_max)
      contact-free    no segment and no arc meets the interior of P (contact test as
                      in _PolygonObstacle.chords_clear; arcs: contact angles with the
                      edges of P, midpoint of each sub-arc must not be inside P)
      witness         an interior point of P is inside the curve
      simple          no self-intersection of the curve sampled with sagitta <= `sagitta`
                      (a numerical test)
    P is enclosed iff the curve is simple, does not meet the interior of P, and the
    witness is inside. Returns a dict with the flags, 'ok' (all of them), the samples,
    and L (exact), D and D_sum (on the samples) of the certified curve."""
    Q = np.asarray(Q, dtype=float)
    poly = np.asarray(poly, dtype=float)
    if 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0:
        poly = poly[::-1].copy()
    m = len(Q)
    rmin = 1.0 / float(kappa_max)
    e_out = np.roll(Q, -1, axis=0) - Q
    l = np.linalg.norm(e_out, axis=1)
    u = e_out / np.maximum(l, 1e-300)[:, None]                   # direction of edge i (Q_i -> Q_i+1)
    u_in = np.roll(u, 1, axis=0)                                 # direction of edge i-1
    l_in = np.roll(l, 1)
    theta = np.arctan2(u_in[:, 0] * u[:, 1] - u_in[:, 1] * u[:, 0], (u_in * u).sum(1))
    t = 0.5 * np.minimum(l_in, l)
    half = np.abs(theta) / 2
    is_arc = half > 1e-12
    t = np.where(is_arc, t, 0.0)                                 # a straight vertex is not filleted
    R = np.where(is_arc, t / np.maximum(np.tan(half), 1e-300), np.inf)
    g = float(np.where(theta > 0, t * np.tan(half / 2), 0.0).max()) if m else 0.0
    Rc = np.where(theta > 0, R + g, R - g)                       # radii of the offset curve
    res = {"g": g}
    res["regular"] = bool(np.all(Rc[is_arc & (theta < 0)] > 0)) if np.any(is_arc & (theta < 0)) else True
    res["radii_ok"] = bool(np.all(Rc[is_arc] >= rmin * (1 - 1e-9))) if is_arc.any() else True
    res["kappa_cert"] = float((1.0 / Rc[is_arc]).max() * rmin) if is_arc.any() else 0.0   # relative to kappa_max
    nR = lambda d: np.stack([d[..., 1], -d[..., 0]], axis=-1)   # outward (right) normal of a ccw direction
    # pieces in order: arc at vertex i (if any), then the segment of edge i
    # Arcs are stored by their start point, radius, start angle a0 (direction from the
    # centre to the start) and signed sweep; points are evaluated relative to the start,
    # p(phi) = start + 2 R sin(phi/2) (-sin(a0+phi/2), cos(a0+phi/2)), which stays accurate
    # for nearly straight vertices, whose radii (and centre distances) are huge.
    pieces = []
    for i in range(m):
        if is_arc[i]:
            p0 = Q[i] - t[i] * u_in[i]
            nl = np.array([-u_in[i, 1], u_in[i, 0]]) * np.sign(theta[i])     # towards the centre
            delta = g if theta[i] > 0 else -g                                # Rc - R, exactly
            start = p0 - delta * nl                                          # offset start point
            a0 = math.atan2(-nl[1], -nl[0])
            pieces.append(("arc", start, Rc[i], a0, float(theta[i]), u_in[i].copy(), u[i].copy()))
        a = Q[i] + t[i] * u[i] + g * nR(u[i])
        b = Q[(i + 1) % m] - t[(i + 1) % m] * u[i] + g * nR(u[i])
        if l[i] - t[i] - t[(i + 1) % m] > 1e-12 * max(1.0, l[i]):   # skip zero-length segments
            pieces.append(("seg", a, b, u[i].copy()))

    def arc_pts(pc, phi):
        _, start, rr, a0, dth = pc[:5]
        phi = np.atleast_1d(phi)
        c = 2 * rr * np.sin(phi / 2)
        return start + c[:, None] * np.stack([-np.sin(a0 + phi / 2), np.cos(a0 + phi / 2)], axis=1)

    def piece_ends(pc):
        if pc[0] == "seg":
            return pc[1], pc[2], pc[3], pc[3]                 # tangent = the edge direction
        a0, dth = pc[3], pc[4]
        s = math.copysign(1.0, dth)
        ta = s * np.array([-math.sin(a0), math.cos(a0)])               # from the arc's own angles,
        tb = s * np.array([-math.sin(a0 + dth), math.cos(a0 + dth)])   # compared with the segments
        return pc[1], arc_pts(pc, dth)[0], ta, tb

    ends = [piece_ends(pc) for pc in pieces]
    scale = max(1.0, float(np.ptp(Q, axis=0).max()))
    gap_pos = max(float(np.hypot(*(ends[k][1] - ends[(k + 1) % len(ends)][0]))) for k in range(len(ends)))
    gap_tan = max(float(np.hypot(*(ends[k][3] - ends[(k + 1) % len(ends)][2]))) for k in range(len(ends)))
    res["closure_C1"] = bool(gap_pos <= 1e-8 * scale and gap_tan <= 1e-6)
    res["gap_pos"], res["gap_tan"] = gap_pos, gap_tan
    # every segment keeps the direction of its edge (t_i + t_{i+1} <= l_i by construction)
    res["closure_C1"] = res["closure_C1"] and bool(np.all(l - t - np.roll(t, -1) >= -1e-9 * np.maximum(l, 1.0)))

    # exact length
    L = 0.0
    for pc in pieces:
        L += float(np.hypot(*(pc[2] - pc[1]))) if pc[0] == "seg" else abs(pc[2] * pc[4])
    res["L"] = L

    # contact-free: segments
    obs = _PolygonObstacle(poly, margin=margin)
    segs = [pc for pc in pieces if pc[0] == "seg" and np.hypot(*(pc[2] - pc[1])) > 1e-12]
    S1 = np.array([pc[1] for pc in segs]); S2 = np.array([pc[2] for pc in segs])
    free = True
    if len(segs):
        free = (not obs.inside(S1).any()) and bool(obs.chords_clear(S1, S2).all())
    # contact-free: arcs (contact angles with every edge of P, then sub-arc midpoints)
    A, B = obs.A, obs.B
    E = B - A
    flat = 1e-7 * scale                       # an arc whose sagitta is below this is tested as its chord
    for pc in pieces:
        if not free:
            break
        if pc[0] != "arc":
            continue
        _, start, rr, a0, dth = pc[:5]
        if rr * (1 - math.cos(dth / 2)) <= flat:
            p_end = arc_pts(pc, dth)
            if obs.inside(np.vstack([start, p_end])).any() or not obs.chords_clear(start[None], p_end)[0]:
                free = False
            continue
        cen = start - rr * np.array([math.cos(a0), math.sin(a0)])
        w = A - cen
        aa = (E ** 2).sum(1)
        bb = 2 * (w * E).sum(1)
        cc = (w ** 2).sum(1) - rr * rr
        disc = bb * bb - 4 * aa * cc
        us = [0.0, 1.0]
        ok = disc >= 0
        sq = np.sqrt(np.maximum(disc, 0))
        for sgn in (-1, 1):
            s = (-bb + sgn * sq) / np.maximum(2 * aa, 1e-300)
            sel = ok & (s >= -1e-9) & (s <= 1 + 1e-9)
            if not sel.any():
                continue
            pts = A[sel] + s[sel, None] * E[sel]
            phi = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
            uu = ((phi - a0) * np.sign(dth)) % (2 * math.pi) / abs(dth)
            us.extend(uu[uu <= 1 + 1e-9].clip(0, 1).tolist())
        us = np.unique(np.round(np.array(us), 12))
        mids = 0.5 * (us[:-1] + us[1:])
        mids = mids[(us[1:] - us[:-1]) * abs(dth) * rr > 1e-9]
        if len(mids):
            ph = a0 + mids * dth
            pts = cen + rr * np.stack([np.cos(ph), np.sin(ph)], axis=1)
            if obs.inside(pts).any():
                free = False
    res["contact_free"] = bool(free)

    # dense samples (sagitta-bounded), used for simplicity, the witness and D, D_sum
    out = []
    for pc in pieces:
        if pc[0] == "seg":
            out.append(pc[1][None])
            continue
        rr, dth = pc[2], pc[4]
        step = 2 * math.acos(max(-1.0, 1 - sagitta / max(rr, 1e-300))) if rr > sagitta else abs(dth)
        n = max(1, int(math.ceil(abs(dth) / max(step, 1e-12))))
        out.append(arc_pts(pc, dth * np.arange(n) / n))
    Sm = np.vstack(out)
    keep = np.r_[True, np.linalg.norm(np.diff(Sm, axis=0), axis=1) > 1e-12]
    Sm = Sm[keep]
    res["samples"] = Sm
    p_in = _interior_point(poly)
    res["witness"] = bool(_MplPath(np.vstack([Sm, Sm[:1]])).contains_point(p_in))
    res["simple"] = _count_crossings(Sm) == 0
    res["ok"] = bool(res["closure_C1"] and res["regular"] and res["radii_ok"] and res["contact_free"]
                     and res["witness"] and res["simple"])
    # deviation of the certified curve (densified at spacing 0.5, as for every curve)
    cd = densify(Sm, 0.5)
    d = obs.dist(cd)
    seg = np.linalg.norm(np.roll(cd, -1, axis=0) - cd, axis=1)
    res["D"] = float(d.max())
    res["D_sum"] = float((0.5 * (d + np.roll(d, -1)) * seg).sum())
    return res


def _count_crossings(P):
    """Number of properly crossing non-adjacent edge pairs of the closed polyline P."""
    P = np.asarray(P, dtype=float)
    m = len(P)
    A, B = P, np.roll(P, -1, axis=0)
    cnt = 0
    jj = np.arange(m)
    step = max(1, 4_000_000 // m)
    for s0 in range(0, m, step):
        i = np.arange(s0, min(m, s0 + step))
        Ai, Bi = A[i][:, None, :], B[i][:, None, :]
        Aj, Bj = A[None], B[None]

        def orient(p, q, r):
            return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
        o1, o2, o3, o4 = orient(Ai, Bi, Aj), orient(Ai, Bi, Bj), orient(Aj, Bj, Ai), orient(Aj, Bj, Bi)
        hit = (o1 * o2 < 0) & (o3 * o4 < 0)
        valid = (jj[None, :] > i[:, None] + 1) & ~((i[:, None] == 0) & (jj[None, :] == m - 1))
        cnt += int((hit & valid).sum())
    return cnt


def _hull_indices(Q):
    """Indices of the convex hull vertices of the closed chain Q (counter-clockwise),
    sorted increasingly (i.e. in chain order for a ccw chain)."""
    order = np.lexsort((Q[:, 1], Q[:, 0]))

    def cross(o, a, b):
        return (Q[a, 0] - Q[o, 0]) * (Q[b, 1] - Q[o, 1]) - (Q[a, 1] - Q[o, 1]) * (Q[b, 0] - Q[o, 0])
    lower, upper = [], []
    for k in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], k) <= 0:
            lower.pop()
        lower.append(k)
    for k in order[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], k) <= 0:
            upper.pop()
        upper.append(k)
    return np.array(sorted(set(lower[:-1] + upper[:-1])))


def _release_pocket(Q, w, l0):
    """Replace the stretch of the ccw chain Q strictly between the two convex-hull
    vertices that bracket vertex w by the straight hull edge (resampled at ~l0).
    The region only grows, so enclosure is kept. If w is itself a hull vertex, the
    deeper of its two adjacent stretches is released. Returns the new chain, or None."""
    m = len(Q)
    H = _hull_indices(Q)
    if len(H) < 3:
        return None
    pos = np.searchsorted(H, w)
    if pos < len(H) and H[pos] == w:                 # w is on the hull: pick a side
        a_l, b_l = H[pos - 1], w
        a_r, b_r = w, H[(pos + 1) % len(H)]
        gap = lambda a, b: (b - a) % m
        a, b = (a_l, b_l) if gap(a_l, b_l) >= gap(a_r, b_r) else (a_r, b_r)
    else:
        a, b = H[pos - 1], H[pos % len(H)]
    n_inner = (b - a) % m - 1
    if n_inner < 1:
        return None
    seg = Q[b] - Q[a]
    k = max(1, int(math.ceil(np.linalg.norm(seg) / l0)))
    new = Q[a] + (np.arange(1, k) / k)[:, None] * seg
    keep = [(b + j) % m for j in range(m - n_inner)]       # b, b+1, ..., a
    return np.vstack([Q[keep], new])


def _resampled_max_curvature(Q, h):
    Qc = np.vstack([Q, Q[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Qc, axis=0), axis=1))])
    n = max(8, int(math.ceil(s[-1] / h)))
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    R = np.stack([np.interp(t, s, Qc[:, 0]), np.interp(t, s, Qc[:, 1])], axis=1)
    return float(vertex_curvature(R).max())


def simulate_curvature_evolution(points, kappa_max, l0=None, target=0.85, eta=0.05, cap=0.25,
                                 tang=0.25, lap=0.25, sharp=0.5, sharp_angle=1.5, split=1.25, merge=0.5,
                                 max_iter=40000, solver="lm", omega=0.5, damping=0.1, hold_iters=25,
                                 stall_iters=1000,
                                 frame_every=0, max_vertices=20000, callback=None, callback_every=0,
                                 debug=None, selfcheck=True, precise_blame=True, contact_noslide=True,
                                 chord_bounds=True, resampled_stop=True, inflate=True, certify=True):
    """Curvature-driven evolution of the polygon `points` (either orientation).

    kappa_max : curvature bound (r = 1 / kappa_max)
    l0        : length scale (default: mean edge length of the input)
    target    : the flow drives |kappa| towards target * kappa_max (slack for the stop test)
    eta       : step, tau = eta * l0^2 (stable for eta < 1/16)
    cap       : largest vertex displacement per iteration, in units of l0
    tang      : strength of the tangential spacing equaliser
    lap, sharp: sharp-vertex term, a Laplacian step of weight lap * min(1, excess/sharp)
                for violating vertices with |theta| > sharp_angle (where the gradient of
                theta degenerates, |theta| close to pi)
    solver    : "lm" (default) - damped Gauss-Newton step on the active excesses, the
                vertices pinned by the polygon are held (active set); omega is the
                fraction of that step taken, damping the Levenberg-Marquardt damping.
                "gradient" - explicit projected gradient descent with step eta*l0^2
                (simple, but slow: the turning spreads diffusively)
    split / merge : edges longer than split*l0 are split, shorter than merge*l0 merged
    frame_every   : if > 0, store every frame_every-th iterate in 'frames' (for animation)
    Switches (all True by default; False is used for the ablation study):
    selfcheck      : reject steps that make the chain self-intersect, and steps after
                     which a fixed interior point of P is no longer inside the chain;
                     with it, every iterate is simple and encloses P (up to the tolerance)
    precise_blame  : blame only the endpoint(s) of a crossing chord that cause the crossing
    contact_noslide: no tangential sliding of vertices that lie on P
    chord_bounds   : inward bounds of the solver from a line search on C1-C2
                     (False: half the distance to P)
    resampled_stop : the stop test also checks the curve resampled at spacing l0
    inflate        : on a stall that no pocket release can resolve (the chain is
                     convex), scale the chain by 1 % about an interior point of P
    certify        : the stop test also requires that the smooth arc-fillet curve of
                     fillet_certificate has curvature <= kappa_max

    Returns a dict: final (counter-clockwise chain), iterations, status
    ('converged' | 'max_iter' | 'diverged'), frames (list, starts with the polygon),
    releases, inflations.
    """
    poly = np.asarray(points, dtype=float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1].copy()
    if l0 is None:
        l0 = float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).mean())
    obs = _PolygonObstacle(poly)
    kmax = float(kappa_max)
    kt = target * kmax
    tau = eta * l0 ** 2
    band = (2 * cap + split) * l0          # only vertices this close to P can break enclosure
    Q = poly.copy()
    frames = [Q.copy()]
    status, iterations = "max_iter", max_iter
    rejected_pts = []                      # (position, expiry iteration) of rejected steps
    best_kap, best_it, releases, inflations = np.inf, 0, 0, 0
    p_in = _interior_point(poly)           # witness: must stay inside the chain

    def encloses_witness(C):
        return bool(_MplPath(np.vstack([C, C[:1]])).contains_point(p_in))

    for it in range(max_iter):
        if callback is not None and callback_every and it % callback_every == 0:
            callback(it, Q)
        G, theta, ell = _curvature_excess_gradient(Q, kt, l0)

        # ---- termination on curvature (native, then at the scale l0) -------
        kap_now = np.abs(theta) / ell
        if it % 10 == 0 and kap_now.max() <= kmax:
            if (not resampled_stop or _resampled_max_curvature(Q, l0) <= kmax) and \
                    (not certify or certificate_curvature(Q)[0] <= kmax):
                status, iterations = "converged", it
                break

        # ---- stall: the excess has a constrained local minimum (e.g. the curve has
        #      entered a pocket it cannot leave with bounded curvature); release the
        #      pocket around the worst vertex by the curve's convex-hull edge ---------
        # progress is measured on the total excess E (the maximum alone can plateau
        # while the rest of the curve is still improving)
        E_now = float((np.maximum(np.abs(theta) - kt * ell, 0.0) ** 2).sum())
        if E_now < 0.98 * best_kap:
            best_kap, best_it = E_now, it
        elif stall_iters and it - best_it >= stall_iters:
            Qn = _release_pocket(Q, int(np.argmax(kap_now)), l0)
            if Qn is not None and selfcheck and not encloses_witness(Qn):
                Qn = None
            # the new hull chords are checked like any other chord (no vertex in P, no
            # chord meeting the interior of P)
            if Qn is not None and (obs.inside(Qn).any() or
                                   not obs.chords_clear(Qn, np.roll(Qn, -1, axis=0)).all()):
                Qn = None
            if Qn is not None:
                Q = Qn
                releases += 1
            elif inflate:
                # convex chain, nothing to release: the curve must grow everywhere
                # (e.g. 2*pi*r > per(conv P)); a similarity about an interior point
                # keeps the chain simple and, for a chain star-shaped about p_in,
                # enlarges the region -- checked like any other step
                Qn = p_in + 1.01 * (Q - p_in)
                ok = not obs.inside(Qn).any() and obs.chords_clear(Qn, np.roll(Qn, -1, axis=0)).all()
                if ok and (not selfcheck or encloses_witness(Qn)):
                    Q = Qn
                    inflations += 1
            rejected_pts = []
            G, theta, ell = _curvature_excess_gradient(Q, kt, l0)
            kap_now = np.abs(theta) / ell
            best_kap, best_it = np.inf, it

        # ---- step: normal part of -tau*grad E, plus tangential equaliser ---
        m = len(Q)
        chord = np.roll(Q, -1, axis=0) - np.roll(Q, 1, axis=0)
        tng = chord / np.maximum(np.linalg.norm(chord, axis=1), 1e-300)[:, None]
        nrm = np.stack([tng[:, 1], -tng[:, 0]], axis=1)          # outward normal (ccw)
        if solver == "lm":
            dP = obs.dist(Q)
            # vertices whose step the projection had to drop in the previous iteration
            # (they have not moved since, so their coordinates match exactly)
            # (kept for hold_iters iterations, matched within 0.02*l0, so the solver
            # does not keep proposing the same infeasible inward step)
            held = np.zeros(m, bool)
            if len(rejected_pts):
                rp = np.array([p for p, t in rejected_pts])
                dd = np.linalg.norm(Q[:, None, :] - rp[None], axis=2).min(1)
                held = dd < 0.02 * l0

            # bound constraint: the largest inward normal displacement that keeps the
            # vertex AND its two chords (neighbours fixed) clear of P, by a short line
            # search over fractions of the clearance; held vertices: 0
            allow = np.zeros(m)
            cand_idx = np.nonzero(dP < band)[0]
            far = dP >= band
            allow[far] = 0.5 * dP[far]
            todo = cand_idx.copy()
            if not chord_bounds:                         # ablation: distance only
                allow[todo] = 0.5 * dP[todo]
                todo = todo[:0]
            for c in (0.5, 0.25, 0.12, 0.06, 0.03):
                if len(todo) == 0:
                    break
                P_new = Q[todo] - (c * dP[todo])[:, None] * nrm[todo]
                ok = ~obs.inside(P_new)
                ok &= obs.chords_clear(Q[(todo - 1) % m], P_new)
                ok &= obs.chords_clear(P_new, Q[(todo + 1) % m])
                allow[todo[ok]] = c * dP[todo[ok]]
                todo = todo[~ok]
            lower = np.where(held, 0.0, -allow / omega)
            u = omega * _lm_normal_step(Q, theta, ell, kt, nrm, lower, damping=damping, l0=l0)
            D = u[:, None] * nrm
        else:
            D = (-tau * (G * nrm).sum(1))[:, None] * nrm
        l_in = np.linalg.norm(Q - np.roll(Q, 1, axis=0), axis=1)
        l_out = np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)
        # tangential spacing equaliser; contact vertices (on P) do not slide, since
        # sliding off a contact point is how a chord would cut a corner of P
        contact = (obs.dist(Q) < 1e-3 * l0) if contact_noslide else np.zeros(m, bool)
        D += (np.where(contact, 0.0, tang) * 0.5 * (l_out - l_in))[:, None] * tng
        # sharp-vertex term: near a reversal (|theta| -> pi) the gradient of theta_i with
        # respect to p_i vanishes, so a vertex at the tip of a narrow notch would stay
        # pinned; a Laplacian step towards the neighbours' midpoint never degenerates
        D_lap = np.zeros_like(Q)
        if lap > 0:
            excess = np.maximum(np.abs(theta) - kt * ell, 0.0)
            wl = lap * np.minimum(1.0, excess / sharp) * (np.abs(theta) > sharp_angle)
            mid = 0.5 * (np.roll(Q, 1, axis=0) + np.roll(Q, -1, axis=0))
            D_lap = wl[:, None] * (mid - Q)
            D += D_lap
        size = np.linalg.norm(D, axis=1)
        D *= np.minimum(1.0, cap * l0 / np.maximum(size, 1e-300))[:, None]
        size = np.linalg.norm(D_lap, axis=1)
        D_lap *= np.minimum(1.0, cap * l0 / np.maximum(size, 1e-300))[:, None]
        fallback_used = np.zeros(m, bool)

        # ---- projection onto the enclosure constraint (polygon = obstacle) --
        near = obs.dist(Q) < band
        for _ in range(2):                               # two rings of neighbours, for the chords
            near = near | np.roll(near, 1) | np.roll(near, -1)
        sub = np.nonzero(near)[0]
        # A rejected step is halved (rounds 0-4), then dropped. The loop only exits
        # when the candidate passes the test (or the remaining violations were already
        # present before the step), so no iterate can lose enclosure through here.
        scale = np.ones(m)
        bad = np.zeros(m, bool)
        for rnd in range(64):
            C = Q + scale[:, None] * D
            bad[:] = False
            if len(sub):
                ins = obs.inside(C[sub])
                bad[sub] = ins
                if not ins.any():                    # vertex positions first, then chords
                    cr = ~obs.chords_clear(C[sub], C[(sub + 1) % m])
                    if cr.any():
                        # blame precisely: does the chord cross with only one end moved?
                        i0 = sub[cr]
                        i1 = (i0 + 1) % m
                        if precise_blame:
                            only0 = ~obs.chords_clear(C[i0], Q[i1])
                            only1 = ~obs.chords_clear(Q[i0], C[i1])
                        else:
                            only0 = only1 = np.zeros(len(i0), bool)
                        neither = ~only0 & ~only1
                        bad[i0[only0 | neither]] = True
                        bad[i1[only1 | neither]] = True
            if not bad.any() and selfcheck:
                # the chain against itself: no proper crossing of two chords
                moved = (scale > 0) & (np.abs(D).sum(1) > 0)
                bad |= _self_cross_blame(C, moved)
            if not bad.any():
                break
            culprit = bad & (scale > 0)
            if not culprit.any():                    # a still vertex is bad because a neighbour moved
                culprit = (np.roll(bad, 1) | np.roll(bad, -1)) & (scale > 0)
                if not culprit.any():
                    break                            # pre-existing state, nothing moving causes it
            if rnd < 5:
                scale = np.where(culprit, scale * 0.5, scale)
            else:
                # fall back once to the Laplacian part alone (the gradient part can
                # point into the polygon at a near-reversal), then drop the step
                fb = culprit & ~fallback_used & (np.abs(D_lap).sum(1) > 0)
                if fb.any():
                    D = D.copy()
                    D[fb] = D_lap[fb]
                    scale[fb] = 1.0
                    fallback_used |= fb
                scale = np.where(culprit & ~fb, 0.0, scale)
        else:
            C = Q                                    # safety net: no valid step found, stay put
            scale = np.zeros(m)
        if selfcheck and C is not Q and not encloses_witness(C):
            C = Q                                    # the step would turn P "outside": refuse it
            scale = np.zeros(m)
        if debug is not None:
            debug.update(Q=Q.copy(), D=D.copy(), scale=scale.copy(), theta=theta.copy(),
                         u=(u.copy() if solver == "lm" else None),
                         lower=(lower.copy() if solver == "lm" else None))
        rejected_pts = [(p, t) for p, t in rejected_pts if t > it]
        if hold_iters > 0:
            rejected_pts += [(p.copy(), it + hold_iters) for p in Q[(scale < 1) & (np.abs(D).sum(1) > 0)]]
        Q = C

        # ---- refinement: split long edges, merge short ones -----------------
        lo = np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)
        idx = np.nonzero(lo > split * l0)[0]
        if len(idx):
            Q = np.insert(Q, idx + 1, 0.5 * (Q[idx] + Q[(idx + 1) % len(Q)]), axis=0)
        lo = np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)
        short = np.nonzero(lo < merge * l0)[0]
        if len(short):
            _, theta, _ = _curvature_excess_gradient(Q, kt, l0)
            m = len(Q)
            kill = np.zeros(m, bool)
            for i in short:
                j = (i + 1) % m
                if kill[[(i - 1) % m, i, j, (j + 1) % m]].any():
                    continue
                cand = i if abs(theta[i]) <= abs(theta[j]) else j
                a, b = (cand - 1) % m, (cand + 1) % m
                if obs.chords_clear(Q[a][None], Q[b][None])[0] and \
                        not (selfcheck and _chord_crosses_chain(Q, a, b)):
                    kill[cand] = True
            if kill.any():
                Qm = Q[~kill]
                # a merge is a step like any other: its chords were checked above, and the
                # interior point of P must still be inside the chain
                if not selfcheck or encloses_witness(Qm):
                    Q = Qm

        if frame_every and (it + 1) % frame_every == 0:
            frames.append(Q.copy())
        if len(Q) > max_vertices or not np.isfinite(Q).all():
            status, iterations = "diverged", it + 1
            break

    if frame_every:
        frames.append(Q.copy())
    return {"final": Q, "iterations": iterations, "status": status, "frames": frames,
            "releases": releases, "inflations": inflations,
            "strict_rejections": obs.strict_rejections}


def chain_length(chain):
    return float(np.linalg.norm(np.roll(chain, -1, axis=0) - chain, axis=1).sum())


def max_distance_to_polygon(chain, polygon, spacing=0.5):
    """max over the (densified) chain of the distance to the polygon boundary."""
    q = densify(chain, spacing)
    A = np.asarray(polygon, dtype=float)
    B = np.roll(A, -1, axis=0)
    AB = B - A
    L2 = (AB ** 2).sum(axis=1)
    worst = 0.0
    for s in range(0, len(q), 1000):
        c = q[s:s + 1000]
        ap = c[:, None, :] - A[None, :, :]
        t = np.clip((ap * AB[None]).sum(axis=2) / L2[None], 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        worst = max(worst, float(np.linalg.norm(c[:, None, :] - proj, axis=2).min(axis=1).max()))
    return worst


def main():
    angleFactor = 1.02 # values bigger than 1. Smaller values means smaller allowed curvature
    sufficientAngleFactor = 1.25
    powerFactor = 0.08
    distanceFactor = 1.15
    anglePow = 1.5
    distanceFromOriginPower = 2
    alpha = 0
    simulation_length = 3000

    # Which method to run and display:
    #   "evolution" - the angle-driven chain evolution (the original simulation)
    #   "hull_disk" - the hull-plus-disk baseline
    #   "both"      - run both, one animation after the other
    #   "curvature" - the curvature-driven evolution (repair of the evolution)
    #   "all"       - evolution, curvature-driven evolution and hull-disk
    method = "both"
    hull_disk_radius = 20      # r = 1 / kappa_max of the baseline
    hull_disk_frames = 60      # number of radius steps in its animation

    # Colouring of the dots in the animations:
    #   "angle"     - original rule: red if the angle is outside the allowed range
    #   "curvature" - red if the vertex violates the curvature bound kappa_max
    color_mode = "curvature"
    kappa_max = 1.0 / hull_disk_radius
    kappaMax = kappa_max if color_mode == "curvature" else None

    sys.setrecursionlimit(5000)

    mode = 1  # 0 - square, 1 - I-shape
    square_size = 100
    num_points = 80

    (
        points,
        isCw,
        square_points_sequence,
        prev_points,
        original_points,
        pointAveDis
    ) = prepare_polygon_simulation(mode, square_size, num_points)

    original_polygon = np.array(points)

    if method in ("evolution", "both", "all"):
        square_points_sequence = simulate_polygon_evolution(
            prev_points,
            original_points,
            isCw,
            square_points_sequence,
            pointAveDis,
            angleFactor,
            sufficientAngleFactor,
            powerFactor,
            distanceFactor,
            anglePow,
            distanceFromOriginPower,
            alpha,
            simulation_length
        )
        final = np.array(square_points_sequence[-1])
        print(f"Evolution:  L = {chain_length(final):.1f},  "
              f"D = {max_distance_to_polygon(final, original_polygon):.1f},  "
              f"vertices with curvature > {kappa_max:g}: "
              f"{int((vertex_curvature(final) > kappa_max * 1.01).sum())} of {len(final)}")
        animate_polygonal_chains(square_points_sequence, isCw, angleFactor, kappaMax)

    if method in ("curvature", "all"):
        res = simulate_curvature_evolution(original_polygon, kappa_max, l0=pointAveDis,
                                           frame_every=25)
        final = res["final"]
        print(f"Curvature evolution: {res['status']} after {res['iterations']} iterations,  "
              f"L = {chain_length(final):.1f},  "
              f"D = {max_distance_to_polygon(final, original_polygon):.1f},  "
              f"vertices with curvature > {kappa_max:g}: "
              f"{int((vertex_curvature(final) > kappa_max * 1.01).sum())} of {len(final)}")
        # the frames are counter-clockwise, hence isCw=False
        animate_polygonal_chains(res["frames"], False, angleFactor, kappaMax)

    if method in ("hull_disk", "both", "all"):
        hull_disk_sequence = simulate_hull_disk(
            points, isCw, hull_disk_radius, hull_disk_frames, pointAveDis
        )
        final = hull_disk_sequence[-1]
        hull_len = chain_length(convex_hull(original_polygon))
        print(f"Hull-disk:  L = {chain_length(final):.1f} "
              f"(= {hull_len:.1f} + 2*pi*{hull_disk_radius} = "
              f"{hull_len + 2 * math.pi * hull_disk_radius:.1f}),  "
              f"D = {max_distance_to_polygon(final, original_polygon):.1f},  "
              f"vertices with curvature > {kappa_max:g}: "
              f"{int((vertex_curvature(final) > kappa_max * 1.01).sum())} of {len(final)}")
        # the hull-disk frames are counter-clockwise, hence isCw=False
        animate_polygonal_chains(hull_disk_sequence, False, angleFactor, kappaMax)

if __name__ == "__main__":
    main()
