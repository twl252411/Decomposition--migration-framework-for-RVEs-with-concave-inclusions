from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import qmc
from .accelerators_numba import get_kernels
from .candidates import (
    filter_candidates_by_radius,
    pair_candidates_from_centers,
    pair_candidates_from_centers_numba,
)
from .collision_kernel import poly_world, sat_mtv, sat_mtv_with_axes
from .geom2d import aabb_from_points, clamp_unit, rot_aabb_world, world_center
from .rve_types import PolygonShape, ShapeTemplate

_PARTS_NB_CACHE: Dict[int, Any] = {}


def _get_parts_nb(template: ShapeTemplate, NList) -> Any:
    key = id(template)
    cached = _PARTS_NB_CACHE.get(key)
    if cached is not None:
        return cached
    parts_nb = NList()
    for p in template.parts:
        parts_nb.append(np.asarray(p, dtype=np.float64))
    _PARTS_NB_CACHE[key] = parts_nb
    return parts_nb


def _segment_intersect_shifted(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    eps: float = 1e-12,
) -> bool:
    def cross(o_x, o_y, p_x, p_y, q_x, q_y):
        return (p_x - o_x) * (q_y - o_y) - (p_y - o_y) * (q_x - o_x)

    def on_segment(p_x, p_y, q_x, q_y, r_x, r_y):
        return (
            min(p_x, r_x) - eps <= q_x <= max(p_x, r_x) + eps
            and min(p_y, r_y) - eps <= q_y <= max(p_y, r_y) + eps
        )

    ab_c = cross(ax, ay, bx, by, cx, cy)
    ab_d = cross(ax, ay, bx, by, dx, dy)
    cd_a = cross(cx, cy, dx, dy, ax, ay)
    cd_b = cross(cx, cy, dx, dy, bx, by)

    if ab_c * ab_d < -eps and cd_a * cd_b < -eps:
        return True

    if abs(ab_c) <= eps and on_segment(ax, ay, cx, cy, bx, by):
        return True
    if abs(ab_d) <= eps and on_segment(ax, ay, dx, dy, bx, by):
        return True
    if abs(cd_a) <= eps and on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if abs(cd_b) <= eps and on_segment(cx, cy, bx, by, dx, dy):
        return True

    return False


def _point_in_polygon_shifted(
    point: np.ndarray,
    poly: np.ndarray,
    shift_x: float,
    shift_y: float,
    eps: float = 1e-12,
) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = poly.shape[0]
    for i in range(n):
        j = (i + 1) % n
        xi = float(poly[i, 0]) + shift_x
        yi = float(poly[i, 1]) + shift_y
        xj = float(poly[j, 0]) + shift_x
        yj = float(poly[j, 1]) + shift_y

        cross_val = (xj - xi) * (y - yi) - (yj - yi) * (x - xi)
        if abs(cross_val) <= eps and (
            min(xi, xj) - eps <= x <= max(xi, xj) + eps
            and min(yi, yj) - eps <= y <= max(yi, yj) + eps
        ):
            return True

        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + eps) + xi
        if intersects:
            inside = not inside
    return inside


def _polygons_intersect_shifted(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    shift_x: float,
    shift_y: float,
    eps: float = 1e-12,
) -> bool:
    na = poly_a.shape[0]
    nb = poly_b.shape[0]
    for i in range(na):
        a0x = float(poly_a[i, 0])
        a0y = float(poly_a[i, 1])
        a1x = float(poly_a[(i + 1) % na, 0])
        a1y = float(poly_a[(i + 1) % na, 1])
        for j in range(nb):
            b0x = float(poly_b[j, 0]) + shift_x
            b0y = float(poly_b[j, 1]) + shift_y
            b1x = float(poly_b[(j + 1) % nb, 0]) + shift_x
            b1y = float(poly_b[(j + 1) % nb, 1]) + shift_y
            if _segment_intersect_shifted(a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y, eps=eps):
                return True

    if _point_in_polygon_shifted(poly_a[0], poly_b, shift_x, shift_y, eps=eps):
        return True
    if _point_in_polygon_shifted(poly_b[0], poly_a, shift_x, shift_y, eps=eps):
        return True
    return False


def _rotate_axes(axes: np.ndarray, c: float, s: float) -> np.ndarray:
    if axes.size == 0:
        return axes
    x = axes[:, 0]
    y = axes[:, 1]
    return np.column_stack((c * x - s * y, s * x + c * y)).astype(np.float64)


def separation_step(
    shapes: List[PolygonShape],
    L: float,
    cell_size: float,
    eps_gjk: float = 1e-8,
    itmax_gjk: int = 64,
    tol_epa: float = 1e-7,
    itmax_epa: int = 48,
    n_jobs: int = 0,
    use_numba: bool = True,
    fallback_concave: Optional[bool] = None,
    candidate_radius_scale: Optional[float] = None,
    return_stats: bool = False,
) -> Tuple[np.ndarray, float]:
    N = len(shapes)
    template = shapes[0].template
    if fallback_concave is None:
        fallback_concave = not template.is_convex
    t = np.asarray([s.translation for s in shapes], dtype=np.float64)
    ang = np.array([float(s.angle) for s in shapes], dtype=np.float64)
    coss = np.cos(ang)
    sins = np.sin(ang)
    lc = template.local_center
    centers = np.empty((N, 2), dtype=np.float64)
    centers[:, 0] = coss * lc[0] - sins * lc[1] + t[:, 0]
    centers[:, 1] = sins * lc[0] + coss * lc[1] + t[:, 1]
    broad_t0 = time.perf_counter() if return_stats else None
    if use_numba:
        ij, nint, _ = pair_candidates_from_centers_numba(centers, L, cell_size)
    else:
        ij, nint, _ = pair_candidates_from_centers(centers, L, cell_size)
    if candidate_radius_scale is not None:
        max_dist = 2.0 * float(template.radius) * float(candidate_radius_scale)
        ij, nint = filter_candidates_by_radius(ij, nint, centers, float(L), max_dist)
    broad_time = 0.0
    if return_stats and broad_t0 is not None:
        broad_time = time.perf_counter() - broad_t0
    if ij.shape[0] == 0:
        if return_stats:
            stats = {
                "t_broad": broad_time,
                "t_narrow": 0.0,
                "n_pairs": 0,
                "n_gjk": 0,
                "n_epa": 0,
            }
            return np.zeros((N, 2), dtype=np.float64), 0.0, stats
        return np.zeros((N, 2), dtype=np.float64), 0.0

    kernels = get_kernels() if use_numba else {}
    if kernels:
        NList = kernels["NList"]
        separation_chunk = kernels["separation_chunk"]

        parts_nb = _get_parts_nb(template, NList)

        P = ij.shape[0]
        if n_jobs is None or n_jobs <= 0:
            n_jobs = min(8, max(1, (P // 3000) + 1))
        if P < 8000:
            n_jobs = 1

        def run_chunk(sel_idx):
            ijc = ij[sel_idx]
            nic = nint[sel_idx]
            return separation_chunk(
                ijc, nic, t, coss, sins,
                float(L),
                float(template.local_center[0]),
                float(template.local_center[1]),
                float(template.radius),
                template.aabb_whole.astype(np.float64),
                template.part_centers.astype(np.float64),
                template.part_radii.astype(np.float64),
                template.aabb_parts.astype(np.float64),
                parts_nb,
                float(eps_gjk),
                int(itmax_gjk),
                float(tol_epa),
                int(itmax_epa),
            )

        narrow_t0 = time.perf_counter() if return_stats else None
        if n_jobs <= 1:
            mv, md, gjk_ct, epa_ct = run_chunk(np.arange(P, dtype=np.int64))
            moves = np.asarray(mv, dtype=np.float64)
            max_depth = float(md)
            if fallback_concave:
                moves, max_depth = _apply_concave_fallback(
                    moves,
                    max_depth,
                    ij,
                    nint,
                    template,
                    t,
                    coss,
                    sins,
                    float(L),
                )
            narrow_time = 0.0
            if return_stats and narrow_t0 is not None:
                narrow_time = time.perf_counter() - narrow_t0
            if return_stats:
                stats = {
                    "t_broad": broad_time,
                    "t_narrow": narrow_time,
                    "n_pairs": int(P),
                    "n_gjk": int(gjk_ct),
                    "n_epa": int(epa_ct),
                }
                return moves, max_depth, stats
            return moves, max_depth

        chunks = np.array_split(np.arange(P, dtype=np.int64), n_jobs)
        moves_total = np.zeros((N, 2), dtype=np.float64)
        md_total = 0.0
        gjk_total = 0
        epa_total = 0
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for mv, md, gjk_ct, epa_ct in ex.map(run_chunk, chunks):
                moves_total += np.asarray(mv, dtype=np.float64)
                if float(md) > md_total:
                    md_total = float(md)
                gjk_total += int(gjk_ct)
                epa_total += int(epa_ct)
        if fallback_concave:
            moves_total, md_total = _apply_concave_fallback(
                moves_total,
                md_total,
                ij,
                nint,
                template,
                t,
                coss,
                sins,
                float(L),
            )
        narrow_time = 0.0
        if return_stats and narrow_t0 is not None:
            narrow_time = time.perf_counter() - narrow_t0
        if return_stats:
            stats = {
                "t_broad": broad_time,
                "t_narrow": narrow_time,
                "n_pairs": int(P),
                "n_gjk": int(gjk_total),
                "n_epa": int(epa_total),
            }
            return moves_total, float(md_total), stats
        return moves_total, float(md_total)

    moves = np.zeros((N, 2), dtype=np.float64)
    max_depth = 0.0

    aabb_whole = template.aabb_whole
    aabb_parts = template.aabb_parts
    part_centers = template.part_centers
    part_normals = template.part_normals
    part_radii = template.part_radii
    parts = template.parts
    r_whole = float(template.radius)
    r_whole_sq = (r_whole + r_whole + 1e-12) ** 2

    lc = template.local_center
    lcx = float(lc[0])
    lcy = float(lc[1])

    world_centers = np.zeros((N, 2), dtype=np.float64)
    aabb_whole_world = np.zeros((N, 4), dtype=np.float64)

    nparts = part_centers.shape[0]
    part_centers_world = np.zeros((N, nparts, 2), dtype=np.float64)
    aabb_parts_world = np.zeros((N, nparts, 4), dtype=np.float64)
    parts_world: List[List[np.ndarray]] = [[None for _ in range(nparts)] for _ in range(N)]
    part_axes_world: List[List[np.ndarray]] = [[None for _ in range(nparts)] for _ in range(N)]

    for i in range(N):
        cA = float(coss[i])
        sA = float(sins[i])
        tAx = float(t[i, 0])
        tAy = float(t[i, 1])
        world_centers[i, 0] = cA * lcx - sA * lcy + tAx
        world_centers[i, 1] = sA * lcx + cA * lcy + tAy
        aabb_whole_world[i] = rot_aabb_world(aabb_whole, cA, sA, tAx, tAy)
        for pa in range(nparts):
            ca = part_centers[pa]
            part_centers_world[i, pa, 0] = cA * ca[0] - sA * ca[1] + tAx
            part_centers_world[i, pa, 1] = sA * ca[0] + cA * ca[1] + tAy
            aabb_parts_world[i, pa] = rot_aabb_world(aabb_parts[pa], cA, sA, tAx, tAy)
            parts_world[i][pa] = poly_world(parts[pa], cA, sA, tAx, tAy)
            part_axes_world[i][pa] = _rotate_axes(part_normals[pa], cA, sA)

    def aabb_overlap_shift(aabb_a: np.ndarray, aabb_b: np.ndarray, dx: float, dy: float, pad: float = 0.0) -> bool:
        return not (
            aabb_a[2] < aabb_b[0] + dx - pad
            or aabb_b[2] + dx < aabb_a[0] - pad
            or aabb_a[3] < aabb_b[1] + dy - pad
            or aabb_b[3] + dy < aabb_a[1] - pad
        )

    narrow_t0 = time.perf_counter() if return_stats else None
    for p in range(ij.shape[0]):
        i = int(ij[p, 0])
        j = int(ij[p, 1])
        nx = int(nint[p, 0])
        ny = int(nint[p, 1])
        dx = float(nx) * L
        dy = float(ny) * L

        dCx = world_centers[i, 0] - (world_centers[j, 0] + dx)
        dCy = world_centers[i, 1] - (world_centers[j, 1] + dy)
        if float(dCx * dCx + dCy * dCy) > r_whole_sq:
            continue

        aabbA = aabb_whole_world[i]
        aabbB = aabb_whole_world[j]
        if not aabb_overlap_shift(aabbA, aabbB, dx, dy, pad=0.0):
            continue

        shift_vec = None
        if dx != 0.0 or dy != 0.0:
            shift_vec = np.array([dx, dy], dtype=np.float64)

        best_d = 0.0
        best_n = None

        for pa in range(nparts):
            ra = float(part_radii[pa])
            cAp = part_centers_world[i, pa]
            aabbAp = aabb_parts_world[i, pa]
            Aw = parts_world[i][pa]
            axesA = part_axes_world[i][pa]

            for pb in range(nparts):
                rb = float(part_radii[pb])
                cBpx = part_centers_world[j, pb, 0] + dx
                cBpy = part_centers_world[j, pb, 1] + dy
                aabbBp = aabb_parts_world[j, pb]
                if not aabb_overlap_shift(aabbAp, aabbBp, dx, dy, pad=0.0):
                    continue
                dPx = float(cAp[0] - cBpx)
                dPy = float(cAp[1] - cBpy)
                sum_r = ra + rb + 1e-12
                if dPx * dPx + dPy * dPy > sum_r * sum_r:
                    continue

                Bw = parts_world[j][pb]
                axesB = part_axes_world[j][pb]
                if shift_vec is not None:
                    Bw = Bw + shift_vec
                dep, nvec = sat_mtv_with_axes(Aw, Bw, axesA, axesB)
                if dep <= 0.0:
                    continue
                if float(nvec[0] * (cAp[0] - cBpx) + nvec[1] * (cAp[1] - cBpy)) < 0.0:
                    nvec = -nvec
                if dep > best_d:
                    best_d = float(dep)
                    best_n = nvec

        if best_n is not None and best_d > 0.0:
            if best_d > max_depth:
                max_depth = best_d
            m = 0.5 * best_d * best_n
            moves[i] += m
            moves[j] -= m

    if fallback_concave:
        moves, max_depth = _apply_concave_fallback(
            moves,
            max_depth,
            ij,
            nint,
            template,
            t,
            coss,
            sins,
            float(L),
        )

    narrow_time = 0.0
    if return_stats and narrow_t0 is not None:
        narrow_time = time.perf_counter() - narrow_t0
    if return_stats:
        stats = {
            "t_broad": broad_time,
            "t_narrow": narrow_time,
            "n_pairs": int(ij.shape[0]),
            "n_gjk": 0,
            "n_epa": 0,
        }
        return moves, float(max_depth), stats
    return moves, float(max_depth)


def _apply_concave_fallback(
    moves: np.ndarray,
    max_depth: float,
    ij: np.ndarray,
    nint: np.ndarray,
    template: ShapeTemplate,
    translations: np.ndarray,
    coss: np.ndarray,
    sins: np.ndarray,
    L: float,
) -> Tuple[np.ndarray, float]:
    if ij.shape[0] == 0:
        return moves, float(max_depth)

    vertices = template.vertices
    if vertices.shape[0] < 3:
        return moves, float(max_depth)

    unique_ids = np.unique(ij).astype(np.int64)
    n_shapes = translations.shape[0]
    base_world = [None] * n_shapes
    base_aabb = [None] * n_shapes
    for idx in unique_ids:
        poly = poly_world(
            vertices,
            float(coss[idx]),
            float(sins[idx]),
            float(translations[idx, 0]),
            float(translations[idx, 1]),
        )
        base_world[int(idx)] = poly
        base_aabb[int(idx)] = aabb_from_points(poly)
    lc = template.local_center
    r_whole = float(template.radius)
    r_whole_sq = (r_whole + r_whole + 1e-12) ** 2
    eps = 1e-12

    for p in range(ij.shape[0]):
        i = int(ij[p, 0])
        j = int(ij[p, 1])
        nx = int(nint[p, 0])
        ny = int(nint[p, 1])

        shift_x = nx * L
        shift_y = ny * L
        poly_a = base_world[i]
        aabb_a = base_aabb[i]
        aabb_b = base_aabb[j]
        if (
            aabb_a[2] < aabb_b[0] + shift_x - eps
            or aabb_b[2] + shift_x < aabb_a[0] - eps
            or aabb_a[3] < aabb_b[1] + shift_y - eps
            or aabb_b[3] + shift_y < aabb_a[1] - eps
        ):
            continue

        cA = float(coss[i])
        sA = float(sins[i])
        cB = float(coss[j])
        sB = float(sins[j])
        tAx = float(translations[i, 0])
        tAy = float(translations[i, 1])
        tBx = float(translations[j, 0] + shift_x)
        tBy = float(translations[j, 1] + shift_y)

        cAw = world_center(lc, cA, sA, tAx, tAy)
        cBw = world_center(lc, cB, sB, tBx, tBy)
        d = cAw - cBw
        d2 = float(d[0] * d[0] + d[1] * d[1])
        if d2 > r_whole_sq:
            continue

        poly_b = base_world[j] + np.array([shift_x, shift_y], dtype=np.float64)
        if not _polygons_intersect_shifted(poly_a, poly_b, 0.0, 0.0, eps=eps):
            continue

        overlap, nvec = sat_mtv(poly_a, poly_b)
        if overlap <= 0.0:
            continue

        if overlap > max_depth:
            max_depth = float(overlap)
        disp = 0.5 * float(overlap) * nvec
        moves[i] += disp
        moves[j] -= disp

    return moves, float(max_depth)


def _clamp_translation_inside(
    template: ShapeTemplate,
    angle: float,
    translation: np.ndarray,
    L: float,
) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    center = world_center(template.local_center, c, s, float(translation[0]), float(translation[1]))
    shift = clamp_unit(center, float(L)) - center
    return (translation + shift).astype(np.float64)


def clamp_shapes_inside(shapes: List[PolygonShape], L: float) -> List[PolygonShape]:
    clamped: List[PolygonShape] = []
    for s in shapes:
        t = _clamp_translation_inside(s.template, float(s.angle), s.translation, L)
        clamped.append(PolygonShape(template=s.template, angle=s.angle, translation=t))
    return clamped

def sample_uniform_2d_halton(n, L, seed=0, scramble=True):
    sampler = qmc.Halton(d=2, scramble=scramble, seed=seed)
    u = sampler.random(n)
    return u * L


def shapes_from_template(template: ShapeTemplate, n: int, L: float, seed: Optional[int] = None) -> List[PolygonShape]:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    pts = sample_uniform_2d_halton(n, L, seed=None if seed is None else int(seed), scramble=True)
    shapes: List[PolygonShape] = []
    for k in range(n):
        t = pts[k]
        t = _clamp_translation_inside(template, float(angles[k]), t, L)
        shapes.append(PolygonShape(template=template, angle=float(angles[k]), translation=t))
    return shapes


def update_shapes(
    shapes: List[PolygonShape],
    moves: np.ndarray,
    L: float,
    step_scale: float = 1.0,
    cap_scale: float = 0.75,
    keep_inside: bool = True,
) -> List[PolygonShape]:
    t = np.asarray([s.translation for s in shapes], dtype=np.float64)
    dn = np.linalg.norm(moves, axis=1)
    cap = cap_scale * float(shapes[0].template.radius + 1e-12)
    sc = np.minimum(1.0, cap / (dn + 1e-12))
    mm = moves * sc[:, None]
    t2 = t + step_scale * mm
    if keep_inside:
        clamped = []
        for i, s in enumerate(shapes):
            t = _clamp_translation_inside(s.template, float(s.angle), t2[i], L)
            clamped.append(PolygonShape(template=s.template, angle=s.angle, translation=t))
        return clamped
    t2 = clamp_unit(t2, L)
    return [PolygonShape(template=s.template, angle=s.angle, translation=t2[i].astype(np.float64)) for i, s in enumerate(shapes)]
