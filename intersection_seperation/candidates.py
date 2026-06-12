from typing import Dict, List, Tuple
import numpy as np

from .accelerators_numba import get_kernels
from .geom2d import clamp_unit, map_min_image
from .rve_types import PolygonShape

_PAIR_KERNELS_CACHE = None


def _get_pair_kernels():
    global _PAIR_KERNELS_CACHE
    if _PAIR_KERNELS_CACHE is not None:
        return _PAIR_KERNELS_CACHE
    kernels = get_kernels()
    if not kernels:
        _PAIR_KERNELS_CACHE = {}
        return _PAIR_KERNELS_CACHE
    nb = kernels["nb"]

    @nb.njit(cache=True, fastmath=True)
    def _pair_candidates_from_centers_numba(centers, L, cell_size):
        n = centers.shape[0]
        if n == 0:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.float64),
            )

        nc = int(np.floor(L / cell_size))
        if nc < 1:
            nc = 1
        inv = nc / L

        head = np.full(nc * nc, -1, dtype=np.int64)
        next_idx = np.full(n, -1, dtype=np.int64)

        for i in range(n):
            ix = int(np.floor(centers[i, 0] * inv)) % nc
            iy = int(np.floor(centers[i, 1] * inv)) % nc
            cid = ix * nc + iy
            next_idx[i] = head[cid]
            head[cid] = i

        count = 0
        neighbor_cells = np.empty(9, dtype=np.int64)
        for i in range(n):
            ix = int(np.floor(centers[i, 0] * inv)) % nc
            iy = int(np.floor(centers[i, 1] * inv)) % nc
            ncell = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cid = ((ix + dx) % nc) * nc + ((iy + dy) % nc)
                    seen = False
                    for k in range(ncell):
                        if neighbor_cells[k] == cid:
                            seen = True
                            break
                    if not seen:
                        neighbor_cells[ncell] = cid
                        ncell += 1
            for k in range(ncell):
                j = head[neighbor_cells[k]]
                while j != -1:
                    if j > i:
                        count += 1
                    j = next_idx[j]

        if count == 0:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.float64),
            )

        ij = np.empty((count, 2), dtype=np.int64)
        nint = np.empty((count, 2), dtype=np.int64)
        diff = np.empty((count, 2), dtype=np.float64)
        idx = 0

        for i in range(n):
            ix = int(np.floor(centers[i, 0] * inv)) % nc
            iy = int(np.floor(centers[i, 1] * inv)) % nc
            ncell = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cid = ((ix + dx) % nc) * nc + ((iy + dy) % nc)
                    seen = False
                    for k in range(ncell):
                        if neighbor_cells[k] == cid:
                            seen = True
                            break
                    if not seen:
                        neighbor_cells[ncell] = cid
                        ncell += 1
            for k in range(ncell):
                j = head[neighbor_cells[k]]
                while j != -1:
                    if j > i:
                        dxv = centers[i, 0] - centers[j, 0]
                        dyv = centers[i, 1] - centers[j, 1]
                        nx = np.floor(dxv / L + 0.5)
                        ny = np.floor(dyv / L + 0.5)
                        diff[idx, 0] = dxv - L * nx
                        diff[idx, 1] = dyv - L * ny
                        nint[idx, 0] = int(nx)
                        nint[idx, 1] = int(ny)
                        ij[idx, 0] = i
                        ij[idx, 1] = j
                        idx += 1
                    j = next_idx[j]

        return ij, nint, diff

    _PAIR_KERNELS_CACHE = {"pair_candidates": _pair_candidates_from_centers_numba}
    return _PAIR_KERNELS_CACHE


def build_cell_list(centers: np.ndarray, L: float, cell_size: float) -> Tuple[Dict[Tuple[int, int], List[int]], int]:
    nc = max(1, int(np.floor(L / cell_size)))
    inv = nc / L
    cells: Dict[Tuple[int, int], List[int]] = {}
    for i, c in enumerate(centers):
        ix = int(np.floor(c[0] * inv)) % nc
        iy = int(np.floor(c[1] * inv)) % nc
        key = (ix, iy)
        cells.setdefault(key, []).append(i)
    return cells, nc


def neighbors_cell_keys(ix: int, iy: int, nc: int) -> List[Tuple[int, int]]:
    out = []
    seen = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = ((ix + dx) % nc, (iy + dy) % nc)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def pair_candidates_from_centers(
    centers: np.ndarray, L: float, cell_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if centers.size == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float64),
        )

    centers = clamp_unit(centers, L)
    cells, nc = build_cell_list(centers, L, cell_size)
    inv = nc / L

    count = 0
    for i, ci in enumerate(centers):
        ix = int(np.floor(ci[0] * inv)) % nc
        iy = int(np.floor(ci[1] * inv)) % nc
        for key in neighbors_cell_keys(ix, iy, nc):
            for j in cells.get(key, []):
                if j <= i:
                    continue
                count += 1

    if count == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float64),
        )

    ij = np.empty((count, 2), dtype=np.int64)
    nint = np.empty((count, 2), dtype=np.int64)
    diff = np.empty((count, 2), dtype=np.float64)
    idx = 0

    for i, ci in enumerate(centers):
        ix = int(np.floor(ci[0] * inv)) % nc
        iy = int(np.floor(ci[1] * inv)) % nc
        for key in neighbors_cell_keys(ix, iy, nc):
            for j in cells.get(key, []):
                if j <= i:
                    continue
                diff_mi, nint_xy = map_min_image(ci - centers[j], L)
                ij[idx, 0] = i
                ij[idx, 1] = j
                nint[idx, 0] = int(nint_xy[0])
                nint[idx, 1] = int(nint_xy[1])
                diff[idx, 0] = float(diff_mi[0])
                diff[idx, 1] = float(diff_mi[1])
                idx += 1
    return ij, nint, diff


def pair_candidates_from_centers_numba(
    centers: np.ndarray, L: float, cell_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernels = _get_pair_kernels()
    if not kernels:
        return pair_candidates_from_centers(centers, L, cell_size)
    centers = clamp_unit(centers, L)
    return kernels["pair_candidates"](centers, float(L), float(cell_size))


def filter_candidates_by_radius(
    ij: np.ndarray,
    nint: np.ndarray,
    centers: np.ndarray,
    L: float,
    max_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if ij.size == 0:
        return ij, nint
    diff = centers[ij[:, 0]] - centers[ij[:, 1]] - L * nint
    dist_sq = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
    max_dist_sq = float(max_dist) * float(max_dist)
    keep = dist_sq <= max_dist_sq
    if not np.any(keep):
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
        )
    return ij[keep], nint[keep]


def pair_candidates(shapes: List[PolygonShape], L: float, cell_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(shapes)
    if N == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.float64),
        )

    angles = np.array([s.angle for s in shapes], dtype=np.float64)
    coss = np.cos(angles)
    sins = np.sin(angles)
    trans = np.vstack([s.translation for s in shapes]).astype(np.float64)
    lc = shapes[0].template.local_center
    centers = np.empty((N, 2), dtype=np.float64)
    centers[:, 0] = coss * lc[0] - sins * lc[1] + trans[:, 0]
    centers[:, 1] = sins * lc[0] + coss * lc[1] + trans[:, 1]
    return pair_candidates_from_centers(centers, L, cell_size)
