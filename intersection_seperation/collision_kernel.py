from typing import Tuple
import numpy as np


def poly_world(P: np.ndarray, c: float, s: float, tx: float, ty: float) -> np.ndarray:
    x = P[:, 0]
    y = P[:, 1]
    X = c * x - s * y + tx
    Y = s * x + c * y + ty
    return np.column_stack((X, Y)).astype(np.float64)


def _axes_from_polygon(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    E = np.roll(P, -1, axis=0) - P
    N = np.column_stack((E[:, 1], -E[:, 0]))
    nrm = np.linalg.norm(N, axis=1)
    keep = nrm > eps
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.float64)
    N = N[keep]
    nrm = nrm[keep] + eps
    return (N.T / nrm).T.astype(np.float64)


def _proj(P: np.ndarray, n: np.ndarray) -> Tuple[float, float]:
    d = P @ n
    return float(d.min()), float(d.max())


def sat_mtv_with_axes(
    A: np.ndarray,
    B: np.ndarray,
    axesA: np.ndarray,
    axesB: np.ndarray,
) -> Tuple[float, np.ndarray]:
    best = 1e300
    best_n = None
    AB = None

    for axes in (axesA, axesB):
        if axes is None or axes.size == 0:
            continue
        if AB is None:
            AB = A.mean(axis=0) - B.mean(axis=0)
        for n in axes:
            amin, amax = _proj(A, n)
            bmin, bmax = _proj(B, n)
            o = min(amax, bmax) - max(amin, bmin)
            if o <= 0.0:
                return 0.0, np.zeros(2, dtype=np.float64)
            if o < best:
                nn = n.copy()
                if float(nn @ AB) < 0.0:
                    nn = -nn
                best = o
                best_n = nn

    if best_n is None:
        return 0.0, np.zeros(2, dtype=np.float64)
    return float(best), best_n.astype(np.float64)


def sat_mtv(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    axesA = _axes_from_polygon(A, eps=eps)
    axesB = _axes_from_polygon(B, eps=eps)
    return sat_mtv_with_axes(A, B, axesA, axesB)
