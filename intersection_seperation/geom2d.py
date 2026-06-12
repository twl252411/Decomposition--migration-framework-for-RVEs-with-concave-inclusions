from typing import Tuple
import numpy as np


def aabb_from_points(P: np.ndarray) -> np.ndarray:
    mn = P.min(axis=0)
    mx = P.max(axis=0)
    return np.array([float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])], dtype=np.float64)


def angle_cs(theta: float) -> Tuple[float, float]:
    return float(np.cos(theta)), float(np.sin(theta))


def clamp_unit(x: np.ndarray, L: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - L * np.floor(x / L)


def convex_hull_center_radius(P: np.ndarray) -> Tuple[np.ndarray, float]:
    c = P.mean(axis=0)
    r = float(np.max(np.linalg.norm(P - c, axis=1)))
    return c.astype(np.float64), r


def map_min_image(d: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=np.float64)
    nint = np.floor(d / L + 0.5).astype(np.int64)
    return d - L * nint, nint


def rot_aabb_world(aabb: np.ndarray, c: float, s: float, tx: float, ty: float) -> np.ndarray:
    minx, miny, maxx, maxy = float(aabb[0]), float(aabb[1]), float(aabb[2]), float(aabb[3])
    x0 = minx * c - miny * s + tx
    y0 = minx * s + miny * c + ty
    x1 = minx * c - maxy * s + tx
    y1 = minx * s + maxy * c + ty
    x2 = maxx * c - miny * s + tx
    y2 = maxx * s + miny * c + ty
    x3 = maxx * c - maxy * s + tx
    y3 = maxx * s + maxy * c + ty
    mnx = min(x0, x1, x2, x3)
    mny = min(y0, y1, y2, y3)
    mxx = max(x0, x1, x2, x3)
    mxy = max(y0, y1, y2, y3)
    return np.array([mnx, mny, mxx, mxy], dtype=np.float64)


def world_center(local_center: np.ndarray, c: float, s: float, tx: float, ty: float) -> np.ndarray:
    lc = local_center
    return np.array([c * lc[0] - s * lc[1] + tx, s * lc[0] + c * lc[1] + ty], dtype=np.float64)
