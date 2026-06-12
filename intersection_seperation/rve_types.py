from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class ShapeTemplate:
    aabb_parts: np.ndarray
    aabb_whole: np.ndarray
    local_center: np.ndarray
    part_centers: np.ndarray
    part_normals: List[np.ndarray]
    part_radii: np.ndarray
    parts: List[np.ndarray]
    radius: float
    vertices: np.ndarray
    is_convex: bool


@dataclass(frozen=True)
class PolygonShape:
    template: ShapeTemplate
    angle: float
    translation: np.ndarray


def local_normals_from_polygon(poly: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    nrm = np.linalg.norm(normals, axis=1)
    keep = nrm > eps
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.float64)
    normals = normals[keep]
    nrm = nrm[keep] + eps
    return (normals.T / nrm).T.astype(np.float64)


def is_polygon_convex(poly: np.ndarray, eps: float = 1e-12) -> bool:
    if poly.shape[0] < 4:
        return True
    sign = 0.0
    n = poly.shape[0]
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        p2 = poly[(i + 2) % n]
        dx1 = p1[0] - p0[0]
        dy1 = p1[1] - p0[1]
        dx2 = p2[0] - p1[0]
        dy2 = p2[1] - p1[1]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) <= eps:
            continue
        if sign == 0.0:
            sign = cross
        elif cross * sign < 0.0:
            return False
    return True
