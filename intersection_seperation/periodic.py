from typing import List, Tuple

import numpy as np

from .geom2d import angle_cs
from .rve_types import PolygonShape


def polygon_vertices_world(shape: PolygonShape) -> np.ndarray:
    c, s = angle_cs(shape.angle)
    v = shape.template.vertices
    x = v[:, 0]
    y = v[:, 1]
    X = c * x - s * y + shape.translation[0]
    Y = s * x + c * y + shape.translation[1]
    return np.column_stack([X, Y]).astype(np.float64)


def periodic_shift_candidates(vertices: np.ndarray, L: float) -> List[Tuple[float, float]]:
    minx = float(vertices[:, 0].min())
    maxx = float(vertices[:, 0].max())
    miny = float(vertices[:, 1].min())
    maxy = float(vertices[:, 1].max())
    eps = 1e-9

    shifts_x = [0.0]
    shifts_y = [0.0]
    if minx <= eps:
        shifts_x.append(L)
    if maxx >= L - eps:
        shifts_x.append(-L)
    if miny <= eps:
        shifts_y.append(L)
    if maxy >= L - eps:
        shifts_y.append(-L)

    shifts = []
    for dx in shifts_x:
        for dy in shifts_y:
            shifts.append((dx, dy))
    return shifts
