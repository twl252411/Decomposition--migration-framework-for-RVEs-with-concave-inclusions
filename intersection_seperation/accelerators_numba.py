from typing import Any, Dict, Optional
import numpy as np

_KERNELS_CACHE: Optional[Dict[str, Any]] = None


def get_kernels() -> Dict[str, Any]:
    global _KERNELS_CACHE
    if _KERNELS_CACHE is not None:
        return _KERNELS_CACHE
    try:
        import numba as nb
    except Exception:
        _KERNELS_CACHE = {}
        return _KERNELS_CACHE

    try:
        from numba.typed import List as NList
    except Exception:
        _KERNELS_CACHE = {}
        return _KERNELS_CACHE

    @nb.njit(cache=True, fastmath=True)
    def _aabb_overlap(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1, pad):
        return not (ax1 < bx0 - pad or bx1 < ax0 - pad or ay1 < by0 - pad or by1 < ay0 - pad)

    @nb.njit(cache=True, fastmath=True)
    def _center_world(lcx, lcy, c, s, tx, ty):
        return c * lcx - s * lcy + tx, s * lcx + c * lcy + ty

    @nb.njit(cache=True, fastmath=True)
    def _epa_edge(ax, ay, bx, by):
        ex = bx - ax
        ey = by - ay
        nx = ey
        ny = -ex
        nn = (nx * nx + ny * ny) ** 0.5
        if nn < 1e-14:
            nx, ny = 1.0, 0.0
            nn = 1.0
        nx /= nn
        ny /= nn
        if nx * ax + ny * ay < 0.0:
            nx = -nx
            ny = -ny
        d = nx * ax + ny * ay
        return nx, ny, d

    @nb.njit(cache=True, fastmath=True)
    def _farthest_point_local(V, dx, dy):
        best = 0
        bestv = V[0, 0] * dx + V[0, 1] * dy
        for i in range(1, V.shape[0]):
            v = V[i, 0] * dx + V[i, 1] * dy
            if v > bestv:
                bestv = v
                best = i
        return V[best, 0], V[best, 1]

    @nb.njit(cache=True, fastmath=True)
    def _gjk_triple_cross(ax, ay, bx, by, cx, cy):
        ac = ax * cx + ay * cy
        bc = bx * cx + by * cy
        return bx * ac - ax * bc, by * ac - ay * bc

    @nb.njit(cache=True, fastmath=True)
    def _minkowski_support(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, dx, dy):
        dAx = cA * dx + sA * dy
        dAy = -sA * dx + cA * dy
        dBx = cB * (-dx) + sB * (-dy)
        dBy = -sB * (-dx) + cB * (-dy)

        aLx, aLy = _farthest_point_local(VA, dAx, dAy)
        bLx, bLy = _farthest_point_local(VB, dBx, dBy)

        aWx = cA * aLx - sA * aLy + tAx
        aWy = sA * aLx + cA * aLy + tAy
        bWx = cB * bLx - sB * bLy + tBx
        bWy = sB * bLx + cB * bLy + tBy
        return aWx - bWx, aWy - bWy

    @nb.njit(cache=True, fastmath=True)
    def _gjk_intersection(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, eps, itmax):
        dx, dy = 1.0, 0.0
        ax, ay = _minkowski_support(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, dx, dy)

        sx = np.empty(3, dtype=np.float64)
        sy = np.empty(3, dtype=np.float64)
        n = 1
        sx[0] = ax
        sy[0] = ay
        dx, dy = -ax, -ay

        for _ in range(itmax):
            ax, ay = _minkowski_support(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, dx, dy)
            if ax * dx + ay * dy < eps:
                return 0, n, sx, sy

            if n < 3:
                sx[n] = ax
                sy[n] = ay
                n += 1
            else:
                sx[0] = sx[1]
                sy[0] = sy[1]
                sx[1] = sx[2]
                sy[1] = sy[2]
                sx[2] = ax
                sy[2] = ay
                n = 3

            if n == 2:
                bx, by = sx[0], sy[0]
                ax2, ay2 = sx[1], sy[1]
                abx, aby = bx - ax2, by - ay2
                aox, aoy = -ax2, -ay2
                if abx * aox + aby * aoy > 0.0:
                    ndx, ndy = _gjk_triple_cross(abx, aby, aox, aoy, abx, aby)
                    if ndx * ndx + ndy * ndy < eps * eps:
                        ndx, ndy = aby, -abx
                    dx, dy = ndx, ndy
                else:
                    sx[0] = ax2
                    sy[0] = ay2
                    n = 1
                    dx, dy = aox, aoy
            else:
                cx, cy = sx[0], sy[0]
                bx, by = sx[1], sy[1]
                ax2, ay2 = sx[2], sy[2]
                abx, aby = bx - ax2, by - ay2
                acx, acy = cx - ax2, cy - ay2
                aox, aoy = -ax2, -ay2

                p1x, p1y = _gjk_triple_cross(acx, acy, abx, aby, abx, aby)
                if p1x * aox + p1y * aoy > 0.0:
                    sx[0], sy[0] = bx, by
                    sx[1], sy[1] = ax2, ay2
                    n = 2
                    dx, dy = p1x, p1y
                    continue

                p2x, p2y = _gjk_triple_cross(abx, aby, acx, acy, acx, acy)
                if p2x * aox + p2y * aoy > 0.0:
                    sx[0], sy[0] = cx, cy
                    sx[1], sy[1] = ax2, ay2
                    n = 2
                    dx, dy = p2x, p2y
                    continue

                return 1, n, sx, sy

        return 1, n, sx, sy

    @nb.njit(cache=True, fastmath=True)
    def _rot_aabb_world(minx, miny, maxx, maxy, c, s, tx, ty):
        x0 = minx * c - miny * s + tx
        y0 = minx * s + miny * c + ty
        x1 = minx * c - maxy * s + tx
        y1 = minx * s + maxy * c + ty
        x2 = maxx * c - miny * s + tx
        y2 = maxx * s + miny * c + ty
        x3 = maxx * c - maxy * s + tx
        y3 = maxx * s + maxy * c + ty
        mnx = x0
        mny = y0
        mxx = x0
        mxy = y0
        if x1 < mnx: mnx = x1
        if x2 < mnx: mnx = x2
        if x3 < mnx: mnx = x3
        if y1 < mny: mny = y1
        if y2 < mny: mny = y2
        if y3 < mny: mny = y3
        if x1 > mxx: mxx = x1
        if x2 > mxx: mxx = x2
        if x3 > mxx: mxx = x3
        if y1 > mxy: mxy = y1
        if y2 > mxy: mxy = y2
        if y3 > mxy: mxy = y3
        return mnx, mny, mxx, mxy

    @nb.njit(cache=True, fastmath=True)
    def _epa_penetration(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, sx, sy, n_simplex, tol, itmax):
        if n_simplex < 2:
            return 0.0, 0.0, 0.0

        px = np.empty(32, dtype=np.float64)
        py = np.empty(32, dtype=np.float64)

        if n_simplex == 2:
            px[0] = sx[0]
            py[0] = sy[0]
            px[1] = sx[1]
            py[1] = sy[1]
            nx, ny, _ = _epa_edge(px[0], py[0], px[1], py[1])
            px[2], py[2] = _minkowski_support(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, nx, ny)
            m = 3
        else:
            px[0] = sx[0]
            py[0] = sy[0]
            px[1] = sx[1]
            py[1] = sy[1]
            px[2] = sx[2]
            py[2] = sy[2]
            m = 3

        for _ in range(itmax):
            best_i = 0
            best_d = 1e300
            best_nx = 0.0
            best_ny = 0.0

            for i in range(m):
                j = 0 if i + 1 == m else i + 1
                nx, ny, d = _epa_edge(px[i], py[i], px[j], py[j])
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_nx = nx
                    best_ny = ny

            supx, supy = _minkowski_support(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, best_nx, best_ny)
            ds = best_nx * supx + best_ny * supy
            if ds - best_d < tol:
                return best_d, best_nx, best_ny

            if m >= 31:
                return best_d, best_nx, best_ny

            insert_at = best_i + 1
            for k in range(m, insert_at, -1):
                px[k] = px[k - 1]
                py[k] = py[k - 1]
            px[insert_at] = supx
            py[insert_at] = supy
            m += 1

        best_d = 1e300
        best_nx = 0.0
        best_ny = 0.0
        for i in range(m):
            j = 0 if i + 1 == m else i + 1
            nx, ny, d = _epa_edge(px[i], py[i], px[j], py[j])
            if d < best_d:
                best_d = d
                best_nx = nx
                best_ny = ny
        return best_d, best_nx, best_ny

    @nb.njit(cache=True, fastmath=True)
    def separation_chunk(
        ij, nint, t, coss, sins,
        L,
        lcx, lcy,
        radius,
        aabb_whole,
        part_centers,
        part_radii,
        aabb_parts,
        parts,
        eps_gjk,
        itmax_gjk,
        tol_epa,
        itmax_epa,
    ):
        N = t.shape[0]
        moves = np.zeros((N, 2), dtype=np.float64)
        max_depth = 0.0
        nparts = part_centers.shape[0]
        gjk_calls = 0
        epa_calls = 0

        for p in range(ij.shape[0]):
            i = ij[p, 0]
            j = ij[p, 1]
            nx = nint[p, 0]
            ny = nint[p, 1]

            tAx = t[i, 0]
            tAy = t[i, 1]
            tBx = t[j, 0] + nx * L
            tBy = t[j, 1] + ny * L

            cA = coss[i]
            sA = sins[i]
            cB = coss[j]
            sB = sins[j]

            cAx, cAy = _center_world(lcx, lcy, cA, sA, tAx, tAy)
            cBx, cBy = _center_world(lcx, lcy, cB, sB, tBx, tBy)

            dxC = cAx - cBx
            dyC = cAy - cBy
            rsum = radius + radius + 1e-12
            if dxC * dxC + dyC * dyC > rsum * rsum:
                continue

            ax0, ay0, ax1, ay1 = _rot_aabb_world(aabb_whole[0], aabb_whole[1], aabb_whole[2], aabb_whole[3], cA, sA, tAx, tAy)
            bx0, by0, bx1, by1 = _rot_aabb_world(aabb_whole[0], aabb_whole[1], aabb_whole[2], aabb_whole[3], cB, sB, tBx, tBy)
            if not _aabb_overlap(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1, 0.0):
                continue

            best_d = 0.0
            best_nx = 0.0
            best_ny = 0.0

            for pa in range(nparts):
                ca0 = part_centers[pa, 0]
                ca1 = part_centers[pa, 1]
                ra = part_radii[pa]
                cAwx = cA * ca0 - sA * ca1 + tAx
                cAwy = sA * ca0 + cA * ca1 + tAy
                pax0, pay0, pax1, pay1 = _rot_aabb_world(aabb_parts[pa, 0], aabb_parts[pa, 1], aabb_parts[pa, 2], aabb_parts[pa, 3], cA, sA, tAx, tAy)

                for pb in range(nparts):
                    cb0 = part_centers[pb, 0]
                    cb1 = part_centers[pb, 1]
                    rb = part_radii[pb]
                    cBwx = cB * cb0 - sB * cb1 + tBx
                    cBwy = sB * cb0 + cB * cb1 + tBy
                    pbx0, pby0, pbx1, pby1 = _rot_aabb_world(aabb_parts[pb, 0], aabb_parts[pb, 1], aabb_parts[pb, 2], aabb_parts[pb, 3], cB, sB, tBx, tBy)
                    if not _aabb_overlap(pax0, pay0, pax1, pay1, pbx0, pby0, pbx1, pby1, 0.0):
                        continue

                    ddx = cAwx - cBwx
                    ddy = cAwy - cBwy
                    rsum_p = ra + rb + 1e-12
                    if ddx * ddx + ddy * ddy > rsum_p * rsum_p:
                        continue

                    VA = parts[pa]
                    VB = parts[pb]
                    gjk_calls += 1
                    hit, n_simplex, sx, sy = _gjk_intersection(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, eps_gjk, itmax_gjk)
                    if hit == 0:
                        continue

                    epa_calls += 1
                    dep, nx2, ny2 = _epa_penetration(VA, VB, cA, sA, tAx, tAy, cB, sB, tBx, tBy, sx, sy, n_simplex, tol_epa, itmax_epa)
                    if dep <= 0.0:
                        continue
                    if nx2 * (cAwx - cBwx) + ny2 * (cAwy - cBwy) < 0.0:
                        nx2 = -nx2
                        ny2 = -ny2
                    if dep > best_d:
                        best_d = dep
                        best_nx = nx2
                        best_ny = ny2

            if best_d > 0.0:
                if best_d > max_depth:
                    max_depth = best_d
                mx = 0.5 * best_d * best_nx
                my = 0.5 * best_d * best_ny
                moves[i, 0] += mx
                moves[i, 1] += my
                moves[j, 0] -= mx
                moves[j, 1] -= my

        return moves, max_depth, gjk_calls, epa_calls

    _KERNELS_CACHE = {
        "nb": nb,
        "NList": NList,
        "separation_chunk": separation_chunk,
    }
    return _KERNELS_CACHE
