"""Rasterizador z-buffer por CPU (numba).

Se hace en CPU a propósito: es determinista, no necesita contexto OpenGL y no
compite por la VRAM con el modelo de difusión. Produce, por píxel, el triángulo
visible, su profundidad métrica y las coordenadas baricéntricas respecto al
triángulo *original* (antes del recorte), con corrección de perspectiva, de modo
que cualquier atributo (normales, UV, ids) se interpola después en numpy.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _clip_near(v, near, clip):
    """Recorta un triángulo en espacio de vista contra el plano z = -near.

    Devuelve (n, pts, bary): hasta 4 vértices del polígono recortado y sus
    baricéntricas respecto al triángulo original.
    """
    pts = np.empty((4, 3))
    bary = np.zeros((4, 3))
    if not clip:
        for i in range(3):
            pts[i] = v[i]
            bary[i, i] = 1.0
        return 3, pts, bary
    n = 0
    for i in range(3):
        j = (i + 1) % 3
        di = -v[i, 2] - near  # >= 0 dentro
        dj = -v[j, 2] - near
        if di >= 0.0:
            pts[n] = v[i]
            bary[n, :] = 0.0
            bary[n, i] = 1.0
            n += 1
        if (di >= 0.0) != (dj >= 0.0):
            t = di / (di - dj)
            pts[n] = v[i] + t * (v[j] - v[i])
            bary[n, :] = 0.0
            bary[n, i] = 1.0 - t
            bary[n, j] = t
            n += 1
    return n, pts, bary


@njit(cache=True)
def _project(p, perspective, sx, sy, width, height):
    """Espacio de vista → (px, py, q, d). q = 1/d en perspectiva, 1 en paralela."""
    d = -p[2]
    if perspective:
        xn = sx * p[0] / d
        yn = sy * p[1] / d
        q = 1.0 / d
    else:
        xn = sx * p[0]
        yn = sy * p[1]
        q = 1.0
    return (xn + 1.0) * 0.5 * width, (1.0 - yn) * 0.5 * height, q, d


@njit(cache=True)
def _clipped(pos, planes, idx):
    """¿Elimina algún plano de sección este punto? idx: índices en planes (-1 = fin)."""
    for k in range(idx.shape[0]):
        j = idx[k]
        if j < 0:
            break
        if planes[j, 0] * pos[0] + planes[j, 1] * pos[1] + planes[j, 2] * pos[2] + planes[j, 3] < 0.0:
            return True
    return False


@njit(cache=True)
def raster_triangles(vpos, perspective, sx, sy, near, width, height, planes, tri_planes):
    """vpos: (N, 3, 3) vértices en espacio de vista (float64).
    planes: (K, 4) planos de sección en vista, se conserva n·x + d >= 0.
    tri_planes: (N, P) índices en planes por triángulo (-1 = ninguno).

    Devuelve depth (H, W) float32 [inf = fondo], tri (H, W) int32 [-1 = fondo],
    bary (H, W, 3) float32.
    """
    depth = np.full((height, width), np.inf, dtype=np.float32)
    tri = np.full((height, width), -1, dtype=np.int32)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    scr = np.empty((4, 4))  # px, py, q, d por vértice recortado
    pos = np.empty(3)

    for t in range(vpos.shape[0]):
        has_planes = tri_planes.shape[1] > 0 and tri_planes[t, 0] >= 0
        n, pts, bc = _clip_near(vpos[t], near, perspective)
        if n < 3:
            continue
        for k in range(n):
            px, py, q, d = _project(pts[k], perspective, sx, sy, width, height)
            scr[k, 0] = px
            scr[k, 1] = py
            scr[k, 2] = q
            scr[k, 3] = d
        # Abanico sobre el polígono recortado
        for f in range(1, n - 1):
            i0, i1, i2 = 0, f, f + 1
            x0, y0 = scr[i0, 0], scr[i0, 1]
            x1, y1 = scr[i1, 0], scr[i1, 1]
            x2, y2 = scr[i2, 0], scr[i2, 1]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-12:
                continue
            xmin = max(int(np.floor(min(x0, x1, x2))), 0)
            xmax = min(int(np.ceil(max(x0, x1, x2))), width - 1)
            ymin = max(int(np.floor(min(y0, y1, y2))), 0)
            ymax = min(int(np.ceil(max(y0, y1, y2))), height - 1)
            if xmin > xmax or ymin > ymax:
                continue
            inv_area = 1.0 / area
            q0, q1, q2 = scr[i0, 2], scr[i1, 2], scr[i2, 2]
            qd0, qd1, qd2 = q0 * scr[i0, 3], q1 * scr[i1, 3], q2 * scr[i2, 3]
            for py in range(ymin, ymax + 1):
                cy = py + 0.5
                for px in range(xmin, xmax + 1):
                    cx = px + 0.5
                    w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv_area
                    w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv_area
                    w2 = 1.0 - w0 - w1
                    if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                        continue
                    qs = w0 * q0 + w1 * q1 + w2 * q2
                    d = (w0 * qd0 + w1 * qd1 + w2 * qd2) / qs
                    if d < depth[py, px]:
                        a0 = w0 * q0 / qs
                        a1 = w1 * q1 / qs
                        a2 = w2 * q2 / qs
                        b0 = a0 * bc[i0, 0] + a1 * bc[i1, 0] + a2 * bc[i2, 0]
                        b1 = a0 * bc[i0, 1] + a1 * bc[i1, 1] + a2 * bc[i2, 1]
                        b2 = a0 * bc[i0, 2] + a1 * bc[i1, 2] + a2 * bc[i2, 2]
                        if has_planes:
                            for c in range(3):
                                pos[c] = b0 * vpos[t, 0, c] + b1 * vpos[t, 1, c] + b2 * vpos[t, 2, c]
                            if _clipped(pos, planes, tri_planes[t]):
                                continue
                        depth[py, px] = d
                        tri[py, px] = t
                        bary[py, px, 0] = b0
                        bary[py, px, 1] = b1
                        bary[py, px, 2] = b2
    return depth, tri, bary


@njit(cache=True)
def raster_edges(vseg, seg_object, depth, perspective, sx, sy, near, rel_tol, abs_tol, planes, seg_planes):
    """Dibuja aristas (M, 2, 3) en espacio de vista con test de profundidad contra
    el z-buffer de las caras. Devuelve (H, W) int32 con el id de objeto o -1."""
    height, width = depth.shape
    out = np.full((height, width), -1, dtype=np.int32)
    pos = np.empty(3)
    for s in range(vseg.shape[0]):
        has_planes = seg_planes.shape[1] > 0 and seg_planes[s, 0] >= 0
        a = vseg[s, 0].copy()
        b = vseg[s, 1].copy()
        if perspective:
            da = -a[2] - near
            db = -b[2] - near
            if da < 0.0 and db < 0.0:
                continue
            if da < 0.0:
                a = a + (da / (da - db)) * (b - a)
            elif db < 0.0:
                b = b + (db / (db - da)) * (a - b)
        ax, ay, aq, ad = _project(a, perspective, sx, sy, width, height)
        bx, by, bq, bd = _project(b, perspective, sx, sy, width, height)
        steps = int(max(abs(bx - ax), abs(by - ay))) + 1
        if steps > 4 * (width + height):  # recorte grosero de segmentos enormes
            steps = 4 * (width + height)
        for i in range(steps + 1):
            t = i / steps
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            ix = int(np.floor(x))
            iy = int(np.floor(y))
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                continue
            q = aq + t * (bq - aq)
            d = (aq * ad + t * (bq * bd - aq * ad)) / q
            if d <= depth[iy, ix] + max(abs_tol, rel_tol * d):
                if has_planes:
                    u = t * bq / q  # parámetro en espacio de vista (corrección de perspectiva)
                    for c in range(3):
                        pos[c] = a[c] + u * (b[c] - a[c])
                    if _clipped(pos, planes, seg_planes[s]):
                        continue
                out[iy, ix] = seg_object[s]
    return out
