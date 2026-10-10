"""Detección de habitaciones sin techo (muy habitual en SketchUp: se modela sin él para ver
el interior desde arriba). Con la cámara dentro a la altura de los ojos se veía cielo abierto
y el sol entraba en vertical (pruebas 0.9-ojos); Cycles añade entonces un techo blanco."""

from __future__ import annotations

import math

import numpy as np


def _hits(tri: np.ndarray, origin: np.ndarray, d: np.ndarray, max_t: float) -> bool:
    """¿Corta el rayo origin + t·d (0 < t < max_t) algún triángulo? (Möller-Trumbore)."""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    return bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < max_t)).any())


def ceiling_height(tris: np.ndarray, eye: np.ndarray, reach: float = 15.0) -> float | None:
    """tris: (N, 3, 3) en metros, z hacia arriba. Altura a la que poner un techo si la cámara
    está dentro de una habitación sin él; None si ya tiene techo o la vista es desde encima
    de las paredes (corte intencionado)."""
    eye = np.asarray(eye, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.float64)
    if not len(tris):
        return None
    near = np.linalg.norm(tris.mean(axis=1)[:, :2] - eye[:2], axis=1) < reach
    above = tris[near & (tris[:, :, 2].max(axis=1) > eye[2])]
    dirs = [np.array([0.0, 0.0, 1.0])]
    for k in range(8):
        a = 2 * math.pi * k / 8
        dirs.append(np.array([math.cos(a) * 0.5, math.sin(a) * 0.5, math.sqrt(0.75)]))
    hits = sum(_hits(above, eye, d, reach) for d in dirs) if len(above) else 0
    if hits >= 6:
        return None  # hay techo
    # Altura de las paredes: caras verticales grandes alrededor de la cámara
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    area = np.linalg.norm(n, axis=1) / 2
    vertical = np.abs(n[:, 2]) < 0.1 * (2 * area + 1e-12)
    walls = near & vertical & (area > 0.25)
    if not walls.any():
        return None
    tops = tris[walls][:, :, 2].max(axis=1)
    top = float(np.percentile(tops, 90))
    return top if top > eye[2] + 0.1 else None
