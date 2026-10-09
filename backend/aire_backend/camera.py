"""Cámara de SketchUp → matriz de vista y proyección equivalentes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    perspective: bool
    vfov: float  # radianes, solo en perspectiva
    ortho_height: float  # metros, solo en paralela
    aspect: float  # ancho / alto de la imagen
    near: float = 0.01  # metros; SketchUp recorta muy cerca del ojo

    @classmethod
    def from_scene(cls, cam: dict, view: dict) -> "Camera":
        aspect = cam.get("aspect_ratio") or 0.0
        if aspect <= 0:
            aspect = view["width"] / view["height"]
        fov = math.radians(cam.get("fov_deg", 35.0))
        if cam.get("fov_is_height", True):
            vfov = fov
        else:
            vfov = 2.0 * math.atan(math.tan(fov / 2.0) / aspect)
        return cls(
            eye=np.asarray(cam["eye"], dtype=np.float64),
            target=np.asarray(cam["target"], dtype=np.float64),
            up=np.asarray(cam["up"], dtype=np.float64),
            perspective=bool(cam.get("perspective", True)),
            vfov=vfov,
            ortho_height=float(cam.get("ortho_height") or 1.0),
            aspect=aspect,
        )

    def resolution(self, width: int) -> tuple[int, int]:
        return int(width), max(1, int(round(width / self.aspect)))

    @property
    def direction(self) -> np.ndarray:
        d = self.target - self.eye
        return d / np.linalg.norm(d)

    def basis(self) -> np.ndarray:
        """Filas: ejes derecha, arriba, atrás (la cámara mira hacia -Z de vista)."""
        f = self.direction
        r = np.cross(f, self.up)
        if np.linalg.norm(r) < 1e-9:  # up paralelo a la dirección
            r = np.cross(f, [0.0, 1.0, 0.0] if abs(f[2]) > 0.9 else [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return np.stack([r, u, -f])

    def view_matrix(self) -> np.ndarray:
        rot = self.basis()
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = -rot @ self.eye
        return m

    def to_view(self, points: np.ndarray) -> np.ndarray:
        """Puntos de mundo (..., 3) → espacio de vista (..., 3)."""
        m = self.view_matrix()
        return points @ m[:3, :3].T + m[:3, 3]

    def projection_params(self) -> tuple[float, float]:
        """Escalas (sx, sy) tales que x_ndc = sx * x / d (perspectiva) o sx * x (paralela)."""
        if self.perspective:
            fy = 1.0 / math.tan(self.vfov / 2.0)
            return fy / self.aspect, fy
        half_h = self.ortho_height / 2.0
        return 1.0 / (half_h * self.aspect), 1.0 / half_h

    def intrinsics(self, width: int, height: int) -> dict:
        """Intrínsecos tipo pinhole en píxeles (útiles para otros modelos 3D/IA)."""
        sx, sy = self.projection_params()
        return {
            "width": width,
            "height": height,
            "fx": sx * width / 2.0,
            "fy": sy * height / 2.0,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "perspective": self.perspective,
        }


def level_view(cam: Camera, max_pitch_deg: float = 40.0):
    """Corrección de verticales (perspectiva de dos puntos, como un objetivo descentrable).

    Las fotos de interiorismo se hacen con la cámara nivelada para que las paredes salgan
    rectas; si la vista de SketchUp mira algo hacia abajo o hacia arriba, las verticales
    convergen y la imagen delata que es 3D. Se nivela la cámara y se desplaza el encuadre
    (shift) para que el centro de la imagen siga en el mismo punto.
    Devuelve (filas derecha, arriba, atrás; shift_y en altos de imagen, como Blender con
    sensor_fit VERTICAL) o None si no procede (paralela, muy picada o ya nivelada)."""
    if not cam.perspective:
        return None
    f = cam.direction
    pitch = math.asin(float(np.clip(f[2], -1.0, 1.0)))
    if abs(pitch) < math.radians(0.5) or abs(pitch) > math.radians(max_pitch_deg):
        return None
    flat = np.array([f[0], f[1], 0.0])
    flat /= np.linalg.norm(flat)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(flat, up)
    shift = math.tan(pitch) / (2.0 * math.tan(cam.vfov / 2.0))
    return np.stack([r, up, -flat]), shift
