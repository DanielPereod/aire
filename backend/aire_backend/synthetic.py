"""Escena de interior sintética en formato aire-scene (para tests y demos sin SketchUp).

    python -m aire_backend.synthetic <carpeta_salida>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .scene import save_scene


class SceneBuilder:
    def __init__(self):
        self.tris, self.norms, self.uvs = [], [], []
        self.tri_obj, self.mat_f, self.mat_b = [], [], []
        self.edges, self.edge_obj = [], []
        self.objects = [{"id": 0, "name": "Modelo", "definition": None, "type": "model", "parent": None, "tag": None}]
        self.materials = [{"id": 0, "name": "<por defecto>", "color": [235, 235, 235, 255], "texture": None}]

    def material(self, name, color, texture=None):
        mid = len(self.materials)
        self.materials.append({"id": mid, "name": name, "color": list(color) + [255], "texture": texture})
        return mid

    def object(self, name, parent=0, definition=None, type_="component", tag=None):
        oid = len(self.objects)
        self.objects.append({"id": oid, "name": name, "definition": definition, "type": type_,
                             "parent": parent, "tag": tag})
        return oid

    def quad(self, a, b, c, d, obj, mat, uv_scale=None, edges=True):
        """Cuadrilátero a-b-c-d en sentido antihorario visto desde su cara frontal."""
        pts = np.asarray([a, b, c, d], dtype=np.float64)
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n /= np.linalg.norm(n)
        if uv_scale:  # proyección plana sobre los ejes del propio quad
            ex = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
            ey = np.cross(n, ex)
            uv = np.stack([(pts - pts[0]) @ ex, (pts - pts[0]) @ ey], axis=1) / uv_scale
        else:
            uv = np.zeros((4, 2))
        for i, j, k in ((0, 1, 2), (0, 2, 3)):
            self.tris.append(pts[[i, j, k]])
            self.norms.append(np.tile(n, (3, 1)))
            self.uvs.append(uv[[i, j, k]])
            self.tri_obj.append(obj)
            self.mat_f.append(mat)
            self.mat_b.append(0)
        if edges:
            for i in range(4):
                self.edges.append(pts[[i, (i + 1) % 4]])
                self.edge_obj.append(obj)

    def box(self, lo, hi, obj, mat):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        v = lambda x, y, z: (x, y, z)  # noqa: E731
        self.quad(v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1), obj, mat)  # arriba
        self.quad(v(x0, y0, z0), v(x0, y1, z0), v(x1, y1, z0), v(x1, y0, z0), obj, mat)  # abajo
        self.quad(v(x0, y0, z0), v(x1, y0, z0), v(x1, y0, z1), v(x0, y0, z1), obj, mat)  # -y
        self.quad(v(x1, y1, z0), v(x0, y1, z0), v(x0, y1, z1), v(x1, y1, z1), obj, mat)  # +y
        self.quad(v(x0, y1, z0), v(x0, y0, z0), v(x0, y0, z1), v(x0, y1, z1), obj, mat)  # -x
        self.quad(v(x1, y0, z0), v(x1, y1, z0), v(x1, y1, z1), v(x1, y0, z1), obj, mat)  # +x

    def arrays(self):
        return {
            "positions": np.asarray(self.tris, dtype=np.float32),
            "normals": np.asarray(self.norms, dtype=np.float32),
            "uvs": np.asarray(self.uvs, dtype=np.float32),
            "tri_object": np.asarray(self.tri_obj, dtype=np.uint32),
            "tri_material_front": np.asarray(self.mat_f, dtype=np.uint32),
            "tri_material_back": np.asarray(self.mat_b, dtype=np.uint32),
            "edge_positions": np.asarray(self.edges, dtype=np.float32).reshape(-1, 2, 3),
            "edge_object": np.asarray(self.edge_obj, dtype=np.uint32),
        }


def build_interior(out_dir: str | Path, width: int = 960, height: int = 640) -> Path:
    """Salón de 5 x 4 x 2,7 m con mesa y una silla anidada (patas como subcomponentes)."""
    out = Path(out_dir)
    (out / "textures").mkdir(parents=True, exist_ok=True)
    # Textura de tarima: tablas alternas, 1 m x 1 m por repetición
    tex = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(8):
        tex[i * 16:(i + 1) * 16] = (150, 105, 70) if i % 2 else (170, 125, 85)
    tex[::16] = (90, 60, 40)
    Image.fromarray(tex).save(out / "textures" / "tarima.png")

    b = SceneBuilder()
    m_floor = b.material("Tarima roble", (160, 115, 78), {"file": "textures/tarima.png", "width_m": 1.0, "height_m": 1.0})
    m_wall = b.material("Pintura blanca", (240, 238, 232))
    m_wood = b.material("Nogal", (110, 70, 45))
    m_fabric = b.material("Tela gris", (120, 120, 125))
    m_metal = b.material("Acero negro", (30, 30, 32))

    W, D, H = 5.0, 4.0, 2.7
    room = b.object("Habitación", type_="group", tag="Arquitectura")
    floor = b.object("Suelo", parent=room, type_="group", tag="Arquitectura")
    walls = b.object("Paredes", parent=room, type_="group", tag="Arquitectura")
    b.quad((0, 0, 0), (W, 0, 0), (W, D, 0), (0, D, 0), floor, m_floor, uv_scale=1.0)
    b.quad((0, 0, H), (0, D, H), (W, D, H), (W, 0, H), walls, m_wall)  # techo
    b.quad((0, D, 0), (W, D, 0), (W, D, H), (0, D, H), walls, m_wall)  # fondo, mira a -y
    b.quad((0, 0, 0), (0, D, 0), (0, D, H), (0, 0, H), walls, m_wall)  # izquierda, mira a +x
    b.quad((W, D, 0), (W, 0, 0), (W, 0, H), (W, D, H), walls, m_wall)  # derecha, mira a -x
    b.quad((W, 0, 0), (0, 0, 0), (0, 0, H), (W, 0, H), walls, m_wall)  # trasera (detrás de la cámara)

    table = b.object("Mesa comedor", definition="Mesa 160x90", tag="Mobiliario")
    b.box((1.7, 2.2, 0.72), (3.3, 3.1, 0.76), table, m_wood)
    for x, y in ((1.75, 2.25), (3.2, 2.25), (1.75, 3.0), (3.2, 3.0)):
        b.box((x, y, 0.0), (x + 0.05, y + 0.05, 0.72), table, m_wood)

    chair = b.object("Silla", definition="Silla tapizada", tag="Mobiliario")
    seat = b.object("Asiento", parent=chair, definition="Asiento")
    b.box((2.3, 1.55, 0.42), (2.75, 2.0, 0.48), seat, m_fabric)
    back = b.object("Respaldo", parent=chair, definition="Respaldo")
    b.box((2.3, 1.5, 0.48), (2.75, 1.55, 0.95), back, m_fabric)
    for i, (x, y) in enumerate(((2.3, 1.5), (2.71, 1.5), (2.3, 1.96), (2.71, 1.96))):
        leg = b.object(f"Pata {i + 1}", parent=chair, definition="Pata acero")
        b.box((x, y, 0.0), (x + 0.04, y + 0.04, 0.42), leg, m_metal)

    meta = {
        "source": {"generator": "aire_backend.synthetic", "model_title": "Salón sintético"},
        "view": {"width": width, "height": height},
        "camera": {
            "eye": [2.5, 0.35, 1.55], "target": [2.5, 3.0, 0.7], "up": [0, 0, 1],
            "perspective": True, "fov_deg": 60.0, "fov_is_height": True,
            "ortho_height": None, "aspect_ratio": 0.0,
        },
        "sun": None,
        "materials": b.materials,
        "objects": b.objects,
    }
    return save_scene(out, meta, b.arrays())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Genera una escena de interior sintética")
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)
    print(build_interior(args.out_dir))


if __name__ == "__main__":
    main()
