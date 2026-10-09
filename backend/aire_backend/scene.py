"""Formato de escena exportado desde SketchUp (aire-scene v1).

Una exportación es una carpeta con:

    scene.json      metadatos: cámara, materiales, árbol de objetos, sol, layout binario
    geometry.bin    buffers little-endian descritos en scene.json["geometry"]["buffers"]
    textures/       texturas de los materiales (PNG/JPG tal y como las escribe SketchUp)
    viewport.png    (opcional) captura del visor a la misma resolución, solo para verificar

Unidades: metros. Ejes: los de SketchUp (Z arriba). Los triángulos vienen en
coordenadas de mundo, con el orden de vértices tal que el producto vectorial
apunta hacia la cara frontal de la cara original de SketchUp.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "aire-scene"
VERSION = 1

# (nombre, dtype, componentes por elemento, ¿por triángulo o por arista?)
BUFFERS = [
    ("positions", "<f4", 9, "tri"),
    ("normals", "<f4", 9, "tri"),
    ("uvs", "<f4", 6, "tri"),
    ("tri_object", "<u4", 1, "tri"),
    ("tri_material_front", "<u4", 1, "tri"),
    ("tri_material_back", "<u4", 1, "tri"),
    ("edge_positions", "<f4", 6, "edge"),
    ("edge_object", "<u4", 1, "edge"),
    ("tri_clip", "<u4", 1, "tri"),  # índice en scene.json["clip_sets"] (0 = sin recorte)
    ("edge_clip", "<u4", 1, "edge"),
]


@dataclass
class Scene:
    root: Path
    meta: dict
    positions: np.ndarray  # (N, 3, 3) float32, mundo
    normals: np.ndarray  # (N, 3, 3) float32, normales por vértice (mundo)
    uvs: np.ndarray  # (N, 3, 2) float32, en repeticiones de textura
    tri_object: np.ndarray  # (N,) uint32, id de objeto hoja
    tri_material_front: np.ndarray  # (N,) uint32
    tri_material_back: np.ndarray  # (N,) uint32
    edge_positions: np.ndarray  # (M, 2, 3) float32
    edge_object: np.ndarray  # (M,) uint32
    tri_clip: np.ndarray  # (N,) uint32, conjunto de planos de sección que recortan el triángulo
    edge_clip: np.ndarray  # (M,) uint32
    _children: dict = field(default_factory=dict, repr=False)

    @property
    def camera(self) -> dict:
        return self.meta["camera"]

    @property
    def view(self) -> dict:
        return self.meta["view"]

    @property
    def section_planes(self) -> list[list[float]]:
        """Planos [nx, ny, nz, d] en mundo (metros): n·x + d = 0."""
        return self.meta.get("section_planes", [])

    @property
    def clip_sets(self) -> list[list[int]]:
        """Conjuntos de índices de section_planes; el 0 es siempre el vacío."""
        return self.meta.get("clip_sets", [[]])

    @property
    def materials(self) -> list[dict]:
        return self.meta["materials"]

    @property
    def objects(self) -> list[dict]:
        return self.meta["objects"]

    def children(self, object_id: int) -> list[int]:
        if not self._children:
            for obj in self.objects:
                parent = obj.get("parent")
                if parent is not None:
                    self._children.setdefault(parent, []).append(obj["id"])
        return self._children.get(object_id, [])

    def descendants(self, object_id: int) -> list[int]:
        """El objeto y todos sus descendientes (para máscaras de componentes anidados)."""
        out, stack = [], [object_id]
        while stack:
            oid = stack.pop()
            out.append(oid)
            stack.extend(self.children(oid))
        return out

    def ancestors(self, object_id: int) -> list[int]:
        out = []
        parent = self.objects[object_id].get("parent")
        while parent is not None:
            out.append(parent)
            parent = self.objects[parent].get("parent")
        return out

    def find_objects(self, query: str) -> list[dict]:
        """Busca por id exacto o por subcadena en nombre / definición / etiqueta."""
        if query.isdigit():
            oid = int(query)
            return [self.objects[oid]] if 0 <= oid < len(self.objects) else []
        q = query.lower()
        hits = []
        for obj in self.objects:
            fields = (obj.get("name"), obj.get("definition"), obj.get("tag"))
            if any(f and q in f.lower() for f in fields):
                hits.append(obj)
        return hits


def load_scene(path: str | Path) -> Scene:
    root = Path(path)
    if root.is_file():
        root = root.parent
    meta = json.loads((root / "scene.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{root}: no es una exportación {FORMAT}")
    if meta.get("version") != VERSION:
        raise ValueError(f"{root}: versión {meta.get('version')} no soportada (esperada {VERSION})")

    geo = meta["geometry"]
    raw = (root / geo["file"]).read_bytes()
    n_tri, n_edge = geo["triangles"], geo["edges"]
    arrays = {}
    for buf in geo["buffers"]:
        count = buf["count"]
        nbytes = count * np.dtype(buf["dtype"]).itemsize
        if buf["offset"] + nbytes > len(raw):
            raise ValueError(f"geometry.bin truncado en el buffer {buf['name']}")
        arrays[buf["name"]] = np.frombuffer(raw, dtype=buf["dtype"], count=count, offset=buf["offset"])

    return Scene(
        root=root,
        meta=meta,
        positions=arrays["positions"].reshape(n_tri, 3, 3),
        normals=arrays["normals"].reshape(n_tri, 3, 3),
        uvs=arrays["uvs"].reshape(n_tri, 3, 2),
        tri_object=arrays["tri_object"],
        tri_material_front=arrays["tri_material_front"],
        tri_material_back=arrays["tri_material_back"],
        edge_positions=arrays["edge_positions"].reshape(n_edge, 2, 3),
        edge_object=arrays["edge_object"],
        tri_clip=arrays.get("tri_clip", np.zeros(n_tri, dtype=np.uint32)),
        edge_clip=arrays.get("edge_clip", np.zeros(n_edge, dtype=np.uint32)),
    )


def save_scene(path: str | Path, meta: dict, arrays: dict[str, np.ndarray]) -> Path:
    """Escribe una exportación. Lo usa la escena sintética; el exportador Ruby
    escribe exactamente el mismo layout (ver sketchup/aire/scene_writer.rb)."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    n_tri = int(np.asarray(arrays["positions"]).reshape(-1, 9).shape[0])
    n_edge = int(np.asarray(arrays["edge_positions"]).reshape(-1, 6).shape[0])

    buffers, chunks, offset = [], [], 0
    for name, dtype, comps, kind in BUFFERS:
        n = n_tri if kind == "tri" else n_edge
        if name not in arrays and name.endswith("_clip"):
            arrays = {**arrays, name: np.zeros(n, dtype=np.uint32)}
        data = np.ascontiguousarray(np.asarray(arrays[name], dtype=dtype).reshape(-1))
        if data.size != n * comps:
            raise ValueError(f"{name}: {data.size} valores, esperados {n * comps}")
        buffers.append({"name": name, "dtype": dtype, "offset": offset, "count": int(data.size)})
        chunks.append(data.tobytes())
        offset += data.nbytes

    meta = dict(meta)
    meta.update(format=FORMAT, version=VERSION, units="m")
    meta["geometry"] = {"file": "geometry.bin", "triangles": n_tri, "edges": n_edge, "buffers": buffers}
    (root / "geometry.bin").write_bytes(b"".join(chunks))
    (root / "scene.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return root
