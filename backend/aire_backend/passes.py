"""Genera los pases de control (G-buffer) a partir de una exportación de SketchUp.

Uso:
    python -m aire_backend.passes <carpeta_export> [--width 1536] [--out <carpeta>]

Salidas (en <carpeta_export>/passes por defecto):
    depth.npy            float32 (H, W), profundidad métrica en metros (0 = fondo)
    depth16.png          uint16, milímetros (exacto hasta 65 m)
    depth_control.png    8 bits, cerca = blanco (convención ControlNet/MiDaS)
    normal.png           normales en espacio de cámara, RGB = (x der, y arriba, z hacia la cámara)
    ids.npy              int32 (H, W), id de objeto hoja (-1 = fondo)
    ids.png              ids con colores estables (para ver/depurar)
    material.npy         int32 (H, W), id de material visible (-1 = fondo)
    albedo.png           color/textura de material sin iluminación
    lines.png            líneas negras sobre blanco (aristas de SketchUp + contornos)
    edges_control.png    líneas blancas sobre negro (convención canny/lineart)
    objects_visible.json objetos visibles: píxeles, bbox, jerarquía
    passes.json          manifiesto con cámara, intrínsecos y ficheros
    overlay_viewport.png (si hay viewport.png) líneas sobre la captura de SketchUp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import Camera
from .raster import raster_edges, raster_triangles
from .scene import Scene, load_scene


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def smooth_normals(positions: np.ndarray, tri_object: np.ndarray, crease_deg: float) -> np.ndarray:
    """Normales suavizadas por vértice (SketchUp solo da la normal de cada cara).

    Promedia las caras del mismo objeto que comparten posición, ponderando por el
    ángulo de cada esquina (así no depende de cómo SketchUp triangule cada cara);
    si la media se aleja de la normal de la cara más que el ángulo de pliegue
    (esquinas de un mueble), se conserva la normal plana.
    """
    p = positions.astype(np.float64)
    unit_n = _normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
    e1 = _normalize(np.roll(p, -1, axis=1) - p)
    e2 = _normalize(np.roll(p, 1, axis=1) - p)
    angle = np.arccos(np.clip(np.einsum("nkc,nkc->nk", e1, e2), -1.0, 1.0))  # (N, 3)
    weighted = (angle[..., None] * unit_n[:, None, :]).reshape(-1, 3)
    q = np.round(p.reshape(-1, 3) / 1e-4).astype(np.uint64)  # posición cuantizada a 0,1 mm
    obj = np.repeat(tri_object.astype(np.uint64), 3)
    # Hash 64 bits de (x, y, z, objeto): una colisión solo mezclaría dos normales
    with np.errstate(over="ignore"):
        key = q[:, 0] * np.uint64(0x9E3779B97F4A7C15)
        key ^= q[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F) + np.uint64(0x165667B19E3779F9)
        key ^= q[:, 2] * np.uint64(0x27D4EB2F165667C5) + (key >> np.uint64(29))
        key ^= obj * np.uint64(0xFF51AFD7ED558CCD) + (key >> np.uint64(31))
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.stack([np.bincount(inv, weights=weighted[:, c]) for c in range(3)], axis=1)
    avg = _normalize(acc)[inv].reshape(-1, 3, 3)
    keep_flat = np.einsum("nkc,nc->nk", avg, unit_n) < np.cos(np.radians(crease_deg))
    out = np.where(keep_flat[..., None], unit_n[:, None, :], avg)
    return out.astype(np.float32)


def id_color(oid: int) -> tuple[int, int, int]:
    """Color estable por id (mismo id → mismo color en todas las vistas)."""
    h = hashlib.md5(str(oid).encode()).digest()
    return 64 + h[0] % 192, 64 + h[1] % 192, 64 + h[2] % 192


class Passes:
    def __init__(self, scene: Scene, width: int | None = None, smooth_angle: float = 35.0,
                 section_keep: str = "auto", height: int | None = None):
        """height fuerza el alto (p. ej. múltiplo de 16 para difusión) conservando el
        FOV vertical; el encuadre horizontal varía en la misma proporción (<1 %)."""
        self.scene = scene
        self.smooth_angle = smooth_angle
        self.section_keep = section_keep
        self.camera = Camera.from_scene(scene.camera, scene.view)
        self.width, self.height = self.camera.resolution(width or scene.view["width"])
        if height:
            self.height = int(height)
            self.camera.aspect = self.width / self.height
        self.arrays: dict[str, np.ndarray] = {}
        self._tex_cache: dict[Path, np.ndarray | None] = {}

    # ------------------------------------------------------------------ render
    def render(self) -> "Passes":
        sc, cam = self.scene, self.camera
        W, H = self.width, self.height
        sx, sy = cam.projection_params()

        planes, set_table = self._section_planes()
        vpos = cam.to_view(sc.positions.astype(np.float64))
        depth, tri, bary = raster_triangles(vpos, cam.perspective, sx, sy, cam.near, W, H,
                                            planes, set_table[sc.tri_clip])
        valid = tri >= 0
        t = np.where(valid, tri, 0)

        # Cara frontal / trasera según el lado que ve la cámara
        p = sc.positions.astype(np.float64)
        gnorm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        if cam.perspective:
            to_eye = cam.eye[None, :] - p[:, 0]
        else:
            to_eye = np.broadcast_to(-cam.direction, p[:, 0].shape)
        front = np.einsum("ij,ij->i", gnorm, to_eye) >= 0.0

        # Normales interpoladas, giradas hacia la cámara si vemos la cara trasera
        normals = sc.normals
        if self.smooth_angle > 0 and len(sc.positions):
            normals = smooth_normals(sc.positions, sc.tri_object, self.smooth_angle)
        n_world = np.einsum("hwk,hwkc->hwc", bary, normals[t].astype(np.float32))
        flip = np.where(front[t], 1.0, -1.0)[..., None]
        n_world = _normalize(n_world * flip)
        n_view = n_world @ cam.basis().T
        n_view[~valid] = 0.0

        obj = np.where(valid, sc.tri_object[t].astype(np.int32), -1)
        mat = np.where(front[t], sc.tri_material_front[t], sc.tri_material_back[t]).astype(np.int32)
        mat[~valid] = -1
        uv = np.einsum("hwk,hwkc->hwc", bary, sc.uvs[t].astype(np.float32))

        edge_obj = np.full((H, W), -1, dtype=np.int32)
        if len(sc.edge_positions):
            vseg = cam.to_view(sc.edge_positions.astype(np.float64))
            edge_obj = raster_edges(vseg, sc.edge_object.astype(np.int32), depth,
                                    cam.perspective, sx, sy, cam.near, 0.004, 0.005,
                                    planes, set_table[sc.edge_clip])

        self.arrays = {
            "depth": np.where(valid, depth, 0.0).astype(np.float32),
            "valid": valid,
            "normal": n_view.astype(np.float32),
            "normal_world": np.where(valid[..., None], n_world, 0.0).astype(np.float32),
            "ids": obj,
            "material": mat,
            "uv": uv,
            "sketchup_edges": edge_obj >= 0,
        }
        self.arrays["albedo"] = self._albedo()
        self.arrays["shaded"] = self._shaded()
        self.arrays["contours"] = self._contours()
        self.arrays["lines"] = self.arrays["sketchup_edges"] | self.arrays["contours"]
        return self

    def _section_planes(self) -> tuple[np.ndarray, np.ndarray]:
        """Planos de sección en espacio de vista, orientados para que el lado que se
        conserva cumpla n·x + d >= 0, y tabla conjunto → índices de plano."""
        cam = self.camera
        world = np.asarray(self.scene.section_planes, dtype=np.float64).reshape(-1, 4)
        rot = cam.basis()
        n_view = world[:, :3] @ rot.T
        d_view = world[:, :3] @ cam.eye + world[:, 3]  # valor del plano en el ojo
        if self.section_keep == "auto":
            # Se oculta el lado de la cámara: así se usan las secciones para mirar dentro
            sign = np.where(d_view > 0, -1.0, 1.0)
        elif self.section_keep == "negative":
            sign = -np.ones(len(world))
        else:
            sign = np.ones(len(world))
        planes = np.concatenate([n_view, d_view[:, None]], axis=1) * sign[:, None]
        sets = self.scene.clip_sets
        width = max(1, max((len(s) for s in sets), default=0))
        table = np.full((len(sets), width), -1, dtype=np.int32)
        for i, s in enumerate(sets):
            table[i, :len(s)] = s
        return np.ascontiguousarray(planes), table

    def _albedo(self) -> np.ndarray:
        mat, uv = self.arrays["material"], self.arrays["uv"]
        out = np.full(mat.shape + (3,), 255, dtype=np.uint8)  # fondo blanco
        for m in np.unique(mat):
            if m < 0:
                continue
            info = self.scene.materials[m]
            sel = mat == m
            rgb = np.asarray(info.get("color", [200, 200, 200])[:3], dtype=np.uint8)
            tex = info.get("texture")
            img = self._texture(tex["file"]) if tex else None
            if img is None:
                out[sel] = rgb
                continue
            th, tw = img.shape[:2]
            u = np.mod(uv[sel, 0], 1.0)
            v = np.mod(uv[sel, 1], 1.0)
            # En SketchUp v = 0 es el borde inferior de la imagen
            ix = np.clip((u * tw).astype(np.int64), 0, tw - 1)
            iy = np.clip(((1.0 - v) * th).astype(np.int64), 0, th - 1)
            out[sel] = img[iy, ix, :3]
        return out

    def _shaded(self) -> np.ndarray:
        """Imagen base fiel al modelo para el render IA: materiales y texturas reales con
        una iluminación sencilla calculada con la geometría (luz de cielo y luz principal
        suave). La IA parte de aquí y solo la «fotorrealiza», así no tiene
        que inventarse colores ni objetos."""
        a = self.arrays
        valid, n = a["valid"], a["normal_world"]
        cam = self.camera
        basis = cam.basis()  # derecha, arriba, atrás (en mundo)
        # Luz de cielo (Z arriba en SketchUp): suelos claros, techos algo más oscuros
        sky = 0.62 + 0.22 * n[..., 2]
        # Luz principal desde arriba, algo por detrás y a un lado de la cámara
        key_dir = _normalize(0.35 * basis[0] + 0.75 * np.array([0.0, 0.0, 1.0]) + 0.55 * basis[2])
        key = 0.32 * np.clip(n @ key_dir, 0.0, 1.0)
        # Sin oclusión en pantalla: con desniveles de profundidad crea halos alrededor de los
        # objetos que la IA copiaría; las sombras de contacto las añade bien la propia IA.
        shade = np.clip(sky + key, 0.0, 1.2)
        lin = (a["albedo"].astype(np.float32) / 255.0) ** 2.2
        out = np.clip(lin * shade[..., None], 0.0, 1.0) ** (1 / 2.2)
        img = (out * 255).round().astype(np.uint8)
        img[~valid] = 255
        return img

    def _texture(self, rel: str) -> np.ndarray | None:
        path = self.scene.root / rel
        if path not in self._tex_cache:
            try:
                self._tex_cache[path] = np.asarray(Image.open(path).convert("RGB"))
            except (OSError, ValueError):
                self._tex_cache[path] = None
        return self._tex_cache[path]

    def _contours(self) -> np.ndarray:
        """Contornos geométricos: cambios de objeto, saltos de profundidad y pliegues."""
        ids, depth, n = self.arrays["ids"], self.arrays["depth"], self.arrays["normal"]
        out = np.zeros(ids.shape, dtype=bool)
        for axis in (0, 1):
            a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            id_jump = ids[a] != ids[b]
            d0, d1 = depth[a], depth[b]
            both = (d0 > 0) & (d1 > 0)
            depth_jump = both & (np.abs(d0 - d1) > 0.03 * np.minimum(d0, d1) + 0.01)
            crease = both & (np.einsum("...c,...c->...", n[a], n[b]) < np.cos(np.radians(35)))
            edge = id_jump | depth_jump | crease
            out[a] |= edge
        return out

    # ------------------------------------------------------------------ objetos
    def visible_objects(self) -> list[dict]:
        ids = self.arrays["ids"]
        sc = self.scene
        leaf, counts = np.unique(ids[ids >= 0], return_counts=True)
        stats: dict[int, dict] = {}
        for oid, cnt in zip(leaf.tolist(), counts.tolist()):
            ys, xs = np.nonzero(ids == oid)
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            for target in [oid] + sc.ancestors(oid):
                s = stats.setdefault(target, {"pixels": 0, "bbox": list(box)})
                s["pixels"] += cnt
                b = s["bbox"]
                s["bbox"] = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])]
        total = ids.size
        out = []
        for oid in sorted(stats):
            o = sc.objects[oid]
            out.append({
                "id": oid,
                "name": o.get("name"),
                "definition": o.get("definition"),
                "type": o.get("type"),
                "tag": o.get("tag"),
                "parent": o.get("parent"),
                "pixels": stats[oid]["pixels"],
                "coverage": round(stats[oid]["pixels"] / total, 5),
                "bbox_xyxy": stats[oid]["bbox"],
            })
        return out

    # ------------------------------------------------------------------ guardar
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        a = self.arrays
        depth, valid = a["depth"], a["valid"]

        np.save(out / "depth.npy", depth)
        Image.fromarray(np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)).save(out / "depth16.png")
        Image.fromarray(depth_control(depth, valid)).save(out / "depth_control.png")

        nrm = ((a["normal"] * 0.5 + 0.5) * 255).round().astype(np.uint8)
        nrm[~valid] = 0
        Image.fromarray(nrm).save(out / "normal.png")

        np.save(out / "ids.npy", a["ids"])
        np.save(out / "material.npy", a["material"])
        Image.fromarray(colorize_ids(a["ids"])).save(out / "ids.png")
        Image.fromarray(a["albedo"]).save(out / "albedo.png")
        Image.fromarray(a["shaded"]).save(out / "shaded.png")

        lines = a["lines"]
        Image.fromarray(np.where(lines, 0, 255).astype(np.uint8)).save(out / "lines.png")
        Image.fromarray(np.where(lines, 255, 0).astype(np.uint8)).save(out / "edges_control.png")

        objects = self.visible_objects()
        (out / "objects_visible.json").write_text(
            json.dumps(objects, indent=2, ensure_ascii=False), encoding="utf-8")

        viewport = self.scene.root / "viewport.png"
        if viewport.exists():
            overlay(viewport, lines).save(out / "overlay_viewport.png")

        manifest = {
            "source": str(self.scene.root),
            "width": self.width,
            "height": self.height,
            "camera": self.scene.camera,
            "intrinsics": self.camera.intrinsics(self.width, self.height),
            "view_matrix": self.camera.view_matrix().tolist(),
            "depth_range_m": [float(depth[valid].min()), float(depth[valid].max())] if valid.any() else None,
            "conventions": {
                "normal": "espacio de cámara, x derecha, y arriba, z hacia la cámara; rgb=(n+1)/2",
                "depth_control": "cerca=blanco, normalizado a percentiles 1-99 de la profundidad visible",
                "ids": "id de objeto hoja de scene.json/objects; -1 = fondo",
            },
            "files": sorted(p.name for p in out.iterdir() if p.name != "passes.json"),
        }
        (out / "passes.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return out


def depth_control(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(depth.shape, dtype=np.uint8)
    if not valid.any():
        return out
    # Inversa de la profundidad: más contraste cerca, como los estimadores tipo MiDaS
    inv = np.zeros_like(depth)
    inv[valid] = 1.0 / depth[valid]
    lo, hi = np.percentile(inv[valid], [1, 99])
    if hi - lo < 1e-9:
        out[valid] = 255
        return out
    norm = np.clip((inv - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = (16 + norm[valid] * 239).astype(np.uint8)
    return out


def colorize_ids(ids: np.ndarray) -> np.ndarray:
    out = np.zeros(ids.shape + (3,), dtype=np.uint8)
    for oid in np.unique(ids):
        if oid >= 0:
            out[ids == oid] = id_color(int(oid))
    return out


def overlay(viewport_path: Path, lines: np.ndarray) -> Image.Image:
    """Pinta nuestras líneas en magenta sobre la captura de SketchUp para comprobar
    que los pases están alineados píxel a píxel con el visor."""
    h, w = lines.shape
    base = Image.open(viewport_path).convert("RGB")
    if base.size != (w, h):
        base = base.resize((w, h), Image.BILINEAR)
    img = np.asarray(base).copy()
    img = (img * 0.6 + 255 * 0.4).astype(np.uint8)
    img[lines] = (255, 0, 160)
    return Image.fromarray(img)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Genera pases de control desde una exportación AIRE")
    ap.add_argument("export_dir")
    ap.add_argument("--width", type=int, default=None, help="ancho en píxeles (por defecto, el de la exportación)")
    ap.add_argument("--out", default=None, help="carpeta de salida (por defecto <export_dir>/passes)")
    ap.add_argument("--section-keep", choices=["auto", "positive", "negative"], default="auto",
                    help="lado de los planos de sección que se conserva (auto: el opuesto a la cámara)")
    ap.add_argument("--smooth-angle", type=float, default=35.0,
                    help="ángulo de pliegue para suavizar normales (0 = normales planas de SketchUp)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    scene = load_scene(args.export_dir)
    t1 = time.perf_counter()
    passes = Passes(scene, args.width, args.smooth_angle, args.section_keep).render()
    t2 = time.perf_counter()
    out = passes.save(args.out or Path(scene.root) / "passes")
    t3 = time.perf_counter()
    print(f"{len(scene.positions):,} triángulos, {len(scene.edge_positions):,} aristas, "
          f"{passes.width}x{passes.height}px")
    print(f"carga {t1 - t0:.2f}s · render {t2 - t1:.2f}s · guardado {t3 - t2:.2f}s → {out}")


if __name__ == "__main__":
    main()
