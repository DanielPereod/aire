"""Máscara exacta de un objeto (y sus subcomponentes) a partir de los pases.

Uso:
    python -m aire_backend.mask <carpeta_export> "silla" [--dilate 4]

Es la base de la edición por objeto ("pon la silla roja"): la máscara sale del
pase de ids, no de un segmentador, así que coincide con el modelo 3D.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .scene import Scene, load_scene


def object_mask(scene: Scene, ids: np.ndarray, object_id: int) -> np.ndarray:
    return np.isin(ids, scene.descendants(object_id))


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * px + 1))
    return np.asarray(img) > 127


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Máscara de un objeto a partir de los pases")
    ap.add_argument("export_dir")
    ap.add_argument("query", help="id o texto a buscar en nombre/definición/etiqueta")
    ap.add_argument("--passes", default=None, help="carpeta de pases (por defecto <export_dir>/passes)")
    ap.add_argument("--dilate", type=int, default=0, help="píxeles de dilatación para inpainting")
    args = ap.parse_args(argv)

    scene = load_scene(args.export_dir)
    passes = Path(args.passes or scene.root / "passes")
    ids = np.load(passes / "ids.npy")
    hits = scene.find_objects(args.query)
    if not hits:
        raise SystemExit(f"Ningún objeto coincide con {args.query!r}")
    for obj in hits:
        mask = dilate(object_mask(scene, ids, obj["id"]), args.dilate)
        if not mask.any():
            print(f"  #{obj['id']} {obj.get('name')!r}: no visible en esta vista")
            continue
        path = passes / f"mask_{obj['id']}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        print(f"  #{obj['id']} {obj.get('name')!r} ({obj.get('definition')}): {mask.sum():,} px → {path}")


if __name__ == "__main__":
    main()
