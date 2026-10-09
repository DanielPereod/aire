"""Métrica de fidelidad estructural: ¿respeta el render la geometría del modelo?

- recall: fracción de las líneas del modelo (aristas + contornos) que tienen un
  borde en el render a menos de `tol` píxeles. Alto = no se ha perdido estructura.
- spurious: fracción de los bordes fuertes del render que están lejos (> 2·tol)
  de cualquier línea del modelo. Incluye texturas (vetas, juntas), así que sirve
  para comparar ajustes entre sí, no como valor absoluto; un salto grande suele
  indicar objetos o huecos inventados.

    python -m aire_backend.fidelity <render.png> <carpeta_pases>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def edge_map(img: Image.Image, keep: float = 0.08) -> np.ndarray:
    """Bordes fuertes: magnitud Sobel por encima del percentil (1 - keep)."""
    g = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 100 * (1 - keep))
    return mag > max(thr, 4.0)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * r + 1))
    return np.asarray(img) > 127


def score(render: Image.Image, lines: np.ndarray, tol: int = 3) -> dict:
    h, w = lines.shape
    if render.size != (w, h):
        render = render.resize((w, h), Image.LANCZOS)
    edges = edge_map(render)
    recall = float((_dilate(edges, tol) & lines).sum() / max(lines.sum(), 1))
    spurious = float((edges & ~_dilate(lines, 2 * tol)).sum() / max(edges.sum(), 1))
    return {"recall": round(recall, 4), "spurious": round(spurious, 4), "tol_px": tol}


def load_lines(passes_dir: Path) -> np.ndarray:
    return np.asarray(Image.open(passes_dir / "lines.png").convert("L")) < 128


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fidelidad estructural de un render frente a los pases")
    ap.add_argument("render")
    ap.add_argument("passes_dir")
    ap.add_argument("--tol", type=int, default=3)
    args = ap.parse_args(argv)
    print(json.dumps(score(Image.open(args.render), load_lines(Path(args.passes_dir)), args.tol), indent=2))


if __name__ == "__main__":
    main()
