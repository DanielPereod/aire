"""Render IA desde una exportación de SketchUp, vía ComfyUI.

    python -m aire_backend.render <carpeta_export> --prompt "salón nórdico, tarde"
    python -m aire_backend.render <carpeta_export> --sweep        # compara ajustes
    python -m aire_backend.render <carpeta_export> --dry-run      # solo genera workflow.json

Requiere ComfyUI en marcha (por defecto http://127.0.0.1:8188) con los modelos
del preset zimage-control (python -m aire_backend.models download ...).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .comfy import ComfyClient, ComfyError
from .fidelity import load_lines, score
from .passes import Passes, supersampled_shaded
from .prompt import build_prompt
from .scene import load_scene
from .workflows import KleinSettings, ZImageSettings, flux2_klein_edit, zimage_control


def round16(x: float) -> int:
    return max(16, int(round(x / 16.0)) * 16)


def prepare_passes(export_dir: str, width: int, section_keep: str = "auto",
                   supersample: int = 2) -> tuple[Passes, Path]:
    """supersample: antialiasing de la imagen base (shaded.png); 1 = sin él."""
    scene = load_scene(export_dir)
    probe = Passes(scene, width, section_keep=section_keep)
    w = round16(width)
    h = round16(w / probe.camera.aspect)
    passes = Passes(scene, w, section_keep=section_keep, height=h).render()
    if supersample > 1:
        passes.arrays["shaded"] = supersampled_shaded(scene, w, h, supersample, section_keep)
    out = passes.save(scene.root / f"passes_{w}x{h}")
    return passes, out


def sweep_grid(base: ZImageSettings) -> list[ZImageSettings]:
    """8 variantes (misma semilla) para afinar. Lo decisivo es cuánto se deja cambiar a la
    IA la imagen base (denoise) frente a cuánto manda la profundidad."""
    grid = [(n, d) for n in (0.4, 0.55, 0.7) for d in (0.6, 0.9)]
    runs = [replace(base, denoise=n, depth_strength=d, lines_strength=0.0) for n, d in grid]
    runs.append(replace(base, denoise=0.55, depth_strength=0.9, lines_strength=0.3))
    runs.append(replace(base, denoise=0.85, depth_strength=0.9, lines_strength=0.0))
    return runs


def label(s) -> str:
    if isinstance(s, KleinSettings):
        size = f"{s.width}×{s.height}"
        return f"FLUX klein · {size} · semilla {s.seed}" + (" · " + s.tag if s.tag else "")
    text = f"depth {s.depth_strength:.2f} · lines {s.lines_strength:.2f} · denoise {s.denoise:.2f}"
    return text + (f" · refine {s.refine:.2f}" if s.refine else "")


def contact_sheet(items: list[tuple[Image.Image, str]], cols: int = 4, tile_w: int = 480) -> Image.Image:
    w0, h0 = items[0][0].size
    tile_h = int(tile_w * h0 / w0)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, text) in enumerate(items):
        x, y = (i % cols) * tile_w, (i // cols) * (tile_h + 34)
        sheet.paste(img.convert("RGB").resize((tile_w, tile_h)), (x, y))
        draw.text((x + 6, y + tile_h + 4), text, fill=(0, 0, 0))
    return sheet


def run_renders(client: ComfyClient, passes: Passes, pdir: Path, runs: list[ZImageSettings], out_dir: Path,
                on_each=None, thumbs: bool = False, base: Path | None = None) -> list[dict]:
    """Sube los pases, renderiza cada ajuste y devuelve resultados con fidelidad.
    on_each(i, n, fase) se llama antes de cada render para informar del avance.
    base: imagen base alternativa a shaded.png (p. ej. un render de Cycles con luz real)."""
    if base is not None:
        fitted = pdir / "base_externa.png"
        Image.open(base).convert("RGB").resize((passes.width, passes.height), Image.Resampling.LANCZOS).save(fitted)
        init = client.upload_image(fitted)
    else:
        init = client.upload_image(pdir / "shaded.png")  # imagen base fiel al modelo
    needs_control = any(isinstance(s, ZImageSettings) for s in runs)
    depth = client.upload_image(pdir / "depth_control.png") if needs_control else None
    lines = client.upload_image(pdir / "edges_control.png") if needs_control else None
    ref_lines = load_lines(pdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, s in enumerate(runs):
        if on_each:
            on_each(i, len(runs))
        prefix = f"aire/{out_dir.name}_{i:02d}"
        if isinstance(s, KleinSettings):
            wf = flux2_klein_edit(s, init, prefix=prefix)
        else:
            wf = zimage_control(s, depth, lines, init, prefix=prefix)
        t0 = time.perf_counter()
        images = client.run(wf, check=(i == 0))
        secs = time.perf_counter() - t0
        img = Image.open(BytesIO(images[0]))
        path = out_dir / f"render_{i:02d}.png"
        img.save(path)
        item = {"file": path.name, "path": str(path), "seconds": round(secs, 1), "label": label(s),
                "fidelity": score(img, ref_lines), "settings": s.to_dict()}
        if thumbs:
            thumb = img.convert("RGB")
            thumb.thumbnail((640, 640))
            tpath = out_dir / f"render_{i:02d}_thumb.jpg"
            thumb.save(tpath, quality=85)
            item["thumb"] = str(tpath)
        results.append(item)
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render IA (Z-Image + ControlNet) desde una exportación AIRE")
    ap.add_argument("export_dir")
    ap.add_argument("--prompt", default="", help="estilo/ambiente; se completa con los objetos y materiales de la escena")
    ap.add_argument("--no-auto-prompt", action="store_true", help="usar solo --prompt")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--depth", type=float, default=0.8, help="fuerza del control de profundidad")
    ap.add_argument("--lines", type=float, default=0.0, help="fuerza del control de líneas (0 = sin líneas)")
    ap.add_argument("--denoise", type=float, default=0.55, help="cuánto cambia la IA la imagen base (0-1)")
    ap.add_argument("--refine", type=float, default=0.0, help="segunda pasada img2img (0 = sin ella)")
    ap.add_argument("--weight-dtype", default="fp8_e4m3fn", help="'default' (bf16) con ≥16 GB de VRAM")
    ap.add_argument("--section-keep", choices=["auto", "positive", "negative"], default="auto")
    ap.add_argument("--sweep", action="store_true", help="probar 8 combinaciones de ajustes con la misma semilla")
    ap.add_argument("--dry-run", action="store_true", help="no conecta con ComfyUI: guarda workflow.json")
    args = ap.parse_args(argv)

    passes, pdir = prepare_passes(args.export_dir, args.width, args.section_keep)
    scene = passes.scene
    prompt = build_prompt(scene, passes.arrays["ids"], passes.arrays["material"], passes.visible_objects(),
                          args.prompt, auto=not args.no_auto_prompt)
    base = ZImageSettings(
        width=passes.width, height=passes.height, prompt=prompt,
        seed=args.seed if args.seed is not None else random.randint(0, 2**31 - 1),
        steps=args.steps, weight_dtype=args.weight_dtype,
        depth_strength=args.depth, lines_strength=args.lines, denoise=args.denoise, refine=args.refine)
    runs = sweep_grid(base) if args.sweep else [base]

    out_dir = scene.root / "renders" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{passes.width}x{passes.height}px · semilla {base.seed}\nprompt: {prompt}\n")

    if args.dry_run:
        for i, s in enumerate(runs):
            wf = zimage_control(s, "aire/depth_control.png", "aire/edges_control.png", "aire/shaded.png")
            (out_dir / f"workflow_{i:02d}.json").write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"workflows (formato API de ComfyUI) en {out_dir}")
        return

    client = ComfyClient(args.server)

    def on_each(i, n):
        print(f"[{i + 1}/{n}] {label(runs[i])}…")

    try:
        results = run_renders(client, passes, pdir, runs, out_dir, on_each)
    except ComfyError as e:
        raise SystemExit(str(e))
    for r in results:
        print(f"  {r['file']}: {r['seconds']}s · recall {r['fidelity']['recall']:.3f} · "
              f"spurious {r['fidelity']['spurious']:.3f}")
    ref_lines = load_lines(pdir)
    sheet_items = [(Image.open(out_dir / r["file"]),
                    f"{r['label']} | recall {r['fidelity']['recall']:.2f} · spur {r['fidelity']['spurious']:.2f}")
                   for r in results]

    (out_dir / "meta.json").write_text(json.dumps({
        "export": str(scene.root), "passes": str(pdir), "prompt": prompt, "runs": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(sheet_items) > 1:
        ref = Image.fromarray(np.where(ref_lines, 0, 255).astype(np.uint8))
        contact_sheet([(ref, "líneas del modelo (referencia)")] + sheet_items, cols=3).save(out_dir / "sweep.png")
    print(f"\n→ {out_dir}")


if __name__ == "__main__":
    main()
