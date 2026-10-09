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
from .passes import Passes
from .prompt import build_prompt
from .scene import load_scene
from .workflows import ZImageSettings, zimage_control


def round16(x: float) -> int:
    return max(16, int(round(x / 16.0)) * 16)


def prepare_passes(export_dir: str, width: int, section_keep: str = "auto") -> tuple[Passes, Path]:
    scene = load_scene(export_dir)
    probe = Passes(scene, width, section_keep=section_keep)
    w = round16(width)
    h = round16(w / probe.camera.aspect)
    passes = Passes(scene, w, section_keep=section_keep, height=h).render()
    out = passes.save(scene.root / f"passes_{w}x{h}")
    return passes, out


def sweep_grid(base: ZImageSettings) -> list[ZImageSettings]:
    grid = itertools.product((0.6, 0.9), (0.0, 0.5), (1.0, 0.9))
    return [replace(base, depth_strength=d, lines_strength=l, denoise=n) for d, l, n in grid]


def label(s: ZImageSettings) -> str:
    return f"depth {s.depth_strength:.2f} · lines {s.lines_strength:.2f} · denoise {s.denoise:.2f}"


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
    ap.add_argument("--lines", type=float, default=0.5, help="fuerza del control de líneas (0 = sin líneas)")
    ap.add_argument("--denoise", type=float, default=1.0, help="<1 parte del albedo (colores del modelo)")
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
        depth_strength=args.depth, lines_strength=args.lines, denoise=args.denoise)
    runs = sweep_grid(base) if args.sweep else [base]

    out_dir = scene.root / "renders" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{passes.width}x{passes.height}px · semilla {base.seed}\nprompt: {prompt}\n")

    if args.dry_run:
        for i, s in enumerate(runs):
            wf = zimage_control(s, "aire/depth_control.png", "aire/edges_control.png", "aire/albedo.png")
            (out_dir / f"workflow_{i:02d}.json").write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"workflows (formato API de ComfyUI) en {out_dir}")
        return

    client = ComfyClient(args.server)
    try:
        depth = client.upload_image(pdir / "depth_control.png")
        lines = client.upload_image(pdir / "edges_control.png")
        albedo = client.upload_image(pdir / "albedo.png")
    except ComfyError as e:
        raise SystemExit(str(e))

    ref_lines = load_lines(pdir)
    results, sheet_items = [], []
    for i, s in enumerate(runs):
        wf = zimage_control(s, depth, lines, albedo, prefix=f"aire/{out_dir.name}_{i:02d}")
        t0 = time.perf_counter()
        try:
            images = client.run(wf, check=(i == 0))
        except ComfyError as e:
            raise SystemExit(str(e))
        secs = time.perf_counter() - t0
        img = Image.open(BytesIO(images[0]))
        path = out_dir / f"render_{i:02d}.png"
        img.save(path)
        fid = score(img, ref_lines)
        results.append({"file": path.name, "seconds": round(secs, 1), "fidelity": fid, "settings": s.to_dict()})
        sheet_items.append((img, f"{label(s)} | recall {fid['recall']:.2f} · spur {fid['spurious']:.2f}"))
        print(f"[{i + 1}/{len(runs)}] {label(s)} → {secs:.1f}s · recall {fid['recall']:.3f} · spurious {fid['spurious']:.3f}")

    (out_dir / "meta.json").write_text(json.dumps({
        "export": str(scene.root), "passes": str(pdir), "prompt": prompt, "runs": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(sheet_items) > 1:
        ref = Image.fromarray(np.where(ref_lines, 0, 255).astype(np.uint8))
        contact_sheet([(ref, "líneas del modelo (referencia)")] + sheet_items, cols=3).save(out_dir / "sweep.png")
    print(f"\n→ {out_dir}")


if __name__ == "__main__":
    main()
