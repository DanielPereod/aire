"""Trabajo de render que lanza la ventana de AIRE en SketchUp.

    python -m aire_backend.job --home <carpeta AIRE> --export <exportación> \\
        --style nordico --light tarde --quality alta --prompt "..." --progress job.json

Arranca ComfyUI si hace falta, genera los pases, renderiza y deja en el fichero
de progreso las imágenes resultantes (con miniaturas para la galería).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from . import comfyctl
from .comfy import ComfyError
from .progress import Progress, UserError
from .prompt import build_prompt
from .render import prepare_passes, run_renders, sweep_grid
from .workflows import ZImageSettings

QUALITY = {
    "rapida": {"width": 1024, "refine": 0.0},
    "alta": {"width": 1536, "refine": 0.3},
    "comparar": {"width": 1024, "refine": 0.3, "sweep": True},
}

# Cuánto se parte de los colores del modelo (albedo): «Como en el modelo» los respeta
# mucho; un estilo (nórdico, japandi…) necesita más libertad para cambiar el ambiente.
DENOISE = {"modelo": 0.82}
DENOISE_STYLE = 0.93

STEPS = ["Preparando la escena", "Arrancando el motor", "Creando la imagen", "Terminando"]


def run_job(home: Path, export: Path, prompt: str, style: str, light: str, quality: str,
            seed: int | None, progress: Progress, variants: int = 1) -> dict:
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("ready"):
        raise UserError("AIRE todavía no está preparado. Pulsa «Preparar AIRE» primero.")
    q = QUALITY.get(quality, QUALITY["alta"])

    progress.step(0, "Leyendo el modelo 3D…")
    passes, pdir = prepare_passes(str(export), q["width"])
    full_prompt = build_prompt(passes.scene, passes.arrays["ids"], passes.arrays["material"],
                               passes.visible_objects(), prompt, style=style, light=light)

    progress.step(1, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(
        f"Arrancando el motor… {int(s)} s (la primera vez tras encender el ordenador tarda más)"))

    base = ZImageSettings(width=passes.width, height=passes.height, prompt=full_prompt,
                          seed=seed if seed is not None else random.randint(0, 2**31 - 1),
                          weight_dtype=cfg.get("weight_dtype", "fp8_e4m3fn"),
                          denoise=DENOISE.get(style, DENOISE_STYLE), refine=q["refine"])
    if q.get("sweep"):
        runs = sweep_grid(base)
    else:
        runs = [replace(base, seed=base.seed + i) for i in range(max(1, variants))]

    out_dir = export.parent / "renders"
    progress.step(2, "", 0)

    def on_each(i, n):
        msg = f"Imagen {i + 1} de {n}…" if n > 1 else "Creando la imagen…"
        progress.update(msg, 100 * i / n)

    try:
        results = run_renders(client, passes, pdir, runs, out_dir, on_each, thumbs=True)
    except ComfyError as e:
        text = str(e)
        if "out of memory" in text.lower() or "OutOfMemory" in text:
            raise UserError("La tarjeta gráfica se ha quedado sin memoria. Cierra otros programas "
                            "(juegos, navegadores con vídeo) y prueba con calidad «Rápida».") from e
        raise

    progress.step(3)
    result = {"folder": str(out_dir), "prompt": full_prompt, "style": style, "light": light,
              "quality": quality, "user_prompt": prompt, "created": time.time(), "images": results}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Trabajo de render de AIRE")
    ap.add_argument("--home", required=True, type=Path)
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", type=Path, default=None, help="texto del usuario en UTF-8 (evita problemas de comillas)")
    ap.add_argument("--style", default="modelo")
    ap.add_argument("--light", default="dia")
    ap.add_argument("--quality", default="alta", choices=sorted(QUALITY))
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--progress", type=Path, required=True)
    args = ap.parse_args(argv)

    log = args.home / "logs" / "render.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress, STEPS)
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    try:
        result = run_job(args.home, args.export, args.prompt, args.style, args.light, args.quality,
                         args.seed, progress, args.variants)
    except BaseException as e:  # noqa: BLE001
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.ctime()}\n")
            traceback.print_exc(file=lf)
        progress.fail(e, log)
        sys.exit(1)
    progress.done(result)


if __name__ == "__main__":
    main()
