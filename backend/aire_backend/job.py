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
from .models import download, missing_files
from .prompt import build_edit_prompt
from .render import prepare_passes, run_renders
from .workflows import KleinSettings

QUALITY = {
    "rapida": {"width": 1024, "steps": 4},
    "alta": {"width": 1536, "steps": 4},
    "comparar": {"width": 1024, "steps": 4, "sweep": True},
}

ENGINE_PRESET = "flux2-klein4b"

STEPS = ["Preparando la escena", "Descargando el modelo FLUX", "Arrancando el motor",
         "Creando la imagen", "Terminando"]


def ensure_models(cfg: dict, progress: Progress, preset: str = ENGINE_PRESET) -> None:
    """Descarga los ficheros del motor que falten (solo la primera vez)."""
    comfy_dir = Path(cfg["comfy_dir"])
    todo = missing_files(preset, comfy_dir)
    if not todo:
        progress.step(1, "Ya descargado")
        return
    progress.step(1, "Solo la primera vez: unos 4 GB", 0)

    def on_progress(f, i, n, done, total):
        pct = 100 * done / total if total else None
        progress.update(f"Fichero {i + 1} de {n}: {done / 2**30:.1f} de {total / 2**30:.1f} GB", pct)

    try:
        download(preset, comfy_dir, on_progress)
    except OSError as e:
        raise UserError("Se ha cortado la descarga del modelo FLUX. Vuelve a pulsar «Crear imagen»: "
                        "seguirá donde se quedó.") from e


def klein_grid(base: KleinSettings, scene_prompt) -> list[KleinSettings]:
    """8 variantes para comparar: instrucción estricta / con algo de atrezo, 4 / 8 pasos, 2 semillas."""
    runs = []
    for creative in (False, True):
        prompt = scene_prompt(creative)
        for steps in (4, 8):
            for k in range(2):
                tag = ("con atrezo" if creative else "estricto")
                runs.append(replace(base, prompt=prompt, steps=steps, seed=base.seed + k, tag=tag))
    return runs


def run_job(home: Path, export: Path, prompt: str, style: str, light: str, quality: str,
            seed: int | None, progress: Progress, variants: int = 1) -> dict:
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("ready"):
        raise UserError("AIRE todavía no está preparado. Pulsa «Preparar AIRE» primero.")
    q = QUALITY.get(quality, QUALITY["alta"])

    progress.step(0, "Leyendo el modelo 3D…")
    passes, pdir = prepare_passes(str(export), q["width"])
    a = passes.arrays
    visible = passes.visible_objects()

    def scene_prompt(creative: bool = False) -> str:
        return build_edit_prompt(passes.scene, a["ids"], a["material"], visible, prompt,
                                 style=style, light=light, creative=creative)

    full_prompt = scene_prompt()
    ensure_models(cfg, progress)

    progress.step(2, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(
        f"Arrancando el motor… {int(s)} s (la primera vez tras encender el ordenador tarda más)"))

    base = KleinSettings(width=passes.width, height=passes.height, prompt=full_prompt, steps=q["steps"],
                         seed=seed if seed is not None else random.randint(0, 2**31 - 1))
    if q.get("sweep"):
        runs = klein_grid(base, scene_prompt)
    else:
        runs = [replace(base, seed=base.seed + i) for i in range(max(1, variants))]

    out_dir = export.parent / "renders"
    progress.step(3, "", 0)

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

    progress.step(4)
    result = {"folder": str(out_dir), "prompt": full_prompt, "style": style, "light": light,
              "quality": quality, "user_prompt": prompt, "engine": ENGINE_PRESET,
              "created": time.time(), "images": results}
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
