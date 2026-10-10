"""«Procesar con IA»: pasa una imagen con luz real (Cycles, modo «Sin IA») por el workflow de Dani
«Architecture FLUX2 · Master Render v2» (workflow_files/README.md) para darle acabado de foto.

    python -m aire_backend.enhance --home <AIRE> --image <job>/renders/render_00.png \\
        --job-dir <carpeta nueva> --progress job.json

Parte del render de Cycles sin el acabado de cámara (luz/cycles_<luz>.png del trabajo de
origen) cuando existe: la IA exagera el grano si se lo damos ya puesto (pruebas 0.11). El
workflow va en dos pasadas, como en el ComfyUI de Dani: render a 0,55 Mpx y ampliación a
1,6 Mpx con refinado suave. Su resultado se deja tal cual (pruebas en el PC, 0.12).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import comfyctl
from .comfy import CRASH_HELP, ComfyError
from .job import ensure_models
from .progress import Progress, UserError
from .render import edge_map, save_views, score
from .workflows import architecture_v2

STEPS = ["Preparando la imagen", "Descargando el modelo", "Arrancando el motor", "Procesando con IA", "Terminando"]
ENGINE = "flux2-klein4b"  # los modelos del workflow de Dani


def source_render(image: Path) -> tuple[Path, dict]:
    """(imagen de partida, result.json del trabajo de origen)."""
    renders = image.parent
    try:
        parent = json.loads((renders / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parent = {}
    raw = renders.parent / "luz" / f"cycles_{parent.get('light') or 'dia'}.png"
    return (raw if raw.exists() else image), parent


def run_enhance(home: Path, image: Path, job_dir: Path, seed: int | None, progress: Progress) -> dict:
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("ready"):
        raise UserError("AIRE todavía no está preparado. Pulsa «Preparar AIRE» primero.")
    if not image.exists():
        raise UserError("No se encuentra la imagen.")
    progress.step(0)
    src_path, parent = source_render(image)
    src = Image.open(src_path).convert("RGB")
    work = job_dir / "trabajo"
    work.mkdir(parents=True, exist_ok=True)
    base_path = work / "base.png"
    src.save(base_path)

    ensure_models(cfg, progress, ENGINE)
    progress.step(2, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(f"Arrancando el motor… {int(s)} s"))
    progress.step(3, "Menos de un minuto")
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    prefix = f"aire/{job_dir.name}"
    client.free()
    t0 = time.perf_counter()
    try:
        first = client.run(architecture_v2("render", client.upload_image(base_path), seed, prefix))
        progress.update("Ampliando…")
        first_path = work / "paso1.png"
        first_path.write_bytes(first[0])
        out = client.run(architecture_v2("upscale", client.upload_image(first_path), seed, prefix))
    except ComfyError as e:
        if "out of memory" in str(e).lower():
            raise UserError("La tarjeta gráfica se ha quedado sin memoria. Cierra otros programas y vuelve "
                            "a intentarlo.") from e
        if "No se puede conectar" in str(e):
            raise UserError(CRASH_HELP) from e
        raise
    secs = time.perf_counter() - t0

    progress.step(4)
    img = Image.open(BytesIO(out[0])).convert("RGB")
    ref = src.resize(img.size, Image.Resampling.LANCZOS)
    out_dir = job_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "render_00.png"
    img.save(path)
    item = {"file": path.name, "path": str(path), "seconds": round(secs, 1), "label": "Procesada con IA",
            "fidelity": score(img, edge_map(ref)), "settings": {"workflow": "architecture_flux2_master_v2",
                                                                "seed": seed}}
    item.update(save_views(img, out_dir, "render_00"))
    info = {"folder": str(out_dir), "kind": "ia", "source": str(image), "workflow": "architecture_flux2_master_v2",
            "user_prompt": parent.get("user_prompt"), "style": parent.get("style"), "light": parent.get("light"),
            "quality": "ia", "engine": ENGINE, "created": time.time(), "images": [item]}
    (out_dir / "result.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Procesar con IA una imagen de AIRE")
    ap.add_argument("--home", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--progress", type=Path, required=True)
    args = ap.parse_args(argv)
    log = args.home / "logs" / "render.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress, STEPS)
    try:
        result = run_enhance(args.home, args.image, args.job_dir, args.seed, progress)
    except BaseException as e:  # noqa: BLE001
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.ctime()} (procesar con IA)\n")
            traceback.print_exc(file=lf)
        progress.fail(e, log)
        sys.exit(1)
    progress.done(result)


if __name__ == "__main__":
    main()
