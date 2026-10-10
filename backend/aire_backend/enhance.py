"""«Procesar con IA»: da acabado de foto a una imagen con luz real (Cycles, modo «Sin IA») con
Qwen Image 2.1 (la variante elegida en «Opciones avanzadas») y el prompt del workflow de Dani
«Architecture FLUX2 · Master Render v2» (workflow_files/README.md).

    python -m aire_backend.enhance --home <AIRE> --image <job>/renders/render_00.png \\
        --job-dir <carpeta nueva> --progress job.json

Parte del render de Cycles sin el acabado de cámara (luz/cycles_<luz>.png del trabajo de
origen) cuando existe: la IA exagera el grano si se lo damos ya puesto (pruebas 0.11). Qwen
trabaja a ~1 Mpx sin deformar (relleno) y el resultado se lleva al tamaño del render.
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
from .job import prepare_qwen
from .progress import Progress, UserError
from .render import edge_map, save_views, score
from .workflows import QwenSettings, pad_to, qwen21_edit, qwen_size, unpad

STEPS = ["Preparando la imagen", "Descargando el modelo", "Arrancando el motor", "Procesando con IA", "Terminando"]
ENGINE = "qwen21"
# Prompt del nodo 8 del workflow de Dani (Master Render v2), adaptado a una imagen que ya es un
# render con luz real en vez de una captura de SketchUp.
PROMPT = ("Convert image 1 into a photorealistic interior architectural photograph. Preserve the exact camera "
          "position, perspective, framing and room geometry of image 1. Keep all walls, openings, ceiling heights, "
          "built-in cabinetry, furniture shapes, counts and positions. Preserve object identities and material "
          "assignments. Do not redesign the room, move objects, add furniture, crop the image or change the lens. "
          "Lighting: mid-afternoon sunlight entering low from the side, casting long soft dappled tree-leaf shadows "
          "across the floor, the cabinet fronts and the table. Bright sunlit patches against gentle shade, soft warm "
          "bounce light, lifted shadows, airy and luminous atmosphere, slight natural haze. Realistic roughness, "
          "subtle reflections on countertops and appliances, crisp contact shadows. High-end editorial interior "
          "photography, full-frame camera, natural look. Return one full-frame image, no collage or text.")


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

    opts = prepare_qwen(home, cfg, progress)
    progress.step(2, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(f"Arrancando el motor… {int(s)} s"))
    progress.step(3, "Menos de un minuto" if opts.get("steps", 25) <= 8 else "Unos 1-2 minutos")
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    prefix = f"aire/{job_dir.name}"
    qw, qh = qwen_size(src.width, src.height)
    padded, qbox = pad_to(src, (qw, qh))
    padded_path = work / "base_qwen.png"
    padded.save(padded_path)
    s = QwenSettings(width=qw, height=qh, prompt=PROMPT, seed=seed, **opts)
    client.free()
    t0 = time.perf_counter()
    try:
        out = client.run(qwen21_edit(s, client.upload_image(padded_path), prefix=prefix))
    except ComfyError as e:
        if "out of memory" in str(e).lower():
            raise UserError("La tarjeta gráfica se ha quedado sin memoria. Cierra otros programas y vuelve "
                            "a intentarlo.") from e
        if "No se puede conectar" in str(e):
            raise UserError(CRASH_HELP) from e
        raise
    secs = time.perf_counter() - t0

    progress.step(4)
    img = unpad(Image.open(BytesIO(out[0])).convert("RGB"), qbox, (qw, qh), src.size)
    ref = src
    out_dir = job_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "render_00.png"
    img.save(path)
    item = {"file": path.name, "path": str(path), "seconds": round(secs, 1), "label": "Procesada con IA",
            "fidelity": score(img, edge_map(ref)), "settings": s.to_dict()}
    item.update(save_views(img, out_dir, "render_00"))
    info = {"folder": str(out_dir), "kind": "ia", "source": str(image), "prompt": PROMPT,
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
