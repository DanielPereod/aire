"""«Ampliar»: amplía una imagen de la galería a 2× con Real-ESRGAN (sale como imagen nueva).

    python -m aire_backend.upscale --home <AIRE> --image <job>/renders/render_00.png \\
        --job-dir <carpeta nueva> [--factor 2] --progress job.json

El modelo amplía a 4×; para 2× se reduce después con Lanczos, que da un resultado más limpio
que ampliar directamente a 2×. Real-ESRGAN deja los bordes nítidos pero alisa las texturas finas
(piedra, tela): se mezcla al 50 % con la imagen ampliada por Lanczos para conservarlas (pruebas
0.13.2 en la RTX 5060).
"""

from __future__ import annotations

import argparse
import json
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
from .render import save_views
from .workflows import upscale_model

STEPS = ["Preparando la imagen", "Descargando el modelo", "Arrancando el motor", "Ampliando", "Terminando"]
PRESET = "upscale-esrgan"
MAX_SIDE = 8192
BLEND = 0.5  # parte de Real-ESRGAN en la mezcla; el resto, la original ampliada con Lanczos  # más grande no lo abre bien casi ningún visor


def run_upscale(home: Path, image: Path, job_dir: Path, factor: int, progress: Progress) -> dict:
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("ready"):
        raise UserError("AIRE todavía no está preparado. Pulsa «Preparar AIRE» primero.")
    if not image.exists():
        raise UserError("No se encuentra la imagen.")
    progress.step(0)
    src = Image.open(image).convert("RGB")
    factor = max(2, min(4, factor))
    if max(src.size) * factor > MAX_SIDE:
        factor = max(1, MAX_SIDE // max(src.size))
        if factor < 2:
            raise UserError("Esta imagen ya es muy grande para ampliarla más.")
    work = job_dir / "trabajo"
    work.mkdir(parents=True, exist_ok=True)
    base = work / "base.png"
    src.save(base)

    ensure_models(cfg, progress, PRESET)
    progress.step(2, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(f"Arrancando el motor… {int(s)} s"))
    progress.step(3, "Unos segundos")
    t0 = time.perf_counter()
    try:
        out = client.run(upscale_model(client.upload_image(base), prefix=f"aire/{job_dir.name}"))
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
    size = (src.width * factor, src.height * factor)
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    img = Image.blend(src.resize(size, Image.Resampling.LANCZOS), img, BLEND)
    out_dir = job_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "render_00.png"
    img.save(path)
    try:
        parent = json.loads((image.parent / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parent = {}
    item = {"file": path.name, "path": str(path), "seconds": round(secs, 1),
            "label": f"Ampliada ×{factor} · {img.width}×{img.height}",
            "settings": {"model": "RealESRGAN_x4plus.pth", "factor": factor, "blend": BLEND}}
    item.update(save_views(img, out_dir, "render_00"))
    info = {"folder": str(out_dir), "kind": "ampliada", "source": str(image),
            "user_prompt": parent.get("user_prompt"), "style": parent.get("style"), "light": parent.get("light"),
            "quality": parent.get("quality"), "engine": "realesrgan", "created": time.time(), "images": [item]}
    (out_dir / "result.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ampliar una imagen de AIRE")
    ap.add_argument("--home", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--progress", type=Path, required=True)
    args = ap.parse_args(argv)
    log = args.home / "logs" / "render.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress, STEPS)
    try:
        result = run_upscale(args.home, args.image, args.job_dir, args.factor, progress)
    except BaseException as e:  # noqa: BLE001
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.ctime()} (ampliar)\n")
            traceback.print_exc(file=lf)
        progress.fail(e, log)
        sys.exit(1)
    progress.done(result)


if __name__ == "__main__":
    main()
