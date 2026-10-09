"""«Procesar con IA»: pasa una imagen con luz real (Cycles, modo «Sin IA») por la IA para darle
acabado de foto, conservando cámara, objetos y colores.

    python -m aire_backend.enhance --home <AIRE> --image <job>/renders/render_00.png \\
        --job-dir <carpeta nueva> --progress job.json

Parte del render de Cycles sin el acabado de cámara (luz/cycles_<luz>.png del trabajo de
origen) cuando existe: la IA exagera el grano si se lo damos ya puesto (pruebas 0.11). Después
fija los colores frente a ese render y vuelve a aplicar el acabado de cámara.
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
from .colormatch import lock_colors
from .comfy import CRASH_HELP, ComfyError
from .job import ensure_models
from .photo import photo_finish
from .progress import Progress, UserError
from .prompt import PHOTO_STYLE, build_edit_prompt
from .render import edge_map, prepare_passes, save_views, score
from .workflows import QwenSettings, pad_to, qwen21_edit, qwen_size, unpad

STEPS = ["Preparando la imagen", "Descargando el modelo", "Arrancando el motor", "Procesando con IA", "Terminando"]
ENGINE = "qwen21"


def source_render(image: Path) -> tuple[Path, dict, Path | None]:
    """(imagen de partida, result.json del trabajo de origen, carpeta de exportación o None)."""
    renders = image.parent
    try:
        parent = json.loads((renders / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parent = {}
    job = renders.parent
    export = job / "export" if (job / "export" / "scene.json").exists() else None
    raw = job / "luz" / f"cycles_{parent.get('light') or 'dia'}.png"
    return (raw if raw.exists() else image), parent, export


def enhance_prompt(export: Path | None, width: int, parent: dict) -> str:
    if export is not None:
        try:
            passes, _ = prepare_passes(str(export), width)
            a = passes.arrays
            return build_edit_prompt(passes.scene, a["ids"], a["material"], passes.visible_objects(),
                                     parent.get("user_prompt") or "", style=parent.get("style") or "",
                                     light=parent.get("light"), albedo=a["albedo"], photo=True)
        except Exception:  # noqa: BLE001 - sin descripción de la escena el prompt general sigue valiendo
            pass
    from .prompt import EDIT_DETAIL, EDIT_KEEP, EDIT_LOOK, PHOTO_LIGHTS
    light = PHOTO_LIGHTS.get(parent.get("light") or "dia", "")
    return " ".join([EDIT_KEEP, EDIT_DETAIL, EDIT_LOOK, PHOTO_STYLE] + ([f"Lighting: {light}."] if light else []))


def run_enhance(home: Path, image: Path, job_dir: Path, seed: int | None, progress: Progress) -> dict:
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("ready"):
        raise UserError("AIRE todavía no está preparado. Pulsa «Preparar AIRE» primero.")
    if not image.exists():
        raise UserError("No se encuentra la imagen.")
    progress.step(0, "Leyendo la escena…")
    src_path, parent, export = source_render(image)
    src = Image.open(src_path).convert("RGB")
    final = Image.open(image).size
    prompt = enhance_prompt(export, final[0], parent)

    work = job_dir / "trabajo"
    work.mkdir(parents=True, exist_ok=True)
    size = qwen_size(*src.size)
    padded, box = pad_to(src, size)
    base_path = work / "base.png"
    padded.save(base_path)

    ensure_models(cfg, progress, ENGINE)
    progress.step(2, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(f"Arrancando el motor… {int(s)} s"))
    progress.step(3, "Unos 1-2 minutos")
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    s = QwenSettings(width=size[0], height=size[1], prompt=prompt, seed=seed, color_lock=1.0, photo=1.0)
    uploaded = client.upload_image(base_path)
    client.free()
    t0 = time.perf_counter()
    try:
        out = client.run(qwen21_edit(s, uploaded, prefix=f"aire/{job_dir.name}"))
    except ComfyError as e:
        if "out of memory" in str(e).lower():
            raise UserError("La tarjeta gráfica se ha quedado sin memoria. Cierra otros programas y vuelve "
                            "a intentarlo.") from e
        if "No se puede conectar" in str(e):
            raise UserError(CRASH_HELP) from e
        raise
    secs = time.perf_counter() - t0

    progress.step(4)
    img = unpad(Image.open(BytesIO(out[0])).convert("RGB"), box, size, final)
    ref = src.resize(final, Image.Resampling.LANCZOS)
    img = photo_finish(lock_colors(img, ref, 1.0), 1.0, seed=seed)
    out_dir = job_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "render_00.png"
    img.save(path)
    item = {"file": path.name, "path": str(path), "seconds": round(secs, 1), "label": "Procesada con IA",
            "fidelity": score(img, edge_map(ref)), "settings": s.to_dict()}
    item.update(save_views(img, out_dir, "render_00"))
    info = {"folder": str(out_dir), "kind": "ia", "source": str(image), "prompt": prompt,
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
