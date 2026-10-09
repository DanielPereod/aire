"""Cambios sobre una imagen ya creada: instrucción de texto, imágenes de referencia y,
opcionalmente, una zona pintada (inpaint).

    python -m aire_backend.edit --home <AIRE> --image render_00.png --job-dir <carpeta> \\
        --prompt-file cambio.txt [--ref sofa.jpg ...] [--mask zona.png] --progress job.json

Con zona pintada se edita solo un recorte alrededor de ella (más resolución para lo pequeño)
y se pega con borde suave: fuera de la zona la imagen queda idéntica, píxel a píxel.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import comfyctl
from .comfy import CRASH_HELP, ComfyError
from .job import ensure_models
from .photo import photo_finish
from .progress import Progress, UserError
from .render import round16, save_views
from .workflows import KleinSettings, QwenSettings, flux2_klein_edit, pad_to, qwen21_edit, qwen_size, unpad

STEPS = ["Preparando la imagen", "Descargando el modelo", "Arrancando el motor", "Aplicando los cambios",
         "Terminando"]

ENGINE = "qwen21"  # mejor con varias referencias; "flux2-klein4b" como alternativa rápida
MAX_WIDTH = 1920  # tamaño máximo de trabajo (8 GB de VRAM)
CROP_SIDE = 1024  # lado largo al que se trabaja un recorte


def edit_prompt(instruction: str, refs: int, local: bool) -> str:
    text = instruction.strip().rstrip(".")
    parts = [f"Edit this interior photograph: {text}." if text else
             "Improve this interior photograph, keeping it the same."]
    if refs == 1:
        parts.append("Use the second image as the reference for the new or changed element: match its "
                     "design, shape, material and color exactly.")
    elif refs > 1:
        parts.append(f"Use the other {refs} images as references for the new or changed elements: match their "
                     "design, shape, material and color exactly.")
    if local:
        parts.append("Blend it seamlessly with the surroundings, with consistent perspective, light and shadows.")
    parts.append("Keep everything else exactly as it is: same camera, framing, room, furniture, materials, "
                 "colors and lighting. Photorealistic interior photograph, natural light, sharp detail.")
    return " ".join(parts)


def mask_box(mask: Image.Image, size: tuple[int, int], margin: float = 0.35, min_side: int = 384):
    """Recorte (x0, y0, x1, y1) alrededor de la zona pintada, con margen de contexto."""
    bbox = mask.point(lambda v: 255 if v > 127 else 0).getbbox()
    if not bbox:
        return None
    w, h = size
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side_x = max((x1 - x0) * (1 + 2 * margin), min_side)
    side_y = max((y1 - y0) * (1 + 2 * margin), min_side)
    x0, x1 = int(max(0, cx - side_x / 2)), int(min(w, cx + side_x / 2))
    y0, y1 = int(max(0, cy - side_y / 2)), int(min(h, cy + side_y / 2))
    return x0, y0, x1, y1


def soft_mask(mask: Image.Image, radius: float) -> Image.Image:
    """Zona pintada algo ensanchada y con borde suave para que el pegado no se note."""
    grow = max(1, int(radius)) * 2 + 1
    m = mask.point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(grow))
    return m.filter(ImageFilter.GaussianBlur(radius))


def work_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    k = long_side / max(w, h)
    return round16(w * k), round16(h * k)


def fit_reference(path: Path, out: Path, side: int = 1024) -> Path:
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.Resampling.LANCZOS)
    w, h = round16(im.width), round16(im.height)
    im.resize((w, h), Image.Resampling.LANCZOS).save(out)
    return out


def run_edit(home: Path, image: Path, prompt: str, refs: list[Path], mask: Path | None, job_dir: Path,
             seed: int | None, progress: Progress, engine: str = ENGINE) -> dict:
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("ready"):
        raise UserError("AIRE todavía no está preparado. Pulsa «Preparar AIRE» primero.")
    if not image.exists():
        raise UserError("No se encuentra la imagen que querías cambiar. Puede que se haya borrado.")
    if not prompt.strip() and not refs:
        raise UserError("Escribe qué quieres cambiar o añade una imagen de referencia.")

    progress.step(0)
    work = job_dir / "trabajo"
    work.mkdir(parents=True, exist_ok=True)
    src = Image.open(image).convert("RGB")
    m = Image.open(mask).convert("L").resize(src.size, Image.Resampling.BILINEAR) if mask else None
    box = mask_box(m, src.size) if m is not None else None
    if box:
        region = src.crop(box)
        w, h = work_size(region.width, region.height, CROP_SIDE)
        upscale = 1.0
    else:
        region = src
        w, h = work_size(src.width, src.height, min(MAX_WIDTH, max(src.width, src.height)))
        upscale = 1.5 if max(w, h) > 1400 else 1.0
    if engine == "qwen21":
        w, h = qwen_size(region.width, region.height)
    base_path = work / "base.png"
    ebox = None
    if engine == "qwen21":  # sin deformar: rellena hasta el múltiplo de 32 en vez de estirar
        padded, ebox = pad_to(region, (w, h))
        padded.save(base_path)
    else:
        region.resize((w, h), Image.Resampling.LANCZOS).save(base_path)
    ref_paths = [fit_reference(r, work / f"ref_{i}.png") for i, r in enumerate(refs)]

    ensure_models(cfg, progress, engine)
    progress.step(2, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(f"Arrancando el motor… {int(s)} s"))

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    text = edit_prompt(prompt, len(refs), box is not None)
    base = client.upload_image(base_path)
    uploaded = tuple(client.upload_image(p) for p in ref_paths)
    prefix = f"aire/{job_dir.name}"
    if engine == "qwen21":
        progress.step(3, "Unos 1-2 minutos")
        client.free()
        s = QwenSettings(width=w, height=h, prompt=text, seed=seed)
        wf = qwen21_edit(s, base, uploaded, prefix=prefix)
    else:
        progress.step(3, "Unos 30-60 s")
        s = KleinSettings(width=w, height=h, prompt=text, steps=4, upscale=upscale, seed=seed)
        wf = flux2_klein_edit(s, base, uploaded, prefix=prefix)
    t0 = time.perf_counter()
    try:
        out = client.run(wf)
    except ComfyError as e:
        if "out of memory" in str(e).lower():
            raise UserError("La tarjeta gráfica se ha quedado sin memoria. Cierra otros programas y prueba "
                            "con una zona más pequeña o menos referencias.") from e
        if "No se puede conectar" in str(e):
            raise UserError(CRASH_HELP) from e
        raise
    secs = time.perf_counter() - t0
    from io import BytesIO
    new = Image.open(BytesIO(out[0])).convert("RGB")
    if ebox is not None:
        new = unpad(new, ebox, (w, h), region.size)

    progress.step(4)
    if box:
        patch = photo_finish(new.resize(region.size, Image.Resampling.LANCZOS), 0.6, seed=s.seed)
        local = m.crop(box)
        alpha = soft_mask(local, max(3.0, max(region.size) / 120))
        result = src.copy()
        result.paste(Image.composite(patch, region, alpha), box[:2])
    else:
        result = photo_finish(new.resize(src.size, Image.Resampling.LANCZOS), 0.6, seed=s.seed)

    out_dir = job_dir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "render_00.png"
    result.save(path)
    item = {"file": path.name, "path": str(path), "seconds": round(secs, 1), "label": "Cambio",
            "settings": s.to_dict()}
    item.update(save_views(result, out_dir, "render_00"))
    parent = _parent_result(image)
    info = {"folder": str(out_dir), "kind": "edit", "source": str(image), "user_prompt": prompt,
            "prompt": s.prompt, "references": len(refs), "masked": box is not None,
            "style": parent.get("style"), "light": parent.get("light"), "quality": parent.get("quality"),
            "engine": engine, "created": time.time(), "images": [item]}
    (out_dir / "result.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def _parent_result(image: Path) -> dict:
    try:
        return json.loads((image.parent / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cambios sobre una imagen de AIRE")
    ap.add_argument("--home", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--ref", type=Path, action="append", default=[])
    ap.add_argument("--mask", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--engine", default=ENGINE, choices=["qwen21", "flux2-klein4b"])
    ap.add_argument("--progress", type=Path, required=True)
    args = ap.parse_args(argv)

    log = args.home / "logs" / "render.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress, STEPS)
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    try:
        result = run_edit(args.home, args.image, args.prompt, args.ref, args.mask, args.job_dir, args.seed, progress,
                          args.engine)
    except BaseException as e:  # noqa: BLE001
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.ctime()} (edición)\n")
            traceback.print_exc(file=lf)
        progress.fail(e, log)
        sys.exit(1)
    progress.done(result)


if __name__ == "__main__":
    main()
