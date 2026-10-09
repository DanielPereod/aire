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

from PIL import Image

from . import comfyctl, cycles_engine, matlib
from .comfy import CRASH_HELP, ComfyError
from .installer import Installer
from .progress import Progress, UserError
from .models import adopt_existing, download, missing_files
from .prompt import build_edit_prompt
from .photo import photo_finish
from .scene import load_scene
from .render import prepare_passes, run_renders, save_views
from .workflows import KleinSettings, QwenSettings, qwen_size

# width = tamaño final; upscale > 1 → 1.ª pasada a width/upscale y 2.ª pasada de detalle
QUALITY = {
# cycles: la imagen base es un render con luz real (Cycles) en vez de la sencilla
    "rapida": {"width": 1024, "steps": 4, "upscale": 1.0},
    "alta": {"width": 1920, "steps": 4, "upscale": 1.5, "cycles": True, "photo": 1.0, "engine": "qwen21"},
    # sin IA: solo el render de Cycles con el acabado de cámara (fiel al modelo al 100 %)
    "real": {"width": 1920, "steps": 0, "upscale": 1.0, "cycles": True, "photo": 1.0, "ai": False, "samples": 512},
    "comparar": {"width": 1920, "steps": 4, "upscale": 1.5, "sweep": True, "cycles": True, "photo": 1.0},
}

ENGINE_PRESET = "flux2-klein4b"
SIZE_GB = {"flux2-klein4b": 4, "qwen21": 16}

STEPS = ["Preparando la escena", "Descargando el modelo", "Calculando la luz real",
         "Arrancando el motor", "Creando la imagen", "Terminando"]


def ensure_models(cfg: dict, progress: Progress, preset: str = ENGINE_PRESET) -> None:
    """Descarga los ficheros del motor que falten (solo la primera vez)."""
    comfy_dir = Path(cfg["comfy_dir"])
    if missing_files(preset, comfy_dir):
        try:
            adopt_existing(preset, comfy_dir)  # p. ej. los de otro ComfyUI del ordenador
        except OSError:
            pass
    todo = missing_files(preset, comfy_dir)
    if not todo:
        progress.step(1, "Ya descargado")
        return
    progress.step(1, f"Solo la primera vez: unos {SIZE_GB.get(preset, 4)} GB", 0)

    def on_progress(f, i, n, done, total):
        pct = 100 * done / total if total else None
        progress.update(f"Fichero {i + 1} de {n}: {done / 2**30:.1f} de {total / 2**30:.1f} GB", pct)

    try:
        download(preset, comfy_dir, on_progress)
    except OSError as e:
        raise UserError("Se ha cortado la descarga del modelo. Vuelve a intentarlo: "
                        "seguirá donde se quedó.") from e


def material_library(home: Path, export: Path, progress: Progress) -> Path | None:
    """Materiales reales (relieve y brillo) para los tipos de esta escena; se descargan la
    primera vez. Sin conexión se sigue sin ellos."""
    try:
        kinds = matlib.kinds_in_scene(load_scene(export))
        matlib.ensure(home, kinds, on_status=lambda i, n, k: progress.update(
            f"Solo la primera vez: descargando materiales reales ({i + 1} de {n})…"))
        return home / "library"
    except Exception as e:  # noqa: BLE001 - es una mejora, no un requisito
        with open(home / "logs" / "render.log", "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.ctime()} biblioteca de materiales no disponible: {e!r}\n")
        lib = home / "library"
        return lib if (lib / "index.json").exists() else None


def light_pass(home: Path, cfg: dict, export: Path, width: int, light: str, progress: Progress,
               samples: int = 256) -> Path | None:
    """Render con luz real (Cycles) para usarlo como imagen base. Si algo falla, se sigue con
    la imagen base sencilla: la imagen sale igual, solo con menos realismo."""
    log = home / "logs" / "render.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not cycles_engine.installed(home, cfg):
            uv = cycles_engine.uv_path(home)
            if not uv.exists():
                return None
            progress.step(2, "Solo la primera vez: instalando el motor de luz (unos 700 MB)…")
            Installer(home, uv, progress, home / "logs" / "install.log").install_cycles()
            cfg.update(json.loads((home / "config.json").read_text(encoding="utf-8")))
            if not cfg.get("cycles_ok"):
                return None
        progress.step(2, "Preparando los materiales…")
        library = material_library(home, export, progress)
        progress.update("Calculando sol, cielo y lámparas…")
        out = export.parent / "luz" / f"cycles_{light}.png"
        cycles_engine.render(home, export, out, width, light if light in ("dia", "tarde", "noche") else "dia",
                             samples=samples, library=library,
                             on_wait=lambda s: progress.update(f"Calculando sol, cielo y lámparas… {int(s)} s"))
        return out
    except Exception as e:  # noqa: BLE001 - la luz real es una mejora, no un requisito
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.ctime()} luz real no disponible: {e!r}\n")
        return None


def klein_grid(base: KleinSettings, with_light: bool = False) -> list[KleinSettings]:
    """8 variantes para comparar (2 semillas).
    Con luz real (Cycles): cuánto puede cambiar la IA el render (1,0 = desde ruido, con el
    render solo como referencia; 0,7 = casi el render), colores fijados y acabado de cámara.
    Sin ella: directo a tamaño final frente a dos pasadas con distinta fuerza de repaso."""
    runs = []
    for k in range(2):
        seed = base.seed + k
        if with_light:
            for d, lock, photo, tag in ((1.0, 0.0, 0.0, "IA sola"),
                                        (1.0, 1.0, 0.0, "colores fijados"),
                                        (1.0, 1.0, 1.0, "colores fijados · acabado de cámara"),
                                        (0.7, 1.0, 1.0, "libertad IA 0.70 · colores fijados · acabado")):
                runs.append(replace(base, seed=seed, base_denoise=d, color_lock=lock, photo=photo,
                                    tag=f"luz real · {tag}"))
            continue
        runs.append(replace(base, seed=seed, upscale=1.0, tag="directo a tamaño final"))
        for d in (0.3, 0.45, 0.6):
            runs.append(replace(base, seed=seed, refine_denoise=d, tag=f"2 pasadas · repaso {d:.2f}"))
    return runs


def render_without_ai(home: Path, cfg: dict, export: Path, passes, light: str, quality: str, prompt: str,
                      style: str, q: dict, progress: Progress) -> dict:
    """Solo luz real (Cycles) y acabado de cámara: sin modelo de IA ni ComfyUI."""
    progress.step(1, "No hace falta")
    base = light_pass(home, cfg, export, passes.width, light, progress, samples=q.get("samples", 256))
    if base is None:
        raise UserError("No se ha podido calcular la luz real. Pulsa «Ver el registro técnico» para ver por qué.")
    progress.step(3, "No hace falta")
    progress.step(4, "Dando el acabado de cámara…")
    t0 = time.perf_counter()
    img = Image.open(base).convert("RGB")
    if q.get("photo", 0) > 0:
        img = photo_finish(img, q["photo"])
    out_dir = export.parent / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "render_00.png"
    img.save(path)
    item = {"file": path.name, "path": str(path), "seconds": round(time.perf_counter() - t0, 1),
            "label": f"Sin IA · luz real (Cycles) · {img.width}×{img.height}", "fidelity": 1.0, "settings": {}}
    item.update(save_views(img, out_dir, "render_00"))
    progress.step(5)
    result = {"folder": str(out_dir), "prompt": "", "style": style, "light": light, "quality": quality,
              "user_prompt": prompt, "engine": "cycles", "created": time.time(), "images": [item]}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_job(home: Path, export: Path, prompt: str, style: str, light: str, quality: str,
            seed: int | None, progress: Progress, variants: int = 1, base_image: Path | None = None) -> dict:
    cfg = json.loads((home / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("ready"):
        raise UserError("AIRE todavía no está preparado. Pulsa «Preparar AIRE» primero.")
    q = QUALITY.get(quality, QUALITY["alta"])

    progress.step(0, "Leyendo el modelo 3D (unos 30 s)…")
    passes, pdir = prepare_passes(str(export), q["width"])
    if not q.get("ai", True):
        return render_without_ai(home, cfg, export, passes, light, quality, prompt, style, q, progress)
    a = passes.arrays
    visible = passes.visible_objects()

    engine = q.get("engine", ENGINE_PRESET)
    full_prompt = build_edit_prompt(passes.scene, a["ids"], a["material"], visible, prompt, style=style, light=light,
                                    albedo=a["albedo"], photo=engine == "qwen21")
    ensure_models(cfg, progress, engine)

    if base_image is None and q.get("cycles"):
        base_image = light_pass(home, cfg, export, passes.width, light, progress)
    else:
        progress.step(2, "No hace falta")

    progress.step(3, "Un momento…")
    client = comfyctl.ensure(home, cfg, on_wait=lambda s: progress.update(
        f"Arrancando el motor… {int(s)} s (la primera vez tras encender el ordenador tarda más)"))

    base = KleinSettings(width=passes.width, height=passes.height, prompt=full_prompt, steps=q["steps"],
                         upscale=q["upscale"], color_lock=1.0 if base_image is not None else 0.0, photo=q.get("photo", 0.0),
                         seed=seed if seed is not None else random.randint(0, 2**31 - 1))
    if engine == "qwen21":
        qw, qh = qwen_size(base.width, base.height)
        base = QwenSettings(width=qw, height=qh, prompt=full_prompt, seed=base.seed,
                            color_lock=base.color_lock, photo=base.photo)
    if q.get("sweep"):
        runs = klein_grid(base, with_light=base_image is not None)
    else:
        runs = [replace(base, seed=base.seed + i) for i in range(max(1, variants))]

    out_dir = export.parent / "renders"
    progress.step(4, "", 0)

    def on_each(i, n):
        msg = f"Imagen {i + 1} de {n}…" if n > 1 else "Creando la imagen…"
        progress.update(msg, 100 * i / n)

    try:
        results = run_renders(client, passes, pdir, runs, out_dir, on_each, thumbs=True, base=base_image)
    except ComfyError as e:
        text = str(e)
        if "out of memory" in text.lower() or "OutOfMemory" in text:
            raise UserError("La tarjeta gráfica se ha quedado sin memoria. Cierra otros programas "
                            "(juegos, navegadores con vídeo) y prueba con calidad «Rápida».") from e
        if "No se puede conectar" in text:
            raise UserError(CRASH_HELP) from e
        raise

    progress.step(5)
    result = {"folder": str(out_dir), "prompt": full_prompt, "style": style, "light": light,
              "quality": quality, "user_prompt": prompt, "engine": engine,
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
    ap.add_argument("--base", type=Path, default=None, help="imagen base en lugar de shaded.png (p. ej. Cycles)")
    ap.add_argument("--progress", type=Path, required=True)
    args = ap.parse_args(argv)

    log = args.home / "logs" / "render.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress, STEPS)
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    try:
        result = run_job(args.home, args.export, args.prompt, args.style, args.light, args.quality,
                         args.seed, progress, args.variants, args.base)
    except BaseException as e:  # noqa: BLE001
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n=== {time.ctime()}\n")
            traceback.print_exc(file=lf)
        progress.fail(e, log)
        sys.exit(1)
    progress.done(result)


if __name__ == "__main__":
    main()
