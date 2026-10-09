"""Catálogo de modelos (ficheros de ComfyUI) y presets por nivel de hardware.

Las URLs y carpetas salen de los blueprints oficiales de ComfyUI (octubre 2026).

    python -m aire_backend.models list
    python -m aire_backend.models download zimage-control --comfy-dir C:\\ComfyUI
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    folder: str  # carpeta dentro de ComfyUI/models
    url: str


FILES = {f.name: f for f in [
    # Z-Image-Turbo (6B, Apache 2.0) + ControlNet Union (profundidad/canny/pose)
    ModelFile("z_image_turbo_bf16.safetensors", "diffusion_models",
              "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    ModelFile("qwen_3_4b.safetensors", "text_encoders",
              "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"),
    ModelFile("ae.safetensors", "vae",
              "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"),
    ModelFile("Z-Image-Turbo-Fun-Controlnet-Union.safetensors", "model_patches",
              "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
    # FLUX.2 [klein] 4B (Apache 2.0): edición con referencias
    ModelFile("flux-2-klein-base-4b-fp8.safetensors", "diffusion_models",
              "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors"),
    ModelFile("flux2-vae.safetensors", "vae",
              "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"),
    # FLUX.2 [dev] (no comercial), para GPUs de 24 GB o más
    ModelFile("flux2_dev_fp8mixed.safetensors", "diffusion_models",
              "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"),
    ModelFile("mistral_3_small_flux2_bf16.safetensors", "text_encoders",
              "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors"),
    # Qwen-Image-Edit-2511 (Apache 2.0), para GPUs de 24 GB o más
    ModelFile("qwen_image_edit_2511_bf16.safetensors", "diffusion_models",
              "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"),
    ModelFile("qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders",
              "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ModelFile("qwen_image_vae.safetensors", "vae",
              "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"),
]}


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    role: str  # "render" (desde pases) o "edit" (edición con referencias/máscara)
    min_vram_gb: int
    files: tuple[str, ...]
    license: str
    notes: str
    settings: dict = field(default_factory=dict)


PRESETS = {p.id: p for p in [
    Preset(
        "zimage-control", "Z-Image-Turbo + ControlNet Union (profundidad + líneas)", "render", 8,
        ("z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors",
         "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
        "Apache 2.0",
        "Render base desde los pases. Con 8-12 GB se carga en FP8 y ComfyUI descarga el "
        "codificador de texto a RAM (recomendado 32 GB de RAM).",
        {"steps": 9, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple", "shift": 3.0},
    ),
    Preset(
        "flux2-klein4b-edit", "FLUX.2 [klein] 4B (edición con referencias)", "edit", 8,
        ("flux-2-klein-base-4b-fp8.safetensors", "qwen_3_4b.safetensors", "flux2-vae.safetensors"),
        "Apache 2.0",
        "Edición y refinado con imágenes de referencia (P3).",
        {"steps": 20, "cfg": 5.0, "sampler": "euler"},
    ),
    Preset(
        "flux2-dev-edit", "FLUX.2 [dev] FP8 (edición, máxima calidad)", "edit", 24,
        ("flux2_dev_fp8mixed.safetensors", "mistral_3_small_flux2_bf16.safetensors", "flux2-vae.safetensors"),
        "FLUX Non-Commercial",
        "Solo con 24 GB o más, o en la nube.",
        {"steps": 20, "guidance": 4.0, "sampler": "euler"},
    ),
    Preset(
        "qwen2511-edit", "Qwen-Image-Edit-2511 (edición)", "edit", 24,
        ("qwen_image_edit_2511_bf16.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image_vae.safetensors"),
        "Apache 2.0",
        "Solo con 24 GB o más, o en la nube.",
        {"steps": 40, "cfg": 4.0, "sampler": "euler", "scheduler": "simple", "shift": 3.1},
    ),
]}


def recommend(vram_gb: float | None) -> dict[str, str | None]:
    """Mejor preset local por función para la VRAM dada (None = sin GPU → nube)."""
    out: dict[str, str | None] = {}
    for role in ("render", "edit"):
        fits = [p for p in PRESETS.values() if p.role == role and vram_gb is not None and p.min_vram_gb <= vram_gb]
        out[role] = max(fits, key=lambda p: p.min_vram_gb).id if fits else None
    return out


def weight_dtype_for(vram_gb: float | None) -> str:
    """Z-Image en bf16 ocupa ~12 GB: por debajo de 16 GB se carga en FP8."""
    return "default" if vram_gb and vram_gb >= 16 else "fp8_e4m3fn"


def download(preset_id: str, comfy_dir: Path) -> None:
    preset = PRESETS[preset_id]
    for name in preset.files:
        f = FILES[name]
        dest = comfy_dir / "models" / f.folder / f.name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ✓ {f.folder}/{f.name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"  ↓ {f.folder}/{f.name}")
        with urllib.request.urlopen(f.url) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 22):
                out.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r    {done / 2**30:.2f} / {total / 2**30:.2f} GB")
                    sys.stdout.flush()
        tmp.replace(dest)
        print()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Modelos de AIRE para ComfyUI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("download")
    d.add_argument("preset", choices=sorted(PRESETS))
    d.add_argument("--comfy-dir", required=True, type=Path, help="carpeta raíz de ComfyUI")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for p in PRESETS.values():
            print(f"{p.id:20s} [{p.role}] ≥{p.min_vram_gb} GB · {p.license}\n    {p.title}\n    {p.notes}")
    else:
        download(args.preset, args.comfy_dir)


if __name__ == "__main__":
    main()
