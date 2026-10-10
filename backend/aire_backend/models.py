"""Catálogo de modelos (ficheros de ComfyUI) y presets por nivel de hardware.

Las URLs y carpetas salen de los blueprints oficiales de ComfyUI (octubre 2026).

    python -m aire_backend.models list
    python -m aire_backend.models download zimage-control --comfy-dir C:\\ComfyUI
"""

from __future__ import annotations

import argparse
import http.client
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    folder: str  # carpeta dentro de ComfyUI/models
    url: str
    size_gb: float = 0.0  # aproximado, para avisar antes de descargar


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
    # FLUX.2 [klein] 4B destilado (Apache 2.0): edición con referencias en 4 pasos
    ModelFile("flux-2-klein-4b-fp8.safetensors", "diffusion_models",
              "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors"),
    # FLUX.2 [klein] 4B base (sin destilar, 20 pasos)
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
    # Qwen Image 2.1 en int8 (plantilla oficial «image_qwen_image_2_1_image_edit»): cabe en 8 GB
    # descargando parte a la RAM; ~80 s por imagen en una RTX 5060
    ModelFile("qwen_image_2.1_int8_convrot.safetensors", "diffusion_models",
              "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 6.8),
    ModelFile("qwen3vl_8b_int8_convrot.safetensors", "text_encoders",
              "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors", 8.7),
    ModelFile("qwen_image_2.1_vae_bf16.safetensors", "vae",
              "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors", 0.6),
    # Turbo: LoRA oficial extraída de Qwen-Image-2.1-Turbo; con el modelo de arriba, 8 pasos en vez de 25
    ModelFile("qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors", "loras",
              "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors", 0.85),
    # Ligero: modelo en GGUF de 4 bits (unsloth) y codificador de texto oficial en w4a8.
    # Necesita el complemento ComfyUI-GGUF (se instala al elegir esta variante)
    ModelFile("qwen-image-2.1-Q4_K_M.gguf", "diffusion_models",
              "https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q4_K_M.gguf", 3.9),
    ModelFile("qwen3vl_8b_w4a8.safetensors", "text_encoders",
              "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_w4a8.safetensors", 5.9),
    # Grande: pesos completos en bf16, para gráficas de 24 GB o más
    ModelFile("qwen_image_2.1_bf16.safetensors", "diffusion_models",
              "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_bf16.safetensors", 13.3),
    ModelFile("qwen3vl_8b_bf16.safetensors", "text_encoders",
              "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_bf16.safetensors", 16.3),
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
        "qwen21", "Qwen Image 2.1 (máxima calidad)", "edit", 8,
        ("qwen_image_2.1_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors",
         "qwen_image_2.1_vae_bf16.safetensors"),
        "ver la ficha del modelo en Hugging Face",
        "Edición con varias imágenes de referencia. Unos 16 GB; con 8 GB de VRAM ComfyUI pasa parte "
        "a la RAM (recomendado 32 GB).",
        {"steps": 25, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    ),
    Preset(
        "qwen21-turbo", "Qwen Image 2.1 Turbo (rápido)", "edit", 8,
        ("qwen_image_2.1_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors",
         "qwen_image_2.1_vae_bf16.safetensors", "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"),
        "ver la ficha del modelo en Hugging Face",
        "El modelo estándar con la LoRA Turbo oficial: 8 pasos en vez de 25, unas 3 veces más rápido.",
        {"steps": 8, "lora": "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"},
    ),
    Preset(
        "qwen21-gguf", "Qwen Image 2.1 ligero (GGUF 4 bits)", "edit", 6,
        ("qwen-image-2.1-Q4_K_M.gguf", "qwen3vl_8b_w4a8.safetensors", "qwen_image_2.1_vae_bf16.safetensors"),
        "ver la ficha del modelo en Hugging Face",
        "Mitad de tamaño que el estándar, para poca memoria. Algo menos de detalle; necesita ComfyUI-GGUF.",
        {"steps": 25, "unet": "qwen-image-2.1-Q4_K_M.gguf", "text_encoder": "qwen3vl_8b_w4a8.safetensors",
         "gguf": True},
    ),
    Preset(
        "qwen21-grande", "Qwen Image 2.1 completo (bf16)", "edit", 24,
        ("qwen_image_2.1_bf16.safetensors", "qwen3vl_8b_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors"),
        "ver la ficha del modelo en Hugging Face",
        "Pesos completos, máxima calidad. Solo con 24 GB de memoria gráfica o más.",
        {"steps": 25, "unet": "qwen_image_2.1_bf16.safetensors", "text_encoder": "qwen3vl_8b_bf16.safetensors"},
    ),
    Preset(
        "flux2-klein4b", "FLUX.2 [klein] 4B (fotorrealiza la imagen base del modelo)", "render", 8,
        ("flux-2-klein-4b-fp8.safetensors", "qwen_3_4b.safetensors", "flux2-vae.safetensors"),
        "Apache 2.0",
        "Modelo de edición: parte de la imagen base del modelo (materiales reales) y la vuelve "
        "fotográfica conservando objetos y distribución. 4 pasos.",
        {"steps": 4, "cfg": 1.0, "sampler": "euler"},
    ),
    Preset(
        "zimage-control", "Z-Image-Turbo + ControlNet Union (profundidad + líneas)", "control", 8,
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


# Variantes de Qwen que el usuario elige en «Opciones avanzadas» (config.json → qwen_variant).
# Solo se descarga la elegida, y la primera vez que se usa.
QWEN_VARIANTS = {"turbo": "qwen21-turbo", "estandar": "qwen21", "ligero": "qwen21-gguf", "grande": "qwen21-grande"}


def default_qwen_variant(vram_gb: float | None) -> str:
    if vram_gb and vram_gb >= 24:
        return "grande"
    if vram_gb is not None and vram_gb < 8:
        return "ligero"
    return "turbo"


def qwen_preset(cfg: dict) -> str:
    """Preset de Qwen elegido (o el recomendado para esta gráfica)."""
    v = cfg.get("qwen_variant")
    if v not in QWEN_VARIANTS:
        v = default_qwen_variant(cfg.get("vram_gb"))
    return QWEN_VARIANTS[v]


def qwen_options(preset_id: str) -> dict:
    """Ajustes de QwenSettings para un preset: modelo, codificador, LoRA, pasos y si es GGUF."""
    st = PRESETS[preset_id].settings
    return {k: st[k] for k in ("steps", "unet", "text_encoder", "lora", "gguf") if k in st}


def download_gb(preset_id: str, comfy_dir: Path) -> float:
    return sum(f.size_gb for f in missing_files(preset_id, comfy_dir))


def weight_dtype_for(vram_gb: float | None) -> str:
    """Z-Image en bf16 ocupa ~12 GB: por debajo de 16 GB se carga en FP8."""
    return "default" if vram_gb and vram_gb >= 16 else "fp8_e4m3fn"


def missing_files(preset_id: str, comfy_dir: Path) -> list[ModelFile]:
    return [FILES[n] for n in PRESETS[preset_id].files
            if not (comfy_dir / "models" / FILES[n].folder / n).is_file()
            or (comfy_dir / "models" / FILES[n].folder / n).stat().st_size == 0]


def other_model_roots() -> list[Path]:
    """Carpetas «models» de otros ComfyUI del ordenador (StabilityMatrix, ComfyUI Desktop,
    portables): si ya tienen un fichero, se aprovecha en vez de descargarlo otra vez."""
    import os
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    patterns = [
        (home / "Downloads", "StabilityMatrix*/Data/Packages/*/models"),
        (home / "Descargas", "StabilityMatrix*/Data/Packages/*/models"),
        (home, "StabilityMatrix*/Data/Packages/*/models"),
        (Path(os.environ.get("APPDATA", home)), "StabilityMatrix/Packages/*/models"),
        (home / "Documents", "ComfyUI/models"),
        (home / "Documentos", "ComfyUI/models"),
        (home / "Documents", "ComfyUI*/ComfyUI/models"),
        (Path("C:/"), "ComfyUI*/ComfyUI/models"),
        (Path("C:/"), "ComfyUI*/models"),
    ]
    roots = []
    for base, pattern in patterns:
        try:
            roots += [p for p in base.glob(pattern) if p.is_dir()]
        except OSError:
            continue
    return roots


def adopt_existing(preset_id: str, comfy_dir: Path, roots: list[Path] | None = None) -> list[str]:
    """Enlaza (o copia si no se puede) los ficheros del preset que ya estén en otro ComfyUI.
    Un enlace duro no ocupa espacio extra ni tarda nada en el mismo disco."""
    import os
    import shutil
    adopted = []
    for f in missing_files(preset_id, comfy_dir):
        for root in roots if roots is not None else other_model_roots():
            src = root / f.folder / f.name
            if not src.is_file() or src.stat().st_size == 0:
                continue
            dest = comfy_dir / "models" / f.folder / f.name
            if dest.resolve() == src.resolve():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dest)
            except OSError:
                shutil.copyfile(src, dest)
            adopted.append(f.name)
            break
    return adopted


def fetch(url: str, dest: Path, on_bytes=None, chunk: int = 1 << 22) -> None:
    """Descarga reanudable: si existe dest.part continúa donde se quedó (Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "AIRE", **({"Range": f"bytes={done}-"} if done else {})})
    with urllib.request.urlopen(req, timeout=30) as r:  # sin datos 30 s = conexión colgada
        if done and r.status != 206:  # el servidor no admite reanudar: desde cero
            done = 0
        total = int(r.headers.get("Content-Length") or 0) + done
        try:
            with open(tmp, "ab" if done else "wb") as out:
                while data := r.read(chunk):
                    out.write(data)
                    done += len(data)
                    if on_bytes:
                        on_bytes(done, total)
        except http.client.HTTPException as e:  # p. ej. IncompleteRead: la conexión se cerró
            raise ConnectionError(f"descarga interrumpida: {e!r}") from e
    # Si la conexión se cierra a medias sin error, no dar por bueno un fichero incompleto
    if total and done < total:
        raise ConnectionError(f"descarga incompleta: {done} de {total} bytes")
    tmp.replace(dest)


def download(preset_id: str, comfy_dir: Path, on_progress=None) -> None:
    """on_progress(fichero, índice, nº ficheros, bytes, total) para mostrar el avance."""
    todo = missing_files(preset_id, comfy_dir)
    for i, f in enumerate(todo):
        dest = comfy_dir / "models" / f.folder / f.name
        if on_progress is None:
            print(f"  ↓ {f.folder}/{f.name}")
            cb = lambda d, t: (sys.stdout.write(f"\r    {d / 2**30:.2f} / {t / 2**30:.2f} GB"), sys.stdout.flush())  # noqa: E731
        else:
            cb = lambda d, t, f=f, i=i: on_progress(f, i, len(todo), d, t)  # noqa: E731
        fetch(f.url, dest, cb)
        if on_progress is None:
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
