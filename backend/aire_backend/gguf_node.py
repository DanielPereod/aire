"""Complemento ComfyUI-GGUF (city96) para la variante «ligero» de Qwen.

Solo se instala si el usuario elige esa variante: se descarga el código en
comfy/ComfyUI/custom_nodes/ComfyUI-GGUF, se instala su dependencia (gguf) en el entorno
de ComfyUI y se reinicia ComfyUI para que cargue el nodo UnetLoaderGGUF.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from . import comfyctl
from .models import fetch
from .progress import UserError

ZIP_URL = "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip"
NODE_DIR = "ComfyUI-GGUF"
NO_WINDOW = 0x08000000


def installed(cfg: dict) -> bool:
    return (Path(cfg["comfy_dir"]) / "custom_nodes" / NODE_DIR / "__init__.py").is_file()


def install(home: Path, cfg: dict, on_status=None) -> None:
    comfy = Path(cfg["comfy_dir"])
    dest = comfy / "custom_nodes" / NODE_DIR
    tmp = comfy / "custom_nodes" / "_gguf_tmp"
    zpath = comfy / "custom_nodes" / "comfyui-gguf.zip"
    log = home / "logs" / "install.log"
    try:
        if on_status:
            on_status("Solo la primera vez: instalando el complemento GGUF…")
        fetch(ZIP_URL, zpath)
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        inner = next(p for p in tmp.iterdir() if p.is_dir())
        shutil.rmtree(dest, ignore_errors=True)
        inner.rename(dest)
        uv = home / "uv" / "uv.exe"
        req = dest / "requirements.txt"
        cmd = [str(uv), "pip", "install", "--python", cfg["comfy_python"], "-r", str(req)]
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"\n$ {' '.join(cmd)}\n")
            lf.flush()
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError(f"uv pip install terminó con código {r.returncode}")
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(dest, ignore_errors=True)
        raise UserError("No se ha podido instalar el complemento para el modelo ligero (GGUF). "
                        "Comprueba la conexión o elige otro modelo en «Opciones avanzadas».") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        zpath.unlink(missing_ok=True)
    comfyctl.stop(home, cfg)  # ComfyUI solo carga complementos al arrancar
