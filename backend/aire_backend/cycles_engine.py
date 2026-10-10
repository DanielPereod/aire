"""Motor de luz real (Cycles) visto desde AIRE: instalación y render en un proceso aparte.

Cycles va como módulo de Python (paquete «bpy» de Blender) en su propio entorno,
`<AIRE>/cycles/venv`, porque necesita otra versión de Python que el resto de AIRE.
Se instala solo la primera vez (unos 700 MB) y no abre ninguna ventana.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

BPY_VERSION = "5.2.2"
PYTHON_VERSION = "3.13"  # el que exige ese bpy
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def venv_dir(home: Path) -> Path:
    return home / "cycles" / "venv"


def python(home: Path) -> Path:
    v = venv_dir(home)
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def uv_path(home: Path) -> Path:
    return home / "uv" / ("uv.exe" if os.name == "nt" else "uv")


def installed(home: Path, cfg: dict) -> bool:
    return bool(cfg.get("cycles_ok")) and python(home).exists()


def install(home: Path, uv: Path, run) -> None:
    """run(cmd, texto) ejecuta un comando sin ventana (el del instalador)."""
    py = python(home)
    if not py.exists():
        run([uv, "venv", venv_dir(home), "--python", PYTHON_VERSION], "Preparando el motor de luz")
    run([uv, "pip", "install", "--python", py, f"bpy=={BPY_VERSION}", "numpy"],
        "Instalando el motor de luz real (unos 700 MB)")


def render(home: Path, export: Path, out: Path, width: int, light: str, samples: int = 256,
           timeout: float = 1800, on_wait=None, library: Path | None = None, dof: bool = True) -> dict:
    """Renderiza con Cycles en un proceso aparte. Devuelve el JSON que escribe cycles_render.py."""
    backend = Path(__file__).resolve().parents[1]
    log = home / "logs" / "cycles.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(python(home)), "-m", "aire_backend.cycles_render", str(export), "--out", str(out),
           "--width", str(width), "--samples", str(samples), "--light", light]
    if library is not None:
        cmd += ["--library", str(library)]
    if not dof:
        cmd.append("--no-dof")
    env = {**os.environ, "PYTHONPATH": str(backend)}
    with open(log, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n=== {time.ctime()}\n$ {' '.join(cmd)}\n")
        lf.flush()
        # cwd en la carpeta de AIRE: nada del directorio actual debe tapar módulos de Blender
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(home),
                                creationflags=NO_WINDOW)
        t0 = time.time()
        while proc.poll() is None:
            if time.time() - t0 > timeout:
                proc.kill()
                raise RuntimeError("Cycles no ha terminado a tiempo")
            if on_wait:
                on_wait(time.time() - t0)
            time.sleep(1)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"Cycles ha fallado (código {proc.returncode}); ver {log}")
    return json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
