"""Instalación automática del motor de render (la lanza la extensión de SketchUp).

Antes de llegar aquí, bootstrap.ps1 ya ha descargado `uv` y creado el entorno
de Python del backend. Este módulo instala, dentro de la carpeta de AIRE:

    comfy/ComfyUI/   código de ComfyUI (versión fijada)
    comfy/venv/      Python de ComfyUI con PyTorch para la GPU NVIDIA
    comfy/ComfyUI/models/...   modelos del preset recomendado
    config.json      rutas y ajustes; "ready": true al terminar

Cada paso es idempotente: si se corta (sin conexión, se apaga el PC), al
volver a pulsar "Preparar AIRE" continúa donde se quedó.

    python -m aire_backend.installer --home <carpeta AIRE> --uv <uv.exe> --progress setup.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from . import comfyctl
from .hardware import gpus, ram_gb
from .models import PRESETS, download, fetch, missing_files, recommend, weight_dtype_for
from .progress import Progress, UserError

# Versión de ComfyUI contra la que se han validado los flujos de AIRE
COMFY_COMMIT = "1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8"
COMFY_ZIPS = [
    f"https://github.com/comfyanonymous/ComfyUI/archive/{COMFY_COMMIT}.zip",
    "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip",
]
# Índices de PyTorch con CUDA, del más nuevo al más antiguo (las RTX 50 necesitan ≥ 12.8)
TORCH_INDEXES = [
    "https://download.pytorch.org/whl/cu130",
    "https://download.pytorch.org/whl/cu128",
]
PYTHON_VERSION = "3.12"
MIN_FREE_GB = 40
COMFY_PORT = 8190  # puerto propio para no chocar con otro ComfyUI del usuario

STEPS = [
    "Preparando el instalador",  # lo hace bootstrap.ps1 (descarga uv y Python)
    "Comprobando tu ordenador",
    "Descargando el motor de render",
    "Instalando el motor de render",
    "Descargando el modelo de inteligencia artificial",
    "Probando que todo funciona",
]

NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


def load_config(home: Path) -> dict:
    p = home / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_config(home: Path, cfg: dict) -> None:
    (home / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class Installer:
    def __init__(self, home: Path, uv: Path, progress: Progress, log: Path, runner=None, fetcher=None):
        self.home = home
        self.uv = uv
        self.progress = progress
        self.log = log
        self.run = runner or self._run
        self.fetch = fetcher or fetch
        self.cfg = load_config(home)

    # ------------------------------------------------------------------ utilidades
    def _run(self, cmd: list, what: str) -> None:
        """Ejecuta un comando sin ventana, con salida al registro."""
        with open(self.log, "a", encoding="utf-8", errors="replace") as lf:
            lf.write(f"\n$ {' '.join(map(str, cmd))}\n")
            lf.flush()
            proc = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT,
                                    creationflags=NO_WINDOW)
            t0 = time.time()
            while proc.poll() is None:
                mins = int((time.time() - t0) // 60)
                self.progress.update(f"{what} (lleva {mins} min)" if mins else what, None)
                time.sleep(1)
        if proc.returncode != 0:
            raise RuntimeError(f"Falló: {' '.join(map(str, cmd))} (código {proc.returncode})")

    @property
    def comfy_root(self) -> Path:
        return self.home / "comfy"

    @property
    def comfy_dir(self) -> Path:
        return self.comfy_root / "ComfyUI"

    @property
    def comfy_venv(self) -> Path:
        return self.comfy_root / "venv"

    # ------------------------------------------------------------------ pasos
    def check_system(self) -> None:
        self.progress.step(1, "Mirando la tarjeta gráfica y el espacio libre…")
        g = gpus()
        vram = max((x["vram_gb"] for x in g), default=None)
        if vram is None:
            raise UserError("No se ha encontrado una tarjeta gráfica NVIDIA. AIRE necesita una para "
                            "crear las imágenes en este ordenador. (Más adelante se podrá usar la nube.)")
        rec = recommend(vram)
        if rec["render"] is None:
            raise UserError(f"La tarjeta gráfica tiene {vram:g} GB de memoria y AIRE necesita al menos 8 GB.")
        free = shutil.disk_usage(self.home).free / 2**30
        pending = not self.cfg.get("models_ok")
        if pending and free < MIN_FREE_GB:
            raise UserError(f"Hace falta más espacio: libera al menos {MIN_FREE_GB} GB en el disco donde "
                            f"está {self.home} (ahora hay {free:.0f} GB libres) y vuelve a intentarlo.")
        self.cfg.update(gpu=g[0]["name"], vram_gb=vram, ram_gb=ram_gb(), preset=rec["render"],
                        edit_preset=rec["edit"], weight_dtype=weight_dtype_for(vram), port=COMFY_PORT)
        save_config(self.home, self.cfg)

    def get_comfy(self) -> None:
        self.progress.step(2, "Descargando ComfyUI…", 0)
        if (self.comfy_dir / "main.py").exists():
            return
        self.comfy_root.mkdir(parents=True, exist_ok=True)
        zpath = self.comfy_root / "comfyui.zip"
        last = None
        for url in COMFY_ZIPS:
            try:
                self.fetch(url, zpath, lambda d, t: self.progress.update(
                    f"Descargando ComfyUI… {d / 2**20:.0f} MB", 100 * d / t if t else None))
                break
            except OSError as e:
                last = e
        else:
            raise UserError("No se ha podido descargar el motor de render. Comprueba la conexión a "
                            "internet y vuelve a intentarlo.") from last
        self.progress.update("Descomprimiendo…", None)
        tmp = self.comfy_root / "_extract"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        inner = next(p for p in tmp.iterdir() if p.is_dir())
        shutil.rmtree(self.comfy_dir, ignore_errors=True)
        inner.rename(self.comfy_dir)
        shutil.rmtree(tmp, ignore_errors=True)
        zpath.unlink(missing_ok=True)

    def install_comfy(self) -> None:
        self.progress.step(3, "Preparando Python…")
        if self.cfg.get("comfy_env_ok"):
            return
        py = venv_python(self.comfy_venv)
        if not py.exists():
            self.run([self.uv, "venv", self.comfy_venv, "--python", PYTHON_VERSION], "Preparando Python")
        last = None
        for index in TORCH_INDEXES:
            try:
                self.run([self.uv, "pip", "install", "--python", py, "torch", "torchvision", "torchaudio",
                          "--index-url", index],
                         "Instalando PyTorch para tu tarjeta gráfica (unos 3 GB, puede tardar)")
                break
            except RuntimeError as e:
                last = e
        else:
            raise UserError("No se ha podido instalar PyTorch. Comprueba la conexión a internet y "
                            "vuelve a intentarlo.") from last
        self.run([self.uv, "pip", "install", "--python", py, "-r", self.comfy_dir / "requirements.txt"],
                 "Instalando el resto del motor")
        self.cfg["comfy_env_ok"] = True
        save_config(self.home, self.cfg)

    def get_models(self) -> None:
        preset = self.cfg["preset"]
        todo = missing_files(preset, self.comfy_dir)
        self.progress.step(4, f"{len(todo)} ficheros por descargar" if todo else "Ya descargado", 0)

        def on_progress(f, i, n, done, total):
            pct = 100 * done / total if total else None
            self.progress.update(f"Fichero {i + 1} de {n}: {done / 2**30:.1f} de {total / 2**30:.1f} GB", pct)

        try:
            download(preset, self.comfy_dir, on_progress)
        except OSError as e:
            raise UserError("Se ha cortado la descarga del modelo. Vuelve a pulsar «Preparar AIRE»: "
                            "continuará donde se quedó.") from e
        self.cfg["models_ok"] = True
        save_config(self.home, self.cfg)

    def smoke_test(self) -> None:
        self.progress.step(5, "Arrancando el motor por primera vez (puede tardar un par de minutos)…")
        self.cfg.update(comfy_dir=str(self.comfy_dir), comfy_python=str(venv_python(self.comfy_venv)))
        save_config(self.home, self.cfg)
        client = comfyctl.ensure(self.home, self.cfg, timeout=900, on_wait=lambda t: self.progress.update(
            f"Arrancando el motor por primera vez… {int(t)} s", None))
        from .comfy import validate
        from .workflows import ZImageSettings, zimage_control
        wf = zimage_control(ZImageSettings(width=512, height=512, prompt="test",
                                           weight_dtype=self.cfg["weight_dtype"]), "x.png", "x.png")
        problems = validate(wf, client.object_info())
        if problems:
            raise RuntimeError("Validación del flujo: " + "; ".join(problems))

    def install(self) -> None:
        self.check_system()
        self.get_comfy()
        self.install_comfy()
        self.get_models()
        self.smoke_test()
        self.cfg["ready"] = True
        self.cfg["preset_title"] = PRESETS[self.cfg["preset"]].title
        save_config(self.home, self.cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Instala el motor de render de AIRE")
    ap.add_argument("--home", required=True, type=Path)
    ap.add_argument("--uv", required=True, type=Path)
    ap.add_argument("--progress", type=Path, default=None)
    args = ap.parse_args(argv)

    args.home.mkdir(parents=True, exist_ok=True)
    log = args.home / "logs" / "install.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress or args.home / "progress" / "setup.json", STEPS)
    try:
        Installer(args.home, args.uv, progress, log).install()
    except BaseException as e:  # noqa: BLE001 - cualquier fallo se muestra al usuario
        with open(log, "a", encoding="utf-8") as lf:
            import traceback
            traceback.print_exc(file=lf)
        progress.fail(e, log)
        sys.exit(1)
    progress.done({"config": str(args.home / "config.json")})


if __name__ == "__main__":
    main()
