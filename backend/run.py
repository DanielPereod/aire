"""Punto de entrada que usa la extensión de SketchUp.

    python run.py <installer|job|comfyctl|passes|render|...> [args]

Añade esta carpeta al path, así el backend se ejecuta directamente desde la
extensión instalada (sin pip install) con el Python que prepara bootstrap.ps1.
"""

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# La caché de numba va a la carpeta de AIRE (la de plugins de SketchUp puede no ser escribible)
_base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_base, "AIRE", "cache", "numba"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: run.py <módulo> [args]")
    module = importlib.import_module(f"aire_backend.{sys.argv[1]}")
    module.main(sys.argv[2:])
