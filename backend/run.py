"""Punto de entrada que usa la extensión de SketchUp.

    python run.py <installer|job|comfyctl|passes|render|...> [args]

Añade esta carpeta al path, así el backend se ejecuta directamente desde la
extensión instalada (sin pip install) con el Python que prepara bootstrap.ps1.
Si algo falla antes de que el módulo pueda informar (p. ej. al importar), el
error se escribe igualmente en el fichero de progreso (--progress) para que la
ventana de AIRE lo muestre en lugar de quedarse esperando.
"""

import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# La caché de numba va a la carpeta de AIRE (la de plugins de SketchUp puede no ser escribible)
_base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_base, "AIRE", "cache", "numba"))


def _arg(name: str):
    args = sys.argv
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


def _report_crash(exc: BaseException) -> None:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    home = _arg("--home")
    log = Path(home) / "logs" / "crash.log" if home else Path(_base) / "AIRE" / "logs" / "crash.log"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.ctime()} {' '.join(sys.argv[1:3])}\n{text}")
    except OSError:
        log = None
    progress = _arg("--progress")
    if not progress:
        return
    try:
        p = Path(progress)
        state = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if state.get("state") in ("done", "error"):
            return  # el módulo ya informó
        state.update(state="error", updated=time.time(), technical=text[-4000:],
                     log=str(log) if log else None,
                     error="Algo ha fallado al arrancar. Prueba de nuevo y, si se repite, envía el "
                           "fichero de registro a quien te ayuda con AIRE.")
        state.setdefault("steps", ["Preparando"])
        state.setdefault("step", 0)
        p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: run.py <módulo> [args]")
    try:
        module = importlib.import_module(f"aire_backend.{sys.argv[1]}")
        module.main(sys.argv[2:])
    except SystemExit as e:
        if e.code not in (0, None):
            _report_crash(e)
        raise
    except BaseException as e:  # noqa: BLE001
        _report_crash(e)
        raise
