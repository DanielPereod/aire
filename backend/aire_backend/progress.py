"""Estado de una tarea larga (instalación, render) en un fichero JSON.

La extensión de SketchUp lo lee cada segundo y lo muestra en la ventana de AIRE.
Los textos van dirigidos a una persona no técnica: qué está pasando y qué hacer.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path


class UserError(Exception):
    """Error con un mensaje pensado para el usuario final (qué pasa y qué hacer)."""


class Progress:
    def __init__(self, path: str | Path | None, steps: list[str]):
        self.path = Path(path) if path else None
        self.steps = steps
        self.state = {
            "state": "running", "steps": steps, "step": 0, "title": steps[0] if steps else "",
            "detail": "", "percent": None, "started": time.time(), "updated": time.time(),
            "pid": os.getpid(), "result": None, "error": None,
        }
        self._last_write = 0.0
        self.write(force=True)

    def write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < 0.5:
            return
        self._last_write = now
        self.state["updated"] = now
        if self.path is None:
            return
        # Informar del progreso nunca debe tumbar la tarea. En Windows, os.replace falla
        # con "Acceso denegado" si SketchUp está leyendo el fichero justo en ese momento:
        # se reintenta y, si no hay suerte, se escribe encima o se deja para la próxima.
        data = json.dumps(self.state, ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            for attempt in range(20):
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1) if attempt < 5 else 0.25)
            self.path.write_text(data, encoding="utf-8")
        except OSError:
            if force:
                self._last_write = 0.0  # que el siguiente write lo vuelva a intentar

    def step(self, index: int, detail: str = "", percent: float | None = None) -> None:
        self.state.update(step=index, title=self.steps[index], detail=detail, percent=percent)
        self.write(force=True)

    def update(self, detail: str | None = None, percent: float | None = None) -> None:
        if detail is not None:
            self.state["detail"] = detail
        self.state["percent"] = percent
        self.write()

    def done(self, result=None) -> None:
        self.state.update(state="done", step=len(self.steps), percent=100.0, result=result, detail="")
        self.write(force=True)

    def fail(self, exc: BaseException, log_path: Path | None = None) -> None:
        if isinstance(exc, UserError):
            msg = str(exc)
        else:
            msg = ("Algo ha fallado. Prueba de nuevo y, si se repite, envía el fichero de registro "
                   "a quien te ayuda con AIRE.")
        self.state.update(state="error", error=msg, technical="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
            log=str(log_path) if log_path else None)
        self.write(force=True)
