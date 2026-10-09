"""Arranca y para el ComfyUI propio de AIRE, en segundo plano y sin ventanas.

    python -m aire_backend.comfyctl status|start|stop --home <carpeta AIRE>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .comfy import ComfyClient
from .progress import UserError

NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
NEW_GROUP = 0x00000200 if os.name == "nt" else 0  # CREATE_NEW_PROCESS_GROUP


def url(cfg: dict) -> str:
    return f"http://127.0.0.1:{cfg.get('port', 8190)}"


def is_up(cfg: dict) -> bool:
    try:
        with urllib.request.urlopen(url(cfg) + "/system_stats", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def start(home: Path, cfg: dict) -> subprocess.Popen:
    log = home / "logs" / "comfyui.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cfg["comfy_python"], "main.py", "--listen", "127.0.0.1", "--port", str(cfg.get("port", 8190)),
           "--disable-auto-launch", "--preview-method", "none", *cfg.get("comfy_args", [])]
    with open(log, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n=== {time.ctime()} $ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=cfg["comfy_dir"], stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=NO_WINDOW | NEW_GROUP,
                                start_new_session=(os.name != "nt"))
    (home / "comfyui.pid").write_text(str(proc.pid))
    return proc


def ensure(home: Path, cfg: dict, timeout: float = 300, on_wait=None) -> ComfyClient:
    """Devuelve un cliente con ComfyUI listo, arrancándolo si hace falta."""
    if not is_up(cfg):
        proc = start(home, cfg)
        t0 = time.time()
        while not is_up(cfg):
            if proc.poll() is not None:
                raise UserError("El motor de render se ha cerrado al arrancar. Reinicia el ordenador y "
                                "vuelve a probar; si se repite, envía el registro logs/comfyui.log.")
            if time.time() - t0 > timeout:
                raise UserError("El motor de render tarda demasiado en arrancar. Vuelve a intentarlo.")
            if on_wait:
                on_wait(time.time() - t0)
            time.sleep(1)
    return ComfyClient(url(cfg))


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             creationflags=NO_WINDOW).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(home: Path) -> bool:
    pid_file = home / "comfyui.pid"
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text().strip() or 0)
    pid_file.unlink(missing_ok=True)
    if not pid or not _alive(pid):
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
    else:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "start", "stop"])
    ap.add_argument("--home", required=True, type=Path)
    args = ap.parse_args(argv)
    cfg = json.loads((args.home / "config.json").read_text(encoding="utf-8"))
    if args.cmd == "status":
        print("up" if is_up(cfg) else "down")
    elif args.cmd == "start":
        ensure(args.home, cfg)
        print("up")
    else:
        print("stopped" if stop(args.home) else "not running")
    sys.exit(0)


if __name__ == "__main__":
    main()
