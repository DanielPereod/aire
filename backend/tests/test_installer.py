import io
import json
import threading
import zipfile
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from aire_backend import installer as inst
from aire_backend import job as jobmod
from aire_backend.comfy import ComfyClient
from aire_backend.models import FILES, PRESETS, fetch
from aire_backend.progress import Progress, UserError
from aire_backend.synthetic import build_interior
from tests.test_render import fake_server  # noqa: F401  (fixture)

Usage = namedtuple("Usage", "total used free")


def comfy_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ComfyUI-abc/main.py", "print('comfy')")
        z.writestr("ComfyUI-abc/requirements.txt", "torch\n")
    return buf.getvalue()


class Fakes:
    def __init__(self, fail_model_once=False):
        self.commands, self.fetched = [], []
        self.fail_model_once = fail_model_once

    def runner(self, cmd, what):
        self.commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            venv = Path(cmd[2])
            py = inst.venv_python(venv)
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
            home = venv.parent.parent  # AIRE/comfy/venv → AIRE
            (venv / "pyvenv.cfg").write_text(f"home = {home / 'python' / 'cpython-3.12'}\nversion = 3.12\n")

    def fetcher(self, url, dest, on_bytes=None):
        self.fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".zip"):
            dest.write_bytes(comfy_zip())
        else:
            if self.fail_model_once:
                self.fail_model_once = False
                raise OSError("conexión cortada")
            dest.write_bytes(b"pesos")
        if on_bytes:
            on_bytes(5, 5)


@pytest.fixture
def env(tmp_path, monkeypatch, fake_server):  # noqa: F811
    monkeypatch.setattr(inst, "gpus", lambda: [{"name": "NVIDIA GeForce RTX 5060", "vram_gb": 8.0, "driver": "1"}])
    monkeypatch.setattr(inst, "ram_gb", lambda: 32.0)
    monkeypatch.setattr(inst.shutil, "disk_usage", lambda p: Usage(1, 0, 500 * 2**30))
    monkeypatch.setattr(inst.comfyctl, "ensure", lambda home, cfg, **kw: ComfyClient(fake_server))
    monkeypatch.setattr("aire_backend.models.fetch", None)  # el instalador no debe usar la red real
    home = tmp_path / "AIRE"
    home.mkdir()
    return home


def make(home, fakes, steps_file=None):
    progress = Progress(steps_file or home / "setup.json", inst.STEPS)
    i = inst.Installer(home, Path("uv.exe"), progress, home / "install.log", runner=fakes.runner, fetcher=fakes.fetcher)
    return i, progress


def patch_model_download(monkeypatch, fakes):
    # download() usa models.fetch: lo redirigimos al fetcher falso
    monkeypatch.setattr("aire_backend.models.fetch", fakes.fetcher)


def test_full_install_and_idempotent(env, monkeypatch):
    fakes = Fakes()
    patch_model_download(monkeypatch, fakes)
    i, progress = make(env, fakes)
    i.install()
    cfg = json.loads((env / "config.json").read_text())
    assert cfg["ready"] and cfg["preset"] == "zimage-control" and cfg["weight_dtype"] == "fp8_e4m3fn"
    assert (env / "comfy" / "ComfyUI" / "main.py").exists()
    for name in PRESETS["zimage-control"].files:
        assert (env / "comfy" / "ComfyUI" / "models" / FILES[name].folder / name).exists()
    torch_cmd = next(c for c in fakes.commands if "torch" in c)
    assert "--index-url" in torch_cmd and "cu13" in torch_cmd[-1]

    # Segunda vez: no vuelve a descargar ni a instalar nada
    fakes2 = Fakes()
    patch_model_download(monkeypatch, fakes2)
    make(env, fakes2)[0].install()
    assert fakes2.fetched == [] and not any("pip" in c for c in fakes2.commands)


def test_resume_after_cut_download(env, monkeypatch):
    """Si la conexión falla una y otra vez, se rinde con un mensaje claro y
    «Continuar» reanuda sin volver a descargar lo que ya estaba."""
    fakes = Fakes()

    def always_cut(url, dest, on_bytes=None):
        if url.endswith(".zip"):
            return fakes.fetcher(url, dest, on_bytes)
        raise ConnectionError("conexión cortada")

    monkeypatch.setattr("aire_backend.models.fetch", always_cut)
    i, _ = make(env, fakes)
    i.sleep = lambda s: None
    with pytest.raises(UserError, match="pulsa «Continuar»"):
        i.install()
    assert not json.loads((env / "config.json").read_text()).get("ready")
    fakes2 = Fakes()
    patch_model_download(monkeypatch, fakes2)
    make(env, fakes2)[0].install()
    assert not any(u.endswith(".zip") for u in fakes2.fetched)  # ComfyUI ya estaba
    assert json.loads((env / "config.json").read_text())["ready"]


def test_friendly_errors(env, monkeypatch):
    fakes = Fakes()
    monkeypatch.setattr(inst, "gpus", lambda: [])
    with pytest.raises(UserError, match="NVIDIA"):
        make(env, fakes)[0].install()
    monkeypatch.setattr(inst, "gpus", lambda: [{"name": "GTX 1650", "vram_gb": 4.0, "driver": "1"}])
    with pytest.raises(UserError, match="al menos 8 GB"):
        make(env, fakes)[0].install()
    monkeypatch.setattr(inst, "gpus", lambda: [{"name": "RTX 5060", "vram_gb": 8.0, "driver": "1"}])
    monkeypatch.setattr(inst.shutil, "disk_usage", lambda p: Usage(1, 0, 10 * 2**30))
    with pytest.raises(UserError, match="libera al menos"):
        make(env, fakes)[0].install()


def test_progress_file_reports_error_for_humans(tmp_path):
    p = Progress(tmp_path / "p.json", ["a", "b"])
    p.fail(UserError("Hace falta más espacio."))
    state = json.loads((tmp_path / "p.json").read_text())
    assert state["state"] == "error" and state["error"] == "Hace falta más espacio."
    p.fail(ValueError("detalle interno"))
    state = json.loads((tmp_path / "p.json").read_text())
    assert "detalle interno" not in state["error"] and "detalle interno" in state["technical"]


class RangeHandler(BaseHTTPRequestHandler):
    data = bytes(range(256)) * 1000

    def log_message(self, *a):
        pass

    def do_GET(self):
        start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0] or 0)
        body = self.data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_fetch_resumes_partial_download(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    dest = tmp_path / "m.safetensors"
    (tmp_path / "m.safetensors.part").write_bytes(RangeHandler.data[:1234])
    seen = []
    fetch(f"http://127.0.0.1:{srv.server_port}/m", dest, lambda d, t: seen.append((d, t)))
    srv.shutdown()
    assert dest.read_bytes() == RangeHandler.data
    assert seen[-1] == (len(RangeHandler.data), len(RangeHandler.data))


def test_render_job_end_to_end(tmp_path, monkeypatch, fake_server):  # noqa: F811
    home = tmp_path / "AIRE"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"ready": True, "weight_dtype": "fp8_e4m3fn", "port": 1}))
    monkeypatch.setattr(jobmod.comfyctl, "ensure", lambda h, c, **kw: ComfyClient(fake_server))
    export = build_interior(tmp_path / "jobs" / "1" / "export", 320, 200)
    progress = Progress(tmp_path / "jobs" / "1" / "job.json", jobmod.STEPS)
    result = jobmod.run_job(home, export, "con plantas", "nordico", "tarde", "rapida", 7, progress, variants=2)
    assert len(result["images"]) == 2
    assert "Scandinavian" in result["prompt"] and "golden hour" in result["prompt"]
    assert result["prompt"].startswith("con plantas")
    for img in result["images"]:
        assert Path(img["path"]).exists() and Path(img["thumb"]).exists()
    assert (Path(result["folder"]) / "result.json").exists()


def test_render_job_requires_install(tmp_path):
    home = tmp_path / "AIRE"
    home.mkdir()
    (home / "config.json").write_text("{}")
    with pytest.raises(UserError, match="Preparar AIRE"):
        jobmod.run_job(home, tmp_path, "", "modelo", "dia", "alta", None, Progress(None, jobmod.STEPS))


def test_packaged_extension_runs_job(tmp_path, fake_server):  # noqa: F811
    """El .rbz contiene un backend que funciona tal cual lo lanza la extensión:
    python <ext>/aire/backend/run.py job ... --progress job.json"""
    import subprocess
    import sys
    import zipfile as zf

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import build_rbz

    rbz = build_rbz.main()
    plugins = tmp_path / "Plugins"
    zf.ZipFile(rbz).extractall(plugins)
    run_py = plugins / "aire" / "backend" / "run.py"
    assert (plugins / "aire" / "ui" / "index.html").exists() and (plugins / "aire" / "bin" / "bootstrap.ps1").exists()

    home = tmp_path / "AIRE"
    home.mkdir()
    port = int(fake_server.rsplit(":", 1)[1])
    (home / "config.json").write_text(json.dumps({"ready": True, "port": port, "weight_dtype": "fp8_e4m3fn"}))
    job = tmp_path / "jobs" / "1"
    export = build_interior(job / "export", 320, 200)
    (job / "prompt.txt").write_text('con "comillas" y\nsaltos de línea, ñ', encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "LOCALAPPDATA": str(tmp_path / "local")}
    proc = subprocess.run([sys.executable, str(run_py), "job", "--home", str(home), "--export", str(export),
                           "--style", "japandi", "--light", "noche", "--quality", "rapida",
                           "--prompt-file", str(job / "prompt.txt"), "--progress", str(job / "job.json")],
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)
    state = json.loads((job / "job.json").read_text(encoding="utf-8"))
    assert proc.returncode == 0, (proc.stderr, state)
    assert state["state"] == "done"
    assert state["result"]["prompt"].startswith('con "comillas" y\nsaltos de línea, ñ')
    assert Path(state["result"]["images"][0]["thumb"]).exists()
    assert (tmp_path / "local" / "AIRE" / "cache" / "numba").exists()  # caché fuera de la extensión


def test_progress_survives_locked_file_on_windows(tmp_path, monkeypatch):
    """SketchUp lee setup.json cada segundo; en Windows os.replace falla con
    «Acceso denegado» si coincide. Nunca debe tumbar la instalación."""
    calls = {"n": 0}
    real_replace = inst.os.replace

    def flaky_replace(a, b):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(5, "Acceso denegado")
        real_replace(a, b)

    monkeypatch.setattr("aire_backend.progress.os.replace", flaky_replace)
    monkeypatch.setattr("aire_backend.progress.time.sleep", lambda s: None)
    p = Progress(tmp_path / "setup.json", ["a", "b"])
    p.step(1, "hola")
    assert json.loads((tmp_path / "setup.json").read_text())["detail"] == "hola"

    def always_locked(a, b):
        raise PermissionError(5, "Acceso denegado")

    monkeypatch.setattr("aire_backend.progress.os.replace", always_locked)
    p.update("sigue", 50)  # escribe encima como último recurso
    p.done({"ok": True})
    assert json.loads((tmp_path / "setup.json").read_text())["state"] == "done"

    monkeypatch.setattr("aire_backend.progress.Path.write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("bloqueado")))
    p.fail(UserError("x"))  # ni siquiera así lanza excepción


def test_foreign_python_venv_is_rebuilt(env, monkeypatch):
    """Un entorno creado con otro Python (p. ej. el de StabilityMatrix) se rehace."""
    fakes = Fakes()
    patch_model_download(monkeypatch, fakes)
    venv = env / "comfy" / "venv"
    inst.venv_python(venv).parent.mkdir(parents=True)
    inst.venv_python(venv).write_text("")
    (venv / "pyvenv.cfg").write_text("home = C:\\Users\\elesa\\Downloads\\StabilityMatrix-win-x64\\Data\\Assets\\Python\n")
    (env / "config.json").write_text(json.dumps({"comfy_env_ok": True}))
    assert not inst.uses_own_python(venv, env)
    make(env, fakes)[0].install()
    assert any(c[1] == "venv" for c in fakes.commands)
    assert inst.uses_own_python(venv, env)


class TruncatingHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "1000")
        self.end_headers()
        self.wfile.write(b"x" * 400)  # se corta a mitad
        self.wfile.flush()
        self.close_connection = True


def test_truncated_download_is_not_accepted(tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), TruncatingHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    dest = tmp_path / "m.safetensors"
    with pytest.raises(ConnectionError):
        fetch(f"http://127.0.0.1:{srv.server_port}/m", dest)
    srv.shutdown()
    assert not dest.exists()  # nunca un modelo corrupto como si estuviera completo
    assert (tmp_path / "m.safetensors.part").stat().st_size == 400  # se reanudará desde aquí


def test_model_download_retries_automatically(env, monkeypatch):
    fakes = Fakes()
    failures = {"left": 2}
    good = fakes.fetcher

    def flaky(url, dest, on_bytes=None):
        if not url.endswith(".zip") and failures["left"]:
            failures["left"] -= 1
            raise ConnectionError("conexión perdida")
        good(url, dest, on_bytes)

    monkeypatch.setattr("aire_backend.models.fetch", flaky)
    i, progress = make(env, fakes)
    i.fetch = flaky
    i.sleep = lambda s: None
    i.install()
    assert json.loads((env / "config.json").read_text())["ready"]
    assert "reintento 2/5" in (env / "install.log").read_text(encoding="utf-8")


def test_speed_and_eta_texts():
    from aire_backend.installer import SpeedMeter, _eta
    m = SpeedMeter()
    m.samples = [(0.0, 0)]
    m.key = "f"
    import time as _t
    rate = m.update("f", 100 * 2**20)  # muchos segundos después de t=0
    assert rate and rate > 0
    assert _eta(30) == "menos de 2 min" and _eta(600) == "10 min" and _eta(5400) == "1 h 30 min"


def test_run_py_reports_early_crash_to_progress(tmp_path):
    """Si el trabajo falla antes de poder informar, la ventana debe enterarse."""
    import subprocess
    import sys
    run_py = Path(__file__).resolve().parents[1] / "run.py"
    progress = tmp_path / "job.json"
    progress.write_text(json.dumps({"state": "running", "steps": ["Preparando la escena"], "step": 0}))
    # --export inexistente: falla dentro de job antes de crear su Progress
    proc = subprocess.run([sys.executable, str(run_py), "job", "--home", str(tmp_path / "nohome"),
                           "--export", str(tmp_path / "nada"), "--progress", str(progress)],
                          capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "LOCALAPPDATA": str(tmp_path)})
    assert proc.returncode != 0
    state = json.loads(progress.read_text(encoding="utf-8"))
    assert state["state"] == "error" and state["error"]


def test_run_py_reports_import_crash(tmp_path):
    import subprocess
    import sys
    run_py = Path(__file__).resolve().parents[1] / "run.py"
    progress = tmp_path / "job.json"
    progress.write_text(json.dumps({"state": "running"}))
    proc = subprocess.run([sys.executable, str(run_py), "no_existe", "--progress", str(progress)],
                          capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "LOCALAPPDATA": str(tmp_path)})
    assert proc.returncode != 0
    state = json.loads(progress.read_text(encoding="utf-8"))
    assert state["state"] == "error" and "ModuleNotFoundError" in state["technical"]
    assert (tmp_path / "AIRE" / "logs" / "crash.log").exists()
