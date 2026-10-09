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
            py = inst.venv_python(Path(cmd[2]))
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")

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
    fakes = Fakes(fail_model_once=True)
    patch_model_download(monkeypatch, fakes)
    i, _ = make(env, fakes)
    with pytest.raises(UserError, match="continuará donde se quedó"):
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
