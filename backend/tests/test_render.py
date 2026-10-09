import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from aire_backend.comfy import ComfyClient, ComfyError, validate
from aire_backend.fidelity import score
from aire_backend.models import FILES, PRESETS, recommend, weight_dtype_for
from aire_backend.passes import Passes
from aire_backend.prompt import build_prompt
from aire_backend.render import prepare_passes, round16, sweep_grid
from aire_backend.scene import load_scene
from aire_backend.synthetic import build_interior
from aire_backend.workflows import ZImageSettings, zimage_control

# object_info mínimo con las firmas reales de ComfyUI (octubre 2026)
OBJECT_INFO = {
    "UNETLoader": {"input": {"required": {"unet_name": [["z_image_turbo_bf16.safetensors"]],
                                          "weight_dtype": [["default", "fp8_e4m3fn"]]}}},
    "CLIPLoader": {"input": {"required": {"clip_name": [["qwen_3_4b.safetensors"]], "type": [["lumina2", "flux2"]]},
                             "optional": {"device": [["default", "cpu"]]}}},
    "VAELoader": {"input": {"required": {"vae_name": [["ae.safetensors"]]}}},
    "ModelPatchLoader": {"input": {"required": {"name": [["Z-Image-Turbo-Fun-Controlnet-Union.safetensors"]]}}},
    "ModelSamplingAuraFlow": {"input": {"required": {"model": ["MODEL"], "shift": ["FLOAT", {}]}}},
    "LoadImage": {"input": {"required": {"image": [[]]}}},
    "ZImageFunControlnet": {"input": {
        "required": {"model": ["MODEL"], "model_patch": ["MODEL_PATCH"], "vae": ["VAE"], "strength": ["FLOAT", {}]},
        "optional": {"image": ["IMAGE"], "inpaint_image": ["IMAGE"], "mask": ["MASK"],
                     "start_percent": ["FLOAT", {}], "end_percent": ["FLOAT", {}]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}], "clip": ["CLIP"]}}},
    "ConditioningZeroOut": {"input": {"required": {"conditioning": ["CONDITIONING"]}}},
    "EmptySD3LatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}}},
    "VAEEncode": {"input": {"required": {"pixels": ["IMAGE"], "vae": ["VAE"]}}},
    "KSampler": {"input": {"required": {k: ["X"] for k in (
        "model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image", "denoise")}}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}}},
    "ReferenceLatent": {"input": {"required": {"conditioning": ["CONDITIONING"]}, "optional": {"latent": ["LATENT"]}}},
    "CFGGuider": {"input": {"required": {"model": ["MODEL"], "positive": ["C"], "negative": ["C"], "cfg": ["FLOAT", {}]}}},
    "RandomNoise": {"input": {"required": {"noise_seed": ["INT", {}]}}},
    "KSamplerSelect": {"input": {"required": {"sampler_name": [["euler"]]}}},
    "Flux2Scheduler": {"input": {"required": {"steps": ["INT", {}], "width": ["INT", {}], "height": ["INT", {}]}}},
    "EmptyFlux2LatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}}},
    "SamplerCustomAdvanced": {"input": {"required": {k: ["X"] for k in ("noise", "guider", "sampler", "sigmas", "latent_image")}}},
}
OBJECT_INFO["UNETLoader"]["input"]["required"]["unet_name"] = [["z_image_turbo_bf16.safetensors", "flux-2-klein-4b-fp8.safetensors"]]
OBJECT_INFO["VAELoader"]["input"]["required"]["vae_name"] = [["ae.safetensors", "flux2-vae.safetensors"]]


def settings(**kw):
    return ZImageSettings(width=512, height=336, prompt="test", **kw)


@pytest.mark.parametrize("s", sweep_grid(settings()) + [settings(refine=0.3), settings(refine=0.3, denoise=1.0)])
def test_workflows_validate(s):
    wf = zimage_control(s, "aire/d.png", "aire/l.png", "aire/a.png")
    assert validate(wf, OBJECT_INFO) == []
    types = [n["class_type"] for n in wf.values()]
    assert types.count("ZImageFunControlnet") == (2 if s.lines_strength > 0 else 1) + (1 if s.refine > 0 else 0)
    assert ("VAEEncode" in types) == (s.denoise < 1.0)
    assert types.count("KSampler") == (2 if s.refine > 0 else 1)


def test_validate_reports_missing_model_and_node():
    wf = zimage_control(settings(vae="otro_vae.safetensors"), "a", "b")
    info = {k: v for k, v in OBJECT_INFO.items() if k != "ModelPatchLoader"}
    problems = validate(wf, info)
    assert any("otro_vae" in p for p in problems)
    assert any("ModelPatchLoader" in p for p in problems)


def test_fidelity_metric():
    lines = np.zeros((100, 100), dtype=bool)
    lines[50, 10:90] = True
    lines[10:90, 30] = True
    perfect = Image.fromarray(np.where(lines, 0, 255).astype(np.uint8))
    blank = Image.new("L", (100, 100), 255)
    assert score(perfect, lines)["recall"] > 0.95
    assert score(blank, lines)["recall"] == 0.0
    shifted = Image.fromarray(np.where(np.roll(lines, 20, axis=0), 0, 255).astype(np.uint8))
    assert score(shifted, lines)["spurious"] > score(perfect, lines)["spurious"]


def test_prompt_lists_parts_and_materials(tmp_path):
    scene = load_scene(build_interior(tmp_path / "s", 320, 200))
    p = Passes(scene).render()
    prompt = build_prompt(scene, p.arrays["ids"], p.arrays["material"], p.visible_objects(), "nordic")
    assert prompt.startswith("nordic")
    assert "Mesa comedor (Nogal)" in prompt and "Suelo (Tarima roble)" in prompt
    assert "Pata" not in prompt and "Habitación" not in prompt


def test_prepare_passes_multiple_of_16(tmp_path):
    root = build_interior(tmp_path / "s", 333, 222)
    passes, out = prepare_passes(str(root), 1000)
    assert passes.width % 16 == 0 and passes.height % 16 == 0
    assert (out / "depth_control.png").exists()
    assert round16(1000) % 16 == 0 and abs(round16(1000) - 1000) <= 8


def test_model_catalog_and_recommendation():
    for p in PRESETS.values():
        assert all(f in FILES for f in p.files)
    assert recommend(8.0) == {"render": "flux2-klein4b", "edit": "flux2-klein4b-edit"}
    assert recommend(None) == {"render": None, "edit": None}
    assert recommend(24)["edit"] in ("flux2-dev-edit", "qwen2511-edit")
    assert weight_dtype_for(8.0) == "fp8_e4m3fn" and weight_dtype_for(24) == "default"


class FakeComfy(BaseHTTPRequestHandler):
    png = None
    queued = []

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/object_info":
            self._send(OBJECT_INFO)
        elif self.path == "/system_stats":
            self._send({"system": {}, "devices": []})
        elif self.path.startswith("/history/"):
            self._send({"p1": {"status": {"status_str": "success", "completed": True},
                               "outputs": {"9": {"images": [{"filename": "r.png", "subfolder": "aire", "type": "output"}]}}}})
        elif self.path.startswith("/view?"):
            self._send(self.png, "image/png")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/upload/image":
            name = body.split(b'name="image"; filename="')[1].split(b'"')[0].decode()
            self._send({"name": name, "subfolder": "aire", "type": "input"})
        elif self.path == "/prompt":
            FakeComfy.queued.append(json.loads(body)["prompt"])
            self._send({"prompt_id": "p1", "number": 0, "node_errors": {}})


@pytest.fixture
def fake_server():
    buf = BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, "PNG")
    FakeComfy.png = buf.getvalue()
    FakeComfy.queued = []
    srv = HTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_client_roundtrip(fake_server, tmp_path):
    client = ComfyClient(fake_server)
    img = tmp_path / "depth.png"
    Image.new("L", (4, 4)).save(img)
    assert client.upload_image(img) == "aire/depth.png"
    images = client.run(zimage_control(settings(), "aire/depth.png", None))
    assert Image.open(BytesIO(images[0])).getpixel((0, 0)) == (10, 20, 30)
    assert len(FakeComfy.queued) == 1


def test_client_refuses_invalid_workflow(fake_server):
    with pytest.raises(ComfyError, match="no está disponible"):
        ComfyClient(fake_server).run(zimage_control(settings(unet="no_existe.safetensors"), "a", "b"))


@pytest.mark.skipif(not os.environ.get("AIRE_COMFY_URL"), reason="AIRE_COMFY_URL no definido")
def test_real_comfy_accepts_all_sweep_workflows(tmp_path):
    """Integración: un ComfyUI real valida (POST /prompt) las 8 variantes del barrido.
    Con ficheros de modelo vacíos la ejecución falla después, pero la validación
    del grafo (nodos, entradas, tipos de enlace, modelos) ocurre antes de encolar."""
    url = os.environ["AIRE_COMFY_URL"]
    client = ComfyClient(url)
    img = tmp_path / "x.png"
    Image.new("RGB", (64, 64)).save(img)
    name = client.upload_image(img)
    info = client.object_info()
    from dataclasses import replace as _replace
    for s in sweep_grid(settings()) + [_replace(x, refine=0.3) for x in sweep_grid(settings())[:2]]:
        wf = zimage_control(s, name, name, name)
        assert validate(wf, info) == []
        req = urllib.request.Request(url + "/prompt", json.dumps({"prompt": wf}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["node_errors"] == {}


def test_klein_edit_workflow_structure():
    from aire_backend.workflows import KleinSettings, flux2_klein_edit
    wf = flux2_klein_edit(KleinSettings(width=1024, height=672, prompt="p", seed=3), "aire/shaded.png")
    assert validate(wf, OBJECT_INFO) == []
    types = [n["class_type"] for n in wf.values()]
    assert types.count("ReferenceLatent") == 2  # positivo y negativo con la imagen base
    clip = next(n for n in wf.values() if n["class_type"] == "CLIPLoader")
    assert clip["inputs"]["type"] == "flux2"
    sched = next(n for n in wf.values() if n["class_type"] == "Flux2Scheduler")
    assert sched["inputs"] == {"steps": 4, "width": 1024, "height": 672}


@pytest.mark.skipif(not os.environ.get("AIRE_COMFY_URL"), reason="AIRE_COMFY_URL no definido")
def test_real_comfy_accepts_klein_workflow(tmp_path):
    from aire_backend.workflows import KleinSettings, flux2_klein_edit
    url = os.environ["AIRE_COMFY_URL"]
    client = ComfyClient(url)
    img = tmp_path / "x.png"
    Image.new("RGB", (64, 64)).save(img)
    name = client.upload_image(img)
    wf = flux2_klein_edit(KleinSettings(width=512, height=512, prompt="test"), name)
    assert validate(wf, client.object_info()) == []
    req = urllib.request.Request(url + "/prompt", json.dumps({"prompt": wf}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        assert json.loads(r.read())["node_errors"] == {}
