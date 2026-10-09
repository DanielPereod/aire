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
    "ImageScale": {"input": {"required": {"image": ["IMAGE"], "upscale_method": [["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]],
                                          "width": ["INT", {}], "height": ["INT", {}], "crop": [["disabled", "center"]]}}},
    "SplitSigmasDenoise": {"input": {"required": {"sigmas": ["SIGMAS"], "denoise": ["FLOAT", {}]}}},
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
    assert "Mesa comedor (brown wood)" in prompt and "Silla (grey fabric)" in prompt
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


def test_klein_two_pass_workflow():
    """Alta calidad: 1.ª pasada a ~1 MP con la base reducida y 2.ª pasada de detalle a tamaño
    final que parte de la foto ampliada y usa la base completa como referencia."""
    from aire_backend.workflows import KleinSettings, flux2_klein_edit
    s = KleinSettings(width=1920, height=1376, prompt="p", seed=3, upscale=1.5, refine_denoise=0.4)
    assert s.first_pass_size() == (1280, 912)
    wf = flux2_klein_edit(s, "aire/shaded.png")
    assert validate(wf, OBJECT_INFO) == []
    by_type = lambda t: [n["inputs"] for n in wf.values() if n["class_type"] == t]  # noqa: E731
    assert [(x["width"], x["height"]) for x in by_type("Flux2Scheduler")] == [(1280, 912), (1920, 1376)]
    assert by_type("Flux2Scheduler")[1]["steps"] == 10 and by_type("SplitSigmasDenoise")[0]["denoise"] == 0.4
    assert by_type("EmptyFlux2LatentImage")[0]["width"] == 1280
    assert len(by_type("SamplerCustomAdvanced")) == 2 and len(by_type("LoadImage")) == 1
    second = by_type("SamplerCustomAdvanced")[1]
    assert second["sigmas"][1] == 1  # low_sigmas: solo los últimos pasos
    assert wf[second["latent_image"][0]]["class_type"] == "VAEEncode"  # parte de la foto ampliada
    assert len(by_type("ReferenceLatent")) == 4
    # Sin ampliación: una sola pasada, como antes
    single = flux2_klein_edit(KleinSettings(width=1024, height=736, prompt="p"), "aire/shaded.png")
    assert [n["class_type"] for n in single.values()].count("SamplerCustomAdvanced") == 1
    assert "ImageScale" not in [n["class_type"] for n in single.values()]


@pytest.mark.skipif(not os.environ.get("AIRE_COMFY_URL"), reason="AIRE_COMFY_URL no definido")
def test_real_comfy_accepts_klein_workflow(tmp_path):
    from aire_backend.workflows import KleinSettings, flux2_klein_edit
    url = os.environ["AIRE_COMFY_URL"]
    client = ComfyClient(url)
    img = tmp_path / "x.png"
    Image.new("RGB", (64, 64)).save(img)
    name = client.upload_image(img)
    for up in (1.0, 1.5):
        wf = flux2_klein_edit(KleinSettings(width=768, height=512, prompt="test", upscale=up), name)
        assert validate(wf, client.object_info()) == []
        req = urllib.request.Request(url + "/prompt", json.dumps({"prompt": wf}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["node_errors"] == {}


def test_color_names_keep_floor_tone():
    from aire_backend.prompt import color_name
    assert color_name((222, 205, 180)) == "beige"
    assert color_name((150, 150, 150)) == "grey"
    assert color_name((70, 100, 140)) == "muted blue"
    assert color_name(None) is None


def test_edit_prompt_names_colors_of_large_surfaces(tmp_path):
    from aire_backend.prompt import build_edit_prompt
    passes, _ = prepare_passes(str(build_interior(tmp_path / "e")), 320)
    a = passes.arrays
    text = build_edit_prompt(passes.scene, a["ids"], a["material"], passes.visible_objects(), albedo=a["albedo"])
    assert "brown at the bottom (Suelo)" in text and "white at the top (Paredes)" in text
    assert "grey" not in text.split("Large surfaces")[0]  # sin palabras de color que empujen


def test_klein_retouch_workflow():
    from aire_backend.workflows import KleinSettings, flux2_klein_retouch
    wf = flux2_klein_retouch(KleinSettings(width=1920, height=1376, prompt="p"), "aire/cycles.png", 0.2)
    assert validate(wf, OBJECT_INFO) == []
    split = next(n for n in wf.values() if n["class_type"] == "SplitSigmasDenoise")
    assert split["inputs"]["denoise"] == 0.2


def test_klein_first_pass_can_start_from_base():
    from aire_backend.workflows import KleinSettings, flux2_klein_edit
    wf = flux2_klein_edit(KleinSettings(width=1920, height=1376, prompt="p", upscale=1.5, base_denoise=0.8),
                          "aire/cycles.png")
    assert validate(wf, OBJECT_INFO) == []
    first = [n for n in wf.values() if n["class_type"] == "SamplerCustomAdvanced"][0]["inputs"]
    assert wf[first["latent_image"][0]]["class_type"] == "VAEEncode" and first["sigmas"][1] == 1
    assert "EmptyFlux2LatentImage" not in [n["class_type"] for n in wf.values()]


def test_lock_colors_takes_reference_hue_keeps_detail():
    import numpy as np
    from PIL import Image
    from aire_backend.colormatch import lock_colors

    # IA: suelo amarillento con una franja de detalle; referencia: gris neutro
    ai = np.full((64, 96, 3), (210, 180, 120), np.uint8)
    ai[:, 40:44] = (120, 100, 60)
    ref = Image.new("RGB", (96, 64), (180, 180, 180))
    out = np.asarray(lock_colors(Image.fromarray(ai), ref, radius=2), np.int16)
    r, g, b = out[10, 10]
    assert abs(r - b) < 12  # el amarillo pasa a neutro
    assert out[10, 41].sum() < out[10, 10].sum() - 150  # la franja oscura (luminancia de la IA) sigue
    same = lock_colors(Image.fromarray(ai), ref, strength=0)
    assert np.array_equal(np.asarray(same), ai)
