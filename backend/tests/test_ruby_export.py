"""Ejecuta el exportador Ruby con un stub de la API de SketchUp y comprueba que
el backend lee exactamente lo esperado (formato, unidades, jerarquía, espejos)."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aire_backend.scene import load_scene

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "sketchup" / "test" / "export_stub_scene.rb"


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    if not shutil.which("ruby"):
        pytest.skip("ruby no disponible")
    out = tmp_path_factory.mktemp("ruby") / "export"
    subprocess.run(["ruby", str(SCRIPT), str(out)], check=True, capture_output=True, text=True)
    return load_scene(out)


def obj_by_name(scene, name):
    return next(o for o in scene.objects if o["name"] == name)


def tris_of(scene, name):
    return scene.positions[scene.tri_object == obj_by_name(scene, name)["id"]]


def test_counts_and_visibility(scene):
    names = [o["name"] for o in scene.objects]
    assert names == ["Stub", "Suelo", "Espejo", "Padre", "Hija"]  # "Invisible" está en etiqueta oculta
    assert len(scene.positions) == 2 * 3  # tres cuadrados visibles
    assert len(scene.edge_positions) == 4 * 3  # la arista suave no se exporta


def test_units_are_meters(scene):
    t = tris_of(scene, "Suelo")
    assert t.min() == pytest.approx(0.0) and t.max() == pytest.approx(1.0, abs=1e-6)


def test_mirror_keeps_front_normal(scene):
    t = tris_of(scene, "Espejo").astype(np.float64)
    assert t[..., 0].max() == pytest.approx(0.0, abs=1e-6) and t[..., 0].min() == pytest.approx(-1.0, abs=1e-6)
    geo = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    assert (geo[:, 2] > 0).all()  # el orden de vértices sigue mirando a +Z
    n = scene.normals[scene.tri_object == obj_by_name(scene, "Espejo")["id"]]
    np.testing.assert_allclose(n.reshape(-1, 3), np.tile([0, 0, 1], (n.size // 3, 1)), atol=1e-6)


def test_nested_transform_and_hierarchy(scene):
    hija, padre = obj_by_name(scene, "Hija"), obj_by_name(scene, "Padre")
    assert hija["parent"] == padre["id"]
    t = tris_of(scene, "Hija")
    # traslación 2 m dentro de una escala x2 y un desplazamiento de 10 m en Y
    assert t[..., 0].min() == pytest.approx(4.0, abs=1e-5) and t[..., 0].max() == pytest.approx(6.0, abs=1e-5)
    assert t[..., 1].min() == pytest.approx(10.0, abs=1e-5)
    np.testing.assert_allclose(hija["bounds_m"]["min"], [4.0, 10.0, 0.0], atol=1e-5)


def test_materials_and_inherited_texture_uv(scene):
    mats = {m["name"]: m for m in scene.materials}
    assert mats["Rojo"]["color"][:3] == [200, 20, 20]
    assert mats["Madera"]["texture"]["width_m"] == pytest.approx(0.5)
    sel = scene.tri_object == obj_by_name(scene, "Suelo")["id"]
    assert (scene.tri_material_front[sel] == mats["Madera"]["id"]).all()
    # 1 m de suelo con textura de 0,5 x 0,25 m → 2 x 4 repeticiones
    uv = scene.uvs[sel].reshape(-1, 2)
    assert uv[:, 0].max() == pytest.approx(2.0, abs=1e-5) and uv[:, 1].max() == pytest.approx(4.0, abs=1e-5)


def test_camera(scene):
    cam = scene.camera
    np.testing.assert_allclose(cam["eye"], [0, -5, 2], atol=1e-6)
    assert cam["perspective"] and cam["fov_deg"] == 50.0
    assert scene.view == {"width": 600, "height": 400, "viewport_width": 1200, "viewport_height": 800}


def test_section_plane_exported_in_world(scene):
    assert len(scene.section_planes) == 1
    np.testing.assert_allclose(scene.section_planes[0], [1, 0, 0, -5.0], atol=1e-6)
    hija = scene.tri_object == obj_by_name(scene, "Hija")["id"]
    suelo = scene.tri_object == obj_by_name(scene, "Suelo")["id"]
    assert all(scene.clip_sets[c] == [0] for c in scene.tri_clip[hija])
    assert (scene.tri_clip[suelo] == 0).all() and scene.clip_sets[0] == []


def test_ruby_window_logic():
    if not shutil.which("ruby"):
        pytest.skip("ruby no disponible")
    proc = subprocess.run(["ruby", str(REPO / "sketchup" / "test" / "window_test.rb")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
