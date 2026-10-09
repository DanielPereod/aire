import math

import numpy as np
import pytest

from aire_backend.camera import Camera
from aire_backend.passes import Passes
from aire_backend.scene import load_scene, save_scene
from aire_backend.synthetic import SceneBuilder, build_interior


def make_scene(tmp_path, builder, camera, width=200, height=100):
    meta = {"view": {"width": width, "height": height}, "camera": camera,
            "materials": builder.materials, "objects": builder.objects}
    return load_scene(save_scene(tmp_path / "scene", meta, builder.arrays()))


def level_camera(eye=(0, 0, 0), fov=60.0, **kw):
    cam = {"eye": list(eye), "target": [eye[0], eye[1] + 1, eye[2]], "up": [0, 0, 1],
           "perspective": True, "fov_deg": fov, "fov_is_height": True, "aspect_ratio": 0.0}
    cam.update(kw)
    return cam


def test_roundtrip(tmp_path):
    scene = load_scene(build_interior(tmp_path / "salon"))
    assert scene.positions.shape[1:] == (3, 3)
    assert len(scene.positions) == len(scene.tri_object) == len(scene.uvs)
    chair = scene.find_objects("silla")[0]
    names = {scene.objects[i]["name"] for i in scene.descendants(chair["id"])}
    assert {"Silla", "Asiento", "Respaldo", "Pata 1", "Pata 4"} <= names


def test_wall_depth_is_metric(tmp_path):
    b = SceneBuilder()
    wall = b.object("Pared")
    b.quad((-10, 3, -10), (10, 3, -10), (10, 3, 10), (-10, 3, 10), wall, 0)  # mira a +y... da igual
    p = Passes(make_scene(tmp_path, b, level_camera())).render()
    d = p.arrays["depth"]
    assert p.arrays["valid"].all()
    np.testing.assert_allclose(d, 3.0, rtol=1e-5)
    # Vemos la cara trasera: la normal se gira hacia la cámara (+z de vista)
    np.testing.assert_allclose(p.arrays["normal"][50, 100], [0, 0, 1], atol=1e-5)


def test_floor_behind_camera_is_clipped_correctly(tmp_path):
    """Un suelo que pasa por detrás del ojo (caso típico en interiores) debe
    recortarse en el plano cercano sin huecos y con la profundidad analítica."""
    h = 1.5
    b = SceneBuilder()
    floor = b.object("Suelo")
    b.quad((-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0), floor, 0)
    W, H = 200, 100
    p = Passes(make_scene(tmp_path, b, level_camera(eye=(0, 0, h)), W, H)).render()
    valid, depth = p.arrays["valid"], p.arrays["depth"]
    sx, sy = p.camera.projection_params()
    assert not valid[: H // 2].any()  # por encima del horizonte no hay suelo
    for row in range(H // 2, H):
        y_ndc = 1.0 - (row + 0.5) / H * 2.0
        expected = h / (-y_ndc / sy)  # rayo (x, y_ndc/sy, -1) corta y_vista = -h
        if expected < 40:  # bien dentro del suelo finito (50 m): fila completa y exacta
            assert valid[row].all(), row
            assert depth[row, W // 2] == pytest.approx(expected, rel=1e-4)
    assert valid[-1].all() and depth[-1, W // 2] < 3.0  # la fila inferior viene del tramo recortado
    np.testing.assert_allclose(p.arrays["normal"][90, 100], [0, 1, 0], atol=1e-5)


def test_box_projects_to_expected_pixels(tmp_path):
    b = SceneBuilder()
    box = b.object("Caja")
    b.box((-0.5, 4.0, -0.5), (0.5, 5.0, 0.5), box, 0)
    W, H = 400, 400
    p = Passes(make_scene(tmp_path, b, level_camera(fov=90.0), W, H)).render()
    # Cara frontal a 4 m, semiancho 0.5 → x_ndc = ±0.125 → píxeles 175..225
    ys, xs = np.nonzero(p.arrays["ids"] == box)
    assert (xs.min(), xs.max() + 1) == (175, 225)
    assert (ys.min(), ys.max() + 1) == (175, 225)
    objs = {o["id"]: o for o in p.visible_objects()}
    assert objs[box]["bbox_xyxy"] == [175, 175, 225, 225]


def test_hidden_edges_are_occluded(tmp_path):
    b = SceneBuilder()
    front = b.object("Delante")
    behind = b.object("Detrás")
    b.quad((-5, 2, -5), (5, 2, -5), (5, 2, 5), (-5, 2, 5), front, 0, edges=False)
    b.edges.append(np.array([(-1, 4, 0), (1, 4, 0)], dtype=float))  # arista tapada
    b.edge_obj.append(behind)
    b.edges.append(np.array([(-1, 2, 0.2), (1, 2, 0.2)], dtype=float))  # arista sobre la cara
    b.edge_obj.append(front)
    p = Passes(make_scene(tmp_path, b, level_camera())).render()
    se = p.arrays["sketchup_edges"]
    assert se.any()
    rows = np.unique(np.nonzero(se)[0])
    assert rows.max() < 50  # solo la de z=0.2 (por encima del horizonte)


def test_ortho_camera(tmp_path):
    b = SceneBuilder()
    box = b.object("Caja")
    b.box((-1, 4, -1), (1, 5, 1), box, 0)
    cam = level_camera(perspective=False, ortho_height=4.0)
    p = Passes(make_scene(tmp_path, b, cam, 200, 200)).render()
    ys, xs = np.nonzero(p.arrays["ids"] == box)
    assert (xs.min(), xs.max() + 1) == (50, 150)
    assert p.arrays["depth"][100, 100] == pytest.approx(4.0)


def test_fov_width_conversion():
    cam = Camera.from_scene(level_camera(fov=90.0, fov_is_height=False), {"width": 200, "height": 100})
    assert cam.vfov == pytest.approx(2 * math.atan(0.5))


def test_smooth_normals_on_cylinder_but_not_box(tmp_path):
    from aire_backend.passes import smooth_normals

    b = SceneBuilder()
    cyl = b.object("Cilindro")
    seg, r = 24, 1.0
    for i in range(seg):
        a0, a1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
        p0 = (r * math.cos(a0), r * math.sin(a0))
        p1 = (r * math.cos(a1), r * math.sin(a1))
        b.quad((*p0, 0), (*p1, 0), (*p1, 1), (*p0, 1), cyl, 0, edges=False)
    box = b.object("Caja")
    b.box((3, 3, 0), (4, 4, 1), box, 0)
    arr = b.arrays()
    n = smooth_normals(arr["positions"], arr["tri_object"], 35.0)
    cyl_sel = arr["tri_object"] == cyl
    # En el cilindro, la normal de cada vértice es radial (no la de la faceta)
    pts = arr["positions"][cyl_sel].reshape(-1, 3)
    radial = pts[:, :2] / np.linalg.norm(pts[:, :2], axis=1, keepdims=True)
    np.testing.assert_allclose(n[cyl_sel].reshape(-1, 3)[:, :2], radial, atol=1e-5)
    # En la caja se conservan las normales planas
    box_sel = arr["tri_object"] == box
    np.testing.assert_allclose(n[box_sel], arr["normals"][box_sel], atol=1e-6)


def test_interior_end_to_end(tmp_path):
    scene = load_scene(build_interior(tmp_path / "salon", 320, 200))
    p = Passes(scene).render()
    out = p.save(tmp_path / "passes")
    for name in ("depth.npy", "depth16.png", "normal.png", "ids.npy", "albedo.png",
                 "lines.png", "edges_control.png", "objects_visible.json", "passes.json"):
        assert (out / name).exists(), name
    visible = {o["name"] for o in p.visible_objects()}
    assert {"Mesa comedor", "Silla", "Suelo", "Paredes"} <= visible
    assert p.arrays["valid"].all()  # interior cerrado: ningún píxel de fondo


def test_section_plane_hides_camera_side(tmp_path):
    """Pared delante de la cámara cortada por un plano de sección: en auto se ve la
    caja de detrás; conservando el otro lado se ve la pared."""
    b = SceneBuilder()
    wall = b.object("Pared")
    box = b.object("Caja")
    b.quad((-5, 1, -5), (5, 1, -5), (5, 1, 5), (-5, 1, 5), wall, 0, edges=False)
    b.box((-0.5, 4, -0.5), (0.5, 5, 0.5), box, 0)
    arr = b.arrays()
    arr["tri_clip"] = np.where(arr["tri_object"] == wall, 1, 0).astype(np.uint32)
    meta = {"view": {"width": 100, "height": 100}, "camera": level_camera(fov=60.0),
            "materials": b.materials, "objects": b.objects,
            "section_planes": [[0.0, 1.0, 0.0, -1.5]], "clip_sets": [[], [0]]}
    scene = load_scene(save_scene(tmp_path / "s", meta, arr))
    auto = Passes(scene, section_keep="auto").render().arrays["ids"]
    assert auto[50, 50] == box and (auto == wall).sum() == 0
    # El plano vale -1.5 en el ojo → "negative" conserva el lado de la cámara
    keep_cam = Passes(scene, section_keep="negative").render().arrays["ids"]
    assert keep_cam[50, 50] == wall
