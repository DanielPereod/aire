import json

from PIL import Image

from aire_backend import matlib


def test_kind_by_name_material_first():
    assert matlib.kind_for("Tarima roble").key == "wood_floor"
    assert matlib.kind_for("Nogal").key == "wood"
    assert matlib.kind_for("Pintura blanca").key == "plaster"
    assert matlib.kind_for("Marmol blanco", "Isla de nogal").key == "marble"
    assert matlib.kind_for("Vidrio") is None and matlib.kind_for("Azulejo metro") is None
    assert matlib.kind_for("Material~11", None, None) is None


CATALOG = {
    "oak_veneer_01": {"name": "Oak Veneer 01", "tags": ["wood"], "categories": ["wood"], "download_count": 5,
                      "dimensions": [1000, 1000]},
    "rotten_wood": {"name": "Rotten wood", "tags": ["wood"], "categories": ["wood"], "download_count": 900},
    "nice_plaster": {"name": "Nice plaster", "tags": ["plaster", "wall"], "categories": ["plaster"],
                     "download_count": 3},
}


def test_choose_prefers_known_then_most_downloaded_match():
    assert matlib.choose(matlib.BY_KEY["wood"], CATALOG) == "oak_veneer_01"
    assert matlib.choose(matlib.BY_KEY["plaster"], CATALOG) == "nice_plaster"  # no hay preferidos: busca
    assert matlib.choose(matlib.BY_KEY["marble"], CATALOG) is None


def test_ensure_downloads_once_and_records_scale(tmp_path, monkeypatch):
    def fake_json(url):
        if url.endswith("assets?t=textures"):
            return CATALOG
        jpg = lambda name: {"2k": {"jpg": {"url": f"https://x/{name}.jpg"}}}  # noqa: E731
        return {"Diffuse": jpg("d"), "nor_gl": jpg("n"), "Rough": jpg("r"), "Displacement": jpg("h")}
    fetched = []

    def fake_fetch(url, dest):
        fetched.append(url)
        Image.new("RGB", (8, 8), (100, 80, 60)).save(dest, "JPEG")
    monkeypatch.setattr(matlib, "_get_json", fake_json)
    index = matlib.ensure(tmp_path, {"wood", "plaster"}, fetch=fake_fetch)
    assert index["wood"]["id"] == "oak_veneer_01" and index["wood"]["size"] == 1.0
    assert set(index["wood"]) >= {"color", "normal", "rough", "mean_color", "mean_rough"}
    assert len(fetched) == 6
    again = matlib.ensure(tmp_path, {"wood"}, fetch=fake_fetch)
    assert again["wood"]["id"] == "oak_veneer_01" and len(fetched) == 6
    assert json.loads((tmp_path / "library" / "index.json").read_text())["plaster"]["id"] == "nice_plaster"


def test_pad_and_unpad_keep_proportion():
    from aire_backend.workflows import pad_to, qwen_size, unpad
    img = Image.new("RGB", (1920, 1385), (10, 20, 30))
    img.paste((250, 0, 0), (900, 600, 1020, 700))
    size = qwen_size(*img.size)
    padded, box = pad_to(img, size)
    assert padded.size == size
    back = unpad(padded, box, size, img.size)
    # el cuadro rojo vuelve al mismo sitio (±3 px), sin desplazamiento por estirar
    import numpy as np
    ys, xs = np.nonzero(np.asarray(back)[..., 0] > 200)
    assert abs(ys.min() - 600) <= 3 and abs(xs.min() - 900) <= 3
