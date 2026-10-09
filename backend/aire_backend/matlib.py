"""Biblioteca de materiales reales para el render con luz real (Cycles).

AIRE reconoce el tipo de cada material por su nombre (y el de la textura y los objetos que lo
usan) y le añade el relieve y el brillo irregular de un material escaneado de verdad. El color
sigue siendo el del modelo: en materiales lisos, la variación del escaneo (veta, poros) se
aplica sobre ese color; en los que ya tienen textura solo se añade el brillo irregular.

Los materiales vienen de Poly Haven (CC0, uso libre). Se descargan la primera vez que una
escena los necesita, a 2K, en <home>/library/<id>/ (unos 10 MB cada uno), y el elegido para
cada tipo se guarda en library/index.json para que el resultado no cambie entre renders.

    python -m aire_backend.matlib --home %LOCALAPPDATA%/AIRE --check     # qué elegiría
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://api.polyhaven.com"
RES = "2k"


@dataclass(frozen=True)
class Kind:
    key: str
    pattern: str  # en el nombre del material, de su textura o de sus objetos
    words: tuple[str, ...]  # para buscarlo en el catálogo si ninguno de los preferidos existe
    prefer: tuple[str, ...] = ()
    size: float = 2.0  # metros que cubre una repetición si el catálogo no lo dice
    tint: float = 0.0  # cuánto de la variación de color del escaneo se aplica sobre un color liso
    normal: float = 0.6  # fuerza del relieve
    avoid: tuple[str, ...] = ()


# El orden importa (lo más específico primero), igual que los presets de cycles_render.
KINDS = [
    Kind("wood_floor", r"parquet|tarima|wood.?floor|floor.?wood|suelo.{0,12}(madera|roble|oak)|laminad|laminate",
         ("wood", "floor"), ("wood_floor", "laminate_floor_02", "wood_floor_worn"), 2.0, 0.8, 0.5),
    # Solo para suelos lisos detectados por su geometría (no por el nombre)
    Kind("floor_stone", r"(?!x)x", ("concrete", "floor"), ("polished_concrete_01", "concrete_floor_02", "grey_tiles"),
         3.0, 0.6, 0.25, ("rough", "dirty", "damaged", "cracked")),
    Kind("marble", r"m[aá]rmol|marble|calacatta|carrara|travertin", ("marble",), ("marble_01",), 2.0, 0.5, 0.15),
    Kind("rattan", r"rat[aá]n|rattan|mimbre|wicker|rejilla|cane\b", ("wicker",), ("wicker_weave", "rattan_weave"),
         0.3, 0.3, 0.9),
    Kind("rug", r"alfombra|rug\b|carpet|moqueta|yute|jute|sisal", ("carpet",), ("carpet", "jute_rug"), 1.0, 0.4, 0.8),
    Kind("leather", r"cuero|leather|\bpiel\b", ("leather",), ("leather_white", "brown_leather"), 0.6, 0.4, 0.5),
    Kind("fabric", r"tela|fabric|tejido|textil|lino|linen|algod|cotton|terciopelo|velvet|boucl|tapiz|upholst|sof[aá]\b",
         ("fabric",), ("fabric_pattern_07", "fabric_pattern_05", "linen"), 0.5, 0.3, 0.7),
    Kind("wood", r"nogal|walnut|roble|oak|madera|wood|fresno|\bash\b|haya|beech|teca|teak|pino|pine|chapa|veneer|cerezo",
         ("wood",), ("oak_veneer_01", "fine_grained_wood", "wood_table_001"), 2.0, 0.7, 0.4,
         ("floor", "bark", "log", "plank", "crate", "pallet", "rotten")),
    Kind("concrete", r"microcement|cemento|concrete|hormig|resin|resina|estuco|stucco",
         ("concrete",), ("concrete_floor_02", "concrete_wall_008"), 3.0, 0.5, 0.3),
    Kind("metal", r"inox|stainless|acero|steel|chrom|cromad|alumin|metal|lat[oó]n|brass|bronce|cobre|copper|\boro\b",
         ("metal", "brushed"), ("brushed_metal", "metal_plate"), 0.5, 0.0, 0.15),
    Kind("plaster", r"pintura|paint|pared|wall|yeso|plaster|techo|ceiling|gotel",
         ("plaster",), ("painted_plaster_wall", "white_plaster_02", "plastered_wall"), 2.0, 0.15, 0.35),
]
BY_KEY = {k.key: k for k in KINDS}
# Cualquier superficie lisa sin tipo reconocido (paredes sin nombre, «Material1»…) se trata
# como pintura: un relieve muy suave que quita el aspecto de plástico perfecto del 3D.
FALLBACK = "plaster"
SKIP = re.compile(r"planta|plant|hoja|leaf|foliage|vidrio|cristal|glass|ventana|window|espejo|mirror|cortina|curtain|visillo|azulejo|tile|"
                  r"cer[aá]mic|ceramic|porcel", re.I)


def kind_for(*names: str | None) -> Kind | None:
    """Tipo de material según el primer nombre que encaje (material antes que objeto)."""
    for name in names:
        if not name:
            continue
        if SKIP.search(name):
            return None
        for k in KINDS:
            if re.search(k.pattern, name, re.I):
                return k
    return None


def kinds_in_scene(scene) -> set[str]:
    """Tipos que necesita una escena (para descargar solo esos)."""
    names_by_obj = {o["id"]: " ".join(filter(None, (o.get("name"), o.get("definition")))) for o in scene.objects}
    used: set[str] = {FALLBACK, "floor_stone", "wood_floor"}  # el suelo se reconoce después, por geometría
    for mid, info in enumerate(scene.materials):
        tex = (info.get("texture") or {}).get("file")
        sel = scene.tri_material_front == mid
        objs = " ".join(names_by_obj.get(int(o), "") for o in set(scene.tri_object[sel][:2000].tolist()))
        k = kind_for(info.get("name"), info.get("internal_name"), tex, objs)
        if k:
            used.add(k.key)
    return used


# --------------------------------------------------------------------------- catálogo
def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "AIRE"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def choose(kind: Kind, catalog: dict) -> str | None:
    """Primer preferido que exista; si no, el más descargado cuyo nombre o etiquetas encajen."""
    for pid in kind.prefer:
        if pid in catalog:
            return pid
    best, best_n = None, -1
    for pid, a in catalog.items():
        text = " ".join([pid, a.get("name", ""), *a.get("tags", []), *a.get("categories", [])]).lower()
        if all(w in text for w in kind.words) and not any(w in text for w in kind.avoid):
            n = a.get("download_count", 0)
            if n > best_n:
                best, best_n = pid, n
    return best


def _pick(files: dict, *keys: str) -> str | None:
    lower = {k.lower(): v for k, v in files.items()}
    for key in keys:
        entry = lower.get(key.lower())
        if entry:
            res = entry.get(RES) or entry.get("1k") or next(iter(entry.values()))
            fmt = res.get("jpg") or res.get("png")
            if fmt and fmt.get("url"):
                return fmt["url"]
    return None


def _mean(path: Path, mode: str):
    from PIL import Image
    im = Image.open(path).convert(mode)
    im.thumbnail((64, 64))
    px = list(im.getdata())
    if mode == "L":
        return sum(px) / len(px) / 255.0
    return [sum(p[i] for p in px) / len(px) / 255.0 for i in range(3)]


def fetch_kind(lib: Path, kind: Kind, catalog: dict, fetch) -> dict | None:
    pid = choose(kind, catalog)
    if not pid:
        return None
    folder = lib / pid
    meta_path = folder / "info.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    files = _get_json(f"{API}/files/{pid}")
    urls = {"color": _pick(files, "Diffuse", "diff", "Color"), "normal": _pick(files, "nor_gl", "Normal"),
            "rough": _pick(files, "Rough", "roughness")}
    if not urls["normal"] and not urls["rough"]:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"id": pid, "kind": kind.key, "source": f"https://polyhaven.com/a/{pid}", "license": "CC0"}
    for role, url in urls.items():
        if url:
            dest = folder / f"{role}.jpg"
            fetch(url, dest)
            meta[role] = dest.name
    dims = catalog.get(pid, {}).get("dimensions")  # milímetros
    meta["size"] = round(max(dims) / 1000.0, 3) if dims else kind.size
    if "color" in meta:
        meta["mean_color"] = _mean(folder / meta["color"], "RGB")
    if "rough" in meta:
        meta["mean_rough"] = _mean(folder / meta["rough"], "L")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def ensure(home: Path, kinds: set[str], on_status=None, fetch=None) -> dict:
    """Descarga lo que falte de la biblioteca para estos tipos. Devuelve el índice
    {tipo: info}. Sin conexión se queda con lo que ya hubiera (el render sale igual, sin ese detalle)."""
    from .models import fetch as http_fetch
    lib = home / "library"
    lib.mkdir(parents=True, exist_ok=True)
    index_path = lib / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    todo = [k for k in kinds if k in BY_KEY and k not in index]
    if todo:
        catalog = _get_json(f"{API}/assets?t=textures")
        for i, key in enumerate(sorted(todo)):
            if on_status:
                on_status(i, len(todo), key)
            meta = fetch_kind(lib, BY_KEY[key], catalog, fetch or http_fetch)
            if meta:
                index[key] = meta
                index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    if "_hdri" not in index:
        try:
            hdri = fetch_hdri(lib, fetch or http_fetch)
            if hdri:
                index["_hdri"] = hdri
                index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        except OSError:
            pass
    return {k: v for k, v in index.items() if k in kinds or k == "_hdri"}


HDRI_PREFER = ("garden_nook", "kloofendal_48d_partly_cloudy_puresky", "sunflowers")
HDRI_WORDS = (("garden",), ("park",), ("trees",))


def choose_hdri(catalog: dict) -> str | None:
    """Exterior para las ventanas: un jardín o parque de día (lo más descargado)."""
    for pid in HDRI_PREFER[:1]:
        if pid in catalog:
            return pid
    for words in HDRI_WORDS:
        best, best_n = None, -1
        for pid, a in catalog.items():
            text = " ".join([pid, a.get("name", ""), *a.get("tags", []), *a.get("categories", [])]).lower()
            if all(w in text for w in words) and "night" not in text and "indoor" not in text:
                if a.get("download_count", 0) > best_n:
                    best, best_n = pid, a.get("download_count", 0)
        if best:
            return best
    return next((p for p in HDRI_PREFER if p in catalog), None)


def fetch_hdri(lib: Path, fetch) -> dict | None:
    pid = choose_hdri(_get_json(f"{API}/assets?t=hdris"))
    if not pid:
        return None
    files = _get_json(f"{API}/files/{pid}")
    entry = files.get("hdri", {})
    url = ((entry.get(RES) or entry.get("1k") or {}).get("hdr") or {}).get("url")
    if not url:
        return None
    dest = lib / f"hdri_{pid}.hdr"
    if not dest.exists():
        fetch(url, dest)
    return {"id": pid, "file": dest.name, "source": f"https://polyhaven.com/a/{pid}", "license": "CC0"}


def load_index(lib: Path | None) -> dict:
    if not lib:
        return {}
    p = Path(lib) / "index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Biblioteca de materiales de AIRE")
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="solo decir qué elegiría para cada tipo")
    ap.add_argument("--all", action="store_true", help="descargar todos los tipos")
    args = ap.parse_args(argv)
    if args.check:
        catalog = _get_json(f"{API}/assets?t=textures")
        for k in KINDS:
            pid = choose(k, catalog)
            a = catalog.get(pid, {})
            print(f"{k.key:11s} → {pid}  ({a.get('name')}, {a.get('dimensions')}, preferido: {pid in k.prefer})")
        print(f"{'hdri':11s} → {choose_hdri(_get_json(f'{API}/assets?t=hdris'))}")
        return
    index = ensure(args.home, set(BY_KEY) if args.all else {FALLBACK},
                   on_status=lambda i, n, k: print(f"[{i + 1}/{n}] {k}", flush=True))
    print(json.dumps({k: v["id"] for k, v in index.items()}, indent=2))


if __name__ == "__main__":
    main()
