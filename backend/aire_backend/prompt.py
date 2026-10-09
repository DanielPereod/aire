"""Prompt automático a partir de la escena (objetos visibles y sus materiales)."""

from __future__ import annotations

import re

import numpy as np

from .scene import Scene

QUALITY = ("real photograph of an interior taken with a full-frame camera, professional interior design "
           "photography for a magazine, natural colors, realistic materials with fine texture and subtle "
           "reflections, soft natural light with gentle shadows and global illumination, clean surfaces "
           "without outlines, high detail, sharp focus, 24mm lens")

_GENERIC = re.compile(r"^(material|color|colour|<.*>|\[.*\]|default|untitled|grupo|group|componente|component)"
                      r"[\s_#-]*\d*$", re.IGNORECASE)


def _meaningful(name: str | None) -> bool:
    return bool(name) and not _GENERIC.match(name.strip()) and not name.strip().isdigit()


def color_name(rgb) -> str | None:
    """Nombre aproximado en inglés del color de un material, para que la IA no lo cambie
    (pruebas 0.7: un suelo beige salía gris)."""
    if not rgb or len(rgb) < 3:
        return None
    import colorsys
    r, g, b = (float(c) / 255.0 for c in rgb[:3])
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    tone = "dark " if l < 0.3 else "light " if l > 0.72 else ""
    if s < 0.12 or (l > 0.9 and s < 0.3):
        if l > 0.9:
            return "white"
        if l < 0.12:
            return "black"
        return f"{tone}grey"
    deg = h * 360
    if (deg < 50 or deg >= 330) and l > 0.55 and s < 0.5:
        return "beige" if deg >= 20 else "light pink"
    if 15 <= deg < 45 and l <= 0.55:
        return f"{tone}brown"
    names = [(15, "red"), (45, "orange"), (65, "yellow"), (160, "green"), (200, "teal"),
             (255, "blue"), (290, "purple"), (330, "pink"), (360, "red")]
    hue = next(n for limit, n in names if deg < limit)
    muted = "muted " if s < 0.35 else ""
    return f"{tone}{muted}{hue}"


_FINISHES = [
    (re.compile(r"lat[oó]n|brass|bronce|bronze", re.I), "brass"),
    (re.compile(r"\boro\b|dorad|gold", re.I), "gold"),
    (re.compile(r"cobre|copper", re.I), "copper"),
    (re.compile(r"inox|stainless|acero|steel|chrom|cromad", re.I), "stainless steel"),
    (re.compile(r"m[aá]rmol|marble", re.I), "marble"),
    (re.compile(r"nogal|walnut|roble|oak|madera|wood", re.I), "wood"),
    (re.compile(r"tela|fabric|tejido|textil|lino|linen|algod", re.I), "fabric"),
    (re.compile(r"rat[aá]n|rattan|mimbre|wicker|caña|cane", re.I), "rattan"),
    (re.compile(r"cuero|leather|piel", re.I), "leather"),
    (re.compile(r"vidrio|cristal|glass", re.I), "glass"),
]


def finish(name: str | None) -> str | None:
    """Tipo de material deducido del nombre (latón, tela…), para que la IA no lo cambie."""
    for pat, word in _FINISHES:
        if name and pat.search(name):
            return word
    return None


def _describe(scene: Scene, m: int, albedo: np.ndarray | None, sel: np.ndarray) -> str:
    info = scene.materials[m]
    if albedo is not None and sel.any():
        lin = ((albedo[sel].astype(np.float32) / 255.0) ** 2.2).mean(axis=0)
        color = color_name(tuple((lin ** (1 / 2.2) * 255).round()))
    else:
        color = color_name(info.get("color"))
    kind = finish(info.get("name"))
    return " ".join(w for w in (color, kind) if w)


def scene_items(scene: Scene, ids: np.ndarray, material: np.ndarray, visible: list[dict],
                min_coverage: float = 0.003, limit: int = 14, albedo: np.ndarray | None = None) -> list[str]:
    """'Objeto (materiales)' para los objetos con nombre que más se ven: el dominante y los
    que cubren ≥15 % del objeto (pruebas 0.8: el remate de latón de las lámparas salía negro)."""
    items, seen = [], set()
    for obj in sorted(visible, key=lambda o: -o["coverage"]):
        if obj["coverage"] < min_coverage or obj["type"] == "model":
            continue
        if obj["coverage"] > 0.4 and scene.children(obj["id"]):
            continue  # contenedor (la habitación entera): mejor describir sus partes
        name = obj["name"] if _meaningful(obj["name"]) else obj.get("definition")
        if not _meaningful(name) or name in seen:
            continue
        # Solo el nivel más alto con nombre: no repetir las patas de la silla
        if any(scene.objects[a].get("name") in seen for a in scene.ancestors(obj["id"])):
            continue
        mask = np.isin(ids, scene.descendants(obj["id"])) & (material >= 0)
        mats, counts = np.unique(material[mask], return_counts=True)
        desc = []
        for m, c in sorted(zip(mats, counts), key=lambda t: -t[1])[:3]:
            if c < 0.15 * counts.sum():
                break
            d = _describe(scene, int(m), albedo, mask & (material == m))
            if d and d not in desc:
                desc.append(d)
        items.append(f"{name} ({', '.join(desc)})" if desc else name)
        seen.add(name)
        if len(items) >= limit:
            break
    return items


def light_hint(scene: Scene) -> str:
    sun = scene.meta.get("sun") or {}
    t = sun.get("time")
    if t and len(t) >= 13:
        hour = int(t[11:13])  # UTC: aproximación, el usuario puede indicarlo en el prompt
        if hour < 8 or hour >= 19:
            return "warm evening light, interior lamps on"
    return "natural daylight through the windows"


STYLES = {
    "modelo": "",
    "nordico": "Scandinavian interior style, light woods, white and beige tones, cozy textiles",
    "japandi": "Japandi interior style, natural wood, linen, muted earthy tones, minimal decor",
    "mediterraneo": "Mediterranean interior style, warm whites, terracotta, natural fibers, rattan",
    "industrial": "industrial loft interior style, black metal, exposed brick, warm leather",
    "clasico": "classic elegant interior style, mouldings, refined fabrics, warm tones",
    "lujo": "luxury contemporary interior, marble, brass details, sophisticated lighting",
    "minimalista": "minimalist interior, clean surfaces, neutral palette, uncluttered",
}

LIGHTS = {
    "dia": "bright natural daylight through the windows",
    "tarde": "warm golden hour sunlight, long soft shadows",
    "noche": "night time, warm interior lamps and ceiling lights on, cozy atmosphere",
}


def build_prompt(scene: Scene, ids: np.ndarray, material: np.ndarray, visible: list[dict],
                 user: str = "", auto: bool = True, style: str = "", light: str | None = None) -> str:
    parts = [user.strip()] if user.strip() else []
    if STYLES.get(style):
        parts.append(STYLES[style])
    if auto:
        items = scene_items(scene, ids, material, visible)
        if items:
            parts.append("The room contains: " + ", ".join(items))
        parts.append(LIGHTS.get(light or "", "") or light_hint(scene))
    parts.append(QUALITY)
    return ". ".join(p for p in parts if p)


# --- Instrucción para modelos de edición (FLUX.2 klein): fotorrealizar sin cambiar nada
EDIT_KEEP = ("Turn this 3D render into a real photograph of the same interior. Keep exactly the same camera "
             "angle, room layout, furniture, objects, shapes, materials, colors and textures: do not add, "
             "remove, move or replace anything.")
# Lo pequeño es lo primero que se pierde (pruebas 0.6: sillas caladas convertidas en punto
# tejido, grifo y fregadero dorados en blanco, botes desaparecidos): se pide expresamente.
EDIT_DETAIL = ("Preserve every small detail exactly as in the image: patterns with gaps keep their gaps, thin "
               "legs and frames stay thin, and handles, taps, sinks, jars, bottles, appliances and small "
               "objects keep their shape, color, finish and number. Every surface keeps its own material and "
               "its exact color and tone: fabric stays the same fabric, wood stays wood and stone stays stone.")
EDIT_LOOK = ("Make it look like a photograph by a professional interior photographer with a full-frame camera: "
             "physically correct light with soft shadows, contact shadows and ambient occlusion in corners, "
             "light falling off naturally across the room, true-to-life reflections on glossy and metal "
             "surfaces, realistic material texture at full resolution, natural colors, subtle real-world "
             "imperfections, no CGI look, no outlines, sharp focus.")
EDIT_CREATIVE = ("You may add a few small decorative props (plants, books, ceramics) that fit the scene, but keep "
                 "all existing furniture and finishes unchanged.")


def _where(ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> str:
    cy, cx = ys.mean() / h, xs.mean() / w
    v = "top" if cy < 0.33 else "bottom" if cy > 0.66 else ""
    hz = "left" if cx < 0.33 else "right" if cx > 0.66 else ""
    return " ".join(p for p in (v, hz) if p) or "center"


def main_surfaces(scene: Scene, ids: np.ndarray, material: np.ndarray, albedo: np.ndarray,
                  min_coverage: float = 0.02, limit: int = 8) -> list[str]:
    """Colores de las superficies grandes, medidos en la imagen base («beige surface at the
    bottom (Suelo)»). Pruebas 0.7: sin esto la IA volvía gris el suelo beige y los muebles azules."""
    h, w = material.shape
    mats, counts = np.unique(material[material >= 0], return_counts=True)
    out = []
    for m, c in sorted(zip(mats, counts), key=lambda t: -t[1]):
        if c / material.size < min_coverage or len(out) >= limit:
            break
        sel = material == m
        lin = ((albedo[sel].astype(np.float32) / 255.0) ** 2.2).mean(axis=0)
        color = color_name(tuple((lin ** (1 / 2.2) * 255).round()))
        if not color:
            continue
        obj_ids, oc = np.unique(ids[sel], return_counts=True)
        obj = scene.objects[int(obj_ids[oc.argmax()])] if len(obj_ids) and obj_ids[oc.argmax()] >= 0 else {}
        name = next((n for n in (obj.get("name"), obj.get("definition"), scene.materials[int(m)].get("name"))
                     if _meaningful(n)), None)
        ys, xs = np.nonzero(sel)
        text = f"{color} at the {_where(ys, xs, h, w)}"
        out.append(f"{text} ({name})" if name else text)
    return out


def build_edit_prompt(scene: Scene, ids: np.ndarray, material: np.ndarray, visible: list[dict],
                      user: str = "", style: str = "", light: str | None = None, creative: bool = False,
                      albedo: np.ndarray | None = None) -> str:
    parts = [EDIT_KEEP, EDIT_DETAIL, EDIT_LOOK]
    if albedo is not None:
        surfaces = main_surfaces(scene, ids, material, albedo)
        if surfaces:
            parts.append("Large surfaces and their exact colors: " + ", ".join(surfaces) + ".")
    items = scene_items(scene, ids, material, visible, albedo=albedo)
    if items:
        parts.append("Objects in the scene: " + ", ".join(items) + ".")
    if STYLES.get(style):
        parts.append(f"Keep the architecture and furniture, but give the decoration a feel of: {STYLES[style]}.")
    parts.append("Lighting: " + (LIGHTS.get(light or "") or light_hint(scene)) + ".")
    if creative:
        parts.append(EDIT_CREATIVE)
    if user.strip():
        parts.append("Also: " + user.strip())
    return " ".join(parts)
