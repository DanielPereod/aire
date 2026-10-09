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


def scene_items(scene: Scene, ids: np.ndarray, material: np.ndarray, visible: list[dict],
                min_coverage: float = 0.003, limit: int = 14) -> list[str]:
    """'Objeto (material dominante)' para los objetos con nombre que más se ven."""
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
        mask = np.isin(ids, scene.descendants(obj["id"]))
        mats, counts = np.unique(material[mask & (material >= 0)], return_counts=True)
        mat_name = scene.materials[int(mats[counts.argmax()])]["name"] if len(mats) else None
        items.append(f"{name} ({mat_name})" if _meaningful(mat_name) else name)
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
EDIT_DETAIL = ("Preserve every small detail exactly as in the image: open or woven patterns keep their gaps "
               "and shape, thin legs and frames stay thin, and handles, taps, sinks, jars, bottles and small "
               "decorative objects keep their shape, color, metal finish and number. Each surface keeps its "
               "own finish (for example a smooth floor stays smooth).")
EDIT_LOOK = ("Make it look like a photograph by a professional interior photographer with a full-frame camera: "
             "physically correct light with soft shadows, contact shadows and ambient occlusion in corners, "
             "light falling off naturally across the room, true-to-life reflections on glossy and metal "
             "surfaces, realistic material texture at full resolution, natural colors, subtle real-world "
             "imperfections, no CGI look, no outlines, sharp focus.")
EDIT_CREATIVE = ("You may add a few small decorative props (plants, books, ceramics) that fit the scene, but keep "
                 "all existing furniture and finishes unchanged.")


def build_edit_prompt(scene: Scene, ids: np.ndarray, material: np.ndarray, visible: list[dict],
                      user: str = "", style: str = "", light: str | None = None, creative: bool = False) -> str:
    parts = [EDIT_KEEP, EDIT_DETAIL, EDIT_LOOK]
    items = scene_items(scene, ids, material, visible)
    if items:
        parts.append("Materials in the scene: " + ", ".join(items) + ".")
    if STYLES.get(style):
        parts.append(f"Keep the architecture and furniture, but give the decoration a feel of: {STYLES[style]}.")
    parts.append("Lighting: " + (LIGHTS.get(light or "") or light_hint(scene)) + ".")
    if creative:
        parts.append(EDIT_CREATIVE)
    if user.strip():
        parts.append("Also: " + user.strip())
    return " ".join(parts)
