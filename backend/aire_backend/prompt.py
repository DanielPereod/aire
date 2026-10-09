"""Prompt automático a partir de la escena (objetos visibles y sus materiales)."""

from __future__ import annotations

import re

import numpy as np

from .scene import Scene

QUALITY = ("photorealistic interior design photograph, architectural photography, "
           "realistic materials and textures, soft global illumination, natural shadows, "
           "high detail, sharp focus, 24mm lens")

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


def build_prompt(scene: Scene, ids: np.ndarray, material: np.ndarray, visible: list[dict],
                 user: str = "", auto: bool = True) -> str:
    parts = [user.strip()] if user.strip() else []
    if auto:
        items = scene_items(scene, ids, material, visible)
        if items:
            parts.append("The room contains: " + ", ".join(items))
        parts.append(light_hint(scene))
    parts.append(QUALITY)
    return ". ".join(p for p in parts if p)
