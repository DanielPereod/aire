"""Fijar los colores de la IA a los de un render de referencia (Cycles).

La IA da el aspecto de foto (detalle, reflejos, textura) pero a veces cambia el tono de
superficies grandes: suelo más amarillo, muebles azules que viran a verde, latón que sale
de acero. Aquí se conserva la luminosidad de la IA y su detalle de color fino, y se toma de
la referencia el color a escala media (desenfocado), que es lo que se ve como «el color».
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageCms, ImageFilter

_SRGB = ImageCms.createProfile("sRGB")
_LAB = ImageCms.createProfile("LAB")
_TO_LAB = ImageCms.buildTransformFromOpenProfiles(_SRGB, _LAB, "RGB", "LAB")
_TO_RGB = ImageCms.buildTransformFromOpenProfiles(_LAB, _SRGB, "LAB", "RGB")


def lock_colors(img: Image.Image, ref: Image.Image, strength: float = 1.0, radius: float | None = None) -> Image.Image:
    """Devuelve img con el color (a, b de Lab) a escala media de ref. strength 0-1."""
    img = img.convert("RGB")
    if strength <= 0:
        return img
    ref = ref.convert("RGB").resize(img.size, Image.Resampling.LANCZOS)
    radius = radius if radius is not None else max(2.0, img.width / 320)  # 6 px a 1920
    L, a, b = ImageCms.applyTransform(img, _TO_LAB).split()
    _, ra, rb = ImageCms.applyTransform(ref, _TO_LAB).split()
    out = [L]
    for c, r in ((a, ra), (b, rb)):
        blur = ImageFilter.GaussianBlur(radius)
        cn = np.asarray(c, np.float32)
        delta = np.asarray(r.filter(blur), np.float32) - np.asarray(c.filter(blur), np.float32)
        out.append(Image.fromarray(np.clip(np.rint(cn + strength * delta), 0, 255).astype(np.uint8)))
    return ImageCms.applyTransform(Image.merge("LAB", out), _TO_RGB)
