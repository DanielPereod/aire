"""Fijar los colores de la IA a los de un render de referencia (Cycles).

La IA da el aspecto de foto (detalle, reflejos, textura) pero a veces cambia el tono de
superficies grandes: suelo más amarillo, muebles azules que viran a verde, latón que sale
de acero. Aquí se conserva la luminosidad de la IA y su detalle, y se toma de la referencia
la cromaticidad (proporción entre rojo, verde y azul) a escala media, que es lo que se ve
como «el color». Al ser una proporción no depende del brillo: si la IA aclara una escena de
noche, la luz cálida sigue igual de cálida.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _linear(img: Image.Image) -> np.ndarray:
    c = np.asarray(img, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(lin: np.ndarray) -> Image.Image:
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return Image.fromarray(np.rint(c * 255).astype(np.uint8))


def _blur(x: np.ndarray, radius: float) -> np.ndarray:
    """Desenfoque gaussiano separable por canal (numpy, sin dependencias)."""
    r = max(1, int(3 * radius))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / radius) ** 2).astype(np.float32)
    k /= k.sum()
    out = x
    for axis in (0, 1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            acc += w * np.take(p, np.arange(i, i + n), axis=axis)
        out = acc
    return out


def _chroma(lin: np.ndarray, radius: float, eps: float = 0.01) -> np.ndarray:
    b = _blur(lin, radius)
    return (b + eps) / ((b @ _LUMA)[..., None] + eps)


def lock_colors(img: Image.Image, ref: Image.Image, strength: float = 1.0, radius: float | None = None) -> Image.Image:
    """Devuelve img con la cromaticidad a escala media de ref (strength 0-1)."""
    img = img.convert("RGB")
    if strength <= 0:
        return img
    ref = ref.convert("RGB").resize(img.size, Image.Resampling.LANCZOS)
    radius = radius if radius is not None else max(2.0, img.width / 320)  # 6 px a 1920
    a, r = _linear(img), _linear(ref)
    gain = np.clip(_chroma(r, radius) / _chroma(a, radius), 0.25, 4.0) ** strength
    return _srgb(a * gain)
