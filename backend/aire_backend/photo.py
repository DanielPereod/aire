"""Acabado fotográfico final: lo que hace una cámara real y un render limpio no tiene.

Halo suave alrededor de las luces fuertes, viñeteado leve del objetivo y grano fino del
sensor. Determinista (grano con semilla fija) y sutil: no cambia colores ni formas.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .colormatch import _blur, _linear, _srgb


def photo_finish(img: Image.Image, strength: float = 1.0, seed: int = 7) -> Image.Image:
    img = img.convert("RGB")
    if strength <= 0:
        return img
    lin = _linear(img)
    h, w = lin.shape[:2]
    # Halo (bloom): solo lo que ya está casi quemado (bombillas, sol en el suelo, ventanas)
    small = max(1, w // 480)
    hot = np.maximum(lin - 0.75, 0.0)[::small, ::small]
    glow = _blur(hot, max(2.0, w / 160 / small))
    glow = np.asarray(Image.fromarray(np.clip(glow * 255 * 4, 0, 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BILINEAR), np.float32) / (255 * 4)
    lin = lin + 0.35 * strength * glow
    # Viñeteado: esquinas ~10 % más oscuras
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x - w / 2) / (w / 2)) ** 2 * 0.5 + ((y - h / 2) / (h / 2)) ** 2 * 0.5
    lin = lin * (1.0 - 0.10 * strength * r2)[..., None]
    out = np.asarray(_srgb(lin), np.float32)
    # Aberración cromática muy leve: rojo y azul algo desplazados hacia los bordes, como un objetivo real
    ca = 0.0007 * strength
    if ca > 0 and min(w, h) > 256:
        for ch, k in ((0, 1.0 + ca), (2, 1.0 - ca)):
            layer = Image.fromarray(np.clip(out[..., ch], 0, 255).astype(np.uint8))
            nw, nh = round(w * k), round(h * k)
            layer = layer.resize((nw, nh), Image.Resampling.BILINEAR)
            if k > 1:
                l, t = (nw - w) // 2, (nh - h) // 2
                layer = layer.crop((l, t, l + w, t + h))
            else:
                canvas = Image.fromarray(np.clip(out[..., ch], 0, 255).astype(np.uint8))
                canvas.paste(layer, ((w - nw) // 2, (h - nh) // 2))
                layer = canvas
            out[..., ch] = np.asarray(layer, np.float32)
    # Grano fino de luminancia, algo más visible en sombras
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    amp = (2.2 + 2.0 * (1.0 - out.mean(axis=2) / 255.0)) * strength
    out = out + (grain * amp)[..., None]
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))
