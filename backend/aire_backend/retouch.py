"""Retoque fotográfico con FLUX.2 klein sobre un render ya hecho (prototipo B: Cycles + IA).

    python -m aire_backend.retouch render.png --out retocado.png --denoise 0.2 [--server URL]
"""

from __future__ import annotations

import argparse
import random
from io import BytesIO
from pathlib import Path

from PIL import Image

from .comfy import ComfyClient
from .workflows import KleinSettings, flux2_klein_retouch

PROMPT = ("Make this interior render look like a real photograph taken by a professional interior "
          "photographer. Keep everything exactly the same: same camera, objects, materials, colors, light "
          "direction and shadows. Only add photographic realism: natural material micro-detail, subtle "
          "imperfections, realistic lens rendering. Do not add, remove or change anything.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Retoque fotográfico suave con FLUX.2 klein")
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--denoise", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--server", default="http://127.0.0.1:8190")
    args = ap.parse_args(argv)
    src = Image.open(args.image).convert("RGB")
    w, h = (max(16, (d // 16) * 16) for d in src.size)
    tmp = Path(args.out).with_suffix(".in.png")
    src.crop((0, 0, w, h)).save(tmp)
    client = ComfyClient(args.server)
    name = client.upload_image(tmp)
    s = KleinSettings(width=w, height=h, prompt=PROMPT,
                      seed=args.seed if args.seed is not None else random.randint(0, 2**31 - 1))
    images = client.run(flux2_klein_retouch(s, name, args.denoise), check=True)
    Image.open(BytesIO(images[0])).save(args.out)
    tmp.unlink(missing_ok=True)
    print(args.out)


if __name__ == "__main__":
    main()
