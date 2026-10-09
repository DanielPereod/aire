"""Empaqueta la extensión de SketchUp en dist/aire-<versión>.rbz.

    python tools/build_rbz.py

Instalar en SketchUp: Extensiones > Extension Manager > Install Extension.
"""

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sketchup"


def main() -> None:
    version = re.search(r"VERSION = '([^']+)'", (SRC / "aire.rb").read_text()).group(1)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"aire-{version}.rbz"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(SRC / "aire.rb", "aire.rb")
        for f in sorted((SRC / "aire").rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(SRC).as_posix())
    print(out)


if __name__ == "__main__":
    main()
