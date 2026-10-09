"""Empaqueta la extensión de SketchUp en dist/aire-<versión>.rbz.

    python tools/build_rbz.py

El .rbz lleva todo lo necesario: la extensión Ruby, la ventana, el script de
arranque y el backend en Python (que se ejecuta con el Python que instala AIRE).
Instalar en SketchUp: Extensiones > Extension Manager > Install Extension.
"""

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sketchup"
BACKEND = ROOT / "backend"


def files_in(folder: Path):
    for f in sorted(folder.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".part"):
            yield f


def main() -> Path:
    version = re.search(r"VERSION = '([^']+)'", (SRC / "aire.rb").read_text()).group(1)
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"aire-{version}.rbz"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(SRC / "aire.rb", "aire.rb")
        for f in files_in(SRC / "aire"):
            z.write(f, f.relative_to(SRC).as_posix())
        z.write(BACKEND / "run.py", "aire/backend/run.py")
        for f in files_in(BACKEND / "aire_backend"):
            z.write(f, ("aire/backend" / f.relative_to(BACKEND)).as_posix())
    print(out)
    return out


if __name__ == "__main__":
    main()
