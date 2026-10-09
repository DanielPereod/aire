# AIRE: render IA para SketchUp

Plugin de SketchUp para renders fotorrealistas de interiorismo con modelos de IA,
guiados por el **modelo 3D** (profundidad, normales, líneas, ids de objeto,
materiales) en lugar de por una captura de pantalla. Incluye edición tipo chat
("pon la silla roja") con máscaras exactas por objeto.

Estado: **P1** (exportador y pases de control) y **P2** (render con Z-Image +
ControlNet vía ComfyUI, con métrica de fidelidad). Ver
[docs/ARQUITECTURA.md](docs/ARQUITECTURA.md), [docs/P1.md](docs/P1.md) y [docs/P2.md](docs/P2.md).

```
sketchup/        extensión Ruby (exportador aire-scene)
backend/         backend Python (pases de control, máscaras; luego difusión y agente)
tools/           empaquetado .rbz
docs/            arquitectura, decisiones y guías de cada prototipo
```

Prueba rápida sin SketchUp:

```bash
pip install -e backend[dev]
python -m aire_backend.synthetic /tmp/salon      # escena de interior sintética
python -m aire_backend.passes /tmp/salon
python -m aire_backend.mask /tmp/salon silla
pytest backend
```
