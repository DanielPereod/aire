# Instalación automática (técnico)

El usuario solo instala el `.rbz`. Todo lo demás lo hace la extensión al pulsar
**Preparar AIRE**, sin consolas ni ventanas y sin que haga falta tener Python.

## Flujo

```
Ventana AIRE (HtmlDialog)  →  window.rb  →  Runner.bootstrap (WScript.Shell, oculto)
  bin/bootstrap.ps1
    1. descarga uv (un único .exe) → %LOCALAPPDATA%\AIRE\uv\uv.exe
    2. uv venv env --python 3.12   (uv descarga su propio Python a AIRE\python)
    3. uv pip install numpy numba pillow
    4. env\Scripts\python.exe <extensión>\aire\backend\run.py installer …
  aire_backend/installer.py
    5. comprueba la GPU NVIDIA (≥ 8 GB) y el espacio libre (≥ 40 GB)
    6. descarga ComfyUI fijado al commit validado (o master si falla)
    7. uv venv comfy\venv + PyTorch CUDA (cu130 → cu128) + requisitos de ComfyUI
    8. descarga los modelos del preset recomendado (reanudable)
    9. arranca ComfyUI en el puerto 8190 y valida el flujo de Z-Image contra él
```

El progreso se escribe en `AIRE\progress\setup.json`; la ventana lo lee cada
segundo. Todos los pasos son idempotentes: «Continuar» retoma donde se quedó.

Render: `window.rb` exporta la vista a `AIRE\jobs\<fecha>\export` y lanza
`pythonw.exe run.py job …` (oculto). `job.py` arranca ComfyUI si hace falta
(`comfyctl`), genera los pases, renderiza y deja las imágenes y miniaturas en
`jobs\<fecha>\renders`. Al cerrar SketchUp se para ComfyUI para liberar la VRAM.

## Carpeta `%LOCALAPPDATA%\AIRE`

| Ruta | Contenido |
|---|---|
| `uv\`, `python\`, `env\` | uv, Python gestionado y entorno del backend |
| `comfy\ComfyUI\`, `comfy\venv\` | ComfyUI y su entorno con PyTorch |
| `comfy\ComfyUI\models\` | modelos (~20–25 GB) |
| `config.json` | GPU, preset, rutas; `"ready": true` cuando está listo |
| `jobs\<fecha>\` | cada render: exportación, pases, imágenes, `result.json` |
| `logs\` | `install.log`, `comfyui.log`, `render.log` |

## Verificación hecha sin Windows

- Instalador y trabajo de render: tests con red, GPU y ComfyUI simulados
  (pasos, reanudación tras corte, mensajes de error), descargas reanudables con
  un servidor HTTP real, y el backend empaquetado en el `.rbz` ejecutado tal cual
  lo lanza la extensión.
- `comfyctl`: arranque y parada en segundo plano de un ComfyUI real (en CPU),
  incluida la detección de que se cierra al arrancar.
- `bootstrap.ps1`: analizado y **ejecutado** con PowerShell 7 en Linux, con `uv`
  y Python falsos y una ruta de usuario con espacios y ñ; caminos de éxito y de error.
- Ventana: lógica Ruby con un stub de SketchUp (instalar, renderizar, galería,
  tareas interrumpidas) y la interfaz HTML en Chromium (estados, modo oscuro,
  clics que llegan a SketchUp).

**Pendiente de probar en Windows real**: Windows PowerShell 5.1 (el que viene con
Windows), `WScript.Shell` desde SketchUp, la instalación real de PyTorch y ComfyUI,
y el render en la RTX 5060. Algún antivirus puede avisar de un PowerShell oculto
que descarga ficheros; si pasa, hay que permitirlo una vez.
