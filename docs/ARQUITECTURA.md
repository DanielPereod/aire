# AIRE: arquitectura y decisiones

Plugin de SketchUp que genera renders fotorrealistas de **interiorismo** con modelos
de IA de generación y edición de imagen, **guiados por el modelo 3D** y no por una
captura de pantalla.

## Decisiones tomadas

| Tema | Decisión |
|---|---|
| Uso | **No comercial** → se pueden usar FLUX.2 [klein] 9B y FLUX.2 [dev] |
| Ejecución | **Local y nube** (la nube, para PCs sin GPU suficiente o para máxima calidad) |
| Plataforma | **Windows** |
| Tipo de escena | **Interiorismo** |
| Ediciones del chat | **Solo sobre el render** (no tocan el modelo de SketchUp) |
| Pruebas | El usuario tiene escenas y una **RTX 5060 (8 GB)** |
| Motor local | **ComfyUI** en modo servidor, controlado por el backend |

## Idea central: G-buffer exacto desde el modelo

Cualquier modelo de difusión acaba recibiendo imágenes 2D. La diferencia está en
cuáles: en lugar de una captura con estilo de SketchUp, el plugin exporta la
geometría y el backend **rasteriza pases exactos** con la cámara de SketchUp:

| Pase | Uso en la IA |
|---|---|
| Profundidad métrica | Control de forma y perspectiva (depth) |
| Normales (suavizadas por ángulo de pliegue) | Control fino de superficies |
| Líneas (aristas de SketchUp + contornos) | Control lineart/canny |
| IDs de objeto (jerárquicos) | Máscaras exactas por objeto para editar ("la silla") |
| Material + albedo (texturas) | Prompts por región y coherencia de materiales |
| Metadatos (nombres, etiquetas, materiales, sol, bbox 3D) | Prompt automático y contexto del agente |

Como los pases salen del modelo, la estructura está fijada: la IA "viste" la
escena, no la inventa.

## Componentes

```
SketchUp (extensión Ruby)
 ├─ exporter.rb      recorre el modelo → aire-scene (triángulos, ids, materiales, cámara, sol, secciones)
 ├─ scene_writer.rb  formato binario (Ruby puro, testeable fuera de SketchUp)
 └─ UI (P4)          HtmlDialog: chat, referencias, galería, selector de modelo
        │ HTTP/WebSocket local (P2+)
Backend local (Python, instalado aparte)
 ├─ passes.py        rasterizador CPU (numba) → pases de control      [P1, hecho]
 ├─ mask.py          máscaras por objeto desde el pase de ids        [P1, hecho]
 ├─ render.py        Z-Image + ControlNet vía ComfyUI, barrido y fidelidad [P2, hecho]
 ├─ agente           LLM con herramientas (buscar objeto, máscara, inpaint, referencias) [P3-P4]
 └─ hardware.py      detección GPU/VRAM/RAM → recomendación de modelo [P2, hecho]
```

**Por qué el rasterizador en CPU**: es determinista, no necesita contexto OpenGL,
no compite por la VRAM con el modelo de difusión y es igual en todas las máquinas.
Rinde ~1 s por millón de triángulos a 1536 px.

## Modelos por nivel de hardware (para validar en P2)

Las cifras de VRAM que publica la comunidad no coinciden entre fuentes. Se medirán
en P2 con las escenas del usuario.

| Nivel | Hardware | Principal | Alternativa |
|---|---|---|---|
| 0 | Sin GPU o < 6 GB | Nube | — |
| 1 | 6–8 GB | SDXL realista + ControlNet Union (depth/lineart) | FLUX.2 klein 4B GGUF Q4 |
| 2 | 12–16 GB | FLUX.2 klein 4B (~13 GB) | Qwen-Image-Edit-2511 GGUF Q4 |
| 3 | 16–24 GB | FLUX.2 klein 9B FP8 (~15 GB) | Qwen-Image-Edit-2511 FP8 |
| 4 | ≥ 24 GB | FLUX.2 dev cuantizado (32B, ~19 GB Q4) | Qwen-Image-Edit-2511 FP8 + LoRA de interiorismo |

Puntos a resolver en P2:
- Si FLUX.2 klein/dev aceptan control por profundidad o normales (ControlNet o
  condicionamiento nativo), o si hay que guiarlos como "edición" de una imagen base
  (albedo + líneas) con referencias.
- Qwen-Image-Edit trae de serie control por profundidad y aristas: es un candidato
  fuerte para la fidelidad estructural.
- Medir la **desviación estructural**: comparar las aristas y la profundidad
  estimadas del render con las de los pases (métrica automática para elegir
  modelo, pasos y fuerza de control).

Nube (nivel 0 o "máxima calidad"): proveedores con FLUX.2 pro/flex y modelos de
edición multimodal. Se elegirán en P2 según calidad, precio y si aceptan imágenes
de control o solo referencias.

## Chat y edición (solo sobre el render)

Un LLM actúa como agente con herramientas:

1. `buscar_objeto("silla")`: busca por nombre, definición, etiqueta y material en
   `objects_visible.json`. Si es ambiguo, pregunta o usa un modelo de visión sobre
   las máscaras.
2. `mascara(id)`: máscara exacta desde el pase de ids, incluidos los subcomponentes.
3. `inpaint(mascara, instrucción, referencias)`: edita solo esa región; el resto
   del render no se toca y la forma se conserva con los pases de control
   recortados a la máscara.
4. `referencia(imagen)`: estilo o un objeto concreto ("este sofá") como imagen
   de referencia del modelo de edición.
5. `global(instrucción)`: cambios de ambiente (hora, luz, estilo) regenerando con
   los mismos pases.

Cada edición es una versión nueva del render (historial con deshacer).

## Hoja de ruta

- **P1, exportador y pases**: hecho. Verificar con escenas reales (ver [P1.md](P1.md)).
- **P2, render**: Z-Image-Turbo + ControlNet Union (profundidad + líneas) vía ComfyUI,
  métrica de fidelidad, barrido de ajustes, detección de hardware y descarga de
  modelos. Hecho; falta medir en la 5060 (ver [P2.md](P2.md)).
- **P3, edición por objeto**: inpainting con máscara de ids y referencias.
- **P4, producto**: UI de chat en HtmlDialog, servidor local, selector de hardware,
  descarga de modelos y proveedor de nube.
