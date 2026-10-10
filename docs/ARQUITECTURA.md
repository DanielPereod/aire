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
| Usuario final | Persona **no técnica**: solo instala el `.rbz`; todo lo demás, desde la ventana de AIRE |

## Motor de render actual (v0.7): FLUX.2 [klein] 4B en modo edición

Lecciones de las pruebas reales con el modelo de la usuaria (cocina):
- Z-Image + ControlNet desde ruido, controles fuertes → respeta la geometría pero parece un
  dibujo (contornos negros, luz plana).
- Mismo flujo, controles suaves → se inventa otra habitación.
- Z-Image img2img desde una imagen base → sigue cambiando demasiado.

Ahora: `shaded.png` (materiales y texturas reales del modelo + luz sencilla) entra como
**imagen de referencia** en FLUX.2 klein 4B (destilado, 4 pasos, cabe en 8 GB) con la
instrucción de convertirla en fotografía sin añadir, quitar ni mover nada. Los modelos de
edición están entrenados para conservar lo que no se pide cambiar. El modelo se descarga
en el primer render (~4 GB) si la instalación era anterior.

Cambios de la 0.7 tras las primeras imágenes con FLUX (detalle perdido, poco realismo,
1024 px):
- **Imagen base limpia.** `shaded.png` se calcula al doble de resolución y se reduce
  (antialiasing), y las texturas se filtran con mipmaps. Antes una textura fina vista de
  lejos salía como píxeles blancos y negros sueltos (la IA hizo terrazo del suelo) y los
  detalles finos (sillas caladas, grifo, botes) llegaban rotos y se perdían.
- **Dos pasadas en «Alta calidad»**: 1280 px (≈1 MP, donde klein respeta mejor la
  composición) y luego ampliación a 1920 px con una segunda pasada que rehace solo los
  últimos pasos (repaso 0,4) con la imagen base a tamaño completo como referencia. Cabe en
  8 GB (≈20 000 tokens en la segunda pasada).
- **Instrucción** que pide conservar expresamente lo pequeño (calados, patas finas,
  tiradores, grifos, botes) y cada acabado, y un aspecto de fotografía real.
- «Comparar» ahora compara directo a 1920 frente a dos pasadas con repaso 0,3/0,45/0,6.

## v0.9: luz real con Cycles + IA («Alta calidad»)

Las pruebas de la 0.7/0.8 mostraron el límite de la IA sola: con una imagen base de luz
plana tenía que inventarse la luz y, de paso, cambiaba materiales según la semilla. Ahora:

1. `cycles_render.py` renderiza la exportación con Cycles (Blender como módulo `bpy`, en su propio
   entorno `cycles/venv`, Python 3.13): geometría y cámara exactas, materiales PBR deducidos
   del nombre (mármol, latón, tela, cristal, cortina…), sol y cielo con la orientación de
   SketchUp, ventanas como portales de luz y lámparas encendidas por la tarde y la noche.
   Revelado automático (exposición, cielo neutro, AgX). ~25 s a 1920 px en una RTX 5060.
2. Ese render es la imagen base de FLUX.2 klein (dos pasadas, 1920 px), que lo convierte
   en fotografía con la luz y los materiales ya resueltos. ~45 s más.
3. **Colores fijados** (`colormatch.py`): la IA a veces cambia el tono de superficies grandes
   (suelo, muebles, latón). Se conserva su luminosidad y detalle y se toma de Cycles la
   cromaticidad (proporción R:G:B en luz lineal) a escala media, que no depende del brillo:
   la luz cálida de noche sigue cálida aunque la IA aclare. Determinista, unos 4 s.
4. **Acabado de cámara** (`photo.py`): halo suave en luces quemadas, viñeteado leve y grano
   fino con semilla fija.

Cycles nivela la cámara y desplaza el encuadre (perspectiva de dos puntos, `camera.level_view`)
para que las verticales salgan rectas como en una foto de interiorismo, si la vista de SketchUp
está inclinada hasta 40°. Si la cámara está dentro de una habitación modelada sin techo (`room.ceiling_height`: rayos
hacia arriba que no chocan con nada), Cycles añade un techo blanco a la altura de las paredes;
si la vista es desde encima de las paredes, no. En las lámparas, una textura de rejilla o con huecos se describe como
ratán aunque el material se llame «tela».

Si Cycles no está o falla, «Alta calidad» sigue con la imagen base sencilla (se anota en
`logs/render.log`). «Preparar AIRE» lo instala (unos 700 MB); en instalaciones anteriores se
instala solo en el primer render de alta calidad.

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
- **Instalación automática y ventana**: hecho (v0.3.0, ver [INSTALACION.md](INSTALACION.md)).
- **P4, producto**: UI de chat en HtmlDialog, servidor local, selector de hardware,
  descarga de modelos y proveedor de nube.

## v0.10: ventana nueva y edición

La ventana tiene tres pestañas: **Crear** (luz, estilo, indicaciones, calidad), **Editar** y
**Galería**. En Editar se describe el cambio, se puede pintar la zona a tocar (inpaint) y
añadir hasta 3 imágenes de referencia. `edit.py` lo resuelve con el motor de edición:

- Con zona pintada, se edita solo un recorte alrededor de ella a 1024 px (más detalle en lo
  pequeño) y se pega con borde suave: fuera de la zona la imagen queda idéntica.
- Sin zona, se rehace la imagen entera siguiendo la instrucción.
- Las referencias entran como imágenes adicionales del modelo de edición («image 2…»).

Cada edición es un trabajo nuevo (`jobs/<fecha>-cambio`) con `kind: edit` y `source`, así la
galería la marca como «Editada» y el editor puede enseñar el «Antes».

### Motor Qwen Image 2.1

«Alta calidad» y las ediciones usan Qwen Image 2.1 en int8 (la plantilla oficial de ComfyUI
`image_qwen_image_2_1_image_edit`, nodos nativos `TextEncodeQwenImage21` y `QwenImage21Cache`):
unos 16 GB de modelos, que se descargan la primera vez o se enlazan si ya están en otro ComfyUI
del ordenador (StabilityMatrix, ComfyUI Desktop…, `models.adopt_existing`). 25 pasos, cfg 1.
La imagen 1 es el render de Cycles (o la imagen a editar) y marca el tamaño; las referencias van
como imágenes 2, 3… «Rápida» sigue con FLUX.2 klein.

### 0.10.2: Qwen a 1 Mpx y modo «Sin IA»

- Qwen Image 2.1 trabaja a ~1 Mpx (presupuesto oficial) con la imagen ya a un tamaño múltiplo de 32 y `resolution = 0`, así el nodo no recorta ni reescala y el encuadre coincide con la base de Cycles. El resultado se amplía con Lanczos al tamaño final antes de fijar colores y del acabado. Antes de cargar Qwen se llama a `/free` para que no conviva en la RAM con otro motor. A 1920 px tardaba unos 190 s por pasada y dejaba el PC con 1 GB de RAM libre.
- Si ComfyUI se cierra a mitad del trabajo, el usuario ve un aviso de falta de memoria en vez de un error de conexión.
- Calidad «Sin IA» (`real`): solo el render de Cycles con el acabado de cámara; no arranca ComfyUI ni descarga modelos.

### 0.11: realismo «tipo foto» con luz real

- **Biblioteca de materiales** (`matlib.py`): tipos reconocidos por el nombre del material, de su textura o de sus objetos (madera, tarima, mármol, tela, cuero, ratán, alfombra, hormigón, metal, pintura). Para cada tipo se descarga de Poly Haven (CC0), la primera vez, un material escaneado a 2K: color, normal y rugosidad, en `<home>/library`. Se elige un preferido; si no existe, el más descargado que encaje. La elección queda en `index.json`. El color sigue siendo el del modelo: en colores lisos se aplica la variación del escaneo (color ÷ su media) y su relieve; con textura propia, solo la rugosidad irregular. Proyección por caja en metros reales. Las superficies lisas sin tipo reciben un relieve de pintura muy suave.
- **Suelo detectado por geometría** (caras hacia arriba en el nivel más bajo, ≥ 2 m²): si no tiene nombre reconocible, piedra pulida o tarima según su color, con algo de reflejo.
- **Cantos redondeados** con el nodo Bevel (4 mm) en el sombreado, sin tocar la geometría.
- **Sol orientado**: con un BVH de la escena se lanzan rayos desde lo que ve la cámara hacia el sol, a 32° (día) o 18° (tarde), cada 10° de azimut. Se elige el que ilumina entre un 4 y un 35 % de la imagen pasando por una ventana; a igualdad, el más cercano al sol de SketchUp. `--sun-from-model` lo desactiva.
- **Exterior HDRI**: un jardín de Poly Haven visto solo por los rayos de cámara, normalizado por la mediana de su mitad inferior.
- **«Alta calidad» con Qwen**: añade al prompt un estilo de foto de catálogo con sol rasante. Fue lo que más acercó el resultado a las fotos de ejemplo de Dani.
- 0.11.1: profundidad de campo leve (f/5.6, enfocada a la mediana de distancia del tercio central; `--no-dof`); imperfecciones de brillo procedurales (manchas de ~20 cm y huellas de ~1 cm) en materiales sin rugosidad escaneada; hojas translúcidas; aberración cromática muy leve en el acabado de cámara.
- 0.11.2 (tras probar en la cocina de Dani):
  - El sol orientado tiene más fuerza (6,5) y los visillos dejan pasar un 40 % del sol directo, porque las manchas de sol salían difusas y grises.
  - Los suelos con textura propia reciben un relieve escaneado suave.
  - El jardín de la ventana se ve menos sobreexpuesto.
  - Sin material de biblioteca para alfombras ni metales.
  - La tela prefiere bouclé o lino.
  - La base de Cycles de «Alta calidad» va sin desenfoque, porque con él la fidelidad bajaba de 0,94 a 0,70.
  - «Sin IA» no prepara los pases de apoyo de la IA.
  - El prompt pide que el ratán no se convierta en cristal.
- 0.11.3:
  - La imagen de Qwen se encaja sin estirar (escala y relleno reflejado hasta el múltiplo de 32), y luego se recorta. Estirarla desplazaba ~16 px los bordes y la fidelidad bajaba a 0,70.
  - Desenfoque a f/11.
  - Sol orientado más bajo: 24° de día y 15° de tarde.
- 0.11.4:
  - El sol se orienta también hacia huecos sin vidrio, como las puertas acristaladas modeladas sin cristal de «7788 cocina». Cuenta como iluminado todo rayo que sale de la escena sin chocar con nada opaco ni con el techo añadido.
  - Con sol orientado, el cielo pesa un 35 % menos, para que las sombras sean más profundas, y el día es algo más cálido.
- 0.11.5: sol orientado más bajo (18° de día, 12° de tarde) y más dorado, buscando la luz rasante de las fotos de ejemplo.
- 0.12.0: «Sin IA» pasa a ser la calidad por defecto (luz real de Cycles). En la galería, el botón «IA» (Procesar con IA) manda la imagen elegida a la IA (Qwen con el prompt de foto), partiendo del render de Cycles sin acabado, y deja una imagen nueva con la etiqueta «Con IA» junto a la original (`enhance.py`).
- 0.12.1: «Procesar con IA» usa el workflow de Dani «Architecture FLUX2 · Master Render v2» (`workflow_files/`): render a 0,55 Mpx en 4 pasos y ampliación a 1,6 Mpx con refinado suave, con FLUX.2 [klein] 4B. Solo nodos nativos, cabe en 8 GB y tarda unos 25 s. El resultado se deja tal cual, sin fijar colores ni acabado.
