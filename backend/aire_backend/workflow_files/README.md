# Workflows de Dani (ComfyUI)

`architecture_flux2_master_v2.ui.json` es una copia sin cambios de
`StabilityMatrix/.../workflows/Architecture_FLUX2/01_Master_Render_v2.app.json`.

Modelos (los mismos del motor FLUX de AIRE): `flux-2-klein-4b-fp8.safetensors`,
`qwen_3_4b.safetensors` (CLIP tipo flux2) y `flux2-vae.safetensors`.

Nodos: todos nativos de ComfyUI, salvo dos de rgthree que solo sirven para la interfaz
(«Image Comparer» y «Fast Groups Bypasser»). Las versiones API los quitan.

Tres pasos que el usuario lanza uno tras otro, cada uno en formato API (`*.api.json`):

1. `1_render`: imagen 1 → ImageScaleToTotalPixels 0,55 Mpx → ReferenceLatent → 4 pasos
   (Flux2Scheduler, euler, BasicGuider) → foto. Entrada: nodo 4 (LoadImage).
2. `2_upscale`: el render del paso 1 a 1,6 Mpx, refinado con 12 pasos y
   SplitSigmasDenoise 0,17. Entrada: nodo 200 (LoadImage). El original enlazaba el
   latente del paso 1; aquí se codifica de nuevo con VAEEncode (nodo 201).
3. `3_inpaint`: imagen y máscara (alfa del LoadImage 112). Usa GrowMask 12, desenfoque y
   SplitSigmasDenoise 0,5; ImageCompositeMasked pega solo la zona. Prompt en el nodo 120.
