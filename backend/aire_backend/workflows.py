"""Constructores de flujos ComfyUI (formato API) a partir de los pases.

Nombres de nodos y entradas comprobados contra el código de ComfyUI (octubre
2026) y sus blueprints oficiales; ComfyClient.run vuelve a validarlos contra el
servidor real antes de encolar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .comfy import Graph


@dataclass
class ZImageSettings:
    width: int
    height: int
    prompt: str
    seed: int = 0
    steps: int = 9
    cfg: float = 1.0
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    shift: float = 3.0
    weight_dtype: str = "fp8_e4m3fn"  # "default" con ≥16 GB de VRAM
    # Se parte de la imagen base del modelo (shaded.png: materiales reales + luz sencilla) y
    # la IA solo la «fotorrealiza» (img2img). Lecciones de las pruebas reales:
    #  - generar desde ruido con controles fuertes → aspecto de dibujo, contornos negros;
    #  - desde ruido con controles suaves → se inventa otra habitación.
    depth_strength: float = 0.8
    depth_end: float = 1.0
    lines_strength: float = 0.0  # las líneas de SketchUp acaban pintadas como contornos
    lines_end: float = 0.5
    denoise: float = 0.55  # cuánto puede cambiar la IA la imagen base (0 = nada, 1 = todo)
    refine: float = 0.0  # > 0: segunda pasada img2img (opcional)
    refine_depth: float = 0.35
    unet: str = "z_image_turbo_bf16.safetensors"
    text_encoder: str = "qwen_3_4b.safetensors"
    vae: str = "ae.safetensors"
    controlnet: str = "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"

    def to_dict(self) -> dict:
        return asdict(self)


def zimage_control(s: ZImageSettings, depth_image: str, lines_image: str | None,
                   init_image: str | None = None, prefix: str = "aire/render") -> dict:
    """Z-Image-Turbo guiado por profundidad (+ líneas) con ControlNet Union.

    depth_image / lines_image / init_image: valores para LoadImage (ya subidos).
    """
    g = Graph()
    unet = g.add("UNETLoader", unet_name=s.unet, weight_dtype=s.weight_dtype)
    clip = g.add("CLIPLoader", clip_name=s.text_encoder, type="lumina2", device="default")
    vae = g.add("VAELoader", vae_name=s.vae)
    patch = g.add("ModelPatchLoader", name=s.controlnet)

    model = g.add("ModelSamplingAuraFlow", model=g.out(unet), shift=s.shift)
    depth = g.add("LoadImage", image=depth_image)
    model = g.add("ZImageFunControlnet", model=g.out(model), model_patch=g.out(patch), vae=g.out(vae),
                  strength=s.depth_strength, image=g.out(depth), start_percent=0.0, end_percent=s.depth_end)
    if lines_image and s.lines_strength > 0:
        lines = g.add("LoadImage", image=lines_image)
        model = g.add("ZImageFunControlnet", model=g.out(model), model_patch=g.out(patch), vae=g.out(vae),
                      strength=s.lines_strength, image=g.out(lines), start_percent=0.0, end_percent=s.lines_end)

    pos = g.add("CLIPTextEncode", text=s.prompt, clip=g.out(clip))
    neg = g.add("ConditioningZeroOut", conditioning=g.out(pos))

    if init_image and s.denoise < 1.0:
        init = g.add("LoadImage", image=init_image)
        latent = g.add("VAEEncode", pixels=g.out(init), vae=g.out(vae))
    else:
        latent = g.add("EmptySD3LatentImage", width=s.width, height=s.height, batch_size=1)

    sample = g.add("KSampler", model=g.out(model), seed=s.seed, steps=s.steps, cfg=s.cfg,
                   sampler_name=s.sampler, scheduler=s.scheduler, positive=g.out(pos), negative=g.out(neg),
                   latent_image=g.out(latent), denoise=s.denoise if init_image else 1.0)

    if s.refine > 0:
        # Refinado: rehace solo el detalle fino (texturas, reflejos, sombras suaves) con un
        # control de profundidad ligero para que la geometría no se mueva.
        rmodel = g.add("ModelSamplingAuraFlow", model=g.out(unet), shift=s.shift)
        rmodel = g.add("ZImageFunControlnet", model=g.out(rmodel), model_patch=g.out(patch), vae=g.out(vae),
                       strength=s.refine_depth, image=g.out(depth), start_percent=0.0, end_percent=0.6)
        sample = g.add("KSampler", model=g.out(rmodel), seed=s.seed + 1, steps=s.steps, cfg=s.cfg,
                       sampler_name=s.sampler, scheduler=s.scheduler, positive=g.out(pos), negative=g.out(neg),
                       latent_image=g.out(sample), denoise=s.refine)
    image = g.add("VAEDecode", samples=g.out(sample), vae=g.out(vae))
    g.add("SaveImage", images=g.out(image), filename_prefix=prefix)
    return g.to_json()
