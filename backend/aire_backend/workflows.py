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


@dataclass
class KleinSettings:
    """width/height: tamaño final. Con upscale > 1 se crea primero a width/upscale (≈1 MP,
    donde FLUX.2 klein respeta mejor la composición) y una segunda pasada a tamaño final
    rehace el detalle fino partiendo de esa imagen ampliada («hires fix»)."""
    width: int
    height: int
    prompt: str
    seed: int = 0
    steps: int = 4
    cfg: float = 1.0
    sampler: str = "euler"
    upscale: float = 1.0
    refine_denoise: float = 0.4  # cuánto rehace la segunda pasada (0 = nada, 1 = todo)
    refine_steps: int = 4
    # < 1: la 1.ª pasada parte de la imagen base (no de ruido puro) y conserva más sus colores;
    # útil cuando la base ya tiene luz real (Cycles)
    base_denoise: float = 1.0
    # 0-1: cuánto se fijan al final los colores a los de la imagen base (solo con base externa)
    color_lock: float = 0.0
    photo: float = 0.0  # 0-1: acabado de cámara (halo en luces, viñeteado, grano)
    unet: str = "flux-2-klein-4b-fp8.safetensors"
    text_encoder: str = "qwen_3_4b.safetensors"
    vae: str = "flux2-vae.safetensors"
    weight_dtype: str = "default"  # el fichero ya viene en FP8
    tag: str = ""  # etiqueta para comparar variantes

    def first_pass_size(self) -> tuple[int, int]:
        if self.upscale <= 1.0:
            return self.width, self.height
        return (max(16, int(round(self.width / self.upscale / 16)) * 16),
                max(16, int(round(self.height / self.upscale / 16)) * 16))

    def to_dict(self) -> dict:
        return asdict(self)


def flux2_klein_edit(s: KleinSettings, base_image: str, extra_refs: tuple[str, ...] = (),
                     prefix: str = "aire/render") -> dict:
    """FLUX.2 [klein] en modo edición (estructura del flujo oficial «Image Edit 4B distilled»):
    la imagen base del modelo entra como referencia y el texto pide fotorrealizarla.

    base_image debe estar al tamaño final (s.width × s.height)."""
    g = Graph()
    unet = g.add("UNETLoader", unet_name=s.unet, weight_dtype=s.weight_dtype)
    clip = g.add("CLIPLoader", clip_name=s.text_encoder, type="flux2", device="default")
    vae = g.add("VAELoader", vae_name=s.vae)
    text = g.add("CLIPTextEncode", text=s.prompt, clip=g.out(clip))
    sampler = g.add("KSamplerSelect", sampler_name=s.sampler)
    base = g.add("LoadImage", image=base_image)
    extras = [g.add("LoadImage", image=r) for r in extra_refs]

    def conditioned(images: list) -> str:
        pos, neg = text, g.add("ConditioningZeroOut", conditioning=g.out(text))
        for img in images:
            lat = g.add("VAEEncode", pixels=g.out(img), vae=g.out(vae))
            pos = g.add("ReferenceLatent", conditioning=g.out(pos), latent=g.out(lat))
            neg = g.add("ReferenceLatent", conditioning=g.out(neg), latent=g.out(lat))
        return g.add("CFGGuider", model=g.out(unet), positive=g.out(pos), negative=g.out(neg), cfg=s.cfg)

    # 1.ª pasada: imagen completa desde ruido, con la imagen base (reducida) como referencia
    w1, h1 = s.first_pass_size()
    ref1 = base
    if (w1, h1) != (s.width, s.height):
        ref1 = g.add("ImageScale", image=g.out(base), upscale_method="area", width=w1, height=h1, crop="disabled")
    guider = conditioned([ref1, *extras])
    noise = g.add("RandomNoise", noise_seed=s.seed)
    if s.base_denoise < 1.0:
        total1 = max(s.steps, round(s.steps / max(s.base_denoise, 0.05)))
        full = g.add("Flux2Scheduler", steps=total1, width=w1, height=h1)
        sigmas = g.add("SplitSigmasDenoise", sigmas=g.out(full), denoise=s.base_denoise)
        sigmas_out = g.out(sigmas, 1)
        latent = g.add("VAEEncode", pixels=g.out(ref1), vae=g.out(vae))
    else:
        sigmas = g.add("Flux2Scheduler", steps=s.steps, width=w1, height=h1)
        sigmas_out = g.out(sigmas)
        latent = g.add("EmptyFlux2LatentImage", width=w1, height=h1, batch_size=1)
    out = g.add("SamplerCustomAdvanced", noise=g.out(noise), guider=g.out(guider), sampler=g.out(sampler),
                sigmas=sigmas_out, latent_image=g.out(latent))
    image = g.add("VAEDecode", samples=g.out(out), vae=g.out(vae))

    if (w1, h1) != (s.width, s.height):
        # 2.ª pasada: amplía la foto y rehace solo los últimos pasos (detalle fino) con la
        # imagen base a tamaño completo como referencia, para recuperar lo pequeño.
        big = g.add("ImageScale", image=g.out(image), upscale_method="lanczos", width=s.width,
                    height=s.height, crop="disabled")
        init = g.add("VAEEncode", pixels=g.out(big), vae=g.out(vae))
        guider2 = conditioned([base, *extras])
        total = max(s.refine_steps, round(s.refine_steps / max(s.refine_denoise, 0.05)))
        sig2 = g.add("Flux2Scheduler", steps=total, width=s.width, height=s.height)
        split = g.add("SplitSigmasDenoise", sigmas=g.out(sig2), denoise=s.refine_denoise)
        noise2 = g.add("RandomNoise", noise_seed=s.seed + 1)
        out2 = g.add("SamplerCustomAdvanced", noise=g.out(noise2), guider=g.out(guider2), sampler=g.out(sampler),
                     sigmas=g.out(split, 1), latent_image=g.out(init))
        image = g.add("VAEDecode", samples=g.out(out2), vae=g.out(vae))
    g.add("SaveImage", images=g.out(image), filename_prefix=prefix)
    return g.to_json()


def flux2_klein_retouch(s: KleinSettings, image: str, denoise: float = 0.2, prefix: str = "aire/retoque") -> dict:
    """Retoque fotográfico suave de un render ya bueno (p. ej. el de Cycles): parte de esa
    imagen, la usa también como referencia y rehace solo los últimos pasos."""
    g = Graph()
    unet = g.add("UNETLoader", unet_name=s.unet, weight_dtype=s.weight_dtype)
    clip = g.add("CLIPLoader", clip_name=s.text_encoder, type="flux2", device="default")
    vae = g.add("VAELoader", vae_name=s.vae)
    text = g.add("CLIPTextEncode", text=s.prompt, clip=g.out(clip))
    img = g.add("LoadImage", image=image)
    lat = g.add("VAEEncode", pixels=g.out(img), vae=g.out(vae))
    pos = g.add("ReferenceLatent", conditioning=g.out(text), latent=g.out(lat))
    neg = g.add("ReferenceLatent", conditioning=g.out(g.add("ConditioningZeroOut", conditioning=g.out(text))),
                latent=g.out(lat))
    guider = g.add("CFGGuider", model=g.out(unet), positive=g.out(pos), negative=g.out(neg), cfg=s.cfg)
    total = max(s.refine_steps, round(s.refine_steps / max(denoise, 0.05)))
    sig = g.add("Flux2Scheduler", steps=total, width=s.width, height=s.height)
    split = g.add("SplitSigmasDenoise", sigmas=g.out(sig), denoise=denoise)
    out = g.add("SamplerCustomAdvanced", noise=g.out(g.add("RandomNoise", noise_seed=s.seed)), guider=g.out(guider),
                sampler=g.out(g.add("KSamplerSelect", sampler_name=s.sampler)), sigmas=g.out(split, 1),
                latent_image=g.out(lat))
    g.add("SaveImage", images=g.out(g.add("VAEDecode", samples=g.out(out), vae=g.out(vae))), filename_prefix=prefix)
    return g.to_json()


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


@dataclass
class QwenSettings:
    """Qwen Image 2.1 en modo edición (plantilla oficial «image_qwen_image_2_1_image_edit»).
    La imagen 1 marca el lienzo: la salida sale de su tamaño (resolution=0, nativa)."""
    width: int
    height: int
    prompt: str
    seed: int = 0
    steps: int = 25
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    negative: str = ""
    resolution: int = 0  # 0 = tamaño nativo de la imagen 1; si no, presupuesto de píxeles (lado²)
    unet: str = "qwen_image_2.1_int8_convrot.safetensors"
    text_encoder: str = "qwen3vl_8b_int8_convrot.safetensors"
    vae: str = "qwen_image_2.1_vae_bf16.safetensors"
    color_lock: float = 0.0
    photo: float = 0.0
    tag: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


QWEN_PIXELS = 1024 * 1024  # presupuesto oficial de Qwen Image 2.1: rápido y cabe en 8 GB


def qwen_size(w: int, h: int, pixels: int = QWEN_PIXELS) -> tuple[int, int]:
    """Tamaño de trabajo de Qwen: ~pixels, misma proporción y múltiplos de 32. Con la imagen ya a
    este tamaño el nodo no la recorta ni la reescala, así que no cambia el encuadre."""
    k = (pixels / (w * h)) ** 0.5
    return max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32)


def pad_to(img, size: tuple[int, int]):
    """Encaja img en size sin deformarla: la escala para que quepa y rellena el resto con su propio
    borde reflejado. Devuelve (imagen, caja del contenido). Estirar a un tamaño múltiplo de 32
    cambiaba la proporción un 1,5 % y desplazaba ~16 px los bordes (pruebas 0.11.2)."""
    import numpy as np
    from PIL import Image
    w, h = size
    k = min(w / img.width, h / img.height)
    cw, ch = min(w, round(img.width * k)), min(h, round(img.height * k))
    scaled = np.asarray(img.convert("RGB").resize((cw, ch), Image.Resampling.LANCZOS))
    left, top = (w - cw) // 2, (h - ch) // 2
    padded = np.pad(scaled, ((top, h - ch - top), (left, w - cw - left), (0, 0)), mode="symmetric")
    return Image.fromarray(padded), (left, top, left + cw, top + ch)


def unpad(img, box: tuple[int, int, int, int], size: tuple[int, int], final: tuple[int, int]):
    """Recorta de la salida (de tamaño size, o proporcional) la caja del contenido y la lleva a final."""
    from PIL import Image
    sx, sy = img.width / size[0], img.height / size[1]
    crop = img.crop((round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy)))
    return crop.resize(final, Image.Resampling.LANCZOS)


def qwen21_edit(s: QwenSettings, image: str, extra_refs: tuple[str, ...] = (), prefix: str = "aire/render") -> dict:
    """image (y las referencias) ya subidas a ComfyUI; image debe estar al tamaño final."""
    g = Graph()
    unet = g.add("UNETLoader", unet_name=s.unet, weight_dtype="default")
    model = g.add("QwenImage21Cache", model=g.out(unet), device="auto", dtype="default")
    clip = g.add("CLIPLoader", clip_name=s.text_encoder, type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=s.vae)
    images = {}
    for i, name in enumerate((image, *extra_refs), start=1):
        images[f"images.image_{i}"] = g.out(g.add("LoadImage", image=name))
    enc = g.add("TextEncodeQwenImage21", clip=g.out(clip), vae=g.out(vae), prompt=s.prompt,
                negative_prompt=s.negative, resolution=s.resolution, **images)
    out = g.add("KSampler", model=g.out(model), positive=g.out(enc, 0), negative=g.out(enc, 1),
                latent_image=g.out(enc, 2), seed=s.seed, steps=s.steps, cfg=s.cfg, sampler_name=s.sampler,
                scheduler=s.scheduler, denoise=1.0)
    dec = g.add("VAEDecode", samples=g.out(out), vae=g.out(vae))
    g.add("SaveImage", images=g.out(dec), filename_prefix=prefix)
    return g.to_json()
