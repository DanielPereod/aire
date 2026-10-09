"""Render físico de una exportación de AIRE con Cycles (Blender), sin IA.

La geometría, la cámara y los materiales salen tal cual del modelo; la luz se calcula:
sol y cielo físicos con la hora y la orientación de SketchUp, ventanas como portales de
luz y, de noche, las lámparas encendidas. El resultado es el mismo en cada render
(determinista); la IA, si se usa, solo retoca después.

Funciona con el Blender de línea de órdenes o con el módulo bpy:

    blender -b --factory-startup -P backend/aire_backend/cycles.py -- <export> --out render.png
    python -m aire_backend.cycles <export> --out render.png        # con «pip install bpy»
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # ejecutado por Blender con -P: importar el paquete por ruta
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "aire_backend"

import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402
import numpy as np  # noqa: E402

from aire_backend.camera import Camera  # noqa: E402
from aire_backend.scene import Scene, load_scene  # noqa: E402

# --------------------------------------------------------------------------- materiales
# (patrón en el nombre del material, del objeto o de la textura) → parámetros PBR.
# El orden importa: lo más específico primero.
PRESETS: list[tuple[str, dict]] = [
    (r"vidrio|cristal|glass|ventana|window|vitro", {"kind": "glass"}),
    (r"cortina|curtain|voile|visillo|courthain|stor", {"kind": "sheer", "roughness": 0.9}),
    (r"lat[oó]n|brass|bronce|bronze|\boro\b|dorad|gold", {"metallic": 1.0, "roughness": 0.28, "tint": (0.93, 0.78, 0.45)}),
    (r"cobre|copper", {"metallic": 1.0, "roughness": 0.3, "tint": (0.95, 0.64, 0.54)}),
    (r"inox|stainless|acero|steel|chrom|cromad|alumin|metal|nabor|grifo|tap\b", {"metallic": 1.0, "roughness": 0.22}),
    (r"m[aá]rmol|marble|quartz|cuarzo|silestone|dekton|granit|porcel[aá]nic|neolith|calacatta|stone|piedra",
     {"roughness": 0.08, "coat": 0.3}),
    (r"azulejo|tile|cer[aá]mic|ceramic|metro", {"roughness": 0.12, "bump": 0.15}),
    (r"lacad|lacquer|laca|ral\d|gloss", {"roughness": 0.25}),
    (r"nogal|walnut|roble|oak|madera|wood|fresno|ash|haya|teca|teak|pino|pine|chapa", {"roughness": 0.45, "bump": 0.25}),
    (r"cuero|leather|piel", {"roughness": 0.5, "bump": 0.1}),
    (r"rat[aá]n|rattan|mimbre|wicker|ca[nñ]a|cane|fibra|jute|yute", {"roughness": 0.7, "bump": 0.5}),
    (r"tela|fabric|tejido|textil|lino|linen|algod|cotton|terciopelo|velvet|bouclé|boucle|tapiz|upholst|bege",
     {"roughness": 0.85, "sheen": 0.6, "bump": 0.35}),
    (r"microcement|cemento|concrete|hormig|resin|resina", {"roughness": 0.55, "bump": 0.05}),
    (r"pintura|paint|pared|wall|yeso|plaster|techo|ceiling|gotel", {"roughness": 0.85}),
    (r"preto|black glass|vitrocer|induc|placa|horno|oven|micro|forno", {"roughness": 0.06, "coat": 0.5}),
    (r"pl[aá]stic|plastic|pvc", {"roughness": 0.35}),
    (r"papel|paper|libro|book", {"roughness": 0.8}),
]
DEFAULT = {"roughness": 0.5}

LAMP_NAMES = re.compile(r"l[aá]mpara|lamp|pendant|colgante|aplique|sconce|plaf[oó]n|foco|spot|downlight|"
                        r"luminaria|light|bombilla|bulb|chandelier|ara[nñ]a", re.I)


def classify(*names: str | None) -> dict:
    """Primer nombre que encaje con algún preset, en orden: el del material manda sobre el del
    objeto (una isla «de mármol» con frente de nogal no debe dar madera pulida)."""
    for name in names:
        if not name:
            continue
        for pat, params in PRESETS:
            if re.search(pat, name, re.I):
                return {**DEFAULT, **params}
    return dict(DEFAULT)


def material_params(scene: Scene, mid: int, object_names: str) -> dict:
    info = scene.materials[mid]
    tex = (info.get("texture") or {}).get("file")
    p = classify(info.get("name"), info.get("internal_name"), tex, object_names)
    # PBR de SketchUp 2025+ si el material lo trae: manda sobre lo deducido por el nombre
    pbr = info.get("pbr") or {}
    if pbr.get("roughness_factor") is not None and pbr.get("workflow") not in (None, 0):
        p["roughness"] = float(pbr["roughness_factor"])
    if pbr.get("metallic_factor") is not None and pbr.get("workflow") not in (None, 0):
        p["metallic"] = float(pbr["metallic_factor"])
    alpha = info.get("alpha", 1.0)
    if alpha is not None and alpha < 0.95 and p.get("kind") is None:
        p["kind"] = "glass" if alpha < 0.6 else "alpha"
        p["alpha"] = alpha
    return p


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def build_material(scene: Scene, mid: int, params: dict, name: str):
    info = scene.materials[mid]
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    out = nodes.get("Material Output")
    rgb = list(_srgb_to_linear(info.get("color", [200, 200, 200])[:3])) + [1.0]

    color_socket = None
    tex = info.get("texture")
    img_path = scene.root / tex["file"] if tex else None
    if img_path and img_path.exists():
        img = bpy.data.images.load(str(img_path), check_existing=True)
        node = nodes.new("ShaderNodeTexImage")
        node.image = img
        node.interpolation = "Cubic"
        color_socket = node.outputs["Color"]
        links.new(color_socket, bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = rgb

    tint = params.get("tint")
    if tint and color_socket is None and params.get("metallic", 0) > 0.5:
        # Metal sin textura y con color de SketchUp poco fiable: el tono físico del metal
        bsdf.inputs["Base Color"].default_value = [*tint, 1.0]
    bsdf.inputs["Roughness"].default_value = params.get("roughness", 0.5)
    bsdf.inputs["Metallic"].default_value = params.get("metallic", 0.0)
    if params.get("coat"):
        bsdf.inputs["Coat Weight"].default_value = params["coat"]
        bsdf.inputs["Coat Roughness"].default_value = 0.03
    if params.get("sheen"):
        bsdf.inputs["Sheen Weight"].default_value = params["sheen"]
    if params.get("bump") and color_socket is not None:
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = params["bump"]
        bump.inputs["Distance"].default_value = 0.002
        links.new(color_socket, bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    kind = params.get("kind")
    if kind == "glass":
        # Vidrio fino de arquitectura: casi todo transparente con un reflejo suave. Deja pasar
        # la luz y la vista sin refracción (sin cáusticas ni ventanas oscuras).
        transp = nodes.new("ShaderNodeBsdfTransparent")
        transp.inputs["Color"].default_value = [0.96, 0.98, 0.97, 1.0]
        gloss = nodes.new("ShaderNodeBsdfGlossy") if "ShaderNodeBsdfGlossy" in dir(bpy.types) else \
            nodes.new("ShaderNodeBsdfAnisotropic")
        gloss.inputs["Roughness"].default_value = 0.0
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 0.08
        links.new(transp.outputs[0], mix.inputs[1])
        links.new(gloss.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], out.inputs["Surface"])
    elif kind == "sheer":
        # Cortina fina: difusa por delante y traslúcida a contraluz
        transl = nodes.new("ShaderNodeBsdfTranslucent")
        if color_socket is not None:
            links.new(color_socket, transl.inputs["Color"])
        else:
            transl.inputs["Color"].default_value = rgb
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 0.45
        links.new(bsdf.outputs[0], mix.inputs[1])
        links.new(transl.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], out.inputs["Surface"])
    elif kind == "alpha":
        bsdf.inputs["Alpha"].default_value = params.get("alpha", 1.0)
    elif params.get("translucent"):
        # Pantalla de lámpara: deja pasar parte de la luz de la bombilla
        transl = nodes.new("ShaderNodeBsdfTranslucent")
        if color_socket is not None:
            links.new(color_socket, transl.inputs["Color"])
        else:
            transl.inputs["Color"].default_value = rgb
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = params["translucent"]
        links.new(bsdf.outputs[0], mix.inputs[1])
        links.new(transl.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


# --------------------------------------------------------------------------- geometría
def _face_material(front: np.ndarray, back: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Material de cada cara. El 0 es «sin material» (el gris/azul por defecto de SketchUp):
    si solo una cara tiene material, se usa para las dos."""
    f = np.where(front == 0, back, front)
    b = np.where(back == 0, f, back)
    return f.astype(np.int64), b.astype(np.int64)


def lamp_objects(scene: Scene) -> list[int]:
    """Objetos de lámpara de nivel más alto (por su nombre)."""
    def names(o):
        return " ".join(filter(None, (o.get("name"), o.get("definition"))))
    out = []
    for obj in scene.objects:
        if LAMP_NAMES.search(names(obj)) and not any(LAMP_NAMES.search(names(scene.objects[a]))
                                                    for a in scene.ancestors(obj["id"])):
            out.append(obj["id"])
    return out


def build_meshes(scene: Scene, visible_tris: np.ndarray) -> tuple[dict, list]:
    """Una malla por par (material delantero, trasero). Devuelve los objetos y las caras de
    cristal agrupadas por objeto de SketchUp (para los portales de luz)."""
    front, back = _face_material(scene.tri_material_front, scene.tri_material_back)
    lamp_tri = np.zeros(len(scene.tri_object), dtype=np.int64)
    for oid in lamp_objects(scene):
        lamp_tri[np.isin(scene.tri_object, scene.descendants(oid))] = 1
    keys = np.stack([front, back, lamp_tri], axis=1)
    names_by_obj = {o["id"]: " ".join(filter(None, (o.get("name"), o.get("definition"))))
                    for o in scene.objects}
    params_cache: dict[tuple[int, str], dict] = {}
    mats_cache: dict[tuple, object] = {}
    objects = {}
    glass_groups: dict[int, list[int]] = {}

    def obj_names(tris: np.ndarray) -> str:
        ids, counts = np.unique(scene.tri_object[tris], return_counts=True)
        top = ids[np.argsort(-counts)[:3]]
        names = []
        for oid in top:
            names.append(names_by_obj.get(int(oid), ""))
            names += [names_by_obj.get(a, "") for a in scene.ancestors(int(oid))]
        return " ".join(names)

    def params_for(mid: int, tris: np.ndarray, lamp: bool) -> dict:
        key = (mid, obj_names(tris), lamp)
        if key not in params_cache:
            p = material_params(scene, mid, key[1])
            if lamp and p.get("metallic", 0) < 0.5 and p.get("kind") is None:
                p["translucent"] = 0.5
            params_cache[key] = p
        return params_cache[key]

    def material_for(mf: int, mb: int, tris: np.ndarray, lamp: bool = False):
        pf, pb = params_for(mf, tris, lamp), params_for(mb, tris, lamp)
        key = (mf, mb, json.dumps(pf, sort_keys=True), json.dumps(pb, sort_keys=True))
        if key in mats_cache:
            return mats_cache[key], pf
        m_front = build_material(scene, mf, pf, f"m{mf}")
        if mf == mb:
            mats_cache[key] = m_front
            return m_front, pf
        m_back = build_material(scene, mb, pb, f"m{mb}")
        # Mezcla por cara visible: dos materiales distintos delante y detrás
        mat = bpy.data.materials.new(f"m{mf}_{mb}")
        nt = mat.node_tree
        nt.nodes.remove(nt.nodes.get("Principled BSDF"))
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        mix = nt.nodes.new("ShaderNodeMixShader")
        gf = nt.nodes.new("ShaderNodeGroup")
        gb = nt.nodes.new("ShaderNodeGroup")
        gf.node_tree, gb.node_tree = _as_group(m_front), _as_group(m_back)
        nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
        nt.links.new(gf.outputs[0], mix.inputs[1])
        nt.links.new(gb.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], nt.nodes.get("Material Output").inputs["Surface"])
        mats_cache[key] = mat
        return mat, pf

    uniq, inverse = np.unique(keys[visible_tris], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    for k, (mf, mb, lamp) in enumerate(uniq):
        tris = visible_tris[inverse == k]
        mat, pf = material_for(int(mf), int(mb), tris, bool(lamp))
        if pf.get("kind") == "glass":
            for oid in np.unique(scene.tri_object[tris]):
                glass_groups.setdefault(int(oid), []).extend(tris[scene.tri_object[tris] == oid].tolist())
        mesh_obj = _make_mesh(scene, tris, mat, f"g{k}")
        if lamp:
            mesh_obj.visible_shadow = False  # la luz de la bombilla sale de la lámpara
        objects[(int(mf), int(mb), len(objects))] = mesh_obj
    return objects, list(glass_groups.values())


def _as_group(mat):
    """Copia el árbol de un material como grupo de nodos (salida: el shader)."""
    group = bpy.data.node_groups.new(mat.name + "_g", "ShaderNodeTree")
    group.interface.new_socket("Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    src = mat.node_tree
    mapping = {}
    for n in src.nodes:
        if n.bl_idname == "ShaderNodeOutputMaterial":
            mapping[n.name] = group.nodes.new("NodeGroupOutput")
            continue
        m = group.nodes.new(n.bl_idname)
        for attr in ("image", "interpolation", "operation"):
            if hasattr(n, attr):
                setattr(m, attr, getattr(n, attr))
        for i, s in enumerate(n.inputs):
            if hasattr(s, "default_value") and i < len(m.inputs):
                try:
                    m.inputs[i].default_value = s.default_value
                except (TypeError, AttributeError):
                    pass
        mapping[n.name] = m
    for link in src.links:
        a, b = mapping[link.from_node.name], mapping[link.to_node.name]
        if b.bl_idname == "NodeGroupOutput":
            group.links.new(a.outputs[link.from_socket.identifier], b.inputs[0])
        else:
            group.links.new(a.outputs[link.from_socket.identifier], b.inputs[link.to_socket.identifier])
    return group


def _make_mesh(scene: Scene, tris: np.ndarray, mat, name: str):
    n = len(tris)
    verts = scene.positions[tris].reshape(-1, 3).astype(np.float32)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(n * 3)
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(n * 3)
    mesh.loops.foreach_set("vertex_index", np.arange(n * 3, dtype=np.int32))
    mesh.polygons.add(n)
    mesh.polygons.foreach_set("loop_start", np.arange(0, n * 3, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(n, 3, dtype=np.int32))
    uv = mesh.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", scene.uvs[tris].reshape(-1, 2).astype(np.float32).ravel())
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)
    mesh.polygons.foreach_set("use_smooth", np.ones(n, dtype=bool))
    nrm = scene.normals[tris].reshape(-1, 3).astype(np.float32)
    mesh.normals_split_custom_set(nrm.reshape(-1, 3).tolist())
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def visible_triangles(scene: Scene) -> np.ndarray:
    """Triángulos sin recortar por planos de sección (los recortados se omiten enteros si
    quedan fuera; los cortados se mantienen: aproximación suficiente para la luz)."""
    keep = np.ones(len(scene.positions), dtype=bool)
    planes = np.asarray(scene.section_planes, dtype=np.float64).reshape(-1, 4)
    for cid, members in enumerate(scene.clip_sets):
        if not members:
            continue
        sel = scene.tri_clip == cid
        p = scene.positions[sel].astype(np.float64)
        inside = np.ones(sel.sum(), dtype=bool)
        for pi in members:
            n, d = planes[pi, :3], planes[pi, 3]
            inside &= ((p @ n) + d >= -1e-6).any(axis=1)
        keep[np.nonzero(sel)[0][~inside]] = False
    return np.nonzero(keep)[0]


# --------------------------------------------------------------------------- cámara y luz
def setup_camera(scene: Scene, width: int) -> tuple[int, int]:
    cam = Camera.from_scene(scene.camera, scene.view)
    w, h = cam.resolution(width)
    data = bpy.data.cameras.new("cam")
    if cam.perspective:
        data.sensor_fit = "VERTICAL"
        data.angle_y = cam.vfov
    else:
        data.type = "ORTHO"
        data.sensor_fit = "VERTICAL"
        data.ortho_scale = cam.ortho_height * max(1.0, w / h)
    data.clip_start = 0.02
    data.clip_end = 1000.0
    obj = bpy.data.objects.new("cam", data)
    bpy.context.scene.collection.objects.link(obj)
    r, u, b = cam.basis()  # derecha, arriba, atrás (mundo)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, b, cam.eye
    obj.matrix_world = Matrix(m.tolist())
    bpy.context.scene.camera = obj
    bs = bpy.context.scene
    bs.render.resolution_x, bs.render.resolution_y = w, h
    bs.render.resolution_percentage = 100
    return w, h


def _look_rotation(direction: np.ndarray) -> list[list[float]]:
    """Matriz cuyo eje -Z local apunta en `direction` (convención de luces de Blender)."""
    z = -direction / np.linalg.norm(direction)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def sun_direction(scene: Scene) -> np.ndarray:
    sun = scene.meta.get("sun") or {}
    d = sun.get("direction")
    if d and np.linalg.norm(d) > 0:
        return np.asarray(d, dtype=np.float64) / np.linalg.norm(d)
    return np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])  # tarde de primavera


def setup_daylight(scene: Scene, light: str) -> None:
    to_sun = sun_direction(scene)
    if light == "tarde":
        # Atardecer: sol bajo y cálido conservando su orientación horizontal
        h = to_sun.copy()
        h[2] = 0.0
        h = h / (np.linalg.norm(h) or 1.0)
        to_sun = h * math.cos(math.radians(12)) + np.array([0, 0, math.sin(math.radians(12))])
    world = bpy.data.worlds.new("cielo")
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False  # el sol va como lámpara: sombras nítidas y menos ruido
    elev = math.asin(max(-1.0, min(1.0, to_sun[2])))
    sky.sun_elevation = max(elev, math.radians(2))
    sky.sun_rotation = math.atan2(to_sun[0], to_sun[1])
    bg = nt.nodes.get("Background")
    bg.inputs["Strength"].default_value = 0.6 if light == "tarde" else 1.0
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    _camera_sees(nt, bg, (1.0, 0.98, 0.95), 40.0)
    bpy.context.scene.world = world

    sun = bpy.data.lights.new("sol", "SUN")
    sun.angle = math.radians(0.53)
    # Proporción sol/cielo realista: con el cielo físico a fuerza 1, unos 4-5 W/m² dan la
    # relación de luz típica (se calibra con la exposición automática al final).
    sun.energy = 4.0 if to_sun[2] > 0.05 else 0.0
    if light == "tarde":
        sun.color = (1.0, 0.72, 0.48)
        sun.energy = 3.0
    obj = bpy.data.objects.new("sol", sun)
    rot = _look_rotation(-to_sun)
    m = np.eye(4)
    m[:3, :3] = rot
    obj.matrix_world = Matrix(m.tolist())
    bpy.context.scene.collection.objects.link(obj)


def _camera_sees(nt, bg, color, strength) -> None:
    """Lo que la cámara ve directamente del exterior (ventanas, huecos de puertas) es un fondo
    claro uniforme, como en la fotografía de interiores; la luz sigue saliendo del cielo."""
    out = nt.nodes.get("World Output")
    flat = nt.nodes.new("ShaderNodeBackground")
    flat.inputs["Color"].default_value = (*color, 1.0)
    flat.inputs["Strength"].default_value = strength
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(flat.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def setup_night() -> None:
    world = bpy.data.worlds.new("noche")
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.02, 0.03, 0.06, 1.0)
    bg.inputs["Strength"].default_value = 0.002
    _camera_sees(nt, bg, (0.03, 0.04, 0.08), 0.02)
    bpy.context.scene.world = world


def add_portals(scene: Scene, glass_groups: list[list[int]], eye: np.ndarray) -> int:
    """Un portal de luz por ventana (rectángulo en el plano del cristal, mirando hacia
    dentro): Cycles muestrea el cielo a través de ellos y el interior sale sin ruido."""
    count = 0
    for tris in glass_groups:
        p = scene.positions[np.asarray(tris)].reshape(-1, 3).astype(np.float64)
        c = p.mean(axis=0)
        _, s, vt = np.linalg.svd(p - c, full_matrices=False)
        normal = vt[2]
        if abs(normal[2]) > 0.5:  # cristal horizontal (mesa, encimera): no es ventana
            continue
        a1, a2 = vt[0], vt[1]
        e1 = (p - c) @ a1
        e2 = (p - c) @ a2
        sx, sy = float(e1.max() - e1.min()), float(e2.max() - e2.min())
        if sx * sy < 0.15:
            continue
        center = c + a1 * (e1.max() + e1.min()) / 2 + a2 * (e2.max() + e2.min()) / 2
        inward = normal if np.dot(eye - center, normal) > 0 else -normal
        data = bpy.data.lights.new(f"portal{count}", "AREA")
        data.shape = "RECTANGLE"
        data.size, data.size_y = sx, sy
        data.cycles.is_portal = True
        obj = bpy.data.objects.new(f"portal{count}", data)
        m = np.eye(4)
        z = -inward  # la luz de área emite hacia -Z local
        x = a1 - z * np.dot(a1, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, center + inward * 0.02
        obj.matrix_world = Matrix(m.tolist())
        bpy.context.scene.collection.objects.link(obj)
        count += 1
    return count


def add_lamps(scene: Scene, visible_tris: np.ndarray, power: float = 60.0) -> int:
    """Una luz cálida dentro de cada lámpara del modelo (por su nombre)."""
    count = 0
    tri_obj = scene.tri_object[visible_tris]
    for oid in lamp_objects(scene):
        sel = np.isin(tri_obj, scene.descendants(oid))
        if not sel.any():
            continue
        p = scene.positions[visible_tris[sel]].reshape(-1, 3)
        lo, hi = p.min(axis=0), p.max(axis=0)
        center = (lo + hi) / 2
        center[2] = lo[2] + (hi[2] - lo[2]) * 0.3  # la bombilla, en la parte baja de la tulipa
        data = bpy.data.lights.new(f"lampara{count}", "POINT")
        data.energy = power
        data.color = (1.0, 0.78, 0.55)  # 2700 K aprox.
        data.shadow_soft_size = 0.06
        lamp = bpy.data.objects.new(f"lampara{count}", data)
        lamp.location = center.tolist()
        bpy.context.scene.collection.objects.link(lamp)
        count += 1
    return count


# --------------------------------------------------------------------------- render
def setup_cycles(samples: int, device: str) -> str:
    bs = bpy.context.scene
    bs.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    used = "CPU"
    for kind in ([device] if device != "auto" else ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]):
        if kind == "CPU":
            break
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except (TypeError, ValueError):
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            used = kind
            break
    bs.cycles.device = "GPU" if used != "CPU" else "CPU"
    bs.cycles.samples = samples
    bs.cycles.use_adaptive_sampling = True
    bs.cycles.adaptive_threshold = 0.01
    bs.cycles.use_denoising = True
    bs.cycles.denoiser = "OPTIX" if used == "OPTIX" else "OPENIMAGEDENOISE"
    bs.cycles.max_bounces = 10
    bs.cycles.diffuse_bounces = 6
    bs.cycles.glossy_bounces = 4
    bs.cycles.transmission_bounces = 8
    bs.cycles.transparent_max_bounces = 16
    bs.cycles.sample_clamp_indirect = 8.0
    bs.cycles.caustics_reflective = False
    bs.cycles.caustics_refractive = False
    bs.view_settings.view_transform = "AgX"
    looks = [i.identifier for i in bs.view_settings.bl_rna.properties["look"].enum_items]
    bs.view_settings.look = next((x for x in ("AgX - Medium High Contrast", "AgX - Base Contrast") if x in looks), "None")
    return used


def develop(exr_path: Path, warmth: float, key: float = 0.18, ev: float | None = None,
            max_ev: float = 4.0, balance: float = 0.85):
    """Revelado como el de una cámara: balance de blancos (mundo gris del interior, con un
    punto de calidez) y exposición que lleva la luminancia media a gris medio, ignorando
    ventanas quemadas y sombras profundas. Devuelve (imagen corregida, EV)."""
    img = bpy.data.images.load(str(exr_path))
    px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    rgba = px.reshape(-1, 4)
    rgb = rgba[:, :3]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    ok = lum > 1e-6
    if ok.any():
        lo, hi = np.percentile(lum[ok], [5, 92])
        core = ok & (lum >= lo) & (lum <= hi)
        mean = rgb[core].mean(axis=0)
        gains = mean.mean() / np.maximum(mean, 1e-6)
        gains = gains ** balance * np.array([1.0 + warmth, 1.0, 1.0 - warmth], dtype=np.float32)
        rgb *= gains.astype(np.float32)
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        if ev is None:
            avg = float(np.exp(np.log(np.maximum(lum[core], 1e-6)).mean()))
            ev = math.log2(key / avg)
    ev = float(np.clip(ev or 0.0, -12.0, max_ev))
    img.pixels.foreach_set(rgba.ravel())
    return img, ev


def render(export_dir: str, out: str, width: int = 1920, samples: int = 256, light: str = "dia",
           device: str = "auto", exposure: float | None = None) -> dict:
    t0 = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = load_scene(export_dir)
    tris = visible_triangles(scene)
    _, glass = build_meshes(scene, tris)
    w, h = setup_camera(scene, width)
    eye = Camera.from_scene(scene.camera, scene.view).eye
    lamps = 0
    if light == "noche":
        setup_night()
        lamps = add_lamps(scene, tris, power=150.0)
    else:
        setup_daylight(scene, light)
        if light == "tarde":
            lamps = add_lamps(scene, tris, power=100.0)
    portals = add_portals(scene, glass, eye)
    used = setup_cycles(samples, device)

    bs = bpy.context.scene
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    exr = out.with_suffix(".exr")
    bs.render.image_settings.file_format = "OPEN_EXR"
    bs.render.image_settings.color_depth = "16"
    bs.render.filepath = str(exr)
    t1 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    t2 = time.perf_counter()

    warmth = {"dia": 0.03, "tarde": 0.08, "noche": 0.0}.get(light, 0.03)
    # De noche la imagen debe quedar oscura: el límite de EV evita «día nublado»
    img, ev = develop(exr, warmth, key=0.18 if light != "noche" else 0.08, ev=exposure,
                      max_ev=4.0 if light != "noche" else 6.0,
                      balance={"dia": 0.85, "tarde": 0.4, "noche": 0.0}.get(light, 0.85))
    bs.view_settings.exposure = ev
    bs.render.image_settings.file_format = "PNG"
    bs.render.image_settings.color_depth = "8"
    bs.render.image_settings.color_management = "FOLLOW_SCENE"
    img.save_render(str(out), scene=bs)
    info = {"out": str(out), "exr": str(exr), "width": w, "height": h, "samples": samples, "light": light,
            "device": used, "portals": portals, "lamps": lamps, "exposure": round(ev, 2),
            "seconds_setup": round(t1 - t0, 1), "seconds_render": round(t2 - t1, 1)}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Render físico (Cycles) de una exportación de AIRE")
    ap.add_argument("export_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--light", choices=["dia", "tarde", "noche"], default="dia")
    ap.add_argument("--device", default="auto", help="auto, OPTIX, CUDA o CPU")
    ap.add_argument("--exposure", type=float, default=None, help="EV fijo (por defecto, automático)")
    args = ap.parse_args(argv)
    info = render(args.export_dir, args.out, args.width, args.samples, args.light, args.device, args.exposure)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
