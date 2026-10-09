"""Cliente mínimo de la API HTTP de ComfyUI (sin dependencias) y validación de
flujos contra /object_info antes de encolarlos.

La validación previa existe porque los nodos y modelos de ComfyUI cambian a
menudo: preferimos un error claro ("falta el modelo X en models/vae") antes que
un fallo a mitad de render.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    pass


class Graph:
    """Grafo en formato API de ComfyUI: {id: {"class_type", "inputs"}}."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add(self, class_type: str, **inputs) -> str:
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": class_type, "inputs": {k: v for k, v in inputs.items() if v is not None}}
        return nid

    @staticmethod
    def out(nid: str, index: int = 0) -> list:
        return [nid, index]

    def to_json(self) -> dict:
        return self.nodes


def _combo_options(spec) -> list | None:
    """Opciones de una entrada tipo combo en /object_info (formato clásico y nuevo)."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def validate(graph: dict, object_info: dict) -> list[str]:
    """Devuelve una lista de problemas legibles (vacía = todo bien)."""
    problems = []
    for nid, node in graph.items():
        ct = node["class_type"]
        info = object_info.get(ct)
        if info is None:
            problems.append(f"Nodo {ct!r} no existe en este ComfyUI (¿versión antigua? actualiza ComfyUI)")
            continue
        spec = {**info.get("input", {}).get("required", {}), **info.get("input", {}).get("optional", {})}
        required = info.get("input", {}).get("required", {})
        for name in required:
            if name not in node["inputs"]:
                problems.append(f"{ct}: falta la entrada obligatoria {name!r}")
        for name, value in node["inputs"].items():
            if name not in spec:
                problems.append(f"{ct}: entrada desconocida {name!r}")
                continue
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in graph:
                    problems.append(f"{ct}.{name}: enlaza con un nodo inexistente {value[0]}")
                continue
            options = _combo_options(spec[name])
            if options is not None and isinstance(value, str) and value not in options:
                if ct == "LoadImage":
                    continue  # se sube justo antes de encolar
                problems.append(f"{ct}.{name} = {value!r} no está disponible "
                                f"(¿falta descargar el modelo? opciones: {options[:5]}{'…' if len(options) > 5 else ''})")
    return problems


class ComfyClient:
    def __init__(self, server: str = "http://127.0.0.1:8188", timeout: float = 30.0):
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    # ------------------------------------------------------------------ HTTP
    def _request(self, path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(self.server + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise ComfyError(f"ComfyUI {e.code} en {path}: {body[:2000]}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"No se puede conectar con ComfyUI en {self.server}: {e.reason}") from e

    def _json(self, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else None
        return json.loads(self._request(path, data, headers))

    # ------------------------------------------------------------------ API
    def object_info(self) -> dict:
        return self._json("/object_info")

    def system_stats(self) -> dict:
        return self._json("/system_stats")

    def upload_image(self, path: Path, subfolder: str = "aire") -> str:
        """Sube una imagen a input/<subfolder>/ y devuelve el valor para LoadImage."""
        boundary = uuid.uuid4().hex
        parts = []
        for key, val in (("overwrite", "true"), ("type", "input"), ("subfolder", subfolder)):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{val}\r\n'.encode())
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f"Content-Type: image/png\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        res = json.loads(self._request("/upload/image", b"".join(parts),
                                       {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def queue(self, graph: dict) -> str:
        res = self._json("/prompt", {"prompt": graph, "client_id": self.client_id})
        if res.get("node_errors"):
            raise ComfyError(f"ComfyUI rechazó el flujo: {json.dumps(res['node_errors'])[:2000]}")
        return res["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 1800.0, poll: float = 1.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = self._json(f"/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message", "") if msgs else ""
                    raise ComfyError(f"Error durante el render: {detail}")
                if status.get("completed", True):
                    return entry
            time.sleep(poll)
        raise ComfyError(f"Tiempo de espera agotado ({timeout:.0f}s)")

    def download(self, image: dict) -> bytes:
        q = urllib.parse.urlencode({"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                                    "type": image.get("type", "output")})
        return self._request(f"/view?{q}")

    def run(self, graph: dict, check: bool = True) -> list[bytes]:
        """Valida, encola, espera y devuelve las imágenes de salida (PNG)."""
        if check:
            problems = validate(graph, self.object_info())
            if problems:
                raise ComfyError("El flujo no es válido para este ComfyUI:\n- " + "\n- ".join(problems))
        entry = self.wait(self.queue(graph))
        images = []
        for out in entry.get("outputs", {}).values():
            for img in out.get("images", []):
                if img.get("type") == "output":
                    images.append(self.download(img))
        return images
