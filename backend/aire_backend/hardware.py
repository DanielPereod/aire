"""Detección de hardware y recomendación de modelos.

    python -m aire_backend.hardware
"""

from __future__ import annotations

import ctypes
import json
import platform
import shutil
import subprocess

from .models import PRESETS, recommend, weight_dtype_for


def gpus() -> list[dict]:
    """GPUs NVIDIA vía nvidia-smi (viene con el driver en Windows y Linux)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in out.strip().splitlines():
        name, mem, driver = [s.strip() for s in line.split(",")]
        result.append({"name": name, "vram_gb": round(int(mem) / 1024, 1), "driver": driver})
    return result


def ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 2**30, 1)
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
        return round(kb / 2**20, 1)
    except (OSError, ValueError, AttributeError):
        return None


def report() -> dict:
    g = gpus()
    vram = max((x["vram_gb"] for x in g), default=None)
    ram = ram_gb()
    rec = recommend(vram)
    warnings = []
    if vram is None:
        warnings.append("No se detecta GPU NVIDIA: se recomienda usar la nube.")
    elif vram < 16 and ram is not None and ram < 32:
        warnings.append(f"Con {vram} GB de VRAM, ComfyUI descarga parte del modelo a RAM; "
                        f"con {ram} GB de RAM puede ir lento o fallar (recomendado 32 GB).")
    return {
        "os": f"{platform.system()} {platform.release()}",
        "gpus": g,
        "ram_gb": ram,
        "recommended": rec,
        "zimage_weight_dtype": weight_dtype_for(vram),
        "warnings": warnings,
    }


def main() -> None:
    r = report()
    print(json.dumps(r, indent=2, ensure_ascii=False))
    for role, pid in r["recommended"].items():
        print(f"\n{role}: {PRESETS[pid].title if pid else 'nube'}")


if __name__ == "__main__":
    main()
