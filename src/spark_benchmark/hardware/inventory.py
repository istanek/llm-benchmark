from __future__ import annotations

import os
import platform
import subprocess
from typing import Literal

from pydantic import BaseModel, Field


class CpuInfo(BaseModel):
    model: str = "unknown"
    logical_cores: int = 1


class MemoryInfo(BaseModel):
    total_mb: int = 0
    kind: Literal["system", "unified"] = "system"


class AcceleratorInfo(BaseModel):
    kind: Literal["gpu", "npu"]
    vendor: str
    model: str
    memory_mb: int | None = None
    memory_kind: Literal["vram", "unified", "shared", "unknown"] = "unknown"


class HardwareInventory(BaseModel):
    """Portable, best-effort description of the machine running a benchmark."""

    schema_version: Literal["hardware-inventory/v1"] = "hardware-inventory/v1"
    os_family: str
    architecture: str
    cpu: CpuInfo
    memory: MemoryInfo
    accelerators: list[AcceleratorInfo] = Field(default_factory=list)


def _memory_total_mb() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return max(0, int((page_size * page_count) / (1024 * 1024)))
    except (AttributeError, OSError, ValueError):
        return 0


def _nvidia_accelerators() -> list[AcceleratorInfo]:
    """Return NVIDIA GPUs when nvidia-smi is available; otherwise no devices."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    devices: list[AcceleratorInfo] = []
    for line in completed.stdout.splitlines():
        name, separator, memory = line.partition(",")
        if not separator or not name.strip():
            continue
        try:
            memory_mb = int(memory.strip())
        except ValueError:
            memory_mb = None
        devices.append(
            AcceleratorInfo(
                kind="gpu",
                vendor="nvidia",
                model=name.strip(),
                memory_mb=memory_mb,
                memory_kind="vram" if memory_mb is not None else "unknown",
            )
        )
    return devices


def _apple_accelerators(os_family: str, architecture: str) -> list[AcceleratorInfo]:
    if os_family != "darwin" or architecture not in {"arm64", "aarch64"}:
        return []
    return [
        AcceleratorInfo(
            kind="gpu",
            vendor="apple",
            model="Apple Silicon GPU",
            memory_kind="unified",
        )
    ]


def _amd_accelerators(os_family: str) -> list[AcceleratorInfo]:
    if os_family != "linux":
        return []
    try:
        completed = subprocess.run(
            ["lspci"], capture_output=True, check=False, text=True, timeout=3.0
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    devices: list[AcceleratorInfo] = []
    for line in completed.stdout.splitlines():
        if "[AMD/ATI]" not in line or "controller" not in line.lower():
            continue
        model = line.split("[AMD/ATI]", 1)[1].strip()
        if model:
            devices.append(
                AcceleratorInfo(
                    kind="gpu", vendor="amd", model=model, memory_kind="unknown"
                )
            )
    return devices


def discover_hardware() -> HardwareInventory:
    """Discover portable host facts without requiring optional vendor libraries."""
    os_family = platform.system().lower() or "unknown"
    architecture = platform.machine().lower() or "unknown"
    accelerators: list[AcceleratorInfo] = []
    if os_family == "darwin":
        accelerators.extend(_apple_accelerators(os_family, architecture))
    elif os_family == "linux":
        accelerators.extend(_nvidia_accelerators())
        accelerators.extend(_amd_accelerators(os_family))
    return HardwareInventory(
        os_family=os_family,
        architecture=architecture,
        cpu=CpuInfo(
            model=platform.processor() or "unknown",
            logical_cores=max(1, os.cpu_count() or 1),
        ),
        memory=MemoryInfo(total_mb=_memory_total_mb()),
        accelerators=accelerators,
    )
