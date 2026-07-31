"""NVIDIA GPU telemetry: NVML when the bindings are importable, else nvidia-smi.

This lived inside ``sustained_throughput.TelemetrySampler`` until 0.6.1, which
left NVIDIA as the one vendor whose collection did not go through the
``telemetry`` package. The sampler now owns only threading and windowing.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

# NVML exposes throttle causes as a bitmask; these are the bits worth naming in
# a report. Missing attributes are skipped, so older bindings still work.
NVML_THROTTLE_BITS: list[tuple[str, str]] = [
    ("hw_slowdown", "nvmlClocksThrottleReasonHwSlowdown"),
    ("hw_thermal_slowdown", "nvmlClocksThrottleReasonHwThermalSlowdown"),
    ("hw_power_brake_slowdown", "nvmlClocksThrottleReasonHwPowerBrakeSlowdown"),
    ("sw_thermal_slowdown", "nvmlClocksThrottleReasonSwThermalSlowdown"),
    ("sw_power_cap", "nvmlClocksThrottleReasonSwPowerCap"),
    ("sync_boost", "nvmlClocksThrottleReasonSyncBoost"),
]

_SMI_QUERY = "power.draw,temperature.gpu,memory.used,clocks.gr"


def _maybe_float(value: str) -> float | None:
    """Parse an nvidia-smi field, tolerating the ``[N/A]`` it emits.

    Unified-memory parts (DGX Spark, Jetson) report ``memory.used`` as
    ``[N/A]`` — a missing field, not a zero, and it must not be charted as one.
    """
    cleaned = value.strip()
    if not cleaned or cleaned in {"[N/A]", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class NvidiaTelemetryCollector:
    """Snapshot-per-poll NVIDIA collector; degrades instead of raising."""

    capabilities = {"gpu_power_w", "gpu_temp_c", "gpu_memory_mb", "gpu_clock_mhz", "throttle_reasons"}

    def __init__(self) -> None:
        self._pynvml: Any | None = None
        self._handle: Any | None = None
        self.source = self._detect()

    def _detect(self) -> str:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return "nvml"
        except Exception:
            self._pynvml = None
            self._handle = None
        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        return "none"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        if self.source == "nvml" and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def snapshot(self) -> dict[str, object]:
        if self.source == "nvml":
            return self._snapshot_nvml()
        if self.source == "nvidia-smi":
            return self._snapshot_smi()
        return {"collector": "nvidia", "source": "unavailable"}

    def _snapshot_nvml(self) -> dict[str, object]:
        pynvml, handle = self._pynvml, self._handle
        if pynvml is None or handle is None:
            return {"collector": "nvidia", "source": "unavailable"}
        sample: dict[str, object] = {"collector": "nvidia", "source": "nvml"}
        try:
            sample["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass
        try:
            sample["gpu_temp_c"] = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass
        try:
            sample["gpu_memory_mb"] = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
        except Exception:
            pass
        try:
            sample["gpu_clock_mhz"] = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            )
        except Exception:
            pass
        try:
            flags = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            sample["throttle_reasons"] = [
                label
                for label, attr in NVML_THROTTLE_BITS
                if getattr(pynvml, attr, 0) and (flags & getattr(pynvml, attr, 0))
            ]
        except Exception:
            pass
        return sample

    def _snapshot_smi(self) -> dict[str, object]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={_SMI_QUERY}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                capture_output=True,
                timeout=2.0,
                text=True,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"collector": "nvidia", "source": "unavailable"}
        if completed.returncode != 0:
            return {"collector": "nvidia", "source": "unavailable"}
        return self.parse_smi_row(completed.stdout)

    @staticmethod
    def parse_smi_row(stdout: str) -> dict[str, object]:
        parts = [part.strip() for part in stdout.strip().split(",")]
        if len(parts) < 4:
            return {"collector": "nvidia", "source": "unavailable"}
        sample: dict[str, object] = {"collector": "nvidia", "source": "nvidia-smi"}
        for key, raw in zip(
            ("gpu_power_w", "gpu_temp_c", "gpu_memory_mb", "gpu_clock_mhz"), parts
        ):
            value = _maybe_float(raw)
            if value is not None:
                sample[key] = value
        return sample
