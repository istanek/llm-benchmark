from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any


class AmdSmiTelemetryCollector:
    """Best-effort AMD telemetry via amd-smi (or legacy rocm-smi)."""

    source = "amd-smi"
    capabilities = {"gpu_power_w", "gpu_temp_c", "gpu_memory_percent", "gpu_memory_mb"}

    def __init__(self) -> None:
        self.executable = shutil.which("amd-smi") or shutil.which("rocm-smi")

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        cleaned = value.strip().replace("%", "").replace("W", "").replace("C", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def parse_payload(self, payload: dict[str, Any]) -> dict[str, object]:
        values: dict[str, float] = {}
        for gpu_data in payload.values():
            if not isinstance(gpu_data, dict):
                continue
            for label, raw_value in gpu_data.items():
                value = self._number(raw_value)
                if value is None:
                    continue
                key = label.lower()
                if "temperature" in key and "gpu_temp_c" not in values:
                    values["gpu_temp_c"] = value
                elif "power" in key and "gpu_power_w" not in values:
                    values["gpu_power_w"] = value
                elif "memory" in key and "%" in label and "gpu_memory_percent" not in values:
                    values["gpu_memory_percent"] = value
                elif "memory" in key and ("mib" in key or "mb" in key) and "gpu_memory_mb" not in values:
                    values["gpu_memory_mb"] = value
        return {"collector": "amd", "source": "amd-smi", **values}

    def snapshot(self) -> dict[str, object]:
        if not self.executable:
            return {"collector": "amd", "source": "unavailable"}
        command = [self.executable, "metric", "--json"] if self.executable.endswith("amd-smi") else [self.executable, "--showtemp", "--showpower", "--json"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=3.0, check=False)
            if completed.returncode != 0:
                return {"collector": "amd", "source": "unavailable"}
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return {"collector": "amd", "source": "unavailable"}
        return self.parse_payload(payload if isinstance(payload, dict) else {})
