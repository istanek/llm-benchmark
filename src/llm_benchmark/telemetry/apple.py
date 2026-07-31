from __future__ import annotations

import re
import shutil
import subprocess


class ApplePowerMetricsCollector:
    """Optional system-level Apple Silicon telemetry; never invokes sudo itself."""

    source = "powermetrics"
    capabilities = {"system_power_w", "cpu_power_w", "gpu_power_w", "thermal_pressure"}
    requires_privilege = True

    def __init__(self) -> None:
        self.executable = shutil.which("powermetrics")

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        if not self.executable:
            return {"collector": "apple", "source": "unavailable", "requires_privilege": True}
        try:
            completed = subprocess.run(
                [self.executable, "--samplers", "smc", "-n", "1", "-i", "1000"],
                capture_output=True, text=True, timeout=5.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"collector": "apple", "source": "unavailable", "requires_privilege": True}
        if completed.returncode != 0:
            return {"collector": "apple", "source": "unavailable", "requires_privilege": True}
        sample: dict[str, object] = {"collector": "apple", "source": "powermetrics", "requires_privilege": True}
        match = re.search(r"System Power:\s*([0-9.]+)\s*W", completed.stdout, flags=re.IGNORECASE)
        if match:
            sample["system_power_w"] = float(match.group(1))
        return sample
