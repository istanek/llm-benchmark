from __future__ import annotations

from typing import Protocol

from llm_benchmark.hardware import HardwareInventory
from llm_benchmark.telemetry.amd import AmdSmiTelemetryCollector
from llm_benchmark.telemetry.apple import ApplePowerMetricsCollector
from llm_benchmark.telemetry.stub import StubTelemetryCollector


class TelemetryCollector(Protocol):
    capabilities: set[str]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def snapshot(self) -> dict[str, object]: ...


def build_telemetry_collector(hardware: HardwareInventory) -> TelemetryCollector:
    """Select a native provider only when its platform/vendor is explicit."""
    vendors = {device.vendor.lower() for device in hardware.accelerators}
    if hardware.os_family == "darwin" and "apple" in vendors:
        return ApplePowerMetricsCollector()
    if "amd" in vendors:
        return AmdSmiTelemetryCollector()
    return StubTelemetryCollector()


def build_collectors(names: list[str]) -> list[StubTelemetryCollector]:
    """Legacy named-collector API kept for existing callers."""
    return [StubTelemetryCollector(name) for name in names]
