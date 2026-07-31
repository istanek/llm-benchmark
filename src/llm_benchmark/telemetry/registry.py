from __future__ import annotations

from typing import Protocol

from llm_benchmark.hardware import HardwareInventory
from llm_benchmark.telemetry.amd import AmdSmiTelemetryCollector
from llm_benchmark.telemetry.apple import ApplePowerMetricsCollector
from llm_benchmark.telemetry.nvidia import NvidiaTelemetryCollector
from llm_benchmark.telemetry.stub import StubTelemetryCollector


class TelemetryCollector(Protocol):
    capabilities: set[str]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def snapshot(self) -> dict[str, object]: ...


def build_telemetry_collector(hardware: HardwareInventory) -> TelemetryCollector:
    """Select a native provider only when its platform/vendor is explicit.

    NVIDIA is checked first: it is the only vendor that reports throttle
    reasons, which the sustained-throughput suite needs to explain a decode
    slowdown. A collector whose tooling is missing reports ``source == "none"``
    and the caller falls back accordingly.
    """
    vendors = {device.vendor.lower() for device in hardware.accelerators}
    if "nvidia" in vendors:
        collector = NvidiaTelemetryCollector()
        if collector.source != "none":
            return collector
    if hardware.os_family == "darwin" and "apple" in vendors:
        return ApplePowerMetricsCollector()
    if "amd" in vendors:
        return AmdSmiTelemetryCollector()
    return StubTelemetryCollector()


def build_collectors(names: list[str]) -> list[StubTelemetryCollector]:
    """Legacy named-collector API kept for existing callers."""
    return [StubTelemetryCollector(name) for name in names]
