from llm_benchmark.hardware import AcceleratorInfo, CpuInfo, HardwareInventory, MemoryInfo
from llm_benchmark.telemetry.amd import AmdSmiTelemetryCollector
from llm_benchmark.telemetry.apple import ApplePowerMetricsCollector
from llm_benchmark.telemetry.registry import build_telemetry_collector
from llm_benchmark.telemetry.stub import StubTelemetryCollector


def _hardware(*, os_family: str, vendor: str) -> HardwareInventory:
    return HardwareInventory(
        os_family=os_family,
        architecture="arm64",
        cpu=CpuInfo(model="test", logical_cores=8),
        memory=MemoryInfo(total_mb=16000),
        accelerators=[AcceleratorInfo(kind="gpu", vendor=vendor, model="test")],
    )


def test_amd_hardware_selects_amd_smi_collector() -> None:
    collector = build_telemetry_collector(_hardware(os_family="linux", vendor="amd"))

    assert isinstance(collector, AmdSmiTelemetryCollector)
    assert {"gpu_power_w", "gpu_temp_c", "gpu_memory_mb"} <= collector.capabilities


def test_apple_hardware_selects_powermetrics_collector() -> None:
    collector = build_telemetry_collector(_hardware(os_family="darwin", vendor="apple"))

    assert isinstance(collector, ApplePowerMetricsCollector)
    assert collector.requires_privilege is True
    assert "system_power_w" in collector.capabilities


def test_unknown_hardware_uses_explicit_no_telemetry_collector() -> None:
    collector = build_telemetry_collector(_hardware(os_family="linux", vendor="other"))

    assert isinstance(collector, StubTelemetryCollector)
    assert collector.snapshot()["collector"] == "unavailable"


def test_amd_smi_collector_normalizes_json_metrics() -> None:
    collector = AmdSmiTelemetryCollector()
    sample = collector.parse_payload(
        {
            "gpu_0": {
                "Temperature (Sensor edge) (C)": "67.0",
                "Average Graphics Package Power (W)": "123.5",
                "GPU Memory Allocated (VRAM%)": "42",
                "GPU Memory Used (MiB)": "2048",
            }
        }
    )

    assert sample["gpu_temp_c"] == 67.0
    assert sample["gpu_power_w"] == 123.5
    assert sample["gpu_memory_percent"] == 42.0
    assert sample["gpu_memory_mb"] == 2048.0
    assert sample["source"] == "amd-smi"
