from llm_benchmark.hardware import AcceleratorInfo, CpuInfo, HardwareInventory, MemoryInfo
from llm_benchmark.sustained_throughput import TelemetrySampler
from llm_benchmark.telemetry.amd import AmdSmiTelemetryCollector
from llm_benchmark.telemetry.apple import ApplePowerMetricsCollector
from llm_benchmark.telemetry.nvidia import NvidiaTelemetryCollector
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


def test_nvidia_hardware_selects_nvidia_collector(monkeypatch) -> None:
    """NVIDIA is picked first — it is the only vendor reporting throttle reasons."""
    monkeypatch.setattr(
        NvidiaTelemetryCollector, "_detect", lambda self: "nvidia-smi", raising=True
    )
    collector = build_telemetry_collector(_hardware(os_family="linux", vendor="nvidia"))

    assert isinstance(collector, NvidiaTelemetryCollector)
    assert "throttle_reasons" in collector.capabilities


def test_nvidia_without_tooling_falls_through_to_the_stub(monkeypatch) -> None:
    monkeypatch.setattr(NvidiaTelemetryCollector, "_detect", lambda self: "none", raising=True)
    collector = build_telemetry_collector(_hardware(os_family="linux", vendor="nvidia"))

    assert isinstance(collector, StubTelemetryCollector)


def test_nvidia_smi_row_treats_na_memory_as_missing_not_zero() -> None:
    """Unified-memory parts report memory.used as [N/A]; 0 MB would be a lie."""
    sample = NvidiaTelemetryCollector.parse_smi_row("75.02, 74, [N/A], 1980\n")

    assert sample["gpu_power_w"] == 75.02
    assert sample["gpu_temp_c"] == 74.0
    assert sample["gpu_clock_mhz"] == 1980.0
    assert "gpu_memory_mb" not in sample
    assert sample["source"] == "nvidia-smi"


def test_sampler_carries_clock_and_throttle_reasons_from_the_collector() -> None:
    """Fields the NVML path used to fill in-line must survive the refactor."""

    class FakeNvidiaCollector:
        source = "nvml"
        capabilities = {"gpu_power_w", "throttle_reasons"}

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def snapshot(self) -> dict:
            return {
                "collector": "nvidia",
                "source": "nvml",
                "gpu_power_w": 91.0,
                "gpu_temp_c": 78.0,
                "gpu_memory_mb": 30000.0,
                "gpu_clock_mhz": 1500.0,
                "throttle_reasons": ["sw_power_cap", "hw_thermal_slowdown"],
            }

    sampler = TelemetrySampler(collector=FakeNvidiaCollector())
    sample = sampler._poll()

    assert sampler.source == "nvml"
    assert sample is not None
    assert sample.gpu_clock_mhz == 1500.0
    assert sample.throttle_reasons == ["sw_power_cap", "hw_thermal_slowdown"]
    assert sample.gpu_mem_used_mb == 30000.0


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
