from spark_benchmark.models import BackendConfig, BackendKind
from spark_benchmark.runners.ollama import OllamaAdapter
from spark_benchmark.runtime import build_environment_snapshot


def test_environment_snapshot_records_a_portable_hardware_inventory() -> None:
    snapshot = build_environment_snapshot(
        platform_config=None,
        backend_config=BackendConfig(
            name=BackendKind.OLLAMA,
            entrypoint="http://localhost:11434/api/generate",
            version="test",
        ),
    )

    hardware = snapshot.hardware
    assert hardware.schema_version == "hardware-inventory/v1"
    assert hardware.os_family
    assert hardware.architecture
    assert hardware.cpu.logical_cores >= 1
    assert hardware.memory.total_mb > 0
    assert hardware.accelerators == [] or hardware.accelerators[0].kind in {"gpu", "npu"}


def test_nvidia_device_with_unreported_memory_is_not_labeled_as_vram(monkeypatch) -> None:
    from spark_benchmark.hardware import inventory

    class Result:
        returncode = 0
        stdout = "NVIDIA GB10, N/A\n"

    monkeypatch.setattr(inventory.subprocess, "run", lambda *args, **kwargs: Result())

    device = inventory._nvidia_accelerators()[0]
    assert device.memory_mb is None
    assert device.memory_kind == "unknown"


def test_ollama_adapter_declares_normalization_capabilities() -> None:
    adapter = OllamaAdapter(
        BackendConfig(
            name=BackendKind.OLLAMA,
            entrypoint="http://localhost:11434/api/generate",
            version="test",
        )
    )

    assert adapter.capabilities.transport == "http"
    assert adapter.capabilities.supports_seed is True
    assert adapter.capabilities.native_metrics == {
        "prefill_tokens",
        "decode_tokens",
        "prefill_time_s",
        "decode_time_s",
    }
    assert adapter.capabilities.metric_limitations["ttft_ms"] == "estimated_from_prefill_time_s"


def test_every_builtin_backend_exposes_capabilities() -> None:
    from spark_benchmark.runners.registry import build_backend

    for backend_kind in (BackendKind.OLLAMA, BackendKind.LLAMACPP, BackendKind.VLLM, BackendKind.TRT_LLM):
        adapter = build_backend(
            BackendConfig(name=backend_kind, entrypoint="test", version="test")
        )
        assert adapter.capabilities.transport in {"http", "subprocess", "unavailable"}
