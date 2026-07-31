from pathlib import Path

from llm_benchmark.config import load_backend, load_experiment, load_model_config
from llm_benchmark.models import BackendKind
from llm_benchmark.runners.openai_compatible import OpenAICompatibleAdapter
from llm_benchmark.runners.registry import build_backend


def test_laguna_experiment_resolves_to_openai_compatible_backend() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(root / "configs/experiments/laguna-smoke.yaml").experiment
    backend = load_backend(root / "configs/backends/openai-compatible.yaml")
    model = load_model_config(root / "configs/models/laguna-s-2.1-iq4-xs.yaml")

    assert experiment.backend is BackendKind.OPENAI_COMPATIBLE
    assert experiment.models == ["laguna-s-2.1-iq4-xs"]
    assert backend.options["endpoint"] == "http://localhost:8888/v1"
    assert model.artifact_path == "Laguna-S-2.1"
    assert isinstance(build_backend(backend), OpenAICompatibleAdapter)
