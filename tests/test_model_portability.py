"""Per-model backend options: reasoning mode and output budget.

These exist because the harness measured three models and hardcoded what
those three needed. `think: False` in the Ollama payload and one shared
`max_tokens` are both fine for a non-reasoning lineup and both silently wrong
for anything else — the run completes and reports a number either way, which
is the failure mode this repo keeps hitting.
"""

from __future__ import annotations

import pytest

from llm_benchmark.models import BackendConfig, ModelConfig, SamplingConfig, sampling_for_model
from llm_benchmark.runners.base import ensure_model_supported
from llm_benchmark.runners.capabilities import BackendCapabilities
from llm_benchmark.runners.ollama import OllamaAdapter


def _model(**overrides) -> ModelConfig:
    base = dict(
        name="probe",
        family="probe",
        revision="probe:latest",
        quantization="ollama-default",
        source="ollama-local",
        context_length=8192,
        artifact_path="probe:latest",
    )
    base.update(overrides)
    return ModelConfig(**base)


def _adapter() -> OllamaAdapter:
    return OllamaAdapter(BackendConfig(name="ollama", entrypoint="ollama", version="test"))


def test_reasoning_defaults_to_off() -> None:
    """Unset means off, so the v1 lineup's recorded numbers stay comparable."""
    adapter = _adapter()
    adapter.load_model(_model())
    assert adapter._build_payload("hi", SamplingConfig())["think"] is False


def test_reasoning_is_forwarded_when_the_model_asks_for_it() -> None:
    adapter = _adapter()
    adapter.load_model(_model(reasoning=True))
    assert adapter._build_payload("hi", SamplingConfig())["think"] is True


def test_reasoning_on_a_backend_without_it_is_refused() -> None:
    """Loudly, rather than measuring the model with reasoning off."""
    capabilities = BackendCapabilities(transport="subprocess", supports_reasoning=False)
    with pytest.raises(ValueError, match="reasoning"):
        ensure_model_supported(capabilities, "llamacpp", _model(reasoning=True))


def test_backend_without_reasoning_accepts_a_model_that_does_not_ask() -> None:
    capabilities = BackendCapabilities(transport="subprocess", supports_reasoning=False)
    ensure_model_supported(capabilities, "llamacpp", _model())
    ensure_model_supported(capabilities, "llamacpp", _model(reasoning=False))


def test_per_model_budget_overrides_the_experiment_budget() -> None:
    sampling = SamplingConfig(max_tokens=1536)
    assert sampling_for_model(sampling, _model(max_output_tokens=8192)).max_tokens == 8192


def test_per_model_budget_is_optional() -> None:
    sampling = SamplingConfig(max_tokens=1536)
    assert sampling_for_model(sampling, _model()).max_tokens == 1536
    assert sampling_for_model(sampling, None).max_tokens == 1536


def test_override_does_not_mutate_the_shared_sampling_config() -> None:
    """Every model in a run shares one SamplingConfig instance."""
    sampling = SamplingConfig(max_tokens=1536)
    sampling_for_model(sampling, _model(max_output_tokens=8192))
    assert sampling.max_tokens == 1536


def test_the_budget_the_backend_sends_is_the_overridden_one() -> None:
    """The recorded request must show the budget actually used, not the shared one."""
    adapter = _adapter()
    adapter.load_model(_model(max_output_tokens=8192))
    payload = adapter._build_payload("hi", sampling_for_model(SamplingConfig(max_tokens=1536), adapter.model))
    assert payload["options"]["num_predict"] == 8192
