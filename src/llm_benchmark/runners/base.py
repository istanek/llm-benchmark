from __future__ import annotations

from typing import Protocol

from llm_benchmark.models import GenerationResult, InferenceMetrics, ModelConfig, SamplingConfig
from llm_benchmark.runners.capabilities import BackendCapabilities


def ensure_model_supported(
    capabilities: BackendCapabilities, backend_name: str, model_config: ModelConfig
) -> None:
    """Refuse a model whose config asks for something this backend cannot do.

    Only ``reasoning`` today. The alternative — ignoring the field — is the
    failure mode this whole harness keeps running into: a run that completes,
    reports a number, and measured something other than what the config says.
    """
    if model_config.reasoning and not capabilities.supports_reasoning:
        raise ValueError(
            f"model {model_config.name!r} sets reasoning: true, but the {backend_name} "
            "backend cannot enable a reasoning pass. Run it on a backend that can, "
            "or drop the field and accept that reasoning is off."
        )


class BackendAdapter(Protocol):
    capabilities: BackendCapabilities

    def load_model(self, model_config: ModelConfig) -> None: ...
    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult: ...
    def get_metrics(self) -> InferenceMetrics: ...
    def unload(self) -> None: ...
