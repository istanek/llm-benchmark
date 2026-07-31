from __future__ import annotations

from typing import Protocol

from spark_benchmark.models import GenerationResult, InferenceMetrics, ModelConfig, SamplingConfig
from spark_benchmark.runners.capabilities import BackendCapabilities


class BackendAdapter(Protocol):
    capabilities: BackendCapabilities

    def load_model(self, model_config: ModelConfig) -> None: ...
    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult: ...
    def get_metrics(self) -> InferenceMetrics: ...
    def unload(self) -> None: ...
