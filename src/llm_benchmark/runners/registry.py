from __future__ import annotations

from llm_benchmark.models import BackendConfig, BackendKind
from llm_benchmark.runners.llamacpp import LlamaCppAdapter
from llm_benchmark.runners.ollama import OllamaAdapter
from llm_benchmark.runners.openai_compatible import OpenAICompatibleAdapter
from llm_benchmark.runners.stub import StubBackendAdapter


def build_backend(config: BackendConfig) -> StubBackendAdapter | LlamaCppAdapter | OllamaAdapter:
    if config.name == BackendKind.LLAMACPP:
        return LlamaCppAdapter(config)
    if config.name == BackendKind.OLLAMA:
        return OllamaAdapter(config)
    if config.name == BackendKind.OPENAI_COMPATIBLE:
        return OpenAICompatibleAdapter(config)
    return StubBackendAdapter(config)
