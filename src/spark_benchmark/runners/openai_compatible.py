from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from spark_benchmark.models import BackendConfig, GenerationResult, InferenceMetrics, ModelConfig, SamplingConfig
from spark_benchmark.runners.capabilities import BackendCapabilities


class OpenAICompatibleAdapter:
    """Non-streaming OpenAI completions adapter for llama.cpp, vLLM and similar servers."""
    capabilities = BackendCapabilities(transport="http", supports_seed=True, native_metrics={"prefill_tokens", "decode_tokens"}, metric_limitations={"ttft_ms": "unavailable_from_non_streaming_completion"})

    def __init__(self, config: BackendConfig) -> None:
        self.config, self.model = config, None
        self.timeout_s = float(config.options.get("request_timeout_s") or 300)
        self.last_metrics = InferenceMetrics(backend_version=config.version)

    def load_model(self, model_config: ModelConfig) -> None:
        self.model = model_config
        self.last_metrics.quantization = model_config.quantization

    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        if self.model is None: raise RuntimeError("Model not loaded")
        endpoint = str(self.config.options.get("endpoint") or self.config.entrypoint).rstrip("/")
        if not endpoint.endswith("/completions"): endpoint += "/completions"
        payload = {"model": self.model.artifact_path or self.model.revision or self.model.name, "prompt": prompt, "max_tokens": params.max_tokens, "temperature": params.temperature, "top_p": params.top_p, "seed": params.seed, "stream": False}
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response: data = json.loads(response.read().decode())
        except urllib.error.URLError as exc: raise RuntimeError(f"OpenAI-compatible request failed to {endpoint}: {exc.reason}") from exc
        usage = data.get("usage") or {}
        self.last_metrics = InferenceMetrics(prefill_tokens=int(usage.get("prompt_tokens") or 0), decode_tokens=int(usage.get("completion_tokens") or 0), decode_time_s=time.perf_counter()-started, backend_version=self.config.version, quantization=self.model.quantization)
        choice = (data.get("choices") or [{}])[0]
        return GenerationResult(prompt=prompt, output=str(choice.get("text") or ""), finish_reason=str(choice.get("finish_reason") or "unknown"), metrics=self.last_metrics, raw={"endpoint": endpoint, "request": payload, "response": data})

    def get_metrics(self) -> InferenceMetrics: return self.last_metrics
    def unload(self) -> None: self.model = None
