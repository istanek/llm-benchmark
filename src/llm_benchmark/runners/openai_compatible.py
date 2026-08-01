from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from llm_benchmark.models import (
    BackendConfig,
    GenerationResult,
    InferenceMetrics,
    ModelConfig,
    SamplingConfig,
    sampling_for_model,
)
from llm_benchmark.runners.base import ensure_model_supported
from llm_benchmark.runners.capabilities import BackendCapabilities


# Capabilities depend on the transport actually used, so they are declared per
# mode and picked in ``__init__`` rather than being a single class attribute.
STREAMING_CAPABILITIES = BackendCapabilities(
    transport="http",
    supports_seed=True,
    supports_streaming=True,
    native_metrics={"prefill_tokens", "decode_tokens", "ttft_ms", "decode_time_s"},
    metric_limitations={
        "decode_tokens": "from_usage_when_reported_else_counted_stream_chunks",
    },
)

NON_STREAMING_CAPABILITIES = BackendCapabilities(
    transport="http",
    supports_seed=True,
    native_metrics={"prefill_tokens", "decode_tokens"},
    metric_limitations={
        "ttft_ms": "unavailable_from_non_streaming_completion",
        # The only clock available is the round trip, which covers queueing,
        # prefill, decode and network. Reporting divides decode_tokens by
        # decode_time_s, so a round trip published here understates tok/s.
        "decode_time_s": "round_trip_only_includes_prefill_and_network",
    },
)


class OpenAICompatibleAdapter:
    """OpenAI completions adapter for llama.cpp, vLLM and similar servers.

    Streams by default: a non-streaming completion cannot separate prefill from
    decode, so it yields neither a true TTFT nor a decode rate. Set
    ``options.stream: false`` to force the single-shot path, which reports the
    round trip and declares that limitation in ``capabilities``.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model: ModelConfig | None = None
        self.timeout_s = float(config.options.get("request_timeout_s") or 300)
        stream_option = config.options.get("stream")
        self.stream = True if stream_option is None else bool(stream_option)
        self.capabilities = STREAMING_CAPABILITIES if self.stream else NON_STREAMING_CAPABILITIES
        self.last_metrics = InferenceMetrics(backend_version=config.version)

    def load_model(self, model_config: ModelConfig) -> None:
        ensure_model_supported(self.capabilities, "openai-compatible", model_config)
        self.model = model_config
        self.last_metrics.quantization = model_config.quantization

    # --- request plumbing ------------------------------------------------

    def endpoint(self) -> str:
        endpoint = str(self.config.options.get("endpoint") or self.config.entrypoint).rstrip("/")
        if not endpoint.endswith("/completions"):
            endpoint += "/completions"
        return endpoint

    def _payload(self, prompt: str, params: SamplingConfig, stream: bool) -> dict[str, Any]:
        assert self.model is not None
        payload: dict[str, Any] = {
            "model": self.model.artifact_path or self.model.revision or self.model.name,
            "prompt": prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed,
            "stream": stream,
        }
        if stream:
            # Servers honouring this send a final usage-only chunk, which is
            # more trustworthy than counting chunks ourselves.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _open(self, payload: dict[str, Any]) -> Any:
        endpoint = self.endpoint()
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_s)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible request failed to {endpoint}: {exc.reason}"
            ) from exc

    # --- generation ------------------------------------------------------

    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        params = sampling_for_model(params, self.model)
        if self.stream:
            return self._generate_streaming(prompt, params)
        return self._generate_once(prompt, params)

    def _generate_streaming(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        payload = self._payload(prompt, params, stream=True)
        pieces: list[str] = []
        chunk_count = 0
        finish_reason = "unknown"
        usage: dict[str, Any] = {}
        first_token_at: float | None = None
        started = time.perf_counter()

        with self._open(payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunk_count += 1
                if event.get("usage"):
                    usage = event["usage"]
                choice = (event.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if not text:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(str(text))

        ended = time.perf_counter()
        ttft_s = (first_token_at - started) if first_token_at is not None else 0.0
        # Decode time excludes the wait for the first token, so
        # decode_tokens / decode_time_s is a decode rate, not an end-to-end rate.
        decode_time_s = (ended - first_token_at) if first_token_at is not None else 0.0
        decode_tokens = int(usage.get("completion_tokens") or 0) or len(pieces)

        self.last_metrics = InferenceMetrics(
            prefill_tokens=int(usage.get("prompt_tokens") or 0),
            decode_tokens=decode_tokens,
            prefill_time_s=ttft_s,
            decode_time_s=decode_time_s,
            ttft_ms=ttft_s * 1000.0,
            backend_version=self.config.version,
            quantization=self.model.quantization if self.model else "",
        )
        return GenerationResult(
            prompt=prompt,
            output="".join(pieces),
            finish_reason=finish_reason,
            metrics=self.last_metrics,
            raw={
                "endpoint": self.endpoint(),
                "request": payload,
                "stream_chunks": chunk_count,
                "usage": usage,
                "usage_reported": bool(usage),
            },
        )

    def _generate_once(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        payload = self._payload(prompt, params, stream=False)
        started = time.perf_counter()
        with self._open(payload) as response:
            data = json.loads(response.read().decode())
        round_trip_s = time.perf_counter() - started

        usage = data.get("usage") or {}
        self.last_metrics = InferenceMetrics(
            prefill_tokens=int(usage.get("prompt_tokens") or 0),
            decode_tokens=int(usage.get("completion_tokens") or 0),
            # No TTFT is observable here, so the round trip is the only clock.
            # See NON_STREAMING_CAPABILITIES.metric_limitations — this is not a
            # pure decode time, and the tok/s derived from it is a lower bound.
            decode_time_s=round_trip_s,
            backend_version=self.config.version,
            quantization=self.model.quantization if self.model else "",
        )
        choice = (data.get("choices") or [{}])[0]
        return GenerationResult(
            prompt=prompt,
            output=str(choice.get("text") or ""),
            finish_reason=str(choice.get("finish_reason") or "unknown"),
            metrics=self.last_metrics,
            raw={"endpoint": self.endpoint(), "request": payload, "response": data},
        )

    def get_metrics(self) -> InferenceMetrics:
        return self.last_metrics

    def unload(self) -> None:
        self.model = None
