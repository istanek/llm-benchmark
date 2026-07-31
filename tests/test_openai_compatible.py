import json

from llm_benchmark.models import BackendConfig, BackendKind, ModelConfig, SamplingConfig
from llm_benchmark.runners.registry import build_backend


def _backend_config(**options) -> BackendConfig:
    return BackendConfig(
        name=BackendKind.OPENAI_COMPATIBLE,
        entrypoint="http://localhost:8888/v1",
        version="test",
        options=options,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="laguna",
        family="laguna",
        revision="Laguna-S-2.1",
        quantization="iq4",
        source="local",
        context_length=131072,
        artifact_path="Laguna-S-2.1",
    )


class _NonStreamingResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _StreamingResponse:
    """Yields raw SSE lines the way a llama.cpp / vLLM server does."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __iter__(self):
        for event in self.events:
            yield event.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_non_streaming_posts_completion_and_normalizes_usage(monkeypatch) -> None:
    from llm_benchmark.runners.openai_compatible import OpenAICompatibleAdapter

    captured = {}

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return _NonStreamingResponse(
            {
                "choices": [{"text": "ok", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
        )

    monkeypatch.setattr("llm_benchmark.runners.openai_compatible.urllib.request.urlopen", urlopen)
    adapter = OpenAICompatibleAdapter(_backend_config(stream=False))
    adapter.load_model(_model_config())
    result = adapter.generate("hello", SamplingConfig(max_tokens=64, temperature=0.0, seed=42))

    assert captured["url"] == "http://localhost:8888/v1/completions"
    assert captured["payload"]["model"] == "Laguna-S-2.1"
    assert captured["payload"]["stream"] is False
    assert result.output == "ok"
    assert result.metrics.prefill_tokens == 12
    assert result.metrics.decode_tokens == 3


def test_non_streaming_declares_that_decode_time_is_a_round_trip() -> None:
    """The single-shot path cannot measure decode; it must say so."""
    from llm_benchmark.runners.openai_compatible import OpenAICompatibleAdapter

    adapter = OpenAICompatibleAdapter(_backend_config(stream=False))
    assert adapter.capabilities.supports_streaming is False
    assert "ttft_ms" in adapter.capabilities.metric_limitations
    assert (
        adapter.capabilities.metric_limitations["decode_time_s"]
        == "round_trip_only_includes_prefill_and_network"
    )


def test_streaming_is_the_default_and_separates_ttft_from_decode(monkeypatch) -> None:
    """decode_time_s must exclude the wait for the first token."""
    from llm_benchmark.runners import openai_compatible
    from llm_benchmark.runners.openai_compatible import OpenAICompatibleAdapter

    captured = {}

    def urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return _StreamingResponse(
            [
                'data: {"choices":[{"text":"Hel"}]}\n',
                'data: {"choices":[{"text":"lo"}]}\n',
                'data: {"choices":[{"text":"","finish_reason":"stop"}]}\n',
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4}}\n',
                "data: [DONE]\n",
            ]
        )

    # Deterministic clock: start=0.0, first token at 1.0, end at 3.0.
    clock = iter([0.0, 1.0, 3.0])
    monkeypatch.setattr(openai_compatible.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr("llm_benchmark.runners.openai_compatible.urllib.request.urlopen", urlopen)

    adapter = OpenAICompatibleAdapter(_backend_config())
    assert adapter.stream is True
    adapter.load_model(_model_config())
    result = adapter.generate("hello", SamplingConfig(max_tokens=64, temperature=0.0, seed=42))

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert result.output == "Hello"
    assert result.finish_reason == "stop"
    assert result.metrics.ttft_ms == 1000.0
    # 3.0 - 1.0, NOT the 3.0 round trip: the old adapter published the round
    # trip as decode_time_s, which understated tok/s by the prefill share.
    assert result.metrics.decode_time_s == 2.0
    assert result.metrics.decode_tokens == 4
    assert result.metrics.prefill_tokens == 12
    assert result.raw["usage_reported"] is True


def test_streaming_falls_back_to_chunk_count_when_usage_is_absent(monkeypatch) -> None:
    from llm_benchmark.runners.openai_compatible import OpenAICompatibleAdapter

    def urlopen(request, timeout):
        return _StreamingResponse(
            [
                'data: {"choices":[{"text":"a"}]}\n',
                'data: {"choices":[{"text":"b"}]}\n',
                'data: {"choices":[{"text":"c","finish_reason":"length"}]}\n',
                "data: [DONE]\n",
            ]
        )

    monkeypatch.setattr("llm_benchmark.runners.openai_compatible.urllib.request.urlopen", urlopen)
    adapter = OpenAICompatibleAdapter(_backend_config())
    adapter.load_model(_model_config())
    result = adapter.generate("hello", SamplingConfig(max_tokens=64))

    assert result.output == "abc"
    assert result.metrics.decode_tokens == 3
    assert result.raw["usage_reported"] is False
    assert (
        adapter.capabilities.metric_limitations["decode_tokens"]
        == "from_usage_when_reported_else_counted_stream_chunks"
    )


def test_streaming_handles_chat_style_delta_chunks(monkeypatch) -> None:
    from llm_benchmark.runners.openai_compatible import OpenAICompatibleAdapter

    def urlopen(request, timeout):
        return _StreamingResponse(
            [
                'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
                ": keepalive\n",
                "\n",
                "data: [DONE]\n",
            ]
        )

    monkeypatch.setattr("llm_benchmark.runners.openai_compatible.urllib.request.urlopen", urlopen)
    adapter = OpenAICompatibleAdapter(_backend_config())
    adapter.load_model(_model_config())
    assert adapter.generate("hello", SamplingConfig()).output == "hi"


def test_registry_builds_openai_compatible_adapter() -> None:
    adapter = build_backend(_backend_config())
    assert adapter.capabilities.transport == "http"
    assert adapter.capabilities.supports_streaming is True
