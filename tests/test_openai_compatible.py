import json

from spark_benchmark.models import BackendConfig, BackendKind, ModelConfig, SamplingConfig
from spark_benchmark.runners.registry import build_backend


def test_openai_compatible_adapter_posts_completion_and_normalizes_usage(monkeypatch) -> None:
    from spark_benchmark.runners.openai_compatible import OpenAICompatibleAdapter

    captured = {}

    class Response:
        def read(self):
            return json.dumps({"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}).encode()
        def __enter__(self): return self
        def __exit__(self, *args): return False

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("spark_benchmark.runners.openai_compatible.urllib.request.urlopen", urlopen)
    adapter = OpenAICompatibleAdapter(BackendConfig(name=BackendKind.OPENAI_COMPATIBLE, entrypoint="http://localhost:8888/v1", version="test"))
    adapter.load_model(ModelConfig(name="laguna", family="laguna", revision="Laguna-S-2.1", quantization="iq4", source="local", context_length=131072, artifact_path="Laguna-S-2.1"))
    result = adapter.generate("hello", SamplingConfig(max_tokens=64, temperature=0.0, seed=42))
    assert captured["url"] == "http://localhost:8888/v1/completions"
    assert captured["payload"]["model"] == "Laguna-S-2.1"
    assert result.output == "ok"
    assert result.metrics.prefill_tokens == 12
    assert result.metrics.decode_tokens == 3


def test_registry_builds_openai_compatible_adapter() -> None:
    adapter = build_backend(BackendConfig(name=BackendKind.OPENAI_COMPATIBLE, entrypoint="http://localhost:8888/v1", version="test"))
    assert adapter.capabilities.transport == "http"
