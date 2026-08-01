from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .hardware import HardwareInventory


class BackendKind(str, Enum):
    LLAMACPP = "llamacpp"
    TRT_LLM = "trt-llm"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai-compatible"


class SamplingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_tokens: int = 2048
    # Optional explicit context-window size (Ollama `options.num_ctx`).
    # None = let the backend use its default. The long_context suite sets
    # this per request so a long prompt actually loads instead of being
    # silently truncated to the server default.
    num_ctx: int | None = None


class ExperimentSpec(BaseModel):
    name: str
    description: str
    platforms: list[str]
    backend: BackendKind
    backend_version: str
    models: list[str]
    suites: list[str]
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    context_lengths: list[int] = Field(default_factory=lambda: [512, 4096, 16384, 65536])
    repetitions: int = 3
    warmup_runs: int = 1

    @model_validator(mode="after")
    def validate_lists(self) -> "ExperimentSpec":
        for field_name in ("platforms", "models", "suites", "context_lengths"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        return self


class ExperimentFile(BaseModel):
    experiment: ExperimentSpec


class PlatformConfig(BaseModel):
    name: str
    display_name: str
    architecture: str
    os: str
    telemetry: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    name: str
    family: str
    revision: str
    quantization: str
    source: str
    context_length: int
    artifact_path: str | None = None
    # Optional grouping key linking quantization variants of the same base
    # model (e.g. "llama-3.3-70b"). Set explicitly in YAML — never inferred
    # from `name` (brittle for odd names). Primarily consumed by the
    # quantization_sweep post-processor; long_context uses it for labels.
    base_model: str | None = None
    # Backend "thinking" / reasoning mode. None keeps the harness default
    # (off), which is how the v1 lineup was measured — leaving it alone means
    # existing numbers stay comparable. Set True for a model whose answer is
    # only produced after a reasoning pass; the backend must support it
    # (BackendCapabilities.supports_reasoning), otherwise the run fails loudly
    # rather than silently measuring a different mode than the config asks for.
    reasoning: bool | None = None
    # Per-model output budget, overriding the experiment's sampling.max_tokens
    # for this model alone. A reasoning model can spend the whole shared budget
    # on its scratchpad and get truncated mid-answer — scored, wrongly, as a
    # model that cannot code. The experiment config sets a budget that suits
    # the suite; this raises it for the models that need more.
    max_output_tokens: int | None = None
    notes: list[str] = Field(default_factory=list)


def sampling_for_model(sampling: SamplingConfig, model: ModelConfig | None) -> SamplingConfig:
    """Apply a model's per-model overrides on top of the experiment sampling.

    Kept here rather than in each backend so every adapter resolves the budget
    the same way, and so the returned config is the one recorded in results —
    an override that does not show up in the manifest is a reproducibility bug.
    """
    if model is None or model.max_output_tokens is None:
        return sampling
    return sampling.model_copy(update={"max_tokens": model.max_output_tokens})


class BackendConfig(BaseModel):
    name: BackendKind
    entrypoint: str
    version: str
    transport: str = "subprocess"
    executable: str | None = None
    default_args: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class InferenceMetrics(BaseModel):
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    ttft_ms: float = 0.0
    peak_memory_mb: float = 0.0
    backend_version: str = ""
    quantization: str = ""


# ``finish_reason`` values meaning the model hit the token budget rather than
# finishing. Backends spell it differently, hence a set.
TRUNCATION_FINISH_REASONS = {"length", "max_tokens", "truncated", "incomplete"}


def is_truncated(finish_reason: str | None) -> bool:
    return str(finish_reason or "").strip().lower() in TRUNCATION_FINISH_REASONS


class GenerationResult(BaseModel):
    prompt: str
    output: str
    finish_reason: str = "unknown"
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)
    raw: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSnapshot(BaseModel):
    platform_name: str
    backend_name: str
    backend_version: str
    python_version: str
    os: str
    hostname: str
    hardware: HardwareInventory
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    experiment: ExperimentSpec
    platform: PlatformConfig
    backend: BackendConfig
    model_names: list[str]
    environment: EnvironmentSnapshot
    results_dir: Path
