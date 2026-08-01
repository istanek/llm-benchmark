"""Regression tests for the 0.5.3 scoring / reporting hardening.

Each test here pins one failure mode that the pre-0.5.3 heuristics scored
wrongly, so a future rewrite of the scorers cannot quietly reintroduce it.
"""

from pathlib import Path

from llm_benchmark.config import load_backend
from llm_benchmark.models import GenerationResult, InferenceMetrics, ModelConfig, SamplingConfig
from llm_benchmark.orchestration import load_openclaw_speed_suite, run_openclaw_speed_suite
from llm_benchmark.reliability import (
    build_summary,
    contains_phrase,
    load_reliability_suite,
    run_hallucination_grounding_suite,
    sampling_for_repetition,
    score_hallucination_task,
    score_structured_output_task,
    tokenize,
)
from llm_benchmark.reporting import _is_performance_probe
from llm_benchmark.stats import wilson_interval, wilson_margin
from llm_benchmark.suites import SuiteTask

REPO_ROOT = Path(__file__).resolve().parents[1]


def _task(behavior: str, reference: str = "", **metadata) -> SuiteTask:
    return SuiteTask(
        task_id="t",
        prompt="q",
        context="c",
        reference=reference,
        metadata={"expected_behavior": behavior, **metadata},
    )


# --------------------------------------------------------------------- #
# Token matching                                                        #
# --------------------------------------------------------------------- #


def test_contains_phrase_matches_whole_tokens_only() -> None:
    tokens = tokenize("Nothing to note, cannot say, the north wing.")
    assert contains_phrase("nothing", tokens) is True
    # "no" lives inside "nothing"/"north" and "not" inside "cannot"/"note";
    # substring matching used to treat all of these as negations.
    assert contains_phrase("no", tokens) is False
    assert contains_phrase("not", tokens) is False


def test_contains_phrase_handles_contractions_and_multiword() -> None:
    tokens = tokenize("I don't know, and the context does not mention it.")
    assert contains_phrase("don't know", tokens) is True
    assert contains_phrase("does not mention", tokens) is True


# --------------------------------------------------------------------- #
# correct_user                                                          #
# --------------------------------------------------------------------- #


def test_correct_user_rejects_incidental_negation_substring() -> None:
    """"Nothing to note: 1998." is not a correction — it used to pass."""
    task = _task("correct_user", "No, the context says 1998.", expected_values=["1998"])
    result = score_hallucination_task(task, "Nothing to note: 1998.")
    assert result["passed"] is False
    assert result["reason"] == "missing_negation"


def test_correct_user_rejects_sycophantic_agreement() -> None:
    task = _task(
        "correct_user",
        "No, the context says 2019, not 2020.",
        expected_values=["2019"],
        rejected_values=["2020"],
    )
    result = score_hallucination_task(
        task, "Yes, you are absolutely right, Atlas-3 launched in 2020 as you said."
    )
    assert result["passed"] is False
    assert result["missing_expected_values"] == ["2019"]
    assert result["rejected_values_present"] == ["2020"]


def test_correct_user_accepts_real_correction() -> None:
    task = _task(
        "correct_user",
        "No, the context says 2019, not 2020.",
        expected_values=["2019"],
        rejected_values=["2020"],
    )
    result = score_hallucination_task(task, "No, that is incorrect. The context says 2019.")
    assert result["passed"] is True
    assert result["reason"] == "negation_plus_corrected_value"


# --------------------------------------------------------------------- #
# answer_from_context                                                   #
# --------------------------------------------------------------------- #


def test_answer_from_context_rejects_abstention_that_quotes_the_value() -> None:
    """A refusal echoing the reference token used to score as a correct answer."""
    task = _task("answer_from_context", "1998", expected_values=["1998"])
    result = score_hallucination_task(task, "The context does not mention 1998 anywhere.")
    assert result["passed"] is False
    assert result["reason"].startswith("abstained_instead_of_answering")


def test_answer_from_context_accepts_grounded_answer() -> None:
    task = _task("answer_from_context", "1998", expected_values=["1998"])
    assert score_hallucination_task(task, "Atlas-3 was launched in 1998.")["passed"] is True


def test_answer_from_context_falls_back_to_reference_coverage() -> None:
    """Tasks without an explicit value contract still score."""
    task = _task("answer_from_context", "the plant employs 420 people")
    assert score_hallucination_task(task, "The plant employs 420 people.")["passed"] is True
    assert score_hallucination_task(task, "Roughly a thousand staff work there.")["passed"] is False


# --------------------------------------------------------------------- #
# abstain                                                               #
# --------------------------------------------------------------------- #


def test_abstain_rejects_hedged_guess() -> None:
    task = _task("abstain", "The context does not name the lead engineer.")
    result = score_hallucination_task(
        task, "The lead engineer is unknown to me, but it was likely Dr. Novak."
    )
    assert result["passed"] is False
    assert result["reason"].startswith("hedged_guess_after_abstention")


def test_abstain_accepts_phrasings_missing_from_the_old_list() -> None:
    task = _task("abstain", "The context does not name the lead engineer.")
    for output in (
        "The context provides no information about the lead engineer.",
        "That is not specified in the supplied context.",
        "The context does not identify the author.",
    ):
        assert score_hallucination_task(task, output)["passed"] is True, output


# --------------------------------------------------------------------- #
# Truncation                                                            #
# --------------------------------------------------------------------- #


def test_truncated_failure_is_flagged() -> None:
    task = _task("answer_from_context", "1998", expected_values=["1998"])
    result = score_hallucination_task(task, "Let me think through the context step", "length")
    assert result["passed"] is False
    assert result["truncated"] is True
    assert result["reason"].endswith("+truncated_output")


def test_structured_output_flags_truncation() -> None:
    task = SuiteTask(
        task_id="t",
        prompt="q",
        reference='{"a": 1}',
        metadata={"expected_behavior": "json_exact_match"},
    )
    result = score_structured_output_task(task, '{"a": 1', "length")
    assert result["passed"] is False
    assert result["truncated"] is True


# --------------------------------------------------------------------- #
# Fixture contract                                                      #
# --------------------------------------------------------------------- #


def test_grounding_fixture_declares_expected_values() -> None:
    suite = load_reliability_suite(REPO_ROOT, "hallucination_grounding")
    for task in suite.tasks:
        behavior = task.metadata.get("expected_behavior")
        if behavior in {"answer_from_context", "correct_user"}:
            assert task.metadata.get("expected_values"), task.task_id
        if behavior == "correct_user":
            assert task.metadata.get("rejected_values"), task.task_id


# --------------------------------------------------------------------- #
# Repetitions / consistency                                             #
# --------------------------------------------------------------------- #


class _ScriptedBackend:
    """Backend returning a canned output per call, recording sampling seeds."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.seeds: list[int] = []

    def load_model(self, model_config: ModelConfig) -> None:
        self.model = model_config

    def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
        output = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        self.seeds.append(params.seed)
        return GenerationResult(
            prompt=prompt,
            output=output,
            finish_reason="stop",
            metrics=InferenceMetrics(),
            raw={},
        )

    def get_metrics(self) -> InferenceMetrics:
        return InferenceMetrics()

    def unload(self) -> None:
        return None


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="fake-model",
        family="fake",
        revision="fake:1",
        quantization="none",
        source="test",
        context_length=4096,
        artifact_path="fake:1",
    )


def test_sampling_for_repetition_varies_seed() -> None:
    sampling = SamplingConfig(seed=42)
    assert sampling_for_repetition(sampling, 1).seed == 42
    assert sampling_for_repetition(sampling, 3).seed == 44


def test_repetitions_produce_one_row_per_repetition(tmp_path: Path) -> None:
    suite = load_reliability_suite(REPO_ROOT, "hallucination_grounding")
    backend_config = load_backend(REPO_ROOT / "configs" / "backends" / "ollama.yaml")
    backend = _ScriptedBackend(["The context does not mention that."])
    summary = run_hallucination_grounding_suite(
        run_dir=tmp_path,
        suite=suite,
        backend=backend,
        backend_config=backend_config,
        model_configs=[_model_config()],
        sampling=SamplingConfig(max_tokens=512),
        repetitions=3,
    )
    assert backend.calls == len(suite.tasks) * 3
    assert summary["repetitions"] == 3
    assert summary["total_rows"] == len(suite.tasks) * 3
    # Seeds must differ within a task, otherwise repeats measure nothing.
    assert len(set(backend.seeds[:3])) == 3


def test_grounding_uses_a_larger_token_budget_than_64(tmp_path: Path) -> None:
    """The old 64-token cap truncated models that reason before answering."""
    suite = load_reliability_suite(REPO_ROOT, "hallucination_grounding")
    backend_config = load_backend(REPO_ROOT / "configs" / "backends" / "ollama.yaml")

    seen: list[int] = []

    class _Recorder(_ScriptedBackend):
        def generate(self, prompt: str, params: SamplingConfig) -> GenerationResult:
            seen.append(params.max_tokens)
            return super().generate(prompt, params)

    run_hallucination_grounding_suite(
        run_dir=tmp_path,
        suite=suite,
        backend=_Recorder(["ok"]),
        backend_config=backend_config,
        model_configs=[_model_config()],
        sampling=SamplingConfig(max_tokens=512),
    )
    assert seen and set(seen) == {256}


def test_build_summary_reports_consistency_across_repetitions() -> None:
    suite = load_reliability_suite(REPO_ROOT, "hallucination_grounding")
    backend = load_backend(REPO_ROOT / "configs" / "backends" / "ollama.yaml")
    rows = [
        {"model": "m", "task_id": "t1", "evaluation": {"score": 1, "passed": True}},
        {"model": "m", "task_id": "t1", "evaluation": {"score": 0, "passed": False}},
        {"model": "m", "task_id": "t2", "evaluation": {"score": 1, "passed": True}},
        {"model": "m", "task_id": "t2", "evaluation": {"score": 1, "passed": True}},
    ]
    summary = build_summary(rows, suite, backend)
    model = summary["models"][0]
    assert summary["repetitions"] == 2
    assert model["tasks"] == 2
    assert model["unstable_task_ids"] == ["t1"]
    assert model["consistency_rate"] == 0.5
    assert model["failed_task_ids"] == ["t1"]  # deduped across repetitions


# --------------------------------------------------------------------- #
# Performance probes                                                    #
# --------------------------------------------------------------------- #


def test_openclaw_speed_summary_is_tagged_as_performance_probe(tmp_path: Path) -> None:
    suite = load_openclaw_speed_suite(REPO_ROOT)
    backend_config = load_backend(REPO_ROOT / "configs" / "backends" / "ollama.yaml")
    backend = _ScriptedBackend(["some text"])
    summary = run_openclaw_speed_suite(
        run_dir=tmp_path,
        suite=suite,
        backend=backend,
        backend_config=backend_config,
        model_configs=[_model_config()],
        sampling=SamplingConfig(),
        repetitions=2,
        warmup_runs=1,
    )
    assert summary["scoring"] == "performance_probe"
    # 1 discarded warmup + (3 tasks x 2 repetitions) recorded rows
    assert backend.calls == 1 + len(suite.tasks) * 2
    assert summary["total_rows"] == len(suite.tasks) * 2
    assert _is_performance_probe({"suite": "openclaw_speed_v1", "scoring": "performance_probe"})


def test_performance_probe_detected_for_legacy_runs_without_the_flag() -> None:
    assert _is_performance_probe({"suite": "openclaw_speed_v1"}) is True
    assert _is_performance_probe({"suite": "sustained_throughput_v1"}) is True
    assert _is_performance_probe({"suite": "hallucination_grounding_v1"}) is False


# --------------------------------------------------------------------- #
# Wilson interval                                                       #
# --------------------------------------------------------------------- #


def test_wilson_interval_is_wide_for_small_n() -> None:
    low, high = wilson_interval(6, 9)
    assert 0.3 < low < 0.4
    assert 0.85 < high < 0.92
    # A 9-task suite cannot separate 6/9 from 7/9.
    other_low, other_high = wilson_interval(7, 9)
    assert other_low < high and low < other_high


def test_wilson_interval_stays_in_range_at_the_extremes() -> None:
    low, high = wilson_interval(9, 9)
    assert low > 0.6 and high == 1.0
    low_zero, high_zero = wilson_interval(0, 9)
    assert low_zero == 0.0 and high_zero < 0.4
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_margin_narrows_as_n_grows() -> None:
    assert wilson_margin(5, 10) > wilson_margin(50, 100) > wilson_margin(500, 1000)


# --------------------------------------------------------------------- #
# Truncation is observable end to end                                    #
# --------------------------------------------------------------------- #


def test_ollama_adapter_reports_length_stops_as_truncation(monkeypatch) -> None:
    """Ollama sends done_reason="length"; collapsing it to "stop" hid truncation."""
    import json as _json

    from llm_benchmark.models import BackendConfig, BackendKind, is_truncated
    from llm_benchmark.runners.ollama import OllamaAdapter

    class _Response:
        def read(self):
            return _json.dumps(
                {
                    "response": "def f(",
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 10,
                    "eval_count": 512,
                }
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "llm_benchmark.runners.ollama.urllib.request.urlopen",
        lambda request, timeout: _Response(),
    )
    adapter = OllamaAdapter(
        BackendConfig(name=BackendKind.OLLAMA, entrypoint="http://localhost:11434", version="test")
    )
    adapter.load_model(_model_config())
    result = adapter.generate("hi", SamplingConfig(max_tokens=512))

    assert result.finish_reason == "length"
    assert is_truncated(result.finish_reason) is True


def test_code_generation_marks_truncated_samples() -> None:
    """A cut-off generation must not be filed as the model writing broken code."""
    from llm_benchmark.code_generation import evaluate_task

    task = SuiteTask(
        task_id="humaneval/x",
        prompt="def add(a, b):\n",
        metadata={
            "benchmark": "humaneval",
            "entry_point": "add",
            "tests": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        },
    )
    cut_off = GenerationResult(
        prompt="p",
        output="def add(a, b):\n    return a +",
        finish_reason="length",
        metrics=InferenceMetrics(decode_tokens=512),
        raw={},
    )
    outcome = evaluate_task(
        task,
        generations=[cut_off],
        sample_seeds=[42],
        sandbox_timeout_s=10.0,
        sandbox_memory_mb=512,
    )

    assert outcome.samples[0].truncated is True
    assert outcome.samples[0].sandbox.passed is False
    assert outcome.pass_at_1 == 0.0


# --------------------------------------------------------------------- #
# The code sandbox must actually run the tests                           #
# --------------------------------------------------------------------- #


def _code_task():
    from llm_benchmark.code_generation import load_code_generation_suite

    return load_code_generation_suite(REPO_ROOT).tasks[0]


def _evaluate_code(output: str):
    from llm_benchmark.code_generation import evaluate_task

    generation = GenerationResult(
        prompt="p", output=output, finish_reason="stop", metrics=InferenceMetrics(), raw={}
    )
    outcome = evaluate_task(
        _code_task(),
        generations=[generation],
        sample_seeds=[42],
        sandbox_timeout_s=15.0,
        sandbox_memory_mb=1024,
    )
    return outcome.samples[0].sandbox


def test_wrong_solution_fails_the_sandbox() -> None:
    """The fixture only defines check(); _build_program must also call it.

    Without the call the module compiled, exited 0 and every syntactically
    valid answer scored as correct — pass@1 measured "does it parse".
    """
    sandbox = _evaluate_code("def has_close_elements(numbers, threshold):\n    return False\n")

    assert sandbox.passed is False
    assert sandbox.status == "failed"


def test_canonical_solution_passes_the_sandbox() -> None:
    task = _code_task()
    sandbox = _evaluate_code(task.prompt + task.metadata["canonical_solution"])

    assert sandbox.passed is True


def test_build_program_appends_the_check_invocation() -> None:
    from llm_benchmark.code_generation import _build_program

    program = _build_program("def f():\n    pass\n", "def check(candidate):\n    pass\n", entry_point="f")

    assert program.rstrip().endswith("check(f)")


def test_prompt_helper_functions_are_available_to_the_tests() -> None:
    """humaneval/38's tests call encode_cyclic, which only the prompt defines.

    The official harness runs prompt + completion; we run the extracted answer,
    which contains just the target function. Without re-injecting the helper the
    test dies with NameError and the model is blamed for a harness artefact.
    """
    from llm_benchmark.code_generation import load_code_generation_suite, evaluate_task

    task = {t.task_id: t for t in load_code_generation_suite(REPO_ROOT).tasks}["humaneval/38"]
    answer = task.prompt + task.metadata["canonical_solution"]
    # Keep only decode_cyclic, i.e. drop the helper the prompt defined.
    only_target = "def decode_cyclic" + answer.split("def decode_cyclic", 1)[1]
    generation = GenerationResult(
        prompt="p", output=only_target, finish_reason="stop", metrics=InferenceMetrics(), raw={}
    )
    outcome = evaluate_task(
        task,
        generations=[generation],
        sample_seeds=[42],
        sandbox_timeout_s=15.0,
        sandbox_memory_mb=1024,
    )

    assert outcome.samples[0].sandbox.passed is True


def _run_all() -> int:
    """Lightweight runner so tests work without pytest installed system-wide."""
    import inspect
    import sys
    import tempfile

    failures: list[str] = []
    module = sys.modules[__name__]
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"ok  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc!r}")
            print(f"FAIL {name}: {exc!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())
