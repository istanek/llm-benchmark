from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from llm_benchmark.models import (
    BackendConfig,
    GenerationResult,
    ModelConfig,
    SamplingConfig,
    is_truncated,
)
from llm_benchmark.results_bundle import write_json, write_result
from llm_benchmark.stats import wilson_interval
from llm_benchmark.suites import SuiteDefinition, SuiteTask, load_suite_definition

ABSTAIN_PHRASES = (
    "does not mention",
    "not mentioned",
    "no mention",
    "does not contain",
    "does not say",
    "doesn't say",
    "does not state",
    "not stated",
    "does not specify",
    "not specified",
    "does not provide",
    "no information",
    "not provided",
    "does not identify",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "do not know",
    "don't know",
    "insufficient information",
    "not enough information",
    "unknown",
    "not answerable",
    "unable to answer from the context",
    "cannot answer from the context",
)

# Markers of a guess bolted onto an abstention ("the context doesn't say, but it
# was probably X"). Abstaining and then fabricating is a hallucination, not an
# abstention, so these veto an otherwise-matching abstention phrase.
HEDGED_GUESS_PHRASES = (
    "but it was likely",
    "but it is likely",
    "most likely",
    "probably",
    "presumably",
    "my guess",
    "i would guess",
    "i believe it was",
    "it could be",
    "perhaps",
)

NEGATION_PHRASES = (
    "no",
    "not",
    "incorrect",
    "inconsistent",
    "false",
    "wrong",
    "does not",
    "is not",
    "isn't",
    "contradicts",
    "unsupported",
)

# Fraction of reference content tokens that must appear in the output before a
# free-text reference counts as reproduced. Only used as a fallback when the
# fixture task carries no explicit ``expected_values`` contract.
REFERENCE_COVERAGE_THRESHOLD = 0.6

# Per-suite output budgets. The grounding cap used to be 64 tokens, which cut
# off models that emit a visible reasoning preamble before their answer and
# scored them as hallucinations. Truncated rows are now flagged in the
# evaluation (``truncated``) so a budget that is still too small is visible in
# the results instead of silently depressing a model's score.
GROUNDING_MAX_TOKENS = 256
STRUCTURED_MAX_TOKENS = 256

# Tokens too common to carry meaning when checking whether a free-text
# reference was reproduced.
_STOPWORDS = frozenset(
    {
        "the", "and", "was", "were", "are", "for", "that", "this", "with", "from",
        "its", "it", "is", "in", "on", "of", "to", "a", "an", "says", "said",
        "context", "states", "state", "not", "but", "has", "have", "had",
    }
)


def fixture_path_for_suite_name(repo_root: Path, suite_name: str) -> Path:
    if suite_name in {"hallucination_grounding", "hallucination_grounding_v1"}:
        return repo_root / "data" / "reliability" / "hallucination_grounding_v1.json"
    if suite_name in {"practical_structured_output", "practical_structured_output_v1"}:
        return repo_root / "data" / "practical" / "practical_structured_output_v1.json"
    if suite_name in {"openclaw_speed", "openclaw_speed_v1"}:
        return repo_root / "data" / "performance" / "openclaw_speed_v1.json"
    if suite_name in {"code_generation", "code_generation_v1"}:
        return repo_root / "data" / "code" / "code_generation_v1.json"
    if suite_name in {"code_generation_mbpp", "code_generation_mbpp_v1", "mbpp"}:
        return repo_root / "data" / "code" / "code_generation_mbpp_v1.json"
    if suite_name in {"sustained_throughput", "sustained_throughput_v1"}:
        return repo_root / "data" / "performance" / "sustained_throughput_v1.json"
    if suite_name in {
        "long_context_retrieval",
        "long_context_retrieval_v1",
        "long_context_retrieval_fast",
    }:
        return repo_root / "data" / "long_context" / "long_context_retrieval_v1.json"
    raise ValueError(f"unsupported suite runner: {suite_name}")


def load_reliability_suite(repo_root: Path, suite_name: str) -> SuiteDefinition:
    return load_suite_definition(fixture_path_for_suite_name(repo_root, suite_name))


def build_grounded_prompt(task: SuiteTask) -> str:
    context = task.context or ""
    return (
        "You must answer using only the supplied context. "
        "If the answer is not in the context, say that clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{task.prompt}"
    )


def build_structured_output_prompt(task: SuiteTask) -> str:
    context = task.context or ""
    return (
        "Return valid JSON only. Do not include markdown fences, prose, or explanations.\n\n"
        f"Input:\n{context}\n\n"
        f"Instruction:\n{task.prompt}"
    )


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def reference_tokens(reference: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize_text(reference)) if len(token) >= 3 or token.isdigit()}


def tokenize(value: str) -> list[str]:
    """Split text into comparable word tokens (order preserved)."""
    return re.findall(r"[a-z0-9]+", normalize_text(value))


def contains_phrase(phrase: str, tokens: list[str]) -> bool:
    """True when *phrase* appears in *tokens* as a whole-token subsequence.

    Substring matching is not usable here: ``"no" in "nothing to note"`` and
    ``"not" in "cannot"`` are both true, which made every sufficiently long
    output look like a negation. Both sides are tokenised the same way so
    contractions ("don't" -> don, t) still line up.
    """
    phrase_tokens = tokenize(phrase)
    if not phrase_tokens:
        return False
    span = len(phrase_tokens)
    return any(tokens[i : i + span] == phrase_tokens for i in range(len(tokens) - span + 1))


def first_matching_phrase(phrases: tuple[str, ...], tokens: list[str]) -> str | None:
    return next((phrase for phrase in phrases if contains_phrase(phrase, tokens)), None)


def reference_coverage(reference: str, tokens: list[str]) -> float:
    """Fraction of the reference's content tokens present in the output."""
    content = {token for token in reference_tokens(reference) if token not in _STOPWORDS}
    if not content:
        return 0.0
    present = {token for token in content if token in set(tokens)}
    return len(present) / len(content)


def _expected_values_present(values: list[str], normalized_output: str) -> tuple[bool, list[str]]:
    """Check the fixture's explicit value contract against the output.

    Values are matched literally (after whitespace/case normalisation) because
    they are typically short and punctuated — "2.3.1" does not survive word
    tokenisation intact.
    """
    missing = [value for value in values if normalize_text(value) not in normalized_output]
    return (not missing), missing


def score_hallucination_task(
    task: SuiteTask, output: str, finish_reason: str | None = None
) -> dict[str, Any]:
    """Score one grounding task.

    Scoring prefers the fixture's explicit contract (``metadata.expected_values``
    / ``metadata.rejected_values``) over guessing from the free-text reference.
    When a task carries no ``expected_values``, the reference must be reproduced
    either literally or with at least ``REFERENCE_COVERAGE_THRESHOLD`` of its
    content tokens.
    """
    behavior = str(task.metadata.get("expected_behavior") or "").strip()
    normalized_output = normalize_text(output)
    normalized_reference = normalize_text(task.reference or "")
    tokens = tokenize(output)
    token_set = set(tokens)
    ref_tokens = reference_tokens(task.reference or "")
    matched_tokens = sorted(token for token in ref_tokens if token in token_set)
    truncated = is_truncated(finish_reason)

    expected_values = [str(value) for value in (task.metadata.get("expected_values") or [])]
    rejected_values = [str(value) for value in (task.metadata.get("rejected_values") or [])]
    rejected_present = [
        value for value in rejected_values if normalize_text(value) in normalized_output
    ]

    if expected_values:
        has_reference, missing_values = _expected_values_present(expected_values, normalized_output)
    else:
        missing_values = []
        has_reference = bool(normalized_reference and normalized_reference in normalized_output) or (
            reference_coverage(task.reference or "", tokens) >= REFERENCE_COVERAGE_THRESHOLD
        )

    abstention = first_matching_phrase(ABSTAIN_PHRASES, tokens)
    hedged_guess = first_matching_phrase(HEDGED_GUESS_PHRASES, tokens)

    passed = False
    reason = "unsupported_behavior"

    if behavior == "answer_from_context":
        if abstention is not None:
            reason = f"abstained_instead_of_answering:{abstention}"
        elif has_reference:
            passed = True
            reason = "expected_value_present" if expected_values else "reference_reproduced"
        else:
            reason = "missing_reference_signal"
    elif behavior == "abstain":
        if abstention is None:
            reason = "missing_abstention_phrase"
        elif hedged_guess is not None:
            reason = f"hedged_guess_after_abstention:{hedged_guess}"
        else:
            passed = True
            reason = f"abstention_phrase:{abstention}"
    elif behavior == "correct_user":
        has_negation = first_matching_phrase(NEGATION_PHRASES, tokens) is not None
        if not has_negation:
            reason = "missing_negation"
        elif not has_reference:
            reason = "missing_corrected_value"
        else:
            passed = True
            reason = "negation_plus_corrected_value"

    if not passed and truncated:
        reason = f"{reason}+truncated_output"

    return {
        "expected_behavior": behavior,
        "passed": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "matched_reference_tokens": matched_tokens,
        "missing_expected_values": missing_values,
        "rejected_values_present": rejected_present,
        "truncated": truncated,
    }


def extract_json_value(output: str) -> tuple[Any | None, str]:
    cleaned = output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^`{3}(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*`{3}$", "", cleaned)
        cleaned = cleaned.strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        trailing = cleaned[index + end :].strip()
        if trailing:
            return None, "trailing_text_after_json"
        return value, "ok"
    return None, "no_json_object_found"


def score_structured_output_task(
    task: SuiteTask, output: str, finish_reason: str | None = None
) -> dict[str, Any]:
    expected_behavior = str(task.metadata.get("expected_behavior") or "").strip()
    truncated = is_truncated(finish_reason)
    if expected_behavior != "json_exact_match":
        return {
            "expected_behavior": expected_behavior,
            "passed": False,
            "score": 0,
            "reason": "unsupported_behavior",
            "matched_reference_tokens": [],
            "truncated": truncated,
        }

    expected = json.loads(task.reference or "null")
    parsed, parse_reason = extract_json_value(output)
    passed = parsed == expected
    reason = "exact_json_match" if passed else parse_reason if parsed is None else "json_value_mismatch"
    if not passed and truncated:
        reason = f"{reason}+truncated_output"
    return {
        "expected_behavior": expected_behavior,
        "passed": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "matched_reference_tokens": [],
        "truncated": truncated,
    }


def build_summary(
    run_rows: list[dict[str, Any]],
    suite: SuiteDefinition,
    backend: BackendConfig,
    scoring: str = "pass_fail",
) -> dict[str, Any]:
    """Fold per-row evaluations into a per-model summary.

    When a suite ran with ``repetitions > 1`` each (model, task) pair produces
    several rows. ``pass_rate`` counts every row, and ``consistency_rate``
    reports the fraction of tasks whose repetitions all agreed — the
    "consistency across repeated runs" axis called for in METHODOLOGY.md.

    ``scoring`` marks how the pass column should be read. Performance probes
    pass ``"performance_probe"`` so reporting does not present a synthetic
    100 % as a quality result.
    """
    per_model: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, list[bool]]] = {}
    for row in run_rows:
        model_name = row["model"]
        bucket = per_model.setdefault(
            model_name,
            {"model": model_name, "passes": 0, "total": 0, "failed_task_ids": []},
        )
        bucket["passes"] += int(row["evaluation"]["score"])
        bucket["total"] += 1
        passed = bool(row["evaluation"]["passed"])
        if not passed and row["task_id"] not in bucket["failed_task_ids"]:
            bucket["failed_task_ids"].append(row["task_id"])
        outcomes.setdefault(model_name, {}).setdefault(row["task_id"], []).append(passed)

    max_repetitions = 1
    for bucket in per_model.values():
        total = bucket["total"] or 1
        bucket["pass_rate"] = round(bucket["passes"] / total, 4)
        per_task = outcomes.get(bucket["model"], {})
        bucket["tasks"] = len(per_task)
        reps = max((len(values) for values in per_task.values()), default=1)
        max_repetitions = max(max_repetitions, reps)
        bucket["repetitions"] = reps
        unstable = sorted(task_id for task_id, values in per_task.items() if len(set(values)) > 1)
        bucket["unstable_task_ids"] = unstable
        bucket["consistency_rate"] = (
            round((len(per_task) - len(unstable)) / len(per_task), 4) if per_task else None
        )
        bucket["truncated_rows"] = sum(
            1
            for row in run_rows
            if row["model"] == bucket["model"] and row["evaluation"].get("truncated")
        )

    return {
        "suite": suite.name,
        "suite_version": suite.version,
        "backend": backend.name.value,
        "scoring": scoring,
        "repetitions": max_repetitions,
        "total_rows": len(run_rows),
        "models": list(per_model.values()),
    }


def write_summary_markdown(run_dir: Path, summary: dict[str, Any]) -> Path:
    lines = [
        f"# {summary['suite']} summary",
        "",
        f"- backend: {summary['backend']}",
        f"- total rows: {summary['total_rows']}",
        f"- repetitions per task: {summary.get('repetitions', 1)}",
        "",
        "| model | passes | total | pass_rate | 95% CI | consistency |",
        "| --- | ---: | ---: | ---: | :---: | ---: |",
    ]
    for model in summary["models"]:
        low, high = wilson_interval(model["passes"], model["total"])
        consistency = model.get("consistency_rate")
        consistency_cell = "-" if consistency is None else f"{consistency:.2%}"
        lines.append(
            f"| {model['model']} | {model['passes']} | {model['total']} | {model['pass_rate']:.2%} "
            f"| {low:.0%}–{high:.0%} | {consistency_cell} |"
        )
        if model["failed_task_ids"]:
            failed = ", ".join(model["failed_task_ids"])
            lines.append(f"| {model['model']} failed tasks | {failed} |  |  |  |  |")
        if model.get("unstable_task_ids"):
            unstable = ", ".join(model["unstable_task_ids"])
            lines.append(f"| {model['model']} unstable tasks | {unstable} |  |  |  |  |")
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def sampling_for_repetition(sampling: SamplingConfig, repetition: int) -> SamplingConfig:
    """Vary the seed per repetition so repeats are not trivially identical.

    At ``temperature = 0`` this measures backend determinism; above it, it
    measures real sampling stability. Either way the number reported by
    ``consistency_rate`` is meaningful rather than guaranteed to be 1.0.
    """
    if repetition <= 1:
        return sampling
    return sampling.model_copy(update={"seed": sampling.seed + repetition - 1})


def run_hallucination_grounding_suite(
    *,
    run_dir: Path,
    suite: SuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    repetitions: int = 1,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suite_sampling = sampling.model_copy(
        update={"max_tokens": min(sampling.max_tokens, GROUNDING_MAX_TOKENS)}
    )
    repetitions = max(1, int(repetitions))
    run_rows: list[dict[str, Any]] = []
    total_tasks = len(suite.tasks)
    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name} for grounding probe")
        backend.load_model(model_config)
        for idx, task in enumerate(suite.tasks, start=1):
            for repetition in range(1, repetitions + 1):
                if progress_callback:
                    suffix = f" rep {repetition}/{repetitions}" if repetitions > 1 else ""
                    progress_callback(
                        f"  {model_config.name} → grounding task {idx}/{total_tasks} ({task.task_id}){suffix}"
                    )
                prompt = build_grounded_prompt(task)
                generation: GenerationResult = backend.generate(
                    prompt, sampling_for_repetition(suite_sampling, repetition)
                )
                evaluation = score_hallucination_task(task, generation.output, generation.finish_reason)
                row = {
                    "suite": suite.name,
                    "suite_version": suite.version,
                    "model": model_config.name,
                    "model_tag": model_config.artifact_path or model_config.revision,
                    "task_id": task.task_id,
                    "repetition": repetition,
                    "tags": task.tags,
                    "prompt": task.prompt,
                    "context": task.context,
                    "reference": task.reference,
                    "generation": generation.model_dump(mode="json"),
                    "evaluation": evaluation,
                }
                write_result(run_dir, row)
                run_rows.append(row)
        if progress_callback:
            progress_callback(f"  unloading {model_config.name} from Ollama")
        backend.unload()

    summary = build_summary(run_rows, suite, backend_config)
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir, summary)
    return summary


def run_practical_structured_output_suite(
    *,
    run_dir: Path,
    suite: SuiteDefinition,
    backend: Any,
    backend_config: BackendConfig,
    model_configs: list[ModelConfig],
    sampling: SamplingConfig,
    repetitions: int = 1,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suite_sampling = sampling.model_copy(
        update={"max_tokens": min(sampling.max_tokens, STRUCTURED_MAX_TOKENS)}
    )
    repetitions = max(1, int(repetitions))
    run_rows: list[dict[str, Any]] = []
    total_tasks = len(suite.tasks)
    for model_config in model_configs:
        if progress_callback:
            progress_callback(f"  loading {model_config.name} for structured-output probe")
        backend.load_model(model_config)
        for idx, task in enumerate(suite.tasks, start=1):
            for repetition in range(1, repetitions + 1):
                if progress_callback:
                    suffix = f" rep {repetition}/{repetitions}" if repetitions > 1 else ""
                    progress_callback(
                        f"  {model_config.name} → structured task {idx}/{total_tasks} ({task.task_id}){suffix}"
                    )
                prompt = build_structured_output_prompt(task)
                generation: GenerationResult = backend.generate(
                    prompt, sampling_for_repetition(suite_sampling, repetition)
                )
                evaluation = score_structured_output_task(task, generation.output, generation.finish_reason)
                row = {
                    "suite": suite.name,
                    "suite_version": suite.version,
                    "model": model_config.name,
                    "model_tag": model_config.artifact_path or model_config.revision,
                    "task_id": task.task_id,
                    "repetition": repetition,
                    "tags": task.tags,
                    "prompt": task.prompt,
                    "context": task.context,
                    "reference": task.reference,
                    "generation": generation.model_dump(mode="json"),
                    "evaluation": evaluation,
                }
                write_result(run_dir, row)
                run_rows.append(row)
        if progress_callback:
            progress_callback(f"  unloading {model_config.name} from Ollama")
        backend.unload()

    summary = build_summary(run_rows, suite, backend_config)
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir, summary)
    return summary
