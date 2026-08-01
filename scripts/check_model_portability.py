#!/usr/bin/env python3
"""Can this harness measure a model it has never seen?

    python3 scripts/check_model_portability.py gpt-oss:120b
    python3 scripts/check_model_portability.py gpt-oss:120b --reasoning --max-tokens 4096

The suites were built around three non-reasoning Ollama models, and several
assumptions about those three were baked in rather than configured. A model
that violates one still produces a complete run and a plausible pass rate --
which is exactly how three scoring bugs survived here for a day. This script
runs a handful of tasks from each suite and reports whether the *harness*
handled the model, separately from whether the model did well:

- empty answers          -> the prompt or the reasoning mode is wrong for it
- truncated answers      -> the shared token budget is too small for it
- unparseable code       -> extract_code cannot find its answer format
- unmatched abstentions  -> the grounding scorer's phrase list misses its wording

A low score on a suite is a result. A high rate of any of the four above means
the number is not a measurement at all, and the script says so.

Exit code is 0 when the model is measurable, 1 when it is not.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_benchmark.code_generation import evaluate_task, load_code_generation_suite  # noqa: E402
from llm_benchmark.config import load_backend, load_model_config  # noqa: E402
from llm_benchmark.model_registry import (  # noqa: E402
    detect_ollama_models,
    synthesize_cloud_model_config,
    synthesize_model_config,
)
from llm_benchmark.models import ModelConfig, SamplingConfig, sampling_for_model  # noqa: E402
from llm_benchmark.reliability import (  # noqa: E402
    build_grounded_prompt,
    build_structured_output_prompt,
    load_reliability_suite,
    score_hallucination_task,
    score_structured_output_task,
)
from llm_benchmark.runners.registry import build_backend  # noqa: E402

# Enough to expose a systematic problem (an empty-output or truncation rate
# shows up immediately), few enough to stay a five-minute check. This is a
# smoke test for the harness, not a measurement of the model.
DEFAULT_TASKS_PER_SUITE = 5

# A rate above this on any diagnostic means the suite's number for this model
# describes the harness, not the model.
BROKEN_RATE = 0.4


@dataclass
class SuiteProbe:
    suite: str
    passed: int = 0
    total: int = 0
    empty: int = 0
    truncated: int = 0
    decode_tokens: list[int] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def rate(self, count: int) -> float:
        return count / self.total if self.total else 0.0


def resolve_model(tag: str, backend_config) -> ModelConfig:
    """Curated YAML if one exists, otherwise synthesise from the Ollama tag."""
    curated = REPO_ROOT / "configs" / "models" / f"{tag}.yaml"
    if curated.exists():
        return load_model_config(curated)
    if tag.endswith("-cloud"):
        return synthesize_cloud_model_config(tag)
    for detected in detect_ollama_models(backend_config):
        if detected.tag == tag:
            return synthesize_model_config(detected)
    raise SystemExit(
        f"{tag!r} is neither a config under configs/models/ nor a tag in Ollama.\n"
        "Pull it first, or pass a curated config name."
    )


def probe_code(backend, suite, sampling: SamplingConfig, limit: int) -> SuiteProbe:
    probe = SuiteProbe(suite="code_generation")
    for task in suite.tasks[:limit]:
        generation = backend.generate(task.prompt, sampling)
        outcome = evaluate_task(
            task,
            generations=[generation],
            sample_seeds=[sampling.seed],
            sandbox_timeout_s=15.0,
            sandbox_memory_mb=1024,
        )
        sample = outcome.samples[0]
        probe.total += 1
        probe.passed += int(sample.sandbox.passed)
        probe.decode_tokens.append(generation.metrics.decode_tokens)
        if not generation.output.strip():
            probe.empty += 1
        if generation.finish_reason == "length":
            probe.truncated += 1
        if not sample.sandbox.passed:
            probe.fail_reasons.append(sample.sandbox.status)
    compile_errors = probe.fail_reasons.count("compile_error")
    if probe.total and compile_errors / probe.total >= BROKEN_RATE:
        probe.notes.append(
            f"{compile_errors}/{probe.total} answers did not compile — extract_code probably "
            "cannot find this model's answer inside its output format"
        )
    return probe


def probe_grounding(backend, suite, sampling: SamplingConfig, limit: int) -> SuiteProbe:
    probe = SuiteProbe(suite="hallucination_grounding")
    for task in suite.tasks[:limit]:
        generation = backend.generate(build_grounded_prompt(task), sampling)
        evaluation = score_hallucination_task(task, generation.output, generation.finish_reason)
        probe.total += 1
        probe.passed += int(evaluation["passed"])
        probe.decode_tokens.append(generation.metrics.decode_tokens)
        if not generation.output.strip():
            probe.empty += 1
        if generation.finish_reason == "length":
            probe.truncated += 1
        if not evaluation["passed"]:
            probe.fail_reasons.append(evaluation["reason"].split(":")[0])
    unmatched = probe.fail_reasons.count("missing_abstention_phrase")
    if unmatched:
        probe.notes.append(
            f"{unmatched}/{probe.total} failures are 'missing_abstention_phrase' — check the raw "
            "answers before believing them: a correct refusal worded outside ABSTAIN_PHRASES "
            "(or written in another language) scores as a hallucination"
        )
    return probe


def probe_structured(backend, suite, sampling: SamplingConfig, limit: int) -> SuiteProbe:
    probe = SuiteProbe(suite="practical_structured_output")
    for task in suite.tasks[:limit]:
        generation = backend.generate(build_structured_output_prompt(task), sampling)
        evaluation = score_structured_output_task(task, generation.output, generation.finish_reason)
        probe.total += 1
        probe.passed += int(evaluation["passed"])
        probe.decode_tokens.append(generation.metrics.decode_tokens)
        if not generation.output.strip():
            probe.empty += 1
        if generation.finish_reason == "length":
            probe.truncated += 1
        if not evaluation["passed"]:
            probe.fail_reasons.append(evaluation["reason"].split("+")[0])
    no_json = probe.fail_reasons.count("no_json_object_found")
    if probe.total and no_json / probe.total >= BROKEN_RATE:
        probe.notes.append(
            f"{no_json}/{probe.total} answers contained no JSON at all — for a reasoning model "
            "this usually means the scratchpad was returned instead of the answer"
        )
    return probe


def report(model: ModelConfig, probes: list[SuiteProbe], sampling: SamplingConfig) -> bool:
    print()
    print(f"model      {model.name}  (tag {model.artifact_path or model.revision})")
    print(f"reasoning  {'on' if model.reasoning else 'off'}")
    origin = "per-model override" if model.max_output_tokens else "probe default"
    print(f"budget     {sampling.max_tokens} tokens ({origin})")
    print()
    print(f"{'suite':28s} {'pass':>7s} {'empty':>7s} {'trunc':>7s} {'median tok':>11s}")
    for probe in probes:
        median = int(statistics.median(probe.decode_tokens)) if probe.decode_tokens else 0
        print(
            f"{probe.suite:28s} {probe.passed:>3d}/{probe.total:<3d} "
            f"{probe.empty:>7d} {probe.truncated:>7d} {median:>11d}"
        )

    problems: list[str] = []
    for probe in probes:
        if probe.rate(probe.empty) >= BROKEN_RATE:
            problems.append(
                f"{probe.suite}: {probe.empty}/{probe.total} answers were empty. The model returned "
                "nothing usable — try --reasoning if it is a reasoning model, since its answer may "
                "only exist after the thinking pass."
            )
        if probe.rate(probe.truncated) >= BROKEN_RATE:
            problems.append(
                f"{probe.suite}: {probe.truncated}/{probe.total} answers hit the token budget. "
                f"Raise it with max_output_tokens in the model config; at {sampling.max_tokens} "
                "tokens this suite scores the budget, not the model."
            )
        problems.extend(f"{probe.suite}: {note}" for note in probe.notes)

    print()
    if not problems:
        print("VERDICT  measurable — no harness-side failure mode above threshold.")
        print("         Pass rates above are a real (if tiny) sample of this model's ability.")
        return True
    print("VERDICT  not measurable as configured:")
    for problem in problems:
        print(f"  - {problem}")
    print()
    print("Suggested configs/models/<name>.yaml additions:")
    if any(p.rate(p.empty) >= BROKEN_RATE for p in probes) and not model.reasoning:
        print("  reasoning: true")
    if any(p.rate(p.truncated) >= BROKEN_RATE for p in probes):
        print(f"  max_output_tokens: {sampling.max_tokens * 4}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="Ollama tag (phi5:14b) or a config name under configs/models/.")
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS_PER_SUITE, help="Tasks per suite.")
    parser.add_argument("--reasoning", action="store_true", help="Enable the backend's reasoning pass.")
    parser.add_argument("--max-tokens", type=int, default=1536, help="Output budget for the probe.")
    parser.add_argument("--backend", default="configs/backends/ollama.yaml")
    args = parser.parse_args()

    backend_config = load_backend(REPO_ROOT / args.backend)
    model = resolve_model(args.tag, backend_config)
    # CLI flags win over the YAML so a config can be trialled before it is written.
    if args.reasoning:
        model = model.model_copy(update={"reasoning": True})
    sampling = SamplingConfig(temperature=0.0, top_p=1.0, seed=42, max_tokens=args.max_tokens)

    # The runner applies max_output_tokens itself; resolve it here too so the
    # report states the budget that was actually sent.
    sampling = sampling_for_model(sampling, model)

    backend = build_backend(backend_config)
    backend.load_model(model)
    try:
        probes = [
            probe_code(backend, load_code_generation_suite(REPO_ROOT), sampling, args.tasks),
            probe_grounding(
                backend, load_reliability_suite(REPO_ROOT, "hallucination_grounding"), sampling, args.tasks
            ),
            probe_structured(
                backend,
                load_reliability_suite(REPO_ROOT, "practical_structured_output"),
                sampling,
                args.tasks,
            ),
        ]
    finally:
        backend.unload()

    return 0 if report(model, probes, sampling) else 1


if __name__ == "__main__":
    raise SystemExit(main())
