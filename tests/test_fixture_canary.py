"""Canary tests: prove the code harness can tell right from wrong.

Two harness bugs shipped on 2026-08-01, both of the same shape — the pipeline
reported pass rates while not actually testing what it claimed:

1. ``_build_program`` never emitted ``check(<entry_point>)``, so a module that
   merely compiled counted as a pass and ``pass@1`` measured parseability.
2. Prompt-defined helpers were missing from the sandbox, so tasks whose tests
   call them died with ``NameError`` and the model took the blame.

Neither was visible from the outside: the numbers looked plausible. The defence
is a canary that runs known-good and known-bad code through the real pipeline
and asserts the verdicts, so "the harness stopped discriminating" fails CI
instead of becoming a published benchmark result.

Kept to a sample of the fixture so CI stays quick; ``--canary-full`` style
exhaustive checking belongs in the conversion script, which already runs all
164 canonical solutions.
"""

from pathlib import Path

from llm_benchmark.code_generation import (
    evaluate_task,
    load_code_generation_suite,
    load_mbpp_mutated_suite,
    load_mbpp_suite,
)
from llm_benchmark.models import GenerationResult, InferenceMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every 12th problem: a deterministic spread over the fixture, ~14 tasks, and
# it includes humaneval/0 plus tasks with prompt-defined helpers.
SAMPLE_STRIDE = 12


def _all_fixture_tasks():
    return (
        load_code_generation_suite(REPO_ROOT).tasks
        + load_mbpp_suite(REPO_ROOT).tasks
        + load_mbpp_mutated_suite(REPO_ROOT).tasks
    )


def _tasks():
    tasks = (
        load_code_generation_suite(REPO_ROOT).tasks
        + load_mbpp_suite(REPO_ROOT).tasks
        + load_mbpp_mutated_suite(REPO_ROOT).tasks
    )
    sample = tasks[::SAMPLE_STRIDE]
    # humaneval/32 (poly) and /38 (encode_cyclic) define helpers in the prompt;
    # pin them explicitly so the sample can never drift off them.
    by_id = {t.task_id: t for t in tasks}
    for task_id in ("humaneval/32", "humaneval/38"):
        if by_id[task_id] not in sample:
            sample.append(by_id[task_id])
    # mbpp/6 and mbpp/18 need a helper function and a module-level constant
    # that precede the entry point — the case extract_code used to slice away.
    for task_id in ("mbpp/6", "mbpp/18"):
        if task_id in by_id and by_id[task_id] not in sample:
            sample.append(by_id[task_id])
    return sample


def _verdict(task, code: str):
    generation = GenerationResult(
        prompt="p", output=code, finish_reason="stop", metrics=InferenceMetrics(), raw={}
    )
    outcome = evaluate_task(
        task,
        generations=[generation],
        sample_seeds=[42],
        sandbox_timeout_s=20.0,
        sandbox_memory_mb=1024,
    )
    return outcome.samples[0].sandbox


def test_canonical_solutions_pass() -> None:
    """Known-good code must pass, or the harness is rejecting correct answers."""
    failures = []
    for task in _tasks():
        sandbox = _verdict(task, task.prompt + task.metadata["canonical_solution"])
        if not sandbox.passed:
            failures.append((task.task_id, sandbox.status, (sandbox.stderr or "")[-160:]))
    assert not failures, f"canonical solutions rejected by the harness: {failures}"


def test_stubbed_solutions_fail() -> None:
    """Known-bad code must fail, or the harness is not running the tests.

    The stub keeps the required signature and returns a constant, so it parses
    and imports cleanly — exactly the shape that slipped through when
    ``check()`` was never invoked.
    """
    passed_anyway = []
    for task in _tasks():
        entry_point = task.metadata["entry_point"]
        stub = f"def {entry_point}(*args, **kwargs):\n    return None\n"
        sandbox = _verdict(task, stub)
        if sandbox.passed:
            passed_anyway.append(task.task_id)
    assert not passed_anyway, (
        "harness accepted a stub returning None — it is not executing the tests: "
        f"{passed_anyway}"
    )


def test_every_task_declares_what_the_sandbox_needs() -> None:
    """A task missing entry_point or tests would raise mid-run, not at load."""
    missing = [
        t.task_id
        for t in _all_fixture_tasks()
        if not t.metadata.get("entry_point") or not t.metadata.get("tests")
    ]
    assert not missing, f"tasks missing entry_point/tests metadata: {missing}"


def _run_all() -> int:
    import inspect
    import sys

    failures: list[str] = []
    module = sys.modules[__name__]
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"ok  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc!r}")
            print(f"FAIL {name}: {exc!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())


# --------------------------------------------------------------------- #
# The mutated fixture must stay a mutation, not a different benchmark    #
# --------------------------------------------------------------------- #


def test_mutants_cover_the_same_problems_as_the_source() -> None:
    """A contamination probe is only readable as a delta, so the two sets have
    to be the same problems. A mutant set missing tasks would make the drop
    partly a difference in which problems were asked."""
    source = {task.task_id for task in load_mbpp_suite(REPO_ROOT).tasks}
    mutants = load_mbpp_mutated_suite(REPO_ROOT).tasks
    covered = {task.metadata["mutation"]["source_task"] for task in mutants}
    assert covered == source


def test_every_mutant_actually_changed_its_entry_point() -> None:
    """The rename is the main lever. A task that kept its name contributes a
    familiar cue and silently weakens the probe."""
    unchanged = [
        task.task_id
        for task in load_mbpp_mutated_suite(REPO_ROOT).tasks
        if task.metadata["entry_point"] == task.metadata["mutation"]["original_entry_point"]
    ]
    assert unchanged == []


def test_mutant_prompts_never_call_the_original_name() -> None:
    """The example must call the new name.

    Checked at call sites only, on purpose: entry points like ``sequence`` and
    ``power`` are ordinary English words, and the problem statement is allowed
    — required, in fact — to keep using them. Renaming those in prose mangled
    the sentence and made tasks harder for reasons unrelated to contamination.
    """
    import re

    leaked = [
        task.task_id
        for task in load_mbpp_mutated_suite(REPO_ROOT).tasks
        if re.search(
            rf"\b{re.escape(task.metadata['mutation']['original_entry_point'])}\b\s*\(",
            task.prompt,
        )
    ]
    assert leaked == []


def test_mutant_prompts_end_with_a_newline() -> None:
    # A prompt ending mid-line breaks a model that echoes it before answering:
    # the closing docstring quotes end up glued to `def f(...)`, which is an
    # unterminated string rather than code. Cost 33 tasks when the generator
    # dropped the trailing newline, and the generator's own validation — which
    # did not simulate the echo — passed all of them.
    missing = [
        task.task_id
        for task in load_mbpp_mutated_suite(REPO_ROOT).tasks
        if not task.prompt.endswith("\n")
    ]
    assert missing == []


def test_mutant_prompts_and_tests_agree_on_the_entry_point() -> None:
    for task in load_mbpp_mutated_suite(REPO_ROOT).tasks:
        entry_point = task.metadata["entry_point"]
        assert entry_point in task.metadata["tests"], task.task_id
        assert entry_point in task.metadata["canonical_solution"], task.task_id
