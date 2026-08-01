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

from llm_benchmark.code_generation import evaluate_task, load_code_generation_suite
from llm_benchmark.models import GenerationResult, InferenceMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every 12th problem: a deterministic spread over the fixture, ~14 tasks, and
# it includes humaneval/0 plus tasks with prompt-defined helpers.
SAMPLE_STRIDE = 12


def _tasks():
    tasks = load_code_generation_suite(REPO_ROOT).tasks
    sample = tasks[::SAMPLE_STRIDE]
    # humaneval/32 (poly) and /38 (encode_cyclic) define helpers in the prompt;
    # pin them explicitly so the sample can never drift off them.
    by_id = {t.task_id: t for t in tasks}
    for task_id in ("humaneval/32", "humaneval/38"):
        if by_id[task_id] not in sample:
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
        for t in load_code_generation_suite(REPO_ROOT).tasks
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
