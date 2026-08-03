"""Preflight: verify a run can run, before it runs.

The sweep of 2026-08-02 measured four models for eight hours and then died on
its last suite because two Project Gutenberg texts had never been downloaded.
These tests pin the checks that would have caught it in milliseconds, and the
two bugs the first version of the preflight itself shipped with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_benchmark.config import load_backend, load_model_config
from llm_benchmark.preflight import (
    Problem,
    check_models,
    check_suite_assets,
    preflight,
    render_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ollama_backend():
    return load_backend(REPO_ROOT / "configs" / "backends" / "ollama.yaml")


# --------------------------------------------------------------------- #
# The check that was missing                                            #
# --------------------------------------------------------------------- #


def test_missing_haystacks_are_caught(tmp_path: Path) -> None:
    """The eight-hour failure, reduced to a unit test.

    The fixture is committed and the corpora are git-ignored, so a fresh
    checkout looks exactly like this: suite definition present, texts absent.
    """
    fixture_dir = tmp_path / "data" / "long_context"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "long_context_retrieval_v1.json").write_text(
        (REPO_ROOT / "data" / "long_context" / "long_context_retrieval_v1.json").read_text()
    )

    problems = check_suite_assets(tmp_path, "long_context_retrieval_fast")
    assert problems, "a checkout with no haystacks must not pass preflight"
    assert any("fetch_haystacks" in problem.fix for problem in problems)
    assert all("missing at" in problem.detail for problem in problems)


def test_present_haystacks_pass(tmp_path: Path) -> None:
    """text_file is relative to the repo root, not to the haystack directory.
    Joining it onto the directory produced a doubled path that could never
    exist, so the first version of this check failed even after a good fetch."""
    fixture = json.loads(
        (REPO_ROOT / "data" / "long_context" / "long_context_retrieval_v1.json").read_text()
    )
    for spec in fixture["haystacks"].values():
        target = tmp_path / spec["text_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("text")
    fixture_dir = tmp_path / "data" / "long_context"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "long_context_retrieval_v1.json").write_text(json.dumps(fixture))

    assert check_suite_assets(tmp_path, "long_context_retrieval_fast") == []


def test_a_matrix_fixture_is_not_asked_for_tasks() -> None:
    """The long-context fixture is needles and haystacks; it legitimately has
    no `tasks` key, and the first preflight reported that as a defect."""
    problems = check_suite_assets(REPO_ROOT, "long_context_retrieval_fast")
    assert not any("no tasks" in problem.detail for problem in problems)


# --------------------------------------------------------------------- #
# Suites and fixtures                                                   #
# --------------------------------------------------------------------- #


def test_every_suite_in_the_full_sweep_passes_preflight(ollama_backend) -> None:
    """If this fails, the sweep config references something that is not there."""
    from llm_benchmark.config import load_experiment

    experiment = load_experiment(REPO_ROOT / "configs" / "experiments" / "full-sweep.yaml").experiment
    problems = [
        problem
        for suite in experiment.suites
        for problem in check_suite_assets(REPO_ROOT, suite)
    ]
    assert problems == [], render_problems(problems)


def test_an_unknown_suite_name_is_reported_not_raised(tmp_path: Path) -> None:
    problems = check_suite_assets(tmp_path, "no_such_suite")
    assert len(problems) == 1
    assert "unsupported suite runner" in problems[0].detail


def test_a_missing_fixture_is_reported(tmp_path: Path) -> None:
    problems = check_suite_assets(tmp_path, "hallucination_grounding")
    assert any("fixture missing" in problem.detail for problem in problems)


# --------------------------------------------------------------------- #
# Models                                                                #
# --------------------------------------------------------------------- #


def test_a_model_without_a_tag_is_caught(ollama_backend) -> None:
    from llm_benchmark.models import ModelConfig

    model = ModelConfig(
        name="tagless",
        family="x",
        revision="",
        quantization="q",
        source="ollama-local",
        context_length=4096,
    )
    problems = check_models(ollama_backend, [model])
    assert any("no artifact_path or revision" in problem.detail for problem in problems)


def test_reasoning_on_a_backend_without_it_is_caught_before_the_run() -> None:
    """Already refused at load_model, but that is hours in. Preflight says so
    while nothing has been loaded."""
    backend = load_backend(REPO_ROOT / "configs" / "backends" / "llamacpp.yaml")
    model = load_model_config(REPO_ROOT / "configs" / "models" / "gpt-oss-120b.yaml")
    problems = check_models(backend, [model.model_copy(update={"reasoning": True})])
    assert any("cannot enable" in problem.detail for problem in problems)


# --------------------------------------------------------------------- #
# Reporting                                                             #
# --------------------------------------------------------------------- #


def test_every_problem_is_reported_at_once(tmp_path: Path) -> None:
    """Stopping at the first would mean finding the second after the next
    eight-hour attempt."""
    problems = preflight(
        tmp_path,
        suite_names=["hallucination_grounding", "long_context_retrieval_fast", "no_such_suite"],
        model_configs=[],
        backend_config=load_backend(REPO_ROOT / "configs" / "backends" / "llamacpp.yaml"),
    )
    subjects = {problem.subject for problem in problems}
    assert {"hallucination_grounding", "long_context_retrieval_fast", "no_such_suite"} <= subjects


def test_the_fix_is_printed_with_the_problem() -> None:
    rendered = render_problems([Problem("suite", "corpus missing", "bash scripts/fetch_haystacks.sh")])
    assert "fix: bash scripts/fetch_haystacks.sh" in rendered
    assert "nothing was run" in rendered
