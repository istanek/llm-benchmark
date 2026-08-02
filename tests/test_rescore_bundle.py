"""Re-scoring stored answers, including the reliability suites.

This tool decides published numbers without re-asking any model, so its
failure modes are quiet by construction: a suite silently skipped, or a run
whose provenance claims it was measured fresh.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rescore():
    spec = importlib.util.spec_from_file_location("rescore_bundle", REPO_ROOT / "scripts" / "rescore_bundle.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rescore_bundle"] = module
    spec.loader.exec_module(module)
    return module


def _grounding_run(root: Path, *, output: str) -> Path:
    """One grounding row, scored as a pass by whatever wrote it."""
    run_dir = root / "hallucination_grounding"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"backend": {"name": "ollama", "entrypoint": "ollama", "version": "test"}})
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite": "hallucination_grounding_v1",
                "suite_version": "0.3.0",
                "models": [{"model": "m", "passes": 1, "total": 1}],
            }
        )
    )
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "model": "m",
                "task_id": "hg-v1-002-abstain-lead-engineer",
                "repetition": 1,
                "generation": {"output": output, "finish_reason": "stop"},
                "evaluation": {"passed": True, "score": 1, "reason": "recorded_by_the_old_scorer"},
            }
        )
        + "\n"
    )
    return run_dir


def test_reliability_rows_are_rescored_not_copied(tmp_path: Path, rescore) -> None:
    """A grounding fix must reach grounding results. The tool handled only the
    code suites at first, so "re-score the bundle" quietly left them alone."""
    run_dir = _grounding_run(tmp_path / "bundle", output="The lead engineer was probably Novak.")
    record = rescore.rescore_reliability_run(
        run_dir, tmp_path / "out" / "hallucination_grounding", "hallucination_grounding_v1"
    )
    assert record is not None
    # Recorded as a pass, but it is a hedged guess after an abstention.
    assert record["before"] == 1
    assert record["after"] == 0


def test_a_correct_abstention_survives_rescoring(tmp_path: Path, rescore) -> None:
    run_dir = _grounding_run(tmp_path / "bundle", output="The context does not mention the lead engineer.")
    record = rescore.rescore_reliability_run(
        run_dir, tmp_path / "out" / "hallucination_grounding", "hallucination_grounding_v1"
    )
    assert record["after"] == 1


def test_rescored_summary_is_written_in_the_original_shape(tmp_path: Path, rescore) -> None:
    """Reports and compare read summary.json; a re-scored bundle that changed
    its shape would be unreadable by everything downstream."""
    run_dir = _grounding_run(tmp_path / "bundle", output="The context does not mention the lead engineer.")
    out_dir = tmp_path / "out" / "hallucination_grounding"
    rescore.rescore_reliability_run(run_dir, out_dir, "hallucination_grounding_v1")

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["suite"] == "hallucination_grounding_v1"
    model = summary["models"][0]
    assert model["model"] == "m"
    assert (model["passes"], model["total"]) == (1, 1)


def test_long_context_is_not_claimed_to_be_rescorable(rescore) -> None:
    """Its verdict depends on how the haystack was assembled at generation
    time, which the row does not record. Re-scoring it would be guessing."""
    assert "long_context" not in " ".join(rescore.RELIABILITY_SUITES)
