"""The report has to say what was asked of each model, not just how it scored.

Reasoning mode and the token budget are per-model now. A table of pass rates
without those columns implies a fair fight: a model handed four times the
budget of the others simply looks better. These tests cover the block that
prevents that reading, and the truncation count that qualifies a pass rate.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_benchmark.reporting import aggregate_runs, render_html_report, render_markdown_report

MANIFEST = {
    "experiment": {"name": "code-generation-v1"},
    "backend": {"name": "ollama"},
    "platform": {"display_name": "local"},
}
PROVENANCE = {
    "schema_version": "harness-provenance/v1",
    "git_commit": "abc123def456789",
    "git_dirty": False,
    "model_options": {
        "thinker": {"reasoning": True, "max_tokens": 8192, "quantization": "MXFP4", "artifact_path": "thinker:120b"},
        "plain": {"reasoning": False, "max_tokens": 1536, "quantization": "Q4_K_M", "artifact_path": "plain:30b"},
    },
}


def _write_run(root: Path, name: str, *, provenance: dict | None, finish_reasons: list[str]) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    manifest = dict(MANIFEST)
    if provenance is not None:
        manifest["provenance"] = provenance
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite": "code_generation_v1",
                "suite_version": "0.3.0",
                "models": [{"model": "plain", "passes": 3, "total": 5}],
            }
        )
    )
    rows = [
        {"model": "plain", "generation": {"finish_reason": reason, "metrics": {"decode_tokens": 100, "decode_time_s": 1.0}}}
        for reason in finish_reasons
    ]
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def test_truncated_answers_are_counted_per_model(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-a", provenance=PROVENANCE, finish_reasons=["stop", "length", "stop", "length", "stop"])
    aggregate = aggregate_runs(tmp_path)
    model = aggregate["suites"][0]["models"][0]
    assert model["truncated"] == 2


def test_markdown_flags_truncation_as_a_floor(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-a", provenance=PROVENANCE, finish_reasons=["stop", "length"])
    report = render_markdown_report(aggregate_runs(tmp_path))
    assert "hit the token budget" in report
    assert "floors" in report


def test_markdown_reports_per_model_options(tmp_path: Path) -> None:
    """The whole point: reasoning on and a raised budget must be visible."""
    _write_run(tmp_path, "run-a", provenance=PROVENANCE, finish_reasons=["stop"])
    report = render_markdown_report(aggregate_runs(tmp_path))
    assert "## Run conditions" in report
    assert "abc123def456" in report
    assert "| thinker | on | 8192 |" in report
    assert "| plain | off | 1536 |" in report


def test_a_dirty_tree_is_called_out(tmp_path: Path) -> None:
    dirty = {**PROVENANCE, "git_dirty": True}
    _write_run(tmp_path, "run-a", provenance=dirty, finish_reasons=["stop"])
    assert "dirty working tree" in render_markdown_report(aggregate_runs(tmp_path))


def test_runs_from_different_commits_are_flagged(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-a", provenance=PROVENANCE, finish_reasons=["stop"])
    _write_run(tmp_path, "run-b", provenance={**PROVENANCE, "git_commit": "999"}, finish_reasons=["stop"])
    aggregate = aggregate_runs(tmp_path)
    assert aggregate["conditions"]["mixed_harness"] is True
    report = render_markdown_report(aggregate)
    assert "different harness commits" in report


def test_runs_without_provenance_say_so(tmp_path: Path) -> None:
    """Every bundle measured before provenance existed lands here."""
    _write_run(tmp_path, "run-a", provenance=None, finish_reasons=["stop"])
    aggregate = aggregate_runs(tmp_path)
    assert aggregate["conditions"]["stamped_runs"] == 0
    report = render_markdown_report(aggregate)
    assert "No provenance recorded" in report


def test_html_report_carries_the_same_block(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-a", provenance=PROVENANCE, finish_reasons=["stop"])
    html = render_html_report(aggregate_runs(tmp_path))
    assert "Run conditions" in html
    assert "thinker:120b" in html
    assert "8192" in html


def test_html_report_still_renders_without_provenance(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-a", provenance=None, finish_reasons=["stop"])
    html = render_html_report(aggregate_runs(tmp_path))
    assert "No provenance recorded" in html
    assert html.strip().startswith("<!")
