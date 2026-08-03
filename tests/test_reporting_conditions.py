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


def test_reused_generations_are_declared_in_both_reports(tmp_path: Path) -> None:
    """A re-scored bundle must not read as a fresh measurement of the models."""
    provenance = {
        **PROVENANCE,
        "generations_from": {
            "bundle": "20260802T051407Z-51ad314a",
            "git_commit": "e36aa36018f7",
            "note": "answers reused; only scoring was re-run",
        },
    }
    _write_run(tmp_path, "run-a", provenance=provenance, finish_reasons=["stop"])
    aggregate = aggregate_runs(tmp_path)
    assert aggregate["conditions"]["generations_from"]["bundle"] == "20260802T051407Z-51ad314a"

    markdown = render_markdown_report(aggregate)
    assert "Answers were not generated by this run" in markdown
    assert "20260802T051407Z-51ad314a" in markdown

    html = render_html_report(aggregate)
    assert "Answers were not generated by this run" in html


def test_unscorable_answers_are_declared_in_the_report(tmp_path: Path) -> None:
    """The denominator shrinks when an answer cannot be read, so the report has
    to say how often that happened."""
    run_dir = tmp_path / "grounding"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({**MANIFEST, "provenance": PROVENANCE}))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite": "hallucination_grounding_v1",
                "suite_version": "0.3.0",
                "models": [
                    {"model": "plain", "passes": 8, "total": 9, "unscorable": 2, "unscorable_task_ids": ["t1", "t2"]}
                ],
            }
        )
    )
    (run_dir / "results.jsonl").write_text("")

    report = render_markdown_report(aggregate_runs(tmp_path))
    assert "could not be scored" in report
    assert "not a measurement" in report


def test_energy_reaches_the_report(tmp_path: Path) -> None:
    """It is written to summary.json at run time and cannot be recovered later,
    so a field that never reaches the report is a measurement nobody sees."""
    run_dir = tmp_path / "code_generation-m"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({**MANIFEST, "provenance": PROVENANCE}))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite": "code_generation_v1",
                "suite_version": "0.3.0",
                "models": [
                    {
                        "model": "thrifty",
                        "passes": 10,
                        "total": 10,
                        "energy": {
                            "energy_j": 1250.0, "seconds": 60.0, "avg_power_w": 35.0,
                            "samples": 120, "telemetry_source": "nvidia-smi",
                            "baseline_power_w": 13.5, "joules_per_solved_task": 125.0,
                            "tasks_per_wh": 28.8,
                        },
                    },
                    {
                        "model": "costly",
                        "passes": 10,
                        "total": 10,
                        "energy": {
                            "energy_j": 18690.0, "seconds": 1056.0, "avg_power_w": 51.3,
                            "samples": 2116, "telemetry_source": "nvidia-smi",
                            "baseline_power_w": 21.7, "joules_per_solved_task": 1869.0,
                            "tasks_per_wh": 1.93,
                        },
                    },
                ],
            }
        )
    )
    (run_dir / "results.jsonl").write_text("")

    report = render_markdown_report(aggregate_runs(tmp_path))
    assert "### Energy" in report
    assert "1869 J" in report and "125 J" in report
    # The lowest baseline, not the first encountered: later models carry the
    # previous one's residual draw.
    assert "host baseline 13.5 W" in report
