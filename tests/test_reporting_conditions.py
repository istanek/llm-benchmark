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


def _priced_suite(leader_energy: float, rival_energy: float, leader_rate=(29, 30), rival_rate=(29, 30)):
    """Two models tied on quality, priced differently."""
    from llm_benchmark.stats import wilson_interval

    models = []
    for name, (passes, total), joules in (
        ("leader", leader_rate, leader_energy),
        ("rival", rival_rate, rival_energy),
    ):
        low, high = wilson_interval(passes, total)
        models.append(
            {
                "model": name,
                "passes": passes,
                "total": total,
                "pass_rate": passes / total,
                "ci_low": low,
                "ci_high": high,
                "energy": {"joules_per_solved_task": joules, "samples": 10},
            }
        )
    return {"suite": "code_v1", "scoring": "pass_fail", "models": models}


def test_a_tie_on_quality_is_decided_by_cost() -> None:
    """The ranking crowns whoever is a fraction of a point ahead, even when the
    interval says that fraction is noise and the runner-up costs 15x less."""
    from llm_benchmark.reporting import cost_aware_pick

    aggregate = {"suites": [_priced_suite(leader_energy=1869.0, rival_energy=125.0)]}
    pick = cost_aware_pick(aggregate, [{"model": "leader"}, {"model": "rival"}])
    assert pick is not None
    assert pick["recommended"] == "rival"
    assert round(pick["ratio"]) == 15


def test_the_leader_keeps_the_recommendation_when_it_is_also_cheapest() -> None:
    from llm_benchmark.reporting import cost_aware_pick

    aggregate = {"suites": [_priced_suite(leader_energy=125.0, rival_energy=1869.0)]}
    assert cost_aware_pick(aggregate, [{"model": "leader"}, {"model": "rival"}]) is None


def test_a_model_outside_the_leaders_interval_is_never_recommended_on_price() -> None:
    """Quality first: cheap and clearly worse is not a recommendation."""
    from llm_benchmark.reporting import cost_aware_pick

    aggregate = {
        "suites": [
            _priced_suite(
                leader_energy=1869.0, rival_energy=125.0,
                leader_rate=(400, 400), rival_rate=(200, 400),
            )
        ]
    }
    assert cost_aware_pick(aggregate, [{"model": "leader"}, {"model": "rival"}]) is None


def test_no_energy_data_means_no_cost_claim() -> None:
    from llm_benchmark.reporting import cost_aware_pick

    suite = _priced_suite(leader_energy=1869.0, rival_energy=125.0)
    for model in suite["models"]:
        model.pop("energy")
    assert cost_aware_pick({"suites": [suite]}, [{"model": "leader"}, {"model": "rival"}]) is None


def _suite_with_speed(name: str, models: list[tuple[str, float, float]]) -> dict:
    """(model, pass_rate, tokens_per_s) — the shape aggregate_runs produces."""
    return {
        "suite": name,
        "scoring": "pass_fail",
        "models": [
            {
                "model": model,
                "passes": int(rate * 100),
                "total": 100,
                "pass_rate": rate,
                "avg_tokens_per_s": tps,
                "avg_ttft_ms": 200.0,
            }
            for model, rate, tps in models
        ],
    }


def test_speed_is_read_from_ordinary_suites_when_the_speed_suite_is_absent() -> None:
    """It used to be read only from openclaw_speed. That suite has not run in
    any recent bundle, so every model scored 0.0 on speed and the weight
    vanished from the total — raising it would have changed nothing."""
    from llm_benchmark.reporting import speed_signals

    aggregate = {"suites": [_suite_with_speed("code_v1", [("fast", 0.9, 75.0), ("slow", 0.9, 10.0)])]}
    ttfts, toks, source = speed_signals(aggregate, ["fast", "slow"])
    assert source == "suite metrics"
    assert toks == {"fast": 75.0, "slow": 10.0}


def test_the_dedicated_speed_suite_still_wins_when_present() -> None:
    from llm_benchmark.reporting import speed_signals

    aggregate = {
        "suites": [
            _suite_with_speed("code_v1", [("m", 0.9, 10.0)]),
            _suite_with_speed("openclaw_speed", [("m", 1.0, 99.0)]),
        ]
    }
    _, toks, source = speed_signals(aggregate, ["m"])
    assert source == "openclaw_speed"
    assert toks == {"m": 99.0}


def test_equal_quality_and_unequal_speed_separates_the_ranking() -> None:
    """The case that prompted the reweighting: two models tied on quality, one
    seven times slower, previously ranked by a coin flip in the third decimal."""
    from llm_benchmark.reporting import _overall_rank_rows

    aggregate = {"suites": [_suite_with_speed("code_generation_v1", [("fast", 0.9, 75.0), ("slow", 0.9, 10.0)])]}
    ranking = _overall_rank_rows(aggregate, ["fast", "slow"])
    assert [row["model"] for row in ranking] == ["fast", "slow"]
    assert ranking[0]["overall_score"] > ranking[1]["overall_score"]


def test_no_speed_signal_redistributes_the_weight_to_quality() -> None:
    """Scoring everyone 0.0 on an unmeasured axis is a claim; "not measured"
    is not. Without speed data the score is quality alone, not quality x 0.6."""
    from llm_benchmark.reporting import _overall_rank_rows

    suite = _suite_with_speed("code_generation_v1", [("m", 0.9, 0.0)])
    for model in suite["models"]:
        model["avg_tokens_per_s"] = None
        model["avg_ttft_ms"] = None
    ranking = _overall_rank_rows({"suites": [suite]}, ["m"])
    assert ranking[0]["overall_score"] == ranking[0]["quality_score"] == 0.9


def test_the_axis_summary_marks_a_quality_tie_rather_than_a_winner() -> None:
    """A composite score always names someone first. The axis view says when
    the run did not actually separate anyone, which is the more useful fact."""
    from llm_benchmark.reporting import axis_summary
    from llm_benchmark.stats import wilson_interval

    models = []
    for name, passes in (("a", 29), ("b", 28)):
        low, high = wilson_interval(passes, 30)
        models.append(
            {"model": name, "passes": passes, "total": 30, "pass_rate": passes / 30,
             "ci_low": low, "ci_high": high, "avg_tokens_per_s": 50.0}
        )
    axes = axis_summary({"suites": [{"suite": "code_v1", "scoring": "pass_fail", "models": models}]})
    quality = next(axis for axis in axes if axis["axis"] == "quality")
    assert quality["separated"] is False
    assert quality["tied_with_leader"] == ["b"]


def test_the_axis_summary_reports_the_spread_where_a_run_does_separate() -> None:
    from llm_benchmark.reporting import axis_summary
    from llm_benchmark.stats import wilson_interval

    models = []
    for name, tps in (("fast", 75.0), ("slow", 10.0)):
        low, high = wilson_interval(29, 30)
        models.append(
            {"model": name, "passes": 29, "total": 30, "pass_rate": 29 / 30,
             "ci_low": low, "ci_high": high, "avg_tokens_per_s": tps,
             "energy": {"joules_per_solved_task": 125.0 if name == "fast" else 1869.0, "samples": 5}}
        )
    axes = axis_summary({"suites": [{"suite": "code_v1", "scoring": "pass_fail", "models": models}]})
    throughput = next(axis for axis in axes if axis["axis"] == "throughput")
    energy = next(axis for axis in axes if axis["axis"] == "energy")
    assert throughput["leader"] == "fast" and "7.5x" in throughput["spread"]
    assert energy["leader"] == "fast" and "15x" in energy["spread"]


def test_axes_appear_before_the_overall_score_in_the_report(tmp_path: Path) -> None:
    """Ordering is the point: the merged number reads as the answer when it is
    printed first."""
    run_dir = tmp_path / "code_generation-m"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({**MANIFEST, "provenance": PROVENANCE}))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite": "code_generation_v1",
                "suite_version": "0.3.0",
                "models": [{"model": "m", "passes": 9, "total": 10}],
            }
        )
    )
    (run_dir / "results.jsonl").write_text("")
    report = render_markdown_report(aggregate_runs(tmp_path))
    assert "## Where they differ" in report
    assert report.index("## Where they differ") < report.index("## code_generation_v1")
