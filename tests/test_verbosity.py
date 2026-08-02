"""Verbosity as a reported axis rather than an error term.

gpt-oss-120b lost 48 MBPP tasks to a token budget chosen for three terser
models, and the suite recorded that as bad code. These tests cover the numbers
that keep the two apart.
"""

from __future__ import annotations

from llm_benchmark.verbosity import collect_verbosity


def _row(model: str, *, tokens: int, finish: str = "stop", passed: bool = True, output: str = "x" * 100, code: str = "x" * 50) -> dict:
    return {
        "model": model,
        "samples": [
            {
                "extracted_code": code,
                "generation": {"output": output, "finish_reason": finish, "metrics": {"decode_tokens": tokens}},
                "sandbox": {"passed": passed},
            }
        ],
    }


def test_length_quantiles_and_median() -> None:
    rows = [_row("m", tokens=t) for t in (100, 200, 300, 400, 1000)]
    stats = collect_verbosity(rows)[0]
    assert stats["median_tokens"] == 300
    assert stats["p90_tokens"] == 1000


def test_a_quantile_on_the_budget_is_reported_as_censored() -> None:
    """p90 == budget does not mean 90 % fit; it means we cannot see past it."""
    rows = [_row("m", tokens=1536, finish="length", passed=False) for _ in range(5)]
    stats = collect_verbosity(rows, budget=1536)[0]
    assert stats["censored"] is True


def test_a_distribution_below_the_budget_is_not_censored() -> None:
    rows = [_row("m", tokens=200) for _ in range(5)]
    assert collect_verbosity(rows, budget=1536)[0]["censored"] is False


def test_answer_density_measures_how_much_was_the_answer() -> None:
    rows = [_row("m", tokens=100, output="y" * 1000, code="y" * 100)]
    assert collect_verbosity(rows)[0]["answer_density"] == 0.1


def test_density_is_capped_when_extraction_re_injects_imports() -> None:
    """Extraction adds prompt imports, so extracted can exceed the raw output."""
    rows = [_row("m", tokens=100, output="y" * 10, code="y" * 500)]
    assert collect_verbosity(rows)[0]["answer_density"] == 1.0


def test_cost_per_solved_task() -> None:
    rows = [_row("m", tokens=100, passed=True), _row("m", tokens=300, passed=False)]
    stats = collect_verbosity(rows)[0]
    assert stats["tokens_per_task"] == 200.0
    assert stats["tokens_per_solved"] == 400.0


def test_the_score_is_bracketed_by_what_truncation_could_hide() -> None:
    """Floor counts truncated answers as failures; ceiling drops them."""
    rows = [_row("m", tokens=100, passed=True) for _ in range(8)]
    rows += [_row("m", tokens=1536, finish="length", passed=False) for _ in range(2)]
    stats = collect_verbosity(rows, budget=1536)[0]
    assert stats["truncated"] == 2
    assert stats["pass_floor"] == 0.8
    assert stats["pass_ceiling"] == 1.0


def test_no_truncation_collapses_the_bracket() -> None:
    rows = [_row("m", tokens=100, passed=True) for _ in range(9)] + [_row("m", tokens=100, passed=False)]
    stats = collect_verbosity(rows)[0]
    assert stats["pass_floor"] == stats["pass_ceiling"] == 0.9


def test_the_flat_generation_shape_is_read_too() -> None:
    """Reliability suites store one generation per row, not a samples list."""
    rows = [
        {
            "model": "m",
            "generation": {"output": "hi", "finish_reason": "stop", "metrics": {"decode_tokens": 42}},
            "evaluation": {"passed": True},
        }
    ]
    stats = collect_verbosity(rows)[0]
    assert stats["tasks"] == 1 and stats["passes"] == 1 and stats["median_tokens"] == 42


def test_models_are_kept_apart() -> None:
    rows = [_row("terse", tokens=100), _row("wordy", tokens=900)]
    stats = {entry["model"]: entry for entry in collect_verbosity(rows)}
    assert stats["terse"]["median_tokens"] == 100
    assert stats["wordy"]["median_tokens"] == 900
