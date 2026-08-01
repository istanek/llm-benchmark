"""Is the new model better than the ones already measured?

The workflow this harness exists for is: point it at a model you have not
tried, and find out where it lands. Re-measuring the incumbents every time
costs hours (a full MBPP pass over three models took 4.5), so the comparison
has to run against stored results.

That only works if both sides were measured the same way, which is why this
module refuses rather than guesses. See ``llm_benchmark.provenance``: this
repo has twice changed what a suite measures without touching a fixture, and
a comparison across that boundary produces a confident, wrong verdict.

Verdicts use non-overlapping 95 % Wilson intervals, the same rule the rest of
the project reports with. Overlapping intervals are a tie — not a ranking with
a small gap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_benchmark.models import HarnessProvenance
from llm_benchmark.provenance import comparability_problems
from llm_benchmark.reporting import passes_total_from_model
from llm_benchmark.stats import wilson_interval


@dataclass(frozen=True)
class SuiteScore:
    suite: str
    suite_version: str
    model: str
    passes: int
    total: int

    @property
    def rate(self) -> float:
        return self.passes / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.passes, self.total)


@dataclass
class Bundle:
    path: Path
    scores: list[SuiteScore]
    provenance: HarnessProvenance | None

    @property
    def models(self) -> list[str]:
        seen: dict[str, None] = {}
        for score in self.scores:
            seen.setdefault(score.model, None)
        return list(seen)

    def score_for(self, suite: str, model: str) -> SuiteScore | None:
        return next((s for s in self.scores if s.suite == suite and s.model == model), None)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_bundle(bundle_dir: Path) -> Bundle:
    """Read every suite result under *bundle_dir*.

    A bundle is one directory per (suite, model) or per suite, each holding a
    ``summary.json`` and a ``manifest.json``. Suites are keyed by the fixture
    name in the summary rather than the directory name, so a bundle that ran
    one model per directory still compares against one that ran three.
    """
    scores: list[SuiteScore] = []
    provenance: HarnessProvenance | None = None
    for suite_dir in sorted(p for p in bundle_dir.iterdir() if p.is_dir()):
        summary = _read_json(suite_dir / "summary.json")
        if not summary:
            continue
        manifest = _read_json(suite_dir / "manifest.json") or {}
        if provenance is None and manifest.get("provenance"):
            provenance = HarnessProvenance.model_validate(manifest["provenance"])
        suite_name = str(summary.get("suite") or suite_dir.name)
        suite_version = str(summary.get("suite_version") or "unknown")
        for model in summary.get("models") or []:
            passes, total = passes_total_from_model(model)
            if not total:
                continue
            scores.append(
                SuiteScore(
                    suite=suite_name,
                    suite_version=suite_version,
                    model=str(model.get("model") or "unknown"),
                    passes=passes,
                    total=total,
                )
            )
    if not scores:
        raise ValueError(f"no suite summaries found under {bundle_dir}")
    return Bundle(path=bundle_dir, scores=scores, provenance=provenance)


def verdict_for(candidate: SuiteScore, baseline: SuiteScore) -> str:
    """better / worse / tie, by 95 % Wilson interval overlap."""
    c_low, c_high = candidate.interval
    b_low, b_high = baseline.interval
    if c_low > b_high:
        return "better"
    if c_high < b_low:
        return "worse"
    return "tie"


@dataclass
class Comparison:
    suite: str
    model: str
    baseline_model: str
    candidate: SuiteScore
    baseline: SuiteScore
    verdict: str
    fixture_mismatch: bool


def compare_bundles(
    candidate: Bundle,
    baseline: Bundle,
    *,
    models: list[str] | None = None,
) -> tuple[list[Comparison], list[str]]:
    """Compare candidate models against every baseline model, suite by suite.

    Returns ``(comparisons, blockers)``. A non-empty ``blockers`` list means
    the comparisons must not be reported as a result — the caller decides
    whether to stop or to print them under an explicit override.
    """
    candidate_models = models or [m for m in candidate.models]
    shared_models = [m for m in candidate_models if m in baseline.models]
    blockers = comparability_problems(
        candidate.provenance, baseline.provenance, models=shared_models
    )

    comparisons: list[Comparison] = []
    for model in candidate_models:
        for score in (s for s in candidate.scores if s.model == model):
            for baseline_score in (s for s in baseline.scores if s.suite == score.suite):
                if baseline_score.model == model and baseline.path == candidate.path:
                    continue
                comparisons.append(
                    Comparison(
                        suite=score.suite,
                        model=model,
                        baseline_model=baseline_score.model,
                        candidate=score,
                        baseline=baseline_score,
                        verdict=verdict_for(score, baseline_score),
                        fixture_mismatch=score.suite_version != baseline_score.suite_version,
                    )
                )
    for comparison in comparisons:
        if comparison.fixture_mismatch:
            blockers.append(
                f"{comparison.suite}: fixture version differs "
                f"({comparison.candidate.suite_version} vs {comparison.baseline.suite_version}) — "
                "the two runs did not answer the same questions"
            )
    # Same wording can arrive from several pairs; report each reason once.
    deduped: list[str] = []
    for problem in blockers:
        if problem not in deduped:
            deduped.append(problem)
    return comparisons, deduped


def render_comparison(
    comparisons: list[Comparison], blockers: list[str], *, forced: bool
) -> str:
    lines: list[str] = []
    if blockers:
        lines.append("REFUSED — these runs are not comparable:" if not forced else "WARNING — compared anyway, over these objections:")
        lines.extend(f"  - {problem}" for problem in blockers)
        lines.append("")
        if not forced:
            lines.append("Re-run the baseline models on the current harness, or pass --force to")
            lines.append("print the numbers anyway. A forced verdict is not evidence.")
            return "\n".join(lines)

    if not comparisons:
        lines.append(
            "No suite appears in both bundles, so there is nothing to compare. "
            "Run the candidate on the same suites as the baseline."
        )
        return "\n".join(lines)

    by_suite: dict[str, list[Comparison]] = {}
    for comparison in comparisons:
        by_suite.setdefault(comparison.suite, []).append(comparison)

    for suite, entries in by_suite.items():
        lines.append(f"## {suite}")
        lines.append("")
        lines.append(f"{'model':16s} {'score':>18s}   {'baseline':16s} {'score':>18s}   verdict")
        for entry in entries:
            c_low, c_high = entry.candidate.interval
            b_low, b_high = entry.baseline.interval
            candidate_cell = f"{entry.candidate.rate:.1%} ({c_low:.0%}-{c_high:.0%})"
            baseline_cell = f"{entry.baseline.rate:.1%} ({b_low:.0%}-{b_high:.0%})"
            # "tie with", "better than" — the preposition carries the meaning,
            # and a table people misread is a table that produces wrong calls.
            phrase = "tie with" if entry.verdict == "tie" else f"{entry.verdict} than"
            lines.append(
                f"{entry.model:16s} {candidate_cell:>18s}   "
                f"{entry.baseline_model:16s} {baseline_cell:>18s}   "
                f"{phrase} {entry.baseline_model}"
            )
        lines.append("")

    ties = sum(1 for c in comparisons if c.verdict == "tie")
    if ties:
        lines.append(
            f"{ties} of {len(comparisons)} pairings are ties: their 95 % intervals overlap, so the "
            "data does not separate them. That is an answer, not a missing one."
        )
    return "\n".join(lines)
