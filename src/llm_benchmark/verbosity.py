"""How much a model says, treated as a result rather than a nuisance.

gpt-oss-120b scored 78.9 % on MBPP where gemma-4 scored 87.6 %, and 48 of its
90 failures were answers cut off at the shared 1536-token budget. The obvious
reading — "it is worse at coding" — may well be wrong. What the run actually
established is that it writes far more, and that a budget chosen for three
terser models truncates it.

The wrong fix is a bigger budget for the verbose model: that hides the cost
rather than reporting it, and makes the comparison unequal in the other
direction. Verbosity is a real property with real consequences — decode time,
money, context pressure — so it belongs in the report as its own axis, next to
the pass rate rather than folded into it.

What is measured here comes entirely from stored runs; nothing needs re-asking:

- **Length distribution** (median / p90 / p99 decode tokens). When a quantile
  lands on the budget the distribution is *censored* — the real value is
  unknown and higher — and that is reported rather than papered over.
- **Answer density**: the fraction of the output that survived extraction, i.e.
  how much of it was the answer rather than prose about the answer.
- **Cost per solved task**: total decode tokens divided by tasks solved. The
  practical question is not only who is right most often, but what being right
  costs.
- **A score bracket**: the pass rate with truncated answers counted as failures
  (a floor, since some were on their way to correct) and with them excluded (a
  ceiling, since some would have failed anyway). The true value is inside; a
  single number pretending to be it is the thing to avoid.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerbosityStats:
    model: str
    tasks: int = 0
    passes: int = 0
    truncated: int = 0
    tokens: list[int] = field(default_factory=list)
    code_shares: list[float] = field(default_factory=list)
    budget: int | None = None

    def _quantile(self, fraction: float) -> int:
        if not self.tokens:
            return 0
        ordered = sorted(self.tokens)
        return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]

    @property
    def median_tokens(self) -> int:
        return int(statistics.median(self.tokens)) if self.tokens else 0

    @property
    def p90_tokens(self) -> int:
        return self._quantile(0.90)

    @property
    def p99_tokens(self) -> int:
        return self._quantile(0.99)

    @property
    def censored(self) -> bool:
        """True when the budget cut the distribution short.

        A p90 sitting exactly on the budget does not mean "90 % of answers fit
        in 1536 tokens" — it means at least 10 % wanted more and we cannot say
        how much more. Reporting such a quantile as a measurement would state
        the budget as if it were a property of the model.
        """
        return bool(self.budget) and self.p90_tokens >= self.budget

    @property
    def answer_density(self) -> float:
        """Median share of the output that survived extraction."""
        return statistics.median(self.code_shares) if self.code_shares else 0.0

    @property
    def tokens_per_task(self) -> float:
        return sum(self.tokens) / self.tasks if self.tasks else 0.0

    @property
    def tokens_per_solved(self) -> float:
        """Decode tokens spent per task actually solved."""
        return sum(self.tokens) / self.passes if self.passes else float("inf")

    @property
    def truncation_rate(self) -> float:
        return self.truncated / self.tasks if self.tasks else 0.0

    @property
    def pass_floor(self) -> float:
        """Truncated answers counted as failures — how the suite scores today."""
        return self.passes / self.tasks if self.tasks else 0.0

    @property
    def pass_ceiling(self) -> float:
        """Truncated answers excluded rather than failed.

        An upper bound, not an estimate: some of those answers would have been
        wrong anyway. Reported alongside the floor so the width of the unknown
        is visible instead of implied.
        """
        scored = self.tasks - self.truncated
        return self.passes / scored if scored else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tasks": self.tasks,
            "passes": self.passes,
            "truncated": self.truncated,
            "truncation_rate": round(self.truncation_rate, 4),
            "median_tokens": self.median_tokens,
            "p90_tokens": self.p90_tokens,
            "p99_tokens": self.p99_tokens,
            "censored": self.censored,
            "answer_density": round(self.answer_density, 4),
            "tokens_per_task": round(self.tokens_per_task, 1),
            "tokens_per_solved": round(self.tokens_per_solved, 1) if self.passes else None,
            "pass_floor": round(self.pass_floor, 4),
            "pass_ceiling": round(self.pass_ceiling, 4),
            "budget": self.budget,
        }


def collect_verbosity(rows: list[dict[str, Any]], *, budget: int | None = None) -> list[dict[str, Any]]:
    """Per-model verbosity stats from ``results.jsonl`` rows.

    Works on code-suite rows (``samples[*].generation``) and on the flat
    ``generation`` shape the reliability suites write, so any suite can be read
    without special-casing at the call site.
    """
    per_model: dict[str, VerbosityStats] = {}
    for row in rows:
        model = row.get("model")
        if not model:
            continue
        stats = per_model.setdefault(model, VerbosityStats(model=model, budget=budget))
        samples = row.get("samples")
        if samples:
            entries = [
                ((sample or {}).get("generation") or {}, (sample or {}).get("sandbox") or {}, sample or {})
                for sample in samples
            ]
        else:
            entries = [(row.get("generation") or {}, {}, {})]

        stats.tasks += 1
        passed = any(sandbox.get("passed") for _, sandbox, _ in entries) or bool(
            (row.get("evaluation") or {}).get("passed")
        )
        stats.passes += int(passed)
        for generation, _, sample in entries:
            metrics = generation.get("metrics") or {}
            stats.tokens.append(int(metrics.get("decode_tokens") or 0))
            if generation.get("finish_reason") == "length":
                stats.truncated += 1
            output = generation.get("output") or ""
            extracted = sample.get("extracted_code") or ""
            if output and extracted:
                # Capped at 1.0: extraction re-injects imports from the prompt,
                # so a terse answer can come out longer than the raw output.
                stats.code_shares.append(min(1.0, len(extracted) / len(output)))
    return [stats.to_dict() for stats in per_model.values()]
