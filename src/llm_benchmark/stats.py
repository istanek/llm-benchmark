"""Small statistical helpers shared by suite runners and reporting.

The suites in this repo are deliberately small (single-digit to low-double-digit
task counts), so a bare pass rate overstates how much the numbers separate two
models. Every pass rate we display should therefore travel with its sample size
and an interval.
"""

from __future__ import annotations

import math

# 95 % two-sided normal quantile.
DEFAULT_Z = 1.959963984540054


def wilson_interval(passes: int, total: int, z: float = DEFAULT_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    remains sane for the small ``n`` and extreme (0 % / 100 %) rates these
    suites routinely produce.
    """
    if total <= 0:
        return 0.0, 1.0
    passes = max(0, min(int(passes), int(total)))
    n = float(total)
    p = passes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    margin = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_margin(passes: int, total: int, z: float = DEFAULT_Z) -> float:
    """Half-width of the Wilson interval, for compact ``x% ± y pp`` display."""
    low, high = wilson_interval(passes, total, z)
    return (high - low) / 2.0


def format_rate_with_ci(passes: int, total: int) -> str:
    """Render a pass rate as ``62% (n=13, 95% CI 36–83%)``."""
    if total <= 0:
        return "-"
    low, high = wilson_interval(passes, total)
    return f"{passes / total:.0%} (n={total}, 95% CI {low:.0%}–{high:.0%})"
