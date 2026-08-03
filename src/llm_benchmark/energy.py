"""What a correct answer costs in joules.

The report already carries decode tokens per solved task, which is a proxy for
cost. Energy is the thing itself, and on this class of machine it is the axis
that decides what you can actually run: gemma-4 prefills a 131k context at 344
tok/s against nemotron-3's 2 498, and gpt-oss-120b spends 5.6x qwen-3.6's
tokens per solved problem. None of that is visible in a pass rate.

Two numbers are reported per model:

- **J/solved** — energy drawn while the model was answering, divided by the
  tasks it got right. Wrong answers cost energy too, so a model that is both
  slow and inaccurate is penalised twice, which is the honest accounting for
  "what does a working result cost me".
- **Tasks/Wh** — the same quantity the other way up, for comparing against a
  power budget rather than a task list.

Both are *measured*, not modelled: power is sampled while the suite runs and
integrated over the model's own window. A figure derived from a datasheet TDP
would be a specification, not a result.

The idle draw of the machine is recorded alongside, because it is not free and
it is not the model's fault. A run that reports 200 J/solved on a host idling
at 15 W is telling you something different from the same figure on a host
idling at 90 W, and subtracting it silently would hide which one you have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnergyWindow:
    """Energy drawn over one measured interval, integrated from power samples."""

    model: str
    joules: float = 0.0
    seconds: float = 0.0
    samples: int = 0
    # Sampled immediately before this model runs, which for any model after
    # the first includes the previous one's residual draw: measured baselines
    # across one four-model bundle were 13.5 / 23.5 / 25.4 / 21.7 W in run
    # order. Only the first is a cold idle. Named "baseline" rather than
    # "idle" so nobody subtracts it believing it is one.
    baseline_watts: float | None = None
    source: str = "none"

    @property
    def average_watts(self) -> float:
        return self.joules / self.seconds if self.seconds else 0.0

    def joules_per_solved(self, solved: int) -> float | None:
        return self.joules / solved if solved else None

    def tasks_per_wh(self, solved: int) -> float | None:
        """Wh = 3600 J. Reported for comparing against a power budget."""
        return (solved / (self.joules / 3600.0)) if self.joules else None

    def to_dict(self, *, solved: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "energy_j": round(self.joules, 1),
            "seconds": round(self.seconds, 1),
            "avg_power_w": round(self.average_watts, 1),
            "samples": self.samples,
            "telemetry_source": self.source,
            "baseline_power_w": round(self.baseline_watts, 1) if self.baseline_watts is not None else None,
        }
        if solved is not None:
            per_solved = self.joules_per_solved(solved)
            payload["joules_per_solved_task"] = round(per_solved, 1) if per_solved is not None else None
            per_wh = self.tasks_per_wh(solved)
            payload["tasks_per_wh"] = round(per_wh, 2) if per_wh is not None else None
        return payload


@dataclass
class EnergyMeter:
    """Samples power for the duration of a model's run and integrates it.

    Integration is trapezoidal over the sample timestamps rather than
    ``average_power x duration``: the poller's cadence drifts under load, and
    the periods where it drifts most are exactly the periods drawing most
    power. Weighting each sample by the interval it actually covers avoids
    over-counting the cheap gaps.
    """

    hz: float = 2.0
    _sampler: Any | None = field(default=None, repr=False)
    _t0: float = 0.0

    def available(self) -> bool:
        return self.source != "none"

    @property
    def source(self) -> str:
        return getattr(self._sampler, "source", "none") if self._sampler else "none"

    def start(self) -> None:
        import time

        # Imported here: sustained_throughput pulls in the whole suite runner,
        # and the code suite should not carry that just to weigh its answers.
        from llm_benchmark.sustained_throughput import TelemetrySampler

        self._sampler = TelemetrySampler(hz=self.hz)
        self._t0 = time.monotonic()
        self._sampler.start(self._t0)

    def stop(self, model: str, *, baseline_watts: float | None = None) -> EnergyWindow:
        import time

        elapsed = time.monotonic() - self._t0
        if self._sampler is None:
            return EnergyWindow(model=model, seconds=elapsed, baseline_watts=baseline_watts)
        self._sampler.stop()
        window = integrate(self._sampler.samples, model=model, fallback_seconds=elapsed)
        window.baseline_watts = baseline_watts
        window.source = self._sampler.source
        return window


def integrate(samples: list[Any], *, model: str, fallback_seconds: float = 0.0) -> EnergyWindow:
    """Trapezoidal integration of ``gpu_power_w`` over sample timestamps."""
    points: list[tuple[float, float]] = []
    for sample in samples:
        # The field is timestamp_s; guessing "t" produced an empty integral
        # and a confident 0.0 J, which is the worst possible failure for a
        # measurement — plausible, silent and wrong.
        timestamp = getattr(sample, "timestamp_s", None)
        # gpu_power_w on NVIDIA/AMD; Apple reports system power only.
        watts = getattr(sample, "gpu_power_w", None)
        if watts is None:
            watts = getattr(sample, "system_power_w", None)
        if watts is None or timestamp is None:
            continue
        points.append((float(timestamp), float(watts)))
    points.sort()
    if len(points) < 2:
        return EnergyWindow(model=model, seconds=fallback_seconds, samples=len(points))

    joules = 0.0
    for (t0, w0), (t1, w1) in zip(points, points[1:]):
        joules += (w0 + w1) / 2.0 * (t1 - t0)
    return EnergyWindow(
        model=model,
        joules=joules,
        seconds=points[-1][0] - points[0][0],
        samples=len(points),
    )


def measure_baseline_watts(seconds: float = 3.0, hz: float = 2.0) -> float | None:
    """Average draw right now, as context for the figures that follow.

    Not necessarily an idle machine: called between models, it sees the
    previous one still settling. The lowest baseline in a bundle is the only
    one that approximates a cold host.
    """
    import time

    from llm_benchmark.sustained_throughput import TelemetrySampler

    sampler = TelemetrySampler(hz=hz)
    if sampler.source == "none":
        return None
    t0 = time.monotonic()
    sampler.start(t0)
    time.sleep(seconds)
    sampler.stop()
    readings = [
        float(watts)
        for sample in sampler.samples
        if (watts := getattr(sample, "gpu_power_w", None) or getattr(sample, "system_power_w", None))
        is not None
    ]
    return sum(readings) / len(readings) if readings else None
