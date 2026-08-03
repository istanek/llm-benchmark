"""Joules per solved task.

Energy is the axis a single-machine benchmark is best placed to measure and
the one nobody can add retroactively — a finished run holds no record of what
it drew. These tests cover the integration and the ways it can quietly report
zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_benchmark.energy import EnergyWindow, integrate


@dataclass
class FakeSample:
    """Mirrors sustained_throughput.TelemetrySample's field names."""

    timestamp_s: float
    gpu_power_w: float | None = None
    system_power_w: float | None = None


def test_constant_power_integrates_to_power_times_time() -> None:
    samples = [FakeSample(timestamp_s=float(i), gpu_power_w=100.0) for i in range(11)]
    window = integrate(samples, model="m")
    assert window.seconds == 10.0
    assert window.joules == 1000.0
    assert window.average_watts == 100.0


def test_a_ramp_is_integrated_trapezoidally() -> None:
    """0 W to 100 W over 10 s is 500 J, not 1000."""
    samples = [FakeSample(timestamp_s=float(i), gpu_power_w=i * 10.0) for i in range(11)]
    assert integrate(samples, model="m").joules == 500.0


def test_uneven_sample_spacing_is_weighted_by_the_gap() -> None:
    """The poller drifts under load, and the periods it drifts through are the
    expensive ones. Weighting by elapsed interval is the whole point of
    integrating rather than averaging."""
    samples = [
        FakeSample(timestamp_s=0.0, gpu_power_w=10.0),
        FakeSample(timestamp_s=1.0, gpu_power_w=10.0),
        FakeSample(timestamp_s=11.0, gpu_power_w=110.0),
    ]
    # 10 J over the first second, then a 10 s trapezoid averaging 60 W.
    assert integrate(samples, model="m").joules == 10.0 + 600.0


def test_the_wrong_field_name_does_not_silently_read_as_zero() -> None:
    """The first version looked for `t` instead of `timestamp_s` and reported a
    confident 0.0 J — plausible, silent and wrong, which is the worst failure a
    measurement can have. A sample with no usable timestamp is dropped, and
    fewer than two points yields no energy claim at all."""

    @dataclass
    class Mislabelled:
        t: float
        gpu_power_w: float

    window = integrate([Mislabelled(t=float(i), gpu_power_w=100.0) for i in range(5)], model="m")
    assert window.samples == 0
    assert window.joules == 0.0


def test_system_power_is_used_when_gpu_power_is_absent() -> None:
    """Apple reports system power only."""
    samples = [FakeSample(timestamp_s=float(i), system_power_w=50.0) for i in range(3)]
    assert integrate(samples, model="m").joules == 100.0


def test_a_single_sample_cannot_be_integrated() -> None:
    window = integrate([FakeSample(timestamp_s=0.0, gpu_power_w=99.0)], model="m", fallback_seconds=7.0)
    assert window.joules == 0.0
    assert window.seconds == 7.0


def test_cost_per_solved_task_counts_energy_spent_on_wrong_answers() -> None:
    """A model that is slow and wrong is penalised twice, which is the honest
    accounting for "what does a working result cost me"."""
    window = EnergyWindow(model="m", joules=6000.0, seconds=60.0)
    assert window.joules_per_solved(10) == 600.0
    assert window.joules_per_solved(20) == 300.0


def test_tasks_per_wh_is_the_same_number_upside_down() -> None:
    window = EnergyWindow(model="m", joules=3600.0, seconds=60.0)
    assert window.tasks_per_wh(10) == 10.0


def test_no_solved_tasks_yields_no_cost_claim_rather_than_infinity() -> None:
    window = EnergyWindow(model="m", joules=6000.0, seconds=60.0)
    assert window.joules_per_solved(0) is None
    payload = window.to_dict(solved=0)
    assert payload["joules_per_solved_task"] is None


def test_idle_draw_is_reported_not_subtracted() -> None:
    """200 J/solved on a host idling at 15 W means something different from the
    same figure at 90 W. Subtracting it silently would hide which one you have."""
    window = EnergyWindow(model="m", joules=1000.0, seconds=10.0, idle_watts=13.7)
    payload = window.to_dict(solved=5)
    assert payload["idle_power_w"] == 13.7
    assert payload["joules_per_solved_task"] == 200.0
