"""Paired per-task comparison.

The marginal rule — overlapping Wilson intervals mean a tie — throws away the
fact that every model answered the same tasks. Pairing recovers that power,
and on the four-model MBPP bundle it turns one reported tie into a separation.
The tests here cover the statistic and, more importantly, the plumbing that
decides which tasks reach it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def paired():
    spec = importlib.util.spec_from_file_location("paired_compare", REPO_ROOT / "scripts" / "paired_compare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["paired_compare"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- #
# McNemar                                                                #
# --------------------------------------------------------------------- #


def test_no_discordant_tasks_cannot_separate_anything(paired) -> None:
    assert paired.mcnemar_exact(0, 0) == 1.0


def test_balanced_flips_are_not_evidence(paired) -> None:
    """Twenty wins each way is a coin, not a ranking."""
    assert paired.mcnemar_exact(20, 20) > 0.9


def test_one_sided_flips_separate_even_at_small_n(paired) -> None:
    """Eight wins to none is significant on eight discordant tasks — the point
    of pairing: the marginal rates could still overlap."""
    assert paired.mcnemar_exact(8, 0) < 0.01


def test_the_p_value_never_exceeds_one(paired) -> None:
    """Doubling the tail overshoots when b == c; it has to be clamped."""
    for b, c in ((1, 1), (2, 2), (5, 5), (13, 13)):
        assert paired.mcnemar_exact(b, c) <= 1.0


# --------------------------------------------------------------------- #
# Truncation probing — the silent one                                    #
# --------------------------------------------------------------------- #


def test_truncation_is_found_inside_the_code_suite_sample_list(paired) -> None:
    """The code suite stores truncation in samples[*], a list. A probe that
    only descends into dicts found nothing, so --exclude-truncated dropped
    zero tasks on the suite where truncation decides the ranking."""
    row = {"model": "m", "task_id": "t", "samples": [{"truncated": True, "generation": {}}]}
    assert paired.row_truncated(row) is True


def test_finish_reason_length_counts_as_truncation(paired) -> None:
    """The backend's own account of why it stopped, which survives a runner
    that forgets to set the derived flag."""
    row = {"model": "m", "task_id": "t", "samples": [{"generation": {"finish_reason": "length"}}]}
    assert paired.row_truncated(row) is True


def test_the_flat_reliability_shape_is_probed_too(paired) -> None:
    row = {"model": "m", "task_id": "t", "generation": {"finish_reason": "length"}}
    assert paired.row_truncated(row) is True
    assert paired.row_truncated({"model": "m", "task_id": "t", "evaluation": {"truncated": True}}) is True


def test_a_complete_answer_is_not_truncated(paired) -> None:
    row = {"model": "m", "task_id": "t", "samples": [{"truncated": False, "generation": {"finish_reason": "stop"}}]}
    assert paired.row_truncated(row) is False


# --------------------------------------------------------------------- #
# Reading rows                                                           #
# --------------------------------------------------------------------- #


def test_pass_is_read_from_the_nested_evaluation(paired, tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"model": "m", "task_id": f"t{i}", "evaluation": {"passed": i % 2 == 0}})
            for i in range(4)
        )
    )
    data = paired.extract(paired.load_rows(path))
    verdicts = paired.collapse(data["m"], "majority")
    assert verdicts == {"t0": True, "t1": False, "t2": True, "t3": False}


def test_repetitions_collapse_by_majority(paired) -> None:
    """Repetitions of one task are not independent samples of the fixture, so
    they fold into a single verdict before pairing."""
    per_task = {"t": [(1, True, False), (2, True, False), (3, False, False)]}
    assert paired.collapse(per_task, "majority") == {"t": True}
    per_task = {"t": [(1, True, False), (2, False, False), (3, False, False)]}
    assert paired.collapse(per_task, "majority") == {"t": False}


def test_pair_mode_keeps_repetitions_apart(paired) -> None:
    per_task = {"t": [(1, True, False), (2, False, False)]}
    assert paired.collapse(per_task, "pair") == {"t@rep1": True, "t@rep2": False}
