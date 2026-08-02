#!/usr/bin/env python3
"""Paired per-task comparison of two models measured on the same fixture.

Why this exists
---------------
The harness currently calls two models "tied" whenever their marginal 95 %
Wilson intervals overlap. That rule is safe but weak: it ignores that every
model answered the *same* tasks. Pairing the outcomes per task recovers that
power. Two models at 85.7 % and 87.6 % with overlapping intervals can still
be separated decisively if one of them wins almost every task the other
loses — and conversely, a 5-point gap can be pure noise if the flips go both
ways.

The statistic is McNemar's exact test on the discordant pairs (tasks where
exactly one model passed), plus a paired-bootstrap CI on the pass-rate
difference. Concordant tasks (both pass / both fail) carry no information
about *which* model is better and are correctly ignored by the test — but
they are reported, because a comparison where 95 % of tasks are concordant
is telling you the fixture barely discriminates.

The script also prints the task-level diff: exactly which task ids flipped
in which direction. Both scorer bugs that invalidated earlier code numbers
(the missing ``check()`` call, first-fence ``extract_code``) would have been
visible in this listing long before they were found by hand — "model X
improved 20 points, but on a disjoint set of tasks" is a red flag that an
aggregate hides.

Usage
-----
Two runs of the same suite (directories containing ``results.jsonl``)::

    paired_compare.py results/benchmarks/<bundle>/<run-A> \
                      results/benchmarks/<bundle>/<run-B>

One results.jsonl holding both models::

    paired_compare.py results.jsonl --model-a qwen-3.6 --model-b gemma-4

No third-party dependencies; exact binomial via ``math.comb``.

Caveats stated up front, in the spirit of METHODOLOGY.md:

* This is a *within-fixture* comparison. It says which model does better on
  these tasks, not on the population the tasks were sampled from — for that
  the Wilson interval on each marginal rate is still the right humility.
* Repetitions of the same task are not independent samples of the task
  population. By default they are collapsed per task by majority vote
  (``--reps majority``); ``--reps pair`` pairs repetition-by-repetition
  instead, which is only meaningful if both runs used the same repetition
  count and seed offsets.
* The exact test is valid at any n, including the n=14 grounding fixture —
  it will simply (and honestly) refuse to reach significance there unless
  the flips are one-sided.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

# Field names are probed in order, so the script tolerates schema drift
# between suite runners without a config file.
TASK_KEYS = ("task_id", "task", "id", "problem_id", "name")
MODEL_KEYS = ("model", "model_name", "model_tag", "tag")
PASS_KEYS = ("passed", "pass", "correct", "ok", "success")
REP_KEYS = ("repetition", "rep", "sample_index", "attempt")
TRUNC_KEYS = ("truncated", "was_truncated")


def _first_key(row: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    # one level of nesting is common (evaluation.passed, sample.truncated)
    for sub in ("evaluation", "result", "sample", "outcome"):
        inner = row.get(sub)
        if isinstance(inner, dict):
            for k in keys:
                if k in inner and inner[k] is not None:
                    return inner[k]
    return None


def row_truncated(row: dict) -> bool:
    """True when any sample in this row hit the token budget.

    Probed explicitly rather than through ``_first_key`` because the code
    suite stores its samples in a *list* (``samples[*].truncated``), and the
    generic probe only descends into dicts. It therefore found nothing and
    ``--exclude-truncated`` silently dropped zero tasks on the one suite where
    truncation matters most — gpt-oss-120b has 52 truncated answers in the
    bundle it reported as unaffected.

    ``finish_reason == "length"`` is checked as well: it is the backend's own
    account of why generation stopped, and it survives even when a runner
    forgets to set the derived flag.
    """
    if bool(_first_key(row, TRUNC_KEYS)):
        return True
    generation = row.get("generation")
    if isinstance(generation, dict) and generation.get("finish_reason") == "length":
        return True
    for sample in row.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        if sample.get("truncated"):
            return True
        inner = sample.get("generation")
        if isinstance(inner, dict) and inner.get("finish_reason") == "length":
            return True
    return False


def load_rows(path: Path) -> list[dict]:
    """Read every results.jsonl under *path* (file or directory)."""
    files: list[Path]
    if path.is_dir():
        files = sorted(path.rglob("results.jsonl"))
        if not files:
            sys.exit(f"error: no results.jsonl found under {path}")
    else:
        files = [path]
    rows: list[dict] = []
    for f in files:
        with f.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(
                        f"warning: {f}:{lineno} is not valid JSON, skipped",
                        file=sys.stderr,
                    )
    return rows


def extract(rows: list[dict]) -> dict[str, dict[str, list[tuple[int, bool, bool]]]]:
    """-> {model: {task_id: [(rep, passed, truncated), ...]}}"""
    out: dict[str, dict[str, list[tuple[int, bool, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skipped = 0
    for row in rows:
        task = _first_key(row, TASK_KEYS)
        model = _first_key(row, MODEL_KEYS)
        passed = _first_key(row, PASS_KEYS)
        if task is None or model is None or passed is None:
            skipped += 1
            continue
        rep = _first_key(row, REP_KEYS)
        trunc = row_truncated(row)
        out[str(model)][str(task)].append(
            (int(rep) if rep is not None else len(out[str(model)][str(task)]),
             bool(passed), trunc)
        )
    if skipped:
        print(
            f"note: {skipped} rows lacked a task/model/pass field and were "
            f"ignored (probe suites score n/a by design)",
            file=sys.stderr,
        )
    return out


# --------------------------------------------------------------------------
# Collapsing repetitions
# --------------------------------------------------------------------------

def collapse(
    per_task: dict[str, list[tuple[int, bool, bool]]], mode: str
) -> dict[str, bool]:
    """Reduce repetitions to one verdict per task id (or per task@rep)."""
    verdicts: dict[str, bool] = {}
    if mode == "majority":
        for task, samples in per_task.items():
            passes = sum(1 for _, p, _ in samples if p)
            verdicts[task] = passes * 2 > len(samples)
    elif mode == "pair":
        for task, samples in per_task.items():
            for rep, p, _ in samples:
                verdicts[f"{task}@rep{rep}"] = p
    else:  # "first"
        for task, samples in per_task.items():
            verdicts[task] = sorted(samples)[0][1]
    return verdicts


def truncated_tasks(per_task: dict[str, list[tuple[int, bool, bool]]]) -> set[str]:
    return {t for t, samples in per_task.items() if any(tr for _, _, tr in samples)}


# --------------------------------------------------------------------------
# Statistics (stdlib only)
# --------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = tasks only model A passed, c = tasks only model B passed. Under H0
    each discordant task is a fair coin, so p = 2 * P(Binom(b+c, .5) <= min).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    a: list[bool], b: list[bool], iters: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95 % percentile CI for pass_rate(A) - pass_rate(B), resampling tasks."""
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        pa = sum(a[i] for i in idx) / n
        pb = sum(b[i] for i in idx) / n
        diffs.append(pa - pb)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters) - 1]
    return lo, hi


def wilson(passes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired per-task comparison of two models on one fixture."
    )
    ap.add_argument("path_a", type=Path, help="results.jsonl or run/bundle dir")
    ap.add_argument(
        "path_b", type=Path, nargs="?",
        help="second run dir; omit if one file holds both models",
    )
    ap.add_argument("--model-a", help="model name in path_a (default: only one)")
    ap.add_argument("--model-b", help="model name in path_b or path_a")
    ap.add_argument(
        "--reps", choices=("majority", "pair", "first"), default="majority",
        help="how to collapse repetitions (default: majority vote per task)",
    )
    ap.add_argument(
        "--exclude-truncated", action="store_true",
        help="drop tasks either model truncated — compare ability, not budget",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    data_a = extract(load_rows(args.path_a))
    data_b = extract(load_rows(args.path_b)) if args.path_b else data_a

    def pick(data, wanted, other, side):
        names = sorted(data)
        if wanted:
            if wanted not in data:
                sys.exit(f"error: model {wanted!r} not in {side}; found: {names}")
            return wanted
        candidates = [n for n in names if n != other]
        if len(candidates) != 1:
            sys.exit(
                f"error: {side} holds {names}; disambiguate with "
                f"--model-{'a' if side == 'A' else 'b'}"
            )
        return candidates[0]

    model_a = pick(data_a, args.model_a, None, "A")
    model_b = pick(data_b, args.model_b, model_a if data_b is data_a else None, "B")

    va = collapse(data_a[model_a], args.reps)
    vb = collapse(data_b[model_b], args.reps)

    common = sorted(set(va) & set(vb))
    only_a, only_b = set(va) - set(vb), set(vb) - set(va)
    if only_a or only_b:
        print(
            f"warning: task sets differ ({len(only_a)} only in A, "
            f"{len(only_b)} only in B) — comparing the {len(common)}-task "
            f"intersection. Same fixture version on both sides?",
            file=sys.stderr,
        )
    if not common:
        sys.exit("error: no common tasks — nothing to pair")

    if args.exclude_truncated:
        drop = truncated_tasks(data_a[model_a]) | truncated_tasks(data_b[model_b])
        before = len(common)
        common = [t for t in common if t.split("@rep")[0] not in drop]
        print(
            f"note: --exclude-truncated dropped {before - len(common)} tasks; "
            f"this compares ability at the *smaller* effective fixture",
            file=sys.stderr,
        )

    a = [va[t] for t in common]
    b = [vb[t] for t in common]
    n = len(common)

    n11 = sum(1 for x, y in zip(a, b) if x and y)
    n00 = sum(1 for x, y in zip(a, b) if not x and not y)
    d_a = [t for t, x, y in zip(common, a, b) if x and not y]   # A won
    d_b = [t for t, x, y in zip(common, a, b) if y and not x]   # B won

    pa, pb = sum(a) / n, sum(b) / n
    lo_a, hi_a = wilson(sum(a), n)
    lo_b, hi_b = wilson(sum(b), n)
    p_value = mcnemar_exact(len(d_a), len(d_b))
    ci_lo, ci_hi = paired_bootstrap_ci(a, b)

    # ---- report ----------------------------------------------------------
    print(f"\nPaired comparison over n = {n} tasks ({args.reps} across reps)\n")
    print(f"  {model_a:<28} {pa:6.1%}  Wilson [{lo_a:.1%}, {hi_a:.1%}]")
    print(f"  {model_b:<28} {pb:6.1%}  Wilson [{lo_b:.1%}, {hi_b:.1%}]")
    marg = "overlap" if not (hi_a < lo_b or hi_b < lo_a) else "no overlap"
    print(f"  marginal intervals: {marg} "
          f"({'tie under the current rule' if marg == 'overlap' else 'already separated'})\n")

    print(f"  concordant  both pass: {n11:>4}   both fail: {n00:>4}")
    print(f"  discordant  only {model_a} passed: {len(d_a):>4}")
    print(f"              only {model_b} passed: {len(d_b):>4}")
    disc = len(d_a) + len(d_b)
    if disc and disc < 0.05 * n:
        print(f"  note: only {disc}/{n} tasks discriminate at all — "
              f"the fixture is nearly saturated for this pair")
    print()
    print(f"  McNemar exact p = {p_value:.4g}")
    print(f"  paired bootstrap 95 % CI on (A - B): "
          f"[{ci_lo:+.1%}, {ci_hi:+.1%}]")

    if p_value < args.alpha:
        winner = model_a if len(d_a) > len(d_b) else model_b
        print(f"\n  verdict: {winner} is better on this fixture "
              f"(p < {args.alpha}), even though the marginal rule "
              f"{'says tie' if marg == 'overlap' else 'agrees'}")
    else:
        print(f"\n  verdict: not separated (p >= {args.alpha}); "
              f"{disc} discordant tasks are too few or too balanced")

    if d_a or d_b:
        print(f"\n  tasks only {model_a} passed ({len(d_a)}):")
        for t in d_a:
            print(f"    + {t}")
        print(f"  tasks only {model_b} passed ({len(d_b)}):")
        for t in d_b:
            print(f"    - {t}")
        print(
            "\n  Read the flip lists before the p-value: flips concentrated"
            "\n  in one task family, or a re-run of the *same* model flipping"
            "\n  many tasks in both directions, is scorer/harness news,"
            "\n  not model news."
        )


if __name__ == "__main__":
    main()
