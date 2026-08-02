#!/usr/bin/env python3
"""Re-score a finished bundle's stored answers with the current scorer.

    python3 scripts/rescore_bundle.py results/benchmarks/<bundle>

Five scoring bugs have been found in this repo, and each one invalidated every
code number measured before it. Re-asking the models costs hours — the last
four-model MBPP pass took seven — while scoring is offline, deterministic, and
depends on nothing the model saw: the prompt and the sampling config do not
reference the scorer. So a scorer fix is applied to the stored generations
instead, and the models are not asked again.

The output is a new bundle. The original is never modified: its numbers are
what that commit produced, and rewriting them in place would erase the evidence
that the scorer ever disagreed with itself.

Provenance records both halves — `git_commit` is the scorer that produced these
numbers, `generations_from` the bundle and commit that produced the answers.
A rescored bundle that claimed to be a full run at the new commit would be the
same category of lie the compare command exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_benchmark.code_generation import (  # noqa: E402
    _aggregate_per_model,
    evaluate_task,
    load_code_generation_suite,
    load_mbpp_mutated_suite,
    load_mbpp_suite,
)
from llm_benchmark.models import GenerationResult, HarnessProvenance  # noqa: E402
from llm_benchmark.provenance import collect_provenance  # noqa: E402
from llm_benchmark.reporting import aggregate_runs, write_report  # noqa: E402
from llm_benchmark.results_bundle import write_json  # noqa: E402

SANDBOX_TIMEOUT_S = 15.0
SANDBOX_MEMORY_MB = 1024


def load_tasks(suite_name: str) -> dict[str, Any]:
    """Fixture tasks keyed by id, for whichever code benchmark the run used."""
    if "mutated" in suite_name:
        loader = load_mbpp_mutated_suite
    elif "mbpp" in suite_name:
        loader = load_mbpp_suite
    else:
        loader = load_code_generation_suite
    return {task.task_id: task for task in loader(REPO_ROOT).tasks}


def rescore_run(run_dir: Path, out_dir: Path) -> dict[str, Any] | None:
    """Re-evaluate one (suite, model) directory. Returns a before/after record."""
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    if not results_path.exists() or not summary_path.exists():
        return None

    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    old_summary = json.loads(summary_path.read_text())
    tasks = load_tasks(str(old_summary.get("suite") or ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict[str, Any]] = []
    outcomes = []
    before = after = 0

    for row in rows:
        task = tasks.get(row["task_id"])
        if task is None:
            # A fixture entry that no longer exists. Kept verbatim rather than
            # dropped, so the row count still matches the original run.
            new_rows.append(row)
            continue
        before += int(any(sample["sandbox"]["passed"] for sample in row["samples"]))
        generations = [
            GenerationResult(
                prompt=sample["generation"].get("prompt", ""),
                output=sample["generation"]["output"],
                finish_reason=sample["generation"].get("finish_reason", "unknown"),
                metrics=sample["generation"].get("metrics") or {},
                raw=sample["generation"].get("raw") or {},
            )
            for sample in row["samples"]
        ]
        outcome = evaluate_task(
            task,
            generations=generations,
            sample_seeds=[sample.get("seed", 42) for sample in row["samples"]],
            sandbox_timeout_s=SANDBOX_TIMEOUT_S,
            sandbox_memory_mb=SANDBOX_MEMORY_MB,
        )
        outcomes.append(outcome)
        after += int(outcome.pass_at_1 >= 1.0)

        new_row = dict(row)
        new_row["samples"] = [
            {
                **original,
                "truncated": sample.truncated,
                "extracted_code": sample.extracted_code,
                "sandbox": {
                    "passed": sample.sandbox.passed,
                    "status": sample.sandbox.status,
                    "duration_s": sample.sandbox.duration_s,
                    "exit_code": sample.sandbox.exit_code,
                    "stderr": sample.sandbox.stderr[-2000:],
                },
            }
            for original, sample in zip(row["samples"], outcome.samples)
        ]
        new_row["evaluation"] = {
            **row.get("evaluation", {}),
            "benchmark": outcome.benchmark,
            "pass_at_1": outcome.pass_at_1,
            "pass_at_k": outcome.pass_at_k_value,
            "pass_at_k_k": outcome.pass_at_k_k,
            "passed": outcome.pass_at_1 >= 1.0,
            "score": 1 if outcome.pass_at_1 >= 1.0 else 0,
            "reason": "all_samples_passed" if outcome.pass_at_1 >= 1.0 else "at_least_one_sample_failed",
        }
        new_rows.append(new_row)

    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in new_rows) + "\n"
    )

    model_name = str((old_summary.get("models") or [{}])[0].get("model") or run_dir.name)
    per_benchmark = _aggregate_per_model(outcomes)
    new_summary = {
        **old_summary,
        "models": [{"model": model_name, "benchmarks": list(per_benchmark.values())}],
    }
    write_json(out_dir / "summary.json", new_summary)
    return {"run": run_dir.name, "model": model_name, "before": before, "after": after, "total": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="Bundle to re-score.")
    parser.add_argument("--output", type=Path, default=None, help="Destination bundle (default: <bundle>-rescored).")
    args = parser.parse_args()

    bundle: Path = args.bundle
    if not bundle.is_dir():
        raise SystemExit(f"not a bundle directory: {bundle}")
    out_bundle: Path = args.output or bundle.with_name(bundle.name + "-rescored")
    if out_bundle.exists():
        raise SystemExit(f"refusing to overwrite {out_bundle}")

    source_commits: set[str] = set()
    records: list[dict[str, Any]] = []
    scorer = collect_provenance(REPO_ROOT)

    for run_dir in sorted(p for p in bundle.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        record = rescore_run(run_dir, out_bundle / run_dir.name)
        if record is None:
            continue
        records.append(record)

        origin = manifest.get("provenance") or {}
        if origin.get("git_commit"):
            source_commits.add(origin["git_commit"])
        provenance = HarnessProvenance(
            git_commit=scorer.git_commit,
            git_dirty=scorer.git_dirty,
            model_options=origin.get("model_options") or {},
            generations_from={
                "bundle": bundle.name,
                "git_commit": origin.get("git_commit"),
                "git_dirty": origin.get("git_dirty"),
                "note": "answers reused; only scoring was re-run",
            },
        )
        write_json(
            out_bundle / run_dir.name / "manifest.json",
            {**manifest, "provenance": provenance.model_dump(mode="json")},
        )

    if not records:
        shutil.rmtree(out_bundle, ignore_errors=True)
        raise SystemExit(f"no scorable runs found under {bundle}")

    print(f"{'model':16s} {'before':>10s} {'after':>10s} {'delta':>8s}")
    for record in records:
        delta = record["after"] - record["before"]
        print(
            f"{record['model']:16s} {record['before']:>4d}/{record['total']:<5d} "
            f"{record['after']:>4d}/{record['total']:<5d} {delta:>+8d}"
        )
    print()
    print(f"generations from {bundle.name} ({', '.join(sorted(source_commits)) or 'no commit recorded'})")
    print(f"scored by        {scorer.git_commit}{' (DIRTY TREE)' if scorer.git_dirty else ''}")

    aggregate = aggregate_runs(out_bundle)
    write_report(out_bundle / "report.md", "both", aggregate)
    print(f"report           {out_bundle / 'report.md'} and report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
