#!/usr/bin/env python3
"""Run the full 164-problem HumanEval suite against several models.

Built for an unattended overnight run:

    PYTHONPATH=src nohup python3 scripts/run_full_code_generation.py \
        --models qwen-3.6 gemma-4 nemotron-3 > /tmp/humaneval.log 2>&1 &

Each model gets its own run directory inside one bundle, so an interrupted
session loses at most the model that was in flight. Re-running with the same
``--output-dir`` skips models that already produced a ``summary.json``
(``--force`` re-runs them). There is no finer-grained resume: the code suite
writes rows as it goes but has no per-task restart, so an interrupted model
starts over.

At the end it prints measured pass@1 per model, which is what
``data/code/reference_scores.yaml`` still needs — those entries are TODO
placeholders and this script deliberately does not write them. A measured
number is not a published reference; fill them from model cards by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_benchmark.code_generation import (  # noqa: E402
    default_reference_scores_path,
    load_code_generation_suite,
    run_code_generation_suite,
)
from llm_benchmark.config import (  # noqa: E402
    load_backend,
    load_experiment,
    load_model_config,
    load_platform,
)
from llm_benchmark.reporting import aggregate_runs, write_report  # noqa: E402
from llm_benchmark.results_bundle import make_run_id, write_manifest  # noqa: E402
from llm_benchmark.runners.registry import build_backend  # noqa: E402
from llm_benchmark.runtime import build_manifest  # noqa: E402

DEFAULT_MODELS = ["qwen-3.6", "gemma-4", "nemotron-3"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model config names under configs/models/.")
    parser.add_argument("--experiment", default="configs/experiments/ollama-baseline.yaml")
    parser.add_argument("--platform", default="configs/platforms/local.yaml")
    parser.add_argument("--backend", default="configs/backends/ollama.yaml")
    parser.add_argument("--task-limit", type=int, default=None, help="Run only the first N problems (smoke check).")
    parser.add_argument("--samples", type=int, default=1, help="Samples per task; >1 enables pass@k.")
    parser.add_argument("--output-dir", default=None, help="Reuse an existing bundle directory to resume.")
    parser.add_argument("--force", action="store_true", help="Re-run models that already have a summary.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = load_experiment(REPO_ROOT / args.experiment).experiment
    platform_config = load_platform(REPO_ROOT / args.platform)
    backend_config = load_backend(REPO_ROOT / args.backend)
    suite = load_code_generation_suite(REPO_ROOT)

    bundle = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results/benchmarks" / make_run_id()
    bundle.mkdir(parents=True, exist_ok=True)

    planned = len(suite.tasks) if args.task_limit is None else min(args.task_limit, len(suite.tasks))
    log(f"bundle: {bundle}")
    log(f"{planned} problems x {args.samples} sample(s) x {len(args.models)} model(s)")
    # Measured on a GB10: qwen-3.6 averages ~2.3 s/problem (~6 min for the
    # full set); gemma-4 decodes ~7x slower, so budget closer to 45 min.
    log("expect ~6 min per model at 75 tok/s, ~45 min at 11 tok/s")

    started = time.time()
    results: dict[str, float | None] = {}
    for index, model_name in enumerate(args.models, start=1):
        run_dir = bundle / f"code_generation-{model_name}"
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and not args.force:
            log(f"[{index}/{len(args.models)}] {model_name}: summary.json exists, skipping (use --force to redo)")
            results[model_name] = _pass_at_1(json.loads(summary_path.read_text()), model_name)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        model_config = load_model_config(REPO_ROOT / f"configs/models/{model_name}.yaml")
        write_manifest(
            run_dir,
            build_manifest(
                experiment=experiment,
                platform_config=platform_config,
                backend_config=backend_config,
                model_names=[model_name],
                results_dir=run_dir,
            ),
        )
        log(f"[{index}/{len(args.models)}] {model_name}: starting")
        model_started = time.time()
        try:
            summary = run_code_generation_suite(
                run_dir=run_dir,
                suite=suite,
                backend=build_backend(backend_config),
                backend_config=backend_config,
                model_configs=[model_config],
                sampling=experiment.sampling,
                num_samples_per_task=args.samples,
                task_limit=args.task_limit,
                reference_scores_path=default_reference_scores_path(REPO_ROOT),
                progress_callback=lambda message: log(message.strip()),
            )
        except Exception as exc:  # noqa: BLE001 - one bad model must not kill the night
            log(f"{model_name}: FAILED after {time.time() - model_started:.0f}s: {exc!r}")
            results[model_name] = None
            continue
        results[model_name] = _pass_at_1(summary, model_name)
        log(f"{model_name}: done in {(time.time() - model_started) / 60:.1f} min")
        for warning in summary.get("reference_warnings") or []:
            log(f"  reference warning: {warning}")

    log(f"all models finished in {(time.time() - started) / 60:.1f} min")

    aggregate = aggregate_runs(bundle)
    write_report(bundle / "report.md", "both", aggregate)
    log(f"report: {bundle / 'report.md'} and report.html")

    print("\nmeasured pass@1 (HumanEval):")
    for model_name, value in results.items():
        rendered = "failed" if value is None else f"{value:.1%}"
        print(f"  {model_name:16} {rendered}")
    print(
        "\ndata/code/reference_scores.yaml still holds TODO placeholders. Fill them"
        "\nfrom published model cards, not from the numbers above — comparing a"
        "\nmeasurement against itself checks nothing."
    )
    return 0


def _pass_at_1(summary: dict, model_name: str) -> float | None:
    for model in summary.get("models") or []:
        if model.get("model") != model_name:
            continue
        for benchmark in model.get("benchmarks") or []:
            if benchmark.get("benchmark") == "humaneval":
                return benchmark.get("pass_at_1")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
