"""Comparing a new model against stored results.

The value of these tests is mostly in the refusals. A comparison that silently
spans two different harnesses produces a verdict indistinguishable from a real
one — which is how this repo reported nemotron-3 at 94.5 % when it scores 74.4.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_benchmark.compare import Bundle, SuiteScore, compare_bundles, load_bundle, verdict_for
from llm_benchmark.models import HarnessProvenance

CLEAN = HarnessProvenance(
    git_commit="a" * 40,
    git_dirty=False,
    model_options={"incumbent": {"reasoning": False, "max_tokens": 1536}},
)


def _bundle(path: Path, scores: list[SuiteScore], provenance: HarnessProvenance | None) -> Bundle:
    return Bundle(path=path, scores=scores, provenance=provenance)


def _score(model: str, passes: int, total: int = 400, suite_version: str = "0.1.0") -> SuiteScore:
    return SuiteScore(suite="code", suite_version=suite_version, model=model, passes=passes, total=total)


# --------------------------------------------------------------------- #
# Verdicts                                                               #
# --------------------------------------------------------------------- #


def test_clearly_separated_scores_are_better_and_worse() -> None:
    assert verdict_for(_score("new", 360), _score("old", 280)) == "better"
    assert verdict_for(_score("new", 280), _score("old", 360)) == "worse"


def test_overlapping_intervals_are_a_tie_not_a_ranking() -> None:
    """A two-point gap on n=400 is noise, and calling it a win is the mistake."""
    assert verdict_for(_score("new", 342), _score("old", 340)) == "tie"


def test_a_tiny_sample_cannot_separate_anything() -> None:
    assert verdict_for(_score("new", 5, total=5), _score("old", 3, total=5)) == "tie"


# --------------------------------------------------------------------- #
# Refusals                                                               #
# --------------------------------------------------------------------- #


def test_different_harness_commits_block_the_comparison(tmp_path: Path) -> None:
    other = CLEAN.model_copy(update={"git_commit": "b" * 40})
    _, blockers = compare_bundles(
        _bundle(tmp_path / "new", [_score("new", 360)], CLEAN),
        _bundle(tmp_path / "old", [_score("incumbent", 280)], other),
    )
    assert any("different harness commits" in problem for problem in blockers)


def test_a_dirty_tree_blocks_the_comparison(tmp_path: Path) -> None:
    dirty = CLEAN.model_copy(update={"git_dirty": True})
    _, blockers = compare_bundles(
        _bundle(tmp_path / "new", [_score("new", 360)], dirty),
        _bundle(tmp_path / "old", [_score("incumbent", 280)], CLEAN),
    )
    assert any("dirty working tree" in problem for problem in blockers)


def test_a_run_without_provenance_blocks_the_comparison(tmp_path: Path) -> None:
    """Every bundle measured before provenance existed lands here."""
    _, blockers = compare_bundles(
        _bundle(tmp_path / "new", [_score("new", 360)], CLEAN),
        _bundle(tmp_path / "old", [_score("incumbent", 280)], None),
    )
    assert any("no provenance stamp" in problem for problem in blockers)


def test_a_different_fixture_version_blocks_the_comparison(tmp_path: Path) -> None:
    _, blockers = compare_bundles(
        _bundle(tmp_path / "new", [_score("new", 360, suite_version="0.2.0")], CLEAN),
        _bundle(tmp_path / "old", [_score("incumbent", 280)], CLEAN),
    )
    assert any("fixture version differs" in problem for problem in blockers)


def test_a_shared_model_run_with_a_different_budget_blocks_the_comparison(tmp_path: Path) -> None:
    """Raising max_output_tokens for one model changes what was asked of it."""
    raised = CLEAN.model_copy(
        update={"model_options": {"incumbent": {"reasoning": False, "max_tokens": 8192}}}
    )
    _, blockers = compare_bundles(
        _bundle(tmp_path / "new", [_score("incumbent", 360)], raised),
        _bundle(tmp_path / "old", [_score("incumbent", 280)], CLEAN),
    )
    assert any("max_tokens differs" in problem for problem in blockers)


def test_matching_provenance_produces_no_blockers(tmp_path: Path) -> None:
    comparisons, blockers = compare_bundles(
        _bundle(tmp_path / "new", [_score("new", 360)], CLEAN),
        _bundle(tmp_path / "old", [_score("incumbent", 280)], CLEAN),
    )
    assert blockers == []
    assert [c.verdict for c in comparisons] == ["better"]


# --------------------------------------------------------------------- #
# Reading a bundle off disk                                              #
# --------------------------------------------------------------------- #


def test_load_bundle_reads_scores_and_provenance(tmp_path: Path) -> None:
    suite_dir = tmp_path / "code_generation-new-model"
    suite_dir.mkdir(parents=True)
    (suite_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite": "code_generation_v1",
                "suite_version": "0.3.0",
                "models": [{"model": "new-model", "passes": 150, "total": 164}],
            }
        )
    )
    (suite_dir / "manifest.json").write_text(json.dumps({"provenance": CLEAN.model_dump(mode="json")}))

    bundle = load_bundle(tmp_path)
    assert bundle.models == ["new-model"]
    score = bundle.score_for("code_generation_v1", "new-model")
    assert score is not None and (score.passes, score.total) == (150, 164)
    assert bundle.provenance is not None and bundle.provenance.git_dirty is False


def test_load_bundle_reads_the_nested_code_suite_shape(tmp_path: Path) -> None:
    """code_generation stores pass_at_1 per benchmark rather than passes/total."""
    suite_dir = tmp_path / "code_generation_mbpp-new-model"
    suite_dir.mkdir(parents=True)
    (suite_dir / "summary.json").write_text(
        json.dumps(
            {
                "suite": "code_generation_mbpp_v1",
                "suite_version": "0.1.0",
                "models": [
                    {"model": "new-model", "benchmarks": [{"pass_at_1": 0.85, "tasks": 426}]}
                ],
            }
        )
    )
    bundle = load_bundle(tmp_path)
    score = bundle.score_for("code_generation_mbpp_v1", "new-model")
    assert score is not None and score.total == 426 and score.passes == 362


# --------------------------------------------------------------------- #
# Every manifest gets stamped                                            #
# --------------------------------------------------------------------- #


def test_build_manifest_stamps_provenance_without_being_asked(tmp_path: Path) -> None:
    """The stamp was opt-in once, and the first real run after it shipped came
    out unstamped: the standalone code-generation runner builds its own
    manifest and nobody had updated the call. Defaulting it is the fix; this
    test is what keeps it defaulted."""
    from llm_benchmark.config import load_backend, load_experiment, load_model_config, load_platform
    from llm_benchmark.runtime import build_manifest

    repo_root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(repo_root / "configs/experiments/code-generation.yaml").experiment
    manifest = build_manifest(
        experiment=experiment,
        platform_config=load_platform(repo_root / "configs/platforms/local.yaml"),
        backend_config=load_backend(repo_root / "configs/backends/ollama.yaml"),
        model_names=["qwen-3.6"],
        results_dir=tmp_path,
    )
    assert manifest.provenance is not None
    assert manifest.provenance.schema_version == "harness-provenance/v1"


def test_build_manifest_records_the_options_each_model_ran_with(tmp_path: Path) -> None:
    from llm_benchmark.config import load_backend, load_experiment, load_model_config, load_platform
    from llm_benchmark.runtime import build_manifest

    repo_root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(repo_root / "configs/experiments/code-generation.yaml").experiment
    model = load_model_config(repo_root / "configs/models/gpt-oss-120b.yaml")
    manifest = build_manifest(
        experiment=experiment,
        platform_config=load_platform(repo_root / "configs/platforms/local.yaml"),
        backend_config=load_backend(repo_root / "configs/backends/ollama.yaml"),
        model_names=[model.name],
        results_dir=tmp_path,
        model_configs=[model],
    )
    options = manifest.provenance.model_options[model.name]
    assert options["reasoning"] is False
    assert options["max_tokens"] == experiment.sampling.max_tokens
    assert options["artifact_path"] == "gpt-oss:120b"
