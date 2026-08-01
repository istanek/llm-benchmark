"""Which measuring device produced a result.

The whole point of the harness is "run a new model, find out whether it is
better". That comparison is against numbers measured earlier, which is only
meaningful if both sides were measured by the same instrument.

That is not a formality here. Two bugs in one day changed what the code suite
measured — first "does the output compile", then, after `check(candidate)` was
actually invoked, "does it pass the tests" — and nemotron-3 moved from 94.5 %
to 74.4 % between them. Both were changes to *code*: the fixture's
`suite_version` sat at 0.1.0 the whole time. A bundle that records only the
fixture version therefore cannot tell you whether its numbers are comparable
with the bundle next to it.

So every run stamps the commit it ran from, whether the tree was dirty, and
the sampling each model actually got. `compare` refuses to produce a verdict
across a mismatch rather than printing a plausible wrong one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from llm_benchmark.models import HarnessProvenance, ModelConfig, SamplingConfig, sampling_for_model


def _git(repo_root: Path, *args: str) -> str | None:
    """Run a git command in *repo_root*, returning None if git or the repo is absent.

    A checkout is not required — the package can be installed and run from
    anywhere. Missing provenance is recorded as missing (and blocks comparison)
    rather than guessed at.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_provenance(
    repo_root: Path,
    *,
    model_configs: list[ModelConfig] | None = None,
    sampling: SamplingConfig | None = None,
) -> HarnessProvenance:
    commit = _git(repo_root, "rev-parse", "HEAD")
    # `status --porcelain` is empty exactly when the tree is clean. An
    # uncommitted change means the commit hash does not identify the code that
    # ran, so a dirty run is never silently comparable with anything.
    status = _git(repo_root, "status", "--porcelain")
    model_options: dict[str, dict[str, object]] = {}
    for model in model_configs or []:
        effective = sampling_for_model(sampling, model) if sampling is not None else None
        model_options[model.name] = {
            "reasoning": bool(model.reasoning),
            "max_tokens": effective.max_tokens if effective is not None else None,
            "quantization": model.quantization,
            "artifact_path": model.artifact_path or model.revision,
        }
    return HarnessProvenance(
        git_commit=commit,
        git_dirty=bool(status) if status is not None else None,
        model_options=model_options,
    )


def comparability_problems(
    candidate: HarnessProvenance | None,
    baseline: HarnessProvenance | None,
    *,
    models: list[str] | None = None,
) -> list[str]:
    """Reasons these two runs must not be compared. Empty list means go ahead.

    Deliberately strict: an unknown commit counts as a mismatch, because "we
    don't know what measured this" and "it was measured the same way" are not
    the same statement.
    """
    problems: list[str] = []
    if candidate is None or baseline is None:
        problems.append(
            "one of the runs carries no provenance stamp (it predates provenance "
            "recording), so there is no way to tell whether the same harness measured both"
        )
        return problems

    if candidate.git_commit is None or baseline.git_commit is None:
        problems.append("at least one run does not record a git commit, so the harness version is unknown")
    elif candidate.git_commit != baseline.git_commit:
        problems.append(
            f"different harness commits: {candidate.git_commit[:12]} vs {baseline.git_commit[:12]} — "
            "a code change can redefine what a suite measures without touching any fixture"
        )
    for label, prov in (("candidate", candidate), ("baseline", baseline)):
        if prov.git_dirty:
            problems.append(f"the {label} run was measured from a dirty working tree, so its commit identifies nothing")
        elif prov.git_dirty is None:
            problems.append(f"the {label} run does not record whether its working tree was clean")

    for model in models or []:
        left = candidate.model_options.get(model)
        right = baseline.model_options.get(model)
        if left is None or right is None:
            continue
        for field in ("reasoning", "max_tokens", "artifact_path", "quantization"):
            if left.get(field) != right.get(field):
                problems.append(
                    f"{model}: {field} differs between runs ({left.get(field)!r} vs {right.get(field)!r}) — "
                    "the model was not asked the same question"
                )
    return problems
