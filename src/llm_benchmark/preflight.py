"""Check that a planned run can actually run, before it starts.

A full sweep spent eight hours measuring four models and then died on the last
suite because two Project Gutenberg texts had never been downloaded. The
missing files were checkable in milliseconds; nothing checked them.

The rule this module encodes: **everything a run needs is verified before the
first model is loaded, and every problem is reported at once.** Failing on the
first one found would mean discovering the second after the next eight-hour
attempt.

What it deliberately does *not* do is repair anything. A preflight that
silently downloads a corpus, or drops the suite it cannot satisfy, decides on
the operator's behalf what the run measures — which is the failure mode this
whole repo keeps finding in other clothes. It reports, with the command that
fixes each problem, and lets the caller decide.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.models import BackendConfig, ModelConfig


@dataclass(frozen=True)
class Problem:
    """One reason a run would fail, and how to fix it."""

    subject: str
    detail: str
    fix: str = ""

    def render(self) -> str:
        line = f"  - {self.subject}: {self.detail}"
        return f"{line}\n      fix: {self.fix}" if self.fix else line


def check_suite_assets(repo_root: Path, suite_name: str) -> list[Problem]:
    """Fixtures and any external files a suite needs at run time."""
    problems: list[Problem] = []

    # Imported here so a missing optional dependency in one suite cannot stop
    # the preflight for every other suite.
    from llm_benchmark.reliability import fixture_path_for_suite_name

    try:
        fixture_path = fixture_path_for_suite_name(repo_root, suite_name)
    except ValueError as exc:
        return [
            Problem(
                suite_name,
                str(exc),
                "check the suite name against fixture_path_for_suite_name()",
            )
        ]

    if not fixture_path.exists():
        return [Problem(suite_name, f"fixture missing at {fixture_path}", "restore it from git")]

    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [Problem(suite_name, f"fixture at {fixture_path} is unreadable: {exc}")]

    if suite_name.startswith("long_context_retrieval"):
        # The haystack corpora are large, git-ignored, and fetched by a script.
        # This is the check whose absence cost the eight-hour sweep.
        # text_file is relative to the repo root, not to the haystack
        # directory — joining it onto the directory produced a doubled path
        # that could never exist, so this check would have "failed" even after
        # a successful fetch.
        for name, spec in (payload.get("haystacks") or {}).items():
            text_file = spec.get("text_file") if isinstance(spec, dict) else None
            if not text_file:
                continue
            path = repo_root / text_file
            if not path.exists():
                problems.append(
                    Problem(
                        suite_name,
                        f"haystack {name!r} missing at {path}",
                        "bash scripts/fetch_haystacks.sh",
                    )
                )
        if not payload.get("needles"):
            problems.append(Problem(suite_name, "fixture declares no needles"))
        return problems

    if "code_generation" in suite_name:
        # A code task without tests or an entry point cannot be scored, and the
        # suite would raise per task rather than up front.
        broken = [
            task.get("task_id", "?")
            for task in payload.get("tasks") or []
            if not (task.get("metadata") or {}).get("tests")
            or not (task.get("metadata") or {}).get("entry_point")
        ]
        if broken:
            problems.append(
                Problem(
                    suite_name,
                    f"{len(broken)} task(s) missing tests or entry_point (first: {broken[0]})",
                    "regenerate the fixture",
                )
            )

    # Task-shaped suites only: the long-context fixture is a matrix of
    # needles and haystacks and legitimately has no `tasks` key. Returned
    # above so this check never sees it.
    if not payload.get("tasks"):
        problems.append(Problem(suite_name, "fixture declares no tasks"))

    return problems


def check_backend(backend_config: BackendConfig) -> list[Problem]:
    """Is the backend answering? Only Ollama is probed; others are assumed up."""
    if backend_config.name.value != "ollama":
        return []

    from llm_benchmark.runners.ollama import resolve_ollama_base

    endpoint = resolve_ollama_base(backend_config.options) + "/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return [
            Problem(
                "backend",
                f"Ollama did not answer at {endpoint} ({exc})",
                "start it, or point $OLLAMA_HOST elsewhere",
            )
        ]
    return []


def _local_ollama_tags(backend_config: BackendConfig) -> set[str] | None:
    from llm_benchmark.runners.ollama import resolve_ollama_base

    endpoint = resolve_ollama_base(backend_config.options) + "/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None
    return {str(entry.get("name")) for entry in payload.get("models") or []}


def check_models(
    backend_config: BackendConfig, model_configs: list[ModelConfig]
) -> list[Problem]:
    """Every model has a usable tag, and the backend can honour its options."""
    from llm_benchmark.runners.ollama import model_is_cloud
    from llm_benchmark.runners.registry import build_backend

    problems: list[Problem] = []
    capabilities = build_backend(backend_config).capabilities

    tags = _local_ollama_tags(backend_config) if backend_config.name.value == "ollama" else None
    for model in model_configs:
        tag = model.artifact_path or model.revision
        if not tag:
            problems.append(Problem(model.name, "no artifact_path or revision to identify the model"))
            continue
        if model.reasoning and not capabilities.supports_reasoning:
            problems.append(
                Problem(
                    model.name,
                    f"asks for reasoning, which the {backend_config.name.value} backend cannot enable",
                    "drop `reasoning: true` or run it on a backend that supports it",
                )
            )
        if tags is not None and not model_is_cloud(model) and tag not in tags:
            problems.append(
                Problem(model.name, f"tag {tag!r} is not pulled locally", f"ollama pull {tag}")
            )
    return problems


def preflight(
    repo_root: Path,
    *,
    suite_names: list[str],
    model_configs: list[ModelConfig],
    backend_config: BackendConfig,
) -> list[Problem]:
    """Every problem that would stop this run, collected in one pass."""
    problems = check_backend(backend_config)
    problems.extend(check_models(backend_config, model_configs))
    for suite_name in suite_names:
        problems.extend(check_suite_assets(repo_root, suite_name))
    return problems


def render_problems(problems: list[Problem]) -> str:
    lines = [f"preflight found {len(problems)} problem(s) — nothing was run:"]
    lines.extend(problem.render() for problem in problems)
    return "\n".join(lines)
