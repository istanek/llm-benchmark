from __future__ import annotations

import platform
import socket
import sys
from pathlib import Path

from llm_benchmark.hardware import discover_hardware
from llm_benchmark.models import (
    BackendConfig,
    EnvironmentSnapshot,
    ExperimentSpec,
    ModelConfig,
    PlatformConfig,
    RunManifest,
)


def build_environment_snapshot(
    platform_config: PlatformConfig | None, backend_config: BackendConfig
) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        platform_name=platform_config.name if platform_config is not None else "auto-detected",
        backend_name=backend_config.name,
        backend_version=backend_config.version,
        python_version=sys.version.split()[0],
        os=platform.platform(),
        hostname=socket.gethostname(),
        hardware=discover_hardware(),
    )


def build_manifest(
    experiment: ExperimentSpec,
    platform_config: PlatformConfig,
    backend_config: BackendConfig,
    model_names: list[str],
    results_dir: Path,
    *,
    repo_root: Path | None = None,
    model_configs: list[ModelConfig] | None = None,
) -> RunManifest:
    """Build the manifest, always stamping provenance.

    ``repo_root`` defaults to the checkout this package was imported from, so a
    caller cannot forget to stamp. It was optional at first, and the very first
    run afterwards produced an unstamped bundle: ``run_full_code_generation.py``
    builds its own manifest and had not been updated, so the field quietly did
    nothing in the one place a baseline actually gets measured. Outside a
    checkout the git lookups return None, which reads as "unknown" and blocks
    comparison — the honest outcome, not a silent pass.

    ``model_configs`` is still optional: without it the stamp records the code
    but not the per-model options, which is worth having on its own.
    """
    # Imported here: provenance imports models, which imports nothing from
    # runtime, and a module-level import would close the loop.
    from llm_benchmark.provenance import collect_provenance

    provenance = collect_provenance(
        repo_root if repo_root is not None else Path(__file__).resolve().parents[2],
        model_configs=model_configs,
        sampling=experiment.sampling,
    )
    return RunManifest(
        experiment=experiment,
        platform=platform_config,
        backend=backend_config,
        model_names=model_names,
        environment=build_environment_snapshot(platform_config, backend_config),
        results_dir=results_dir,
        provenance=provenance,
    )
