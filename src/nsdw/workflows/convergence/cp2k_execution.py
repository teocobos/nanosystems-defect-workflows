"""Local execution orchestration for CP2K convergence studies."""

from __future__ import annotations

from pathlib import Path

from nsdw.models.result import NSDWResult
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)
from nsdw.workflows.single_point import (
    run_cp2k_single_point,
)


class CP2KConvergenceExecutionError(RuntimeError):
    """Raised when a CP2K convergence campaign cannot be executed."""


def run_cp2k_convergence_campaign(
    *,
    campaign_directory: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
) -> tuple[NSDWResult, ...]:
    """Run all calculations in a generated CP2K convergence campaign."""

    campaign_directory = (
        Path(campaign_directory)
        .expanduser()
        .resolve()
    )

    manifest = load_convergence_manifest(
        campaign_directory / "manifest.json"
    )

    if manifest.calculator != "cp2k":
        raise CP2KConvergenceExecutionError(
            "CP2K convergence execution requires a "
            f"CP2K manifest, got {manifest.calculator!r}."
        )

    results: list[NSDWResult] = []

    for candidate in manifest.candidates:
        working_directory = (
            campaign_directory / candidate.directory
        )

        input_file = Path(candidate.input_file)

        output_file = input_file.with_suffix(".out")

        result = run_cp2k_single_point(
            calculation_id=candidate.label,
            working_directory=working_directory,
            input_file=input_file,
            output_file=output_file,
            executable=executable,
            environment=environment,
        )

        results.append(result)

    return tuple(results)
