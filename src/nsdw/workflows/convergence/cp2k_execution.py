"""Local execution orchestration for CP2K convergence studies."""

from __future__ import annotations

from pathlib import Path
from pymatgen.core import Structure

from nsdw.models.result import NSDWResult
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)
from nsdw.workflows.single_point import (
    run_cp2k_single_point,
)
from nsdw.calculators.cp2k import (
    adapt_cp2k_result,
    parse_cp2k_input,
    parse_cp2k_output,
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

    structure_path = (
        campaign_directory / "structure.json"
    )

    if not structure_path.is_file():
        raise CP2KConvergenceExecutionError(
            "CP2K convergence campaign is missing its "
            f"canonical structure file: {structure_path}"
        )

    try:
        structure = Structure.from_file(
            structure_path
        )

    except Exception as exc:
        raise CP2KConvergenceExecutionError(
            "Could not load the canonical structure for "
            f"CP2K convergence campaign: {structure_path}"
        ) from exc

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
            structure=structure,
        )

        results.append(result)

    return tuple(results)


def collect_existing_cp2k_convergence_results(
    *,
    campaign_directory: str | Path,
    output_file: str | Path = "calculation.out",
) -> tuple[NSDWResult, ...]:
    """Collect completed CP2K convergence results without executing CP2K."""

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
            "CP2K convergence collection requires a "
            f"CP2K manifest, got {manifest.calculator!r}."
        )

    structure_path = (
        campaign_directory / "structure.json"
    )

    if not structure_path.is_file():
        raise CP2KConvergenceExecutionError(
            "CP2K convergence campaign is missing its "
            f"canonical structure file: {structure_path}"
        )

    try:
        structure = Structure.from_file(
            structure_path
        )

    except Exception as exc:
        raise CP2KConvergenceExecutionError(
            "Could not load the canonical structure for "
            f"CP2K convergence campaign: {structure_path}"
        ) from exc

    output_file = Path(output_file)

    results: list[NSDWResult] = []

    for candidate in manifest.candidates:
        working_directory = (
            campaign_directory / candidate.directory
        )

        input_path = (
            working_directory / candidate.input_file
        )

        candidate_output_path = (
            output_file
            if output_file.is_absolute()
            else working_directory / output_file
        )

        if not input_path.is_file():
            raise CP2KConvergenceExecutionError(
                "CP2K convergence candidate is missing "
                f"its input file: {input_path}"
            )

        if not candidate_output_path.is_file():
            raise CP2KConvergenceExecutionError(
                "CP2K convergence candidate is missing "
                f"its output file: {candidate_output_path}"
            )

        parsed_input = parse_cp2k_input(
            input_path
        )

        parsed_output = parse_cp2k_output(
            candidate_output_path
        )

        result = adapt_cp2k_result(
            parsed_output,
            structure=structure,
            input_settings=parsed_input,
            input_path=input_path,
            output_path=candidate_output_path,
            calculation_id=candidate.label,
        )

        results.append(result)

    return tuple(results)