"""Generate CP2K calculation packages for convergence studies."""

from __future__ import annotations

from pathlib import Path

from pymatgen.core import Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
)
from nsdw.calculators.cp2k.package import (
    write_cp2k_package,
)
from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.models import (
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)
import shutil

class CP2KConvergenceGenerationError(RuntimeError):
    """Raised when a CP2K convergence study cannot be generated."""


def _candidate_config(
    *,
    base_config: CP2KInputConfig,
    parameter: ConvergenceParameter,
    value: Quantity | str | tuple[int, int, int] | None,
    project_name: str,
) -> CP2KInputConfig:
    """Return a CP2K configuration for one convergence candidate."""

    if value is None:
        raise CP2KConvergenceGenerationError(
            "CP2K convergence candidates must define a value."
        )

    updates: dict[str, object] = {
        "project_name": project_name,
    }

    if parameter == ConvergenceParameter.CUTOFF:
        if not isinstance(value, Quantity):
            raise CP2KConvergenceGenerationError(
                "CP2K cutoff convergence requires Quantity values."
            )

        updates["cutoff_ry"] = value.value

    elif parameter == ConvergenceParameter.RELATIVE_CUTOFF:
        if not isinstance(value, Quantity):
            raise CP2KConvergenceGenerationError(
                "CP2K relative-cutoff convergence requires "
                "Quantity values."
            )

        updates["relative_cutoff_ry"] = value.value

    elif parameter == ConvergenceParameter.KPOINTS:
        if not isinstance(value, tuple):
            raise CP2KConvergenceGenerationError(
                "CP2K k-point convergence requires tuple values."
            )

        updates["k_points"] = value

    elif parameter == ConvergenceParameter.BASIS:
        raise CP2KConvergenceGenerationError(
            "CP2K basis-set convergence generation is not yet supported."
        )

    else:
        raise CP2KConvergenceGenerationError(
            f"Unsupported CP2K convergence parameter: {parameter}"
        )

    return base_config.model_copy(
        update=updates,
    )


def generate_cp2k_convergence_study(
    *,
    structure: Structure,
    base_config: CP2KInputConfig,
    study: ConvergenceStudyDefinition,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Generate one portable CP2K package per convergence candidate."""

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_existed_before = output_directory.exists()

    packages: list[Path] = []

    try:
        for candidate in study.candidates:
            project_name = (
                f"{base_config.project_name}-{candidate.label}"
            )

            config = _candidate_config(
                base_config=base_config,
                parameter=study.parameter,
                value=candidate.value,
                project_name=project_name,
            )

            candidate_directory = (
                output_directory / candidate.label
            )

            package = write_cp2k_package(
                structure=structure,
                config=config,
                output_directory=candidate_directory,
            )

            packages.append(package)

    except Exception:
        if not output_existed_before and output_directory.exists():
            shutil.rmtree(output_directory)

        raise

    return tuple(packages)