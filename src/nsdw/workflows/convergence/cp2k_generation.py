"""Generate CP2K calculation packages for convergence studies."""

from __future__ import annotations

import shutil
from pathlib import Path

from pymatgen.core import Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
)
from nsdw.calculators.cp2k.package import (
    write_cp2k_package,
)
from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.manifest import (
    ConvergenceManifestCandidate,
    ConvergenceStudyManifest,
    write_convergence_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)



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
        # Preserve the canonical periodic structure used by every
        # convergence candidate. Candidate XYZ files contain only
        # execution coordinates and do not preserve the lattice.
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        structure.to(
            filename=output_directory / "structure.json",
            fmt="json",
        )

        # Generate one CP2K package for each convergence candidate.
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

        # Build the persistent manifest describing the generated study.
        manifest_candidates: list[
            ConvergenceManifestCandidate
        ] = []

        for candidate, package in zip(
            study.candidates,
            packages,
            strict=True,
        ):
            if candidate.value is None:
                raise CP2KConvergenceGenerationError(
                    "Generated convergence candidates must "
                    "define a value."
                )

            input_files = tuple(
                package.glob("*.inp")
            )

            coordinate_files = tuple(
                package.glob("*.xyz")
            )

            if len(input_files) != 1:
                raise CP2KConvergenceGenerationError(
                    "Expected exactly one CP2K input file "
                    f"in {package}."
                )

            if len(coordinate_files) != 1:
                raise CP2KConvergenceGenerationError(
                    "Expected exactly one coordinate file "
                    f"in {package}."
                )

            manifest_candidates.append(
                ConvergenceManifestCandidate(
                    label=candidate.label,
                    order=candidate.order,
                    value=candidate.value,
                    directory=package.relative_to(
                        output_directory
                    ).as_posix(),
                    input_file=input_files[0].name,
                    coordinate_file=(
                        coordinate_files[0].name
                    ),
                )
            )

        manifest = ConvergenceStudyManifest(
            calculator="cp2k",
            parameter=study.parameter,
            criterion=study.criterion,
            candidates=tuple(
                manifest_candidates
            ),
        )

        write_convergence_manifest(
            manifest=manifest,
            path=output_directory / "manifest.json",
        )

    except Exception:
        # Only remove the campaign directory if this invocation
        # created it. Never delete a directory that existed before
        # generation began.
        if (
            not output_existed_before
            and output_directory.exists()
        ):
            shutil.rmtree(output_directory)

        raise

    return tuple(packages)