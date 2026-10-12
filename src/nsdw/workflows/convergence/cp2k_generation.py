"""Generate CP2K calculation packages for convergence studies."""

from __future__ import annotations

import shutil
from pathlib import Path

from pymatgen.core import Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
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




def _validate_basis_candidate(
    *,
    base_config: CP2KInputConfig,
    candidate: CP2KBasisPotentialConfig,
) -> None:
    """Require basis convergence to preserve pseudopotentials."""

    reference = base_config.basis_potential

    if reference is None:
        raise CP2KConvergenceGenerationError(
            "Basis convergence requires a base basis/potential configuration."
        )

    if candidate.potential_file != reference.potential_file:
        raise CP2KConvergenceGenerationError(
            "Basis candidates must preserve the pseudopotential file."
        )

    reference_potentials = {
        kind.element: kind.potential
        for kind in reference.kinds
    }

    candidate_potentials = {
        kind.element: kind.potential
        for kind in candidate.kinds
    }

    if (
        len(reference_potentials) != len(reference.kinds)
        or len(candidate_potentials) != len(candidate.kinds)
        or candidate_potentials != reference_potentials
    ):
        raise CP2KConvergenceGenerationError(
            "Basis candidates must preserve element coverage "
            "and pseudopotential assignments."
        )


def _candidate_config(
    *,
    base_config: CP2KInputConfig,
    parameter: ConvergenceParameter,
    value: Quantity | str | tuple[int, int, int] | None,
    project_name: str,
    basis_candidates: dict[str, CP2KBasisPotentialConfig] | None = None,
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
        if basis_candidates is None:
            raise CP2KConvergenceGenerationError(
                "CP2K basis-set convergence generation is not yet supported "
                "without explicit basis candidates."
            )

        if not isinstance(value, str) or value not in basis_candidates:
            raise CP2KConvergenceGenerationError(
                f"Unknown CP2K basis candidate: {value!r}"
            )

        candidate_basis = basis_candidates[value]

        _validate_basis_candidate(
            base_config=base_config,
            candidate=candidate_basis,
        )

        updates["basis_potential"] = candidate_basis

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
    basis_candidates: dict[str, CP2KBasisPotentialConfig] | None = None,
) -> tuple[Path, ...]:
    """Generate one portable CP2K package per convergence candidate."""

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_existed_before = output_directory.exists()

    packages: list[Path] = []
    candidate_configs: list[CP2KInputConfig] = []

    try:
        # Preserve the canonical periodic structure used by every
        # convergence candidate. Candidate XYZ files contain only
        # execution coordinates and do not preserve the lattice.
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_path = output_directory / "base-config.json"

        if config_path.exists():
            raise CP2KConvergenceGenerationError(
                "Convergence base configuration already exists: "
                f"{config_path}"
            )

        config_path.write_text(
            base_config.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
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
                basis_candidates=basis_candidates,
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
            candidate_configs.append(config)

        # Build the persistent manifest describing the generated study.
        manifest_candidates: list[
            ConvergenceManifestCandidate
        ] = []

        for candidate, package, config in zip(
            study.candidates,
            packages,
            candidate_configs,
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
                    basis_potential=(
                        config.basis_potential
                        if study.parameter == ConvergenceParameter.BASIS
                        else None
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