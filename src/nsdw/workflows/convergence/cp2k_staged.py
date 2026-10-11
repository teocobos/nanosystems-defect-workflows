"""Resumable CP2K convergence workflow stages."""

from __future__ import annotations

from pathlib import Path

from nsdw.calculators.cp2k.generation_models import CP2KInputConfig
from nsdw.calculators.cp2k.input_parser import parse_cp2k_input
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)
from nsdw.workflows.convergence.models import ConvergenceParameter


class CP2KConvergenceStageError(RuntimeError):
    """Raised when a convergence stage cannot safely advance."""


def validate_cutoff_campaign_configuration(
    *,
    campaign_directory: str | Path,
    base_config: CP2KInputConfig,
) -> None:
    """Check the configuration against all existing cutoff inputs."""

    directory = Path(campaign_directory).expanduser().resolve()
    manifest = load_convergence_manifest(directory / "manifest.json")

    if manifest.calculator != "cp2k":
        raise CP2KConvergenceStageError(
            "Expected a CP2K convergence campaign."
        )

    if manifest.parameter != ConvergenceParameter.CUTOFF:
        raise CP2KConvergenceStageError(
            "Expected a cutoff convergence campaign."
        )

    if not manifest.candidates:
        raise CP2KConvergenceStageError(
            "Cutoff campaign contains no candidates."
        )

    if base_config.basis_potential is None:
        raise CP2KConvergenceStageError(
            "A basis/potential configuration is required."
        )

    for candidate in manifest.candidates:
        input_path = (
            directory / candidate.directory / candidate.input_file
        )

        parsed = parse_cp2k_input(input_path)

        expected_project = (
            f"{base_config.project_name}-{candidate.label}"
        )

        if parsed.project_name != expected_project:
            raise CP2KConvergenceStageError(
                f"Project mismatch in {input_path}"
            )

        if parsed.run_type is None or (
            parsed.run_type.value != base_config.run_type
        ):
            raise CP2KConvergenceStageError(
                f"Run-type mismatch in {input_path}"
            )

        if parsed.charge != base_config.charge:
            raise CP2KConvergenceStageError(
                f"Charge mismatch in {input_path}"
            )

        if parsed.multiplicity != base_config.multiplicity:
            raise CP2KConvergenceStageError(
                f"Multiplicity mismatch in {input_path}"
            )

        if parsed.cutoff_ry != candidate.value.value:
            raise CP2KConvergenceStageError(
                f"Cutoff mismatch in {input_path}"
            )

        if parsed.relative_cutoff_ry != base_config.relative_cutoff_ry:
            raise CP2KConvergenceStageError(
                f"Relative-cutoff mismatch in {input_path}"
            )

        if parsed.k_points != base_config.k_points:
            raise CP2KConvergenceStageError(
                f"K-point mismatch in {input_path}"
            )

        expected_scf = base_config.scf

        if (
            parsed.eps_scf is None
            or abs(parsed.eps_scf - expected_scf.eps_scf)
            > 1.0e-10 * expected_scf.eps_scf
        ):
            raise CP2KConvergenceStageError(
                f"EPS_SCF mismatch in {input_path}"
            )

        scf_fields = (
            ("scf_solver", "solver"),
            ("scf_guess", "scf_guess"),
            ("max_scf", "max_scf"),
            ("outer_scf_max", "outer_scf_max"),
        )

        for parsed_name, config_name in scf_fields:
            actual = getattr(parsed, parsed_name)
            expected = getattr(expected_scf, config_name)

            if actual != expected:
                raise CP2KConvergenceStageError(
                    f"{parsed_name} mismatch in {input_path}: "
                    f"{actual!r} != {expected!r}"
                )

        if expected_scf.solver == "OT":
            ot_fields = (
                "ot_minimizer",
                "ot_preconditioner",
            )

            for field in ot_fields:
                if getattr(parsed, field) != getattr(expected_scf, field):
                    raise CP2KConvergenceStageError(
                        f"{field} mismatch in {input_path}"
                    )

            if (
                parsed.energy_gap is None
                or abs(parsed.energy_gap - expected_scf.energy_gap)
                > 1.0e-10 * expected_scf.energy_gap
            ):
                raise CP2KConvergenceStageError(
                    f"OT energy-gap mismatch in {input_path}"
                )

        if (
            parsed.xc_functional is None
            or parsed.xc_functional.upper()
            != base_config.functional.value.upper()
        ):
            raise CP2KConvergenceStageError(
                f"Functional mismatch in {input_path}"
            )

        expected_kinds = {
            kind.element: (kind.basis_set, kind.potential)
            for kind in base_config.basis_potential.kinds
        }

        actual_kinds = {
            kind.element: (kind.basis_set, kind.potential)
            for kind in parsed.kinds
        }

        if actual_kinds != expected_kinds:
            raise CP2KConvergenceStageError(
                f"Basis/potential mismatch in {input_path}"
            )


def advance_cp2k_convergence_campaign(
    *,
    completed_campaign_directory: str | Path,
    next_campaign_directory: str | Path,
    base_config: CP2KInputConfig | None = None,
) -> tuple[Path, ...]:
    """Generate a relative-cutoff stage from validated cutoff evidence.

    Does not execute CP2K, submit jobs, or modify the source campaign.
    """
    import hashlib
    import json
    from types import SimpleNamespace

    from pymatgen.core import Structure

    from nsdw.structures.result import calculate_structure_hash
    from nsdw.workflows.convergence.cp2k_evidence import (
        validate_cp2k_calculation_evidence,
    )
    from nsdw.workflows.convergence.cp2k_generation import (
        generate_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.manifest import (
        convergence_study_from_manifest,
    )
    from nsdw.workflows.convergence.recipes import (
        apply_selected_cutoff,
        build_standard_cp2k_relative_cutoff_study,
        selected_ry_value,
    )

    source = Path(completed_campaign_directory).expanduser().resolve()
    destination = Path(next_campaign_directory).expanduser().resolve()

    if not source.is_dir():
        raise CP2KConvergenceStageError(
            f"Source campaign does not exist: {source}"
        )

    if destination.exists():
        raise CP2KConvergenceStageError(
            f"Destination already exists: {destination}"
        )

    if destination == source or source in destination.parents:
        raise CP2KConvergenceStageError(
            "Destination must not be inside the source campaign."
        )

    manifest = load_convergence_manifest(source / "manifest.json")

    if manifest.calculator != "cp2k":
        raise CP2KConvergenceStageError(
            "Source campaign must use CP2K."
        )

    if manifest.parameter != ConvergenceParameter.CUTOFF:
        raise CP2KConvergenceStageError(
            "Source campaign must be a cutoff study."
        )

    config_path = source / "base-config.json"

    if config_path.is_file():
        stored_config = CP2KInputConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )

        if base_config is not None and base_config != stored_config:
            raise CP2KConvergenceStageError(
                "Supplied configuration differs from persisted configuration."
            )

        config = stored_config
    else:
        if base_config is None:
            raise CP2KConvergenceStageError(
                "Historical campaign requires an explicit base_config."
            )

        config = base_config

    validate_cutoff_campaign_configuration(
        campaign_directory=source,
        base_config=config,
    )

    report_path = source / "convergence-report.json"

    if not report_path.is_file():
        raise CP2KConvergenceStageError(
            "Completed convergence report is missing."
        )

    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)

    if report.get("energy_tolerance_satisfied") is not True:
        raise CP2KConvergenceStageError(
            "Cutoff convergence tolerance was not satisfied."
        )

    selected_label = report.get("selected_candidate_label")

    if not isinstance(selected_label, str) or not selected_label:
        raise CP2KConvergenceStageError(
            "Convergence report has no selected candidate."
        )

    study = convergence_study_from_manifest(manifest)

    analysis = SimpleNamespace(
        selected_candidate_label=selected_label,
    )

    selected_cutoff = selected_ry_value(
        study=study,
        analysis=analysis,
    )

    validated = validate_cp2k_calculation_evidence(
        project_root=source.parent,
        report_path=report_path,
        expected_parameter="cutoff",
        expected_value=selected_cutoff,
    )

    structure_path = source / "structure.json"
    structure = Structure.from_file(structure_path)

    structure_hash = calculate_structure_hash(structure)

    if (
        structure_hash != validated["structure_hash"]
        or len(structure) != validated["n_atoms"]
    ):
        raise CP2KConvergenceStageError(
            "Canonical structure does not match validated evidence."
        )

    next_config = apply_selected_cutoff(
        base_config=config,
        study=study,
        analysis=analysis,
    )

    next_study = build_standard_cp2k_relative_cutoff_study()

    import os
    import shutil
    import tempfile

    # Stage output beside the destination so publication remains
    # on the same filesystem.
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )

    try:
        staged_packages = generate_cp2k_convergence_study(
            structure=structure,
            base_config=next_config,
            study=next_study,
            output_directory=staging,
        )

        provenance = {
            "schema_version": 1,
            "source_campaign": str(source),
            "source_report_sha256": hashlib.sha256(
                report_bytes
            ).hexdigest(),
            "source_structure_hash": structure_hash,
            "source_parameter": "cutoff",
            "selected_candidate_label": selected_label,
            "selected_cutoff_ry": selected_cutoff,
            "next_parameter": "relative_cutoff",
            "relative_cutoff_values_ry": [
                candidate.value.value
                for candidate in next_study.candidates
            ],
        }

        (staging / "stage-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )

        _publish_directory_no_replace(
            staging=staging,
            destination=destination,
        )

        return tuple(
            destination / package.relative_to(staging)
            for package in staged_packages
        )

    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _publish_directory_no_replace(
    staging: Path,
    destination: Path,
) -> None:
    """Atomically publish a directory without replacing an existing path.

    Requires Linux renameat2 with RENAME_NOREPLACE support.
    """
    import ctypes
    import errno
    import os

    libc = ctypes.CDLL(None, use_errno=True)

    renameat2 = getattr(libc, "renameat2", None)

    if renameat2 is None:
        raise CP2KConvergenceStageError(
            "Atomic no-replace directory publication is unavailable."
        )

    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int

    at_fdcwd = -100
    rename_noreplace = 1

    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )

    if result != 0:
        error = ctypes.get_errno()

        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise CP2KConvergenceStageError(
                f"Destination already exists: {destination}"
            )

        raise CP2KConvergenceStageError(
            "Atomic stage publication failed: "
            f"{os.strerror(error)}"
        )
