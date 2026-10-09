"""Independent CP2K evidence checks for convergence campaigns."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
from nsdw.calculators.cp2k.input_parser import parse_cp2k_input
from nsdw.calculators.cp2k.models import CP2KSCFStatus
from nsdw.calculators.cp2k.parser import parse_cp2k_output
from nsdw.structures.result import calculate_structure_hash
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
    validate_convergence_evidence,
)
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)
from nsdw.workflows.convergence.cp2k_structure_evidence import (
    validate_cp2k_structure_evidence,
)
from nsdw.workflows.convergence.cp2k_output_evidence import (
    validate_cp2k_single_point_output,
)


def _file_within(root: Path, path: Path) -> Path:
    """Resolve paths and reject missing files or directory escapes."""
    resolved = path.expanduser().resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConvergenceEvidenceValidationError(
            f"Evidence path escapes project root: {path}"
        ) from exc

    if not resolved.is_file():
        raise ConvergenceEvidenceValidationError(
            f"Missing evidence file: {resolved}"
        )

    return resolved


def _output_path(
    root: Path,
    directory: Path,
    input_file: Path,
) -> Path:
    """Reject ambiguous output naming rather than guessing."""
    derived = directory / input_file.with_suffix(".out").name
    legacy = directory / "calculation.out"

    choices = {
        path.resolve()
        for path in (derived, legacy)
        if path.is_file()
    }

    if len(choices) != 1:
        raise ConvergenceEvidenceValidationError(
            "Expected exactly one unambiguous CP2K output "
            f"for input {input_file.name}; found {len(choices)}."
        )

    return _file_within(root, next(iter(choices)))


def _positive_finite(value: Any, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ConvergenceEvidenceValidationError(
            f"Invalid {description}."
        )

    return float(value)


def _input_signature(parsed: Any, parameter: str) -> tuple:
    """Compare settings that must not change during a campaign."""
    fields = (
        "run_type",
        "charge",
        "multiplicity",
        "xc_functional",
        "eps_scf",
        "scf_solver",
        "k_points",
        "kinds",
        "basis_set_file",
        "potential_file",
        "admm",
    )

    if parameter != "cutoff":
        fields += ("cutoff_ry",)

    if parameter != "relative_cutoff":
        fields += ("relative_cutoff_ry",)

    if parameter == "kpoints":
        fields = tuple(
            field for field in fields if field != "k_points"
        )

    return tuple(
        (field, getattr(parsed, field))
        for field in fields
    )


def validate_cp2k_calculation_evidence(
    *,
    project_root: str | Path,
    report_path: str | Path,
    expected_parameter: str,
    expected_value: float | tuple[int, int, int],
    expected_structure_hash: str | None = None,
    expected_n_atoms: int | None = None,
) -> dict[str, Any]:
    """Validate report values against CP2K input/output evidence.

    This is an additional verification layer. It does not establish
    coordinate-file identity, resolve CP2K INCLUDE dependencies, or
    authorize methodology promotion.
    """
    root = Path(project_root).expanduser().resolve()
    report_file = Path(report_path).expanduser()

    if not report_file.is_absolute():
        report_file = root / report_file

    report_file = _file_within(root, report_file)

    report = validate_convergence_evidence(
        project_root=root,
        report_path=report_file,
        expected_parameter=expected_parameter,
        expected_value=expected_value,
        expected_structure_hash=expected_structure_hash,
        expected_n_atoms=expected_n_atoms,
    )

    campaign = report_file.parent
    manifest_path = _file_within(
        root, campaign / "manifest.json"
    )
    structure_path = _file_within(
        root, campaign / "structure.json"
    )

    try:
        manifest = load_convergence_manifest(manifest_path)
        structure = Structure.from_file(structure_path)
        actual_hash = calculate_structure_hash(structure)
    except Exception as exc:
        raise ConvergenceEvidenceValidationError(
            "Cannot reconstruct canonical campaign structure."
        ) from exc

    if (
        actual_hash != report["structure_hash"]
        or len(structure) != report["n_atoms"]
    ):
        raise ConvergenceEvidenceValidationError(
            "Canonical structure differs from convergence report."
        )

    baseline_signature = None
    used_directories: set[Path] = set()
    used_inputs: set[Path] = set()
    used_outputs: set[Path] = set()

    for candidate, recorded in zip(
        manifest.candidates,
        report["candidates"],
        strict=True,
    ):
        directory = (campaign / candidate.directory).resolve()

        try:
            directory.relative_to(root)
            directory.relative_to(campaign)
        except ValueError as exc:
            raise ConvergenceEvidenceValidationError(
                "Candidate directory escapes its campaign."
            ) from exc

        if directory in used_directories:
            raise ConvergenceEvidenceValidationError(
                "Duplicate candidate directory."
            )

        used_directories.add(directory)

        input_path = _file_within(
            root, directory / candidate.input_file
        )

        if input_path.parent != directory:
            raise ConvergenceEvidenceValidationError(
                "Candidate input is outside its directory."
            )

        output_path = _output_path(
            root, directory, input_path
        )

        if (
            input_path in used_inputs
            or output_path in used_outputs
        ):
            raise ConvergenceEvidenceValidationError(
                "Candidate evidence files are reused."
            )

        used_inputs.add(input_path)
        used_outputs.add(output_path)

        validate_cp2k_structure_evidence(
            project_root=root,
            campaign_directory=campaign,
            candidate_directory=directory,
            input_filename=candidate.input_file,
            coordinate_filename=candidate.coordinate_file,
        )

        try:
            parsed_input = parse_cp2k_input(input_path)
            parsed_output = parse_cp2k_output(output_path)
        except Exception as exc:
            raise ConvergenceEvidenceValidationError(
                f"Cannot parse CP2K evidence for {candidate.label}."
            ) from exc

        parameter = manifest.parameter.value

        if parameter == "cutoff":
            actual = parsed_input.cutoff_ry
            expected = candidate.value.value
        elif parameter == "relative_cutoff":
            actual = parsed_input.relative_cutoff_ry
            expected = candidate.value.value
        elif parameter == "kpoints":
            actual = parsed_input.k_points
            expected = tuple(candidate.value)
        else:
            raise ConvergenceEvidenceValidationError(
                f"Unsupported CP2K evidence parameter: {parameter}"
            )

        if parameter == "kpoints":
            if actual != expected:
                raise ConvergenceEvidenceValidationError(
                    f"Incorrect k-point mesh for {candidate.label}."
                )
            if parsed_input.scf_solver != "DIAGONALIZATION":
                raise ConvergenceEvidenceValidationError(
                    "Explicit k-point evidence requires DIAGONALIZATION."
                )
        elif (
            actual is None
            or not math.isclose(
                _positive_finite(actual, "CP2K parameter"),
                _positive_finite(expected, "manifest parameter"),
                rel_tol=1e-12,
                abs_tol=1e-10,
            )
        ):
            raise ConvergenceEvidenceValidationError(
                f"Incorrect {parameter} for {candidate.label}."
            )

        signature = _input_signature(parsed_input, parameter)

        if baseline_signature is None:
            baseline_signature = signature
        elif signature != baseline_signature:
            raise ConvergenceEvidenceValidationError(
                "Non-varied CP2K settings differ between candidates."
            )

        if (
            parsed_output.normal_termination is not True
            or parsed_output.scf.status != CP2KSCFStatus.CONVERGED
        ):
            raise ConvergenceEvidenceValidationError(
                f"CP2K calculation did not complete successfully: "
                f"{candidate.label}."
            )

        verified_energy_hartree = (
            validate_cp2k_single_point_output(output_path)
        )

        for field in (
            "project_name",
            "run_type",
            "charge",
            "multiplicity",
        ):
            input_value = getattr(parsed_input, field)
            output_value = getattr(parsed_output, field)

            if (
                input_value is None
                or output_value is None
                or input_value != output_value
            ):
                raise ConvergenceEvidenceValidationError(
                    f"CP2K input/output {field} identity mismatch "
                    f"or missing metadata: {candidate.label}."
                )

        if parsed_input.run_type.value != "ENERGY":
            raise ConvergenceEvidenceValidationError(
                f"CP2K convergence requires ENERGY run type: "
                f"{candidate.label}."
            )

        energy_hartree = (
            parsed_output.energy.total_energy_hartree
        )

        if energy_hartree is None:
            raise ConvergenceEvidenceValidationError(
                f"Missing final CP2K energy: {candidate.label}."
            )

        if not math.isclose(
            energy_hartree,
            verified_energy_hartree,
            rel_tol=1e-12,
            abs_tol=1e-10,
        ):
            raise ConvergenceEvidenceValidationError(
                f"Final CP2K energy provenance mismatch: "
                f"{candidate.label}."
            )

        actual_energy_ev = (
            _positive_finite(energy_hartree, "CP2K energy")
            * HARTREE_TO_EV
        )

        reported_energy_ev = _positive_finite(
            recorded["total_energy_ev"],
            "reported energy",
        )

        if not math.isclose(
            actual_energy_ev,
            reported_energy_ev,
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ConvergenceEvidenceValidationError(
                f"CP2K output energy differs from report: "
                f"{candidate.label}."
            )

    return report
