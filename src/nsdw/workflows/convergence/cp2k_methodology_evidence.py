"""Independent CP2K methodology provenance-path validation."""

from __future__ import annotations

from pathlib import Path
import json
import math

from nsdw.calculators.cp2k.input_parser import parse_cp2k_input
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)

from nsdw.workflows.convergence.cp2k_evidence import (
    validate_cp2k_calculation_evidence,
)

from nsdw.project.models import CP2KProductionMethodology
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


REPORT_FIELDS = (
    "cutoff_report",
    "relative_cutoff_report",
    "kpoint_report",
    "final_cutoff_verification_report",
    "final_relative_cutoff_verification_report",
)


def resolve_methodology_evidence_paths(
    *,
    project_root: str | Path,
    candidate_path: str | Path,
    methodology: CP2KProductionMethodology,
) -> dict[str, Path]:
    """Resolve report paths without permitting project escapes.

    Report paths in a methodology candidate are relative to the
    candidate artifact's parent directory.

    This function validates paths only. It does not establish
    scientific correctness of report contents.
    """
    root = Path(project_root).expanduser().resolve()
    candidate = Path(candidate_path).expanduser()

    if not candidate.is_absolute():
        candidate = root / candidate

    candidate = candidate.resolve()

    if not root.is_dir():
        raise ConvergenceEvidenceValidationError(
            f"Project root does not exist: {root}"
        )

    if not candidate.is_relative_to(root):
        raise ConvergenceEvidenceValidationError(
            "Methodology candidate escapes the project root."
        )

    if not candidate.is_file():
        raise ConvergenceEvidenceValidationError(
            f"Methodology candidate is missing: {candidate}"
        )

    provenance = methodology.provenance

    if (
        provenance.source != "convergence"
        or provenance.workflow != "standard_cp2k_convergence"
    ):
        raise ConvergenceEvidenceValidationError(
            "Unsupported methodology provenance."
        )

    required = {
        "cutoff_report",
        "relative_cutoff_report",
    }

    if methodology.k_points is not None:
        required.update({
            "kpoint_report",
            "final_cutoff_verification_report",
            "final_relative_cutoff_verification_report",
        })
    else:
        if any(
            getattr(provenance, field) is not None
            for field in (
                "kpoint_report",
                "final_cutoff_verification_report",
                "final_relative_cutoff_verification_report",
            )
        ):
            raise ConvergenceEvidenceValidationError(
                "Gamma-only methodology has k-point evidence."
            )

    if methodology.k_points is not None:
        if methodology.scf.solver != "DIAGONALIZATION":
            raise ConvergenceEvidenceValidationError(
                "K-point methodology requires diagonalisation."
            )

    resolved: dict[str, Path] = {}
    used_paths: set[Path] = set()

    for field in REPORT_FIELDS:
        raw = getattr(provenance, field)

        if raw is None:
            if field in required:
                raise ConvergenceEvidenceValidationError(
                    f"Required methodology report is missing: {field}"
                )
            continue

        if not isinstance(raw, str) or not raw.strip():
            raise ConvergenceEvidenceValidationError(
                f"Invalid methodology report path: {field}"
            )

        relative = Path(raw)

        if (
            relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ConvergenceEvidenceValidationError(
                f"Unsafe methodology report path: {field}"
            )

        path = (candidate.parent / relative).resolve()

        if not path.is_relative_to(root):
            raise ConvergenceEvidenceValidationError(
                f"Methodology report escapes project: {field}"
            )

        if not path.is_relative_to(candidate.parent):
            raise ConvergenceEvidenceValidationError(
                f"Methodology report escapes workflow: {field}"
            )

        if not path.is_file():
            raise ConvergenceEvidenceValidationError(
                f"Methodology report does not exist: {field}"
            )

        if path.name != "convergence-report.json":
            raise ConvergenceEvidenceValidationError(
                f"Unexpected methodology report filename: {field}"
            )

        if path in used_paths:
            raise ConvergenceEvidenceValidationError(
                "Methodology reports must have distinct paths."
            )

        used_paths.add(path)
        resolved[field] = path

    return resolved



def _selected_report_value(
    report_path: Path,
    parameter: str,
):
    """Read the selection only to supply the independent validator."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        label = report["selected_candidate_label"]
        matches = [
            row for row in report["candidates"]
            if row["label"] == label
        ]

        if len(matches) != 1:
            raise ValueError("Invalid selected candidate")

        value = matches[0]["value"]

        if parameter == "kpoints":
            if (
                not isinstance(value, list)
                or len(value) != 3
                or any(type(v) is not int or v <= 0 for v in value)
            ):
                raise ValueError("Invalid selected k-point mesh")
            return tuple(value)

        if (
            not isinstance(value, dict)
            or value.get("unit") != "Ry"
        ):
            raise ValueError("Invalid selected numerical value")

        numeric = value["value"]
        if (
            isinstance(numeric, bool)
            or not isinstance(numeric, (int, float))
            or not math.isfinite(numeric)
            or numeric <= 0
        ):
            raise ValueError("Invalid selected numerical value")

        return float(numeric)

    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConvergenceEvidenceValidationError(
            f"Cannot determine selected value from {report_path}."
        ) from exc


def _require_tail_stable_value(
    *,
    report: dict,
    parameter: str,
    proposed_value: float,
) -> None:
    """Check a production value independently of report selection."""
    candidates = report["candidates"]
    matches = [
        (index, row)
        for index, row in enumerate(candidates)
        if (
            isinstance(row.get("value"), dict)
            and row["value"].get("unit") == "Ry"
            and type(row["value"].get("value")) in (int, float)
            and math.isclose(
                row["value"]["value"],
                proposed_value,
                rel_tol=1e-12,
                abs_tol=1e-10,
            )
        )
    ]

    if len(matches) != 1:
        raise ConvergenceEvidenceValidationError(
            f"Proposed {parameter} is absent from verification grid."
        )

    index, candidate = matches[0]

    if index == len(candidates) - 1:
        raise ConvergenceEvidenceValidationError(
            f"Proposed {parameter} has no higher-cost verification."
        )

    tolerance = report["energy_tolerance_ev_per_atom"]
    energies = [
        row["energy_ev_per_atom"]
        for row in candidates
    ]

    difference = max(
        abs(energies[index] - higher)
        for higher in energies[index + 1:]
    )

    if (
        candidate["tail_stable"] is not True
        or difference > tolerance
    ):
        raise ConvergenceEvidenceValidationError(
            f"Proposed {parameter} is not tail-stable."
        )


def validate_cp2k_methodology_evidence(
    *,
    project_root: str | Path,
    candidate_path: str | Path,
    methodology: CP2KProductionMethodology,
) -> dict[str, dict]:
    """Independently validate each referenced convergence campaign.

    Read-only. This function does not authorize production promotion.
    """
    if methodology.status != "candidate":
        raise ConvergenceEvidenceValidationError(
            "Methodology evidence validation requires candidate status."
        )

    paths = resolve_methodology_evidence_paths(
        project_root=project_root,
        candidate_path=candidate_path,
        methodology=methodology,
    )

    stages = (
        ("cutoff_report", "cutoff", methodology.cutoff_ry, True),
        (
            "relative_cutoff_report",
            "relative_cutoff",
            methodology.relative_cutoff_ry,
            True,
        ),
        (
            "kpoint_report",
            "kpoints",
            methodology.k_points,
            True,
        ),
        (
            "final_cutoff_verification_report",
            "cutoff",
            methodology.cutoff_ry,
            False,
        ),
        (
            "final_relative_cutoff_verification_report",
            "relative_cutoff",
            methodology.relative_cutoff_ry,
            False,
        ),
    )

    validated: dict[str, dict] = {}
    structure_hash = None
    n_atoms = None

    for field, parameter, proposed, require_selection in stages:
        if field not in paths:
            continue

        selected = _selected_report_value(
            paths[field],
            parameter,
        )

        report = validate_cp2k_calculation_evidence(
            project_root=project_root,
            report_path=paths[field],
            expected_parameter=parameter,
            expected_value=selected,
            expected_structure_hash=structure_hash,
            expected_n_atoms=n_atoms,
        )

        if structure_hash is None:
            structure_hash = report["structure_hash"]
            n_atoms = report["n_atoms"]

        if require_selection:
            if parameter == "kpoints":
                matches = selected == proposed
            else:
                matches = math.isclose(
                    selected,
                    proposed,
                    rel_tol=1e-12,
                    abs_tol=1e-10,
                )

            if not matches:
                raise ConvergenceEvidenceValidationError(
                    f"Methodology {field} selection mismatch."
                )
        else:
            _require_tail_stable_value(
                report=report,
                parameter=parameter,
                proposed_value=proposed,
            )

        validated[field] = report

    _validate_cross_campaign_settings(
        paths=paths,
        methodology=methodology,
    )

    return validated



_COMMON_FIELDS = (
    "run_type",
    "charge",
    "multiplicity",
    "xc_functional",
    "eps_scf",
    "kinds",
    "basis_set_file",
    "potential_file",
    "admm",
)


def _campaign_inputs(report_path: Path):
    """Load every candidate input from a validated campaign."""
    campaign = report_path.parent
    manifest = load_convergence_manifest(
        campaign / "manifest.json"
    )

    parsed = []

    for candidate in manifest.candidates:
        directory = (
            campaign / candidate.directory
        ).resolve()

        if not directory.is_relative_to(campaign.resolve()):
            raise ConvergenceEvidenceValidationError(
                "Campaign candidate directory escapes campaign."
            )

        path = (
            directory / candidate.input_file
        ).resolve()

        if not path.is_relative_to(directory):
            raise ConvergenceEvidenceValidationError(
                "Campaign input escapes candidate directory."
            )

        if not path.is_file():
            raise ConvergenceEvidenceValidationError(
                "Missing campaign CP2K input."
            )

        parsed.append(parse_cp2k_input(path))

    if not parsed:
        raise ConvergenceEvidenceValidationError(
            "Campaign has no CP2K inputs."
        )

    return parsed


def _same_numeric(actual, expected):
    return (
        actual is not None
        and expected is not None
        and math.isclose(
            actual,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-10,
        )
    )





_SCF_COMMON_FIELDS = (
    "scf_guess",
    "max_scf",
)

_SCF_OT_FIELDS = (
    "outer_scf_max",
    "ot_minimizer",
    "ot_preconditioner",
)


def _validate_scf_identity(
    *,
    parsed_input,
    methodology: CP2KProductionMethodology,
) -> None:
    """Validate explicitly recorded, solver-relevant SCF settings."""
    expected = methodology.scf

    if parsed_input.scf_solver != expected.solver:
        raise ConvergenceEvidenceValidationError(
            "Production SCF solver differs from CP2K evidence."
        )

    if not _same_numeric(
        parsed_input.eps_scf,
        expected.eps_scf,
    ):
        raise ConvergenceEvidenceValidationError(
            "Production SCF tolerance differs from CP2K evidence."
        )

    for field in _SCF_COMMON_FIELDS:
        actual = getattr(parsed_input, field)
        proposed = getattr(expected, field)

        if actual is None or actual != proposed:
            raise ConvergenceEvidenceValidationError(
                f"Production SCF {field} differs from CP2K evidence."
            )

    if expected.solver == "OT":
        for field in _SCF_OT_FIELDS:
            actual = getattr(parsed_input, field)
            proposed = getattr(expected, field)

            if actual is None or actual != proposed:
                raise ConvergenceEvidenceValidationError(
                    f"Production SCF {field} differs from CP2K evidence."
                )

        if not _same_numeric(
            parsed_input.energy_gap,
            expected.energy_gap,
        ):
            raise ConvergenceEvidenceValidationError(
                "Production SCF energy_gap differs from CP2K evidence."
            )


def _scf_evidence_signature(parsed_input) -> tuple:
    """Represent all explicitly evidenced solver-relevant SCF settings."""
    common = (
        parsed_input.scf_solver,
        parsed_input.scf_guess,
        parsed_input.eps_scf,
        parsed_input.max_scf,
    )

    if parsed_input.scf_solver == "OT":
        return common + (
            parsed_input.outer_scf_max,
            parsed_input.ot_minimizer,
            parsed_input.ot_preconditioner,
            parsed_input.energy_gap,
        )

    return common


def _validate_methodology_input_identity(
    *,
    parsed_input,
    methodology: CP2KProductionMethodology,
) -> None:
    """Match production settings against parsed CP2K input evidence."""

    functional = methodology.functional
    expected_functional = getattr(
        functional, "value", functional
    )

    if (
        parsed_input.xc_functional is None
        or parsed_input.xc_functional != expected_functional
    ):
        raise ConvergenceEvidenceValidationError(
            "Production XC functional differs from CP2K evidence."
        )

    basis = methodology.basis_potential

    if basis is None:
        raise ConvergenceEvidenceValidationError(
            "Production methodology has no basis/potential configuration."
        )

    if (
        parsed_input.basis_set_file != basis.basis_set_file
        or parsed_input.potential_file != basis.potential_file
    ):
        raise ConvergenceEvidenceValidationError(
            "Production basis/potential files differ from CP2K evidence."
        )

    expected_kinds = tuple(
        sorted(
            (
                kind.element,
                kind.basis_set,
                kind.potential,
            )
            for kind in basis.kinds
        )
    )

    actual_kinds = tuple(
        sorted(
            (
                kind.element,
                kind.basis_set,
                kind.potential,
            )
            for kind in (parsed_input.kinds or ())
        )
    )

    if actual_kinds != expected_kinds:
        raise ConvergenceEvidenceValidationError(
            "Production atomic basis/potential assignments "
            "differ from CP2K evidence."
        )

def _validate_cross_campaign_settings(
    *,
    paths: dict[str, Path],
    methodology: CP2KProductionMethodology,
) -> None:
    """Verify fixed settings across the chained CP2K campaigns."""
    inputs = {
        field: _campaign_inputs(path)
        for field, path in paths.items()
    }

    reference = inputs["cutoff_report"][0]

    _validate_methodology_input_identity(
        parsed_input=reference,
        methodology=methodology,
    )

    # Initial cutoff and relative-cutoff studies must share
    # the same SCF configuration.
    initial_signature = _scf_evidence_signature(reference)

    for field in ("cutoff_report", "relative_cutoff_report"):
        for candidate in inputs[field]:
            if _scf_evidence_signature(candidate) != initial_signature:
                raise ConvergenceEvidenceValidationError(
                    f"Initial SCF configuration mismatch: {field}."
                )

    if methodology.k_points is None:
        production_fields = ("relative_cutoff_report",)
    else:
        production_fields = (
            "kpoint_report",
            "final_cutoff_verification_report",
            "final_relative_cutoff_verification_report",
        )

    for field in production_fields:
        for candidate in inputs[field]:
            _validate_scf_identity(
                parsed_input=candidate,
                methodology=methodology,
            )

    for field, candidates in inputs.items():
        for candidate in candidates:
            for setting in _COMMON_FIELDS:
                if getattr(candidate, setting) != getattr(
                    reference, setting
                ):
                    raise ConvergenceEvidenceValidationError(
                        f"Cross-campaign {setting} mismatch: {field}."
                    )

    # The initial two campaigns must use the same sampling and
    # SCF solver. A solver change is allowed only at the k-point stage.
    initial_mesh = reference.k_points
    initial_solver = reference.scf_solver

    for field in ("cutoff_report", "relative_cutoff_report"):
        for candidate in inputs[field]:
            if (
                candidate.k_points != initial_mesh
                or candidate.scf_solver != initial_solver
            ):
                raise ConvergenceEvidenceValidationError(
                    f"Initial sampling/solver mismatch: {field}."
                )

    for candidate in inputs["relative_cutoff_report"]:
        if not _same_numeric(
            candidate.cutoff_ry,
            methodology.cutoff_ry,
        ):
            raise ConvergenceEvidenceValidationError(
                "Relative-cutoff campaign uses wrong fixed cutoff."
            )

    if methodology.k_points is None:
        return

    for candidate in inputs["kpoint_report"]:
        if (
            not _same_numeric(
                candidate.cutoff_ry,
                methodology.cutoff_ry,
            )
            or not _same_numeric(
                candidate.relative_cutoff_ry,
                methodology.relative_cutoff_ry,
            )
            or candidate.scf_solver != "DIAGONALIZATION"
        ):
            raise ConvergenceEvidenceValidationError(
                "K-point campaign uses incorrect fixed settings."
            )

    final_stages = (
        "final_cutoff_verification_report",
        "final_relative_cutoff_verification_report",
    )

    for field in final_stages:
        for candidate in inputs[field]:
            if (
                candidate.k_points != methodology.k_points
                or candidate.scf_solver != "DIAGONALIZATION"
            ):
                raise ConvergenceEvidenceValidationError(
                    f"Incorrect final k-point mesh or solver: {field}."
                )

    for candidate in inputs["final_cutoff_verification_report"]:
        if not _same_numeric(
            candidate.relative_cutoff_ry,
            methodology.relative_cutoff_ry,
        ):
            raise ConvergenceEvidenceValidationError(
                "Final cutoff verification uses wrong relative cutoff."
            )

    for candidate in inputs[
        "final_relative_cutoff_verification_report"
    ]:
        if not _same_numeric(
            candidate.cutoff_ry,
            methodology.cutoff_ry,
        ):
            raise ConvergenceEvidenceValidationError(
                "Final relative-cutoff verification uses wrong cutoff."
            )
