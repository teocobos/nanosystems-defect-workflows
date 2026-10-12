"""Read-only inspection of generated CP2K convergence campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path

from nsdw.calculators.cp2k.input_parser import parse_cp2k_input
from nsdw.calculators.cp2k.parser import parse_cp2k_output
from nsdw.calculators.cp2k.models import CP2KSCFStatus
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)
from nsdw.workflows.convergence.cp2k_output_evidence import (
    validate_cp2k_single_point_output,
)


from nsdw.workflows.convergence.cp2k_structure_evidence import validate_cp2k_structure_evidence


class CP2KCandidateState(str, Enum):
    COMPLETED = "completed"
    PENDING = "pending"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass(frozen=True)
class CP2KCandidateStatus:
    label: str
    state: CP2KCandidateState
    reason: str
    input_path: Path
    output_path: Path | None


@dataclass(frozen=True)
class CP2KCampaignStatus:
    campaign_directory: Path
    candidates: tuple[CP2KCandidateStatus, ...]

    @property
    def all_completed(self) -> bool:
        return bool(self.candidates) and all(
            candidate.state == CP2KCandidateState.COMPLETED
            for candidate in self.candidates
        )


def _within(directory: Path, path: Path) -> Path:
    """Resolve a path and reject escapes from its expected directory."""
    resolved = path.resolve()

    try:
        resolved.relative_to(directory)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes expected directory: {path}"
        ) from exc

    return resolved



def _validate_candidate_input(candidate, parameter, input_path):
    """Validate candidate identity and its varied numerical parameter."""
    from nsdw.models.quantity import Quantity
    from nsdw.workflows.convergence.models import ConvergenceParameter

    parsed = parse_cp2k_input(input_path)

    if parsed.run_type is None or parsed.run_type.value != "ENERGY":
        raise ValueError("Candidate must use RUN_TYPE ENERGY.")

    if parsed.project_name is None:
        raise ValueError("Candidate CP2K project name is missing.")

    if not parsed.project_name.endswith(f"-{candidate.label}"):
        raise ValueError("Candidate project name does not match manifest label.")

    if parameter in (
        ConvergenceParameter.CUTOFF,
        ConvergenceParameter.RELATIVE_CUTOFF,
    ):
        if not isinstance(candidate.value, Quantity):
            raise ValueError("Candidate requires a numerical Quantity.")

        if candidate.value.unit.lower() != "ry":
            raise ValueError("Candidate convergence value must use Ry.")

        field = (
            "cutoff_ry"
            if parameter == ConvergenceParameter.CUTOFF
            else "relative_cutoff_ry"
        )

        actual = getattr(parsed, field)

        if actual is None or not math.isclose(
            actual,
            candidate.value.value,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ValueError(
                f"Candidate {field} differs from manifest value."
            )

    return parsed


def _validate_fixed_configuration(
    parsed,
    candidate,
    parameter,
    base_config,
) -> None:
    """Validate candidate settings against its reference configuration."""
    from nsdw.workflows.convergence.models import ConvergenceParameter

    def require_equal(field, actual, expected):
        if actual != expected:
            raise ValueError(
                f"{field} differs from base-config.json: "
                f"{actual!r} != {expected!r}"
            )

    def require_float(field, actual, expected):
        if actual is None or not math.isclose(
            actual,
            expected,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{field} differs from base-config.json: "
                f"{actual!r} != {expected!r}"
            )

    require_equal(
        "project_name",
        parsed.project_name,
        f"{base_config.project_name}-{candidate.label}",
    )

    require_equal(
        "run_type",
        parsed.run_type.value if parsed.run_type else None,
        base_config.run_type,
    )

    require_equal("charge", parsed.charge, base_config.charge)
    require_equal(
        "multiplicity",
        parsed.multiplicity,
        base_config.multiplicity,
    )

    if parameter != ConvergenceParameter.CUTOFF:
        require_float(
            "cutoff_ry",
            parsed.cutoff_ry,
            base_config.cutoff_ry,
        )

    if parameter != ConvergenceParameter.RELATIVE_CUTOFF:
        require_float(
            "relative_cutoff_ry",
            parsed.relative_cutoff_ry,
            base_config.relative_cutoff_ry,
        )

    if parameter != ConvergenceParameter.KPOINTS:
        require_equal(
            "k_points",
            parsed.k_points,
            base_config.k_points,
        )

    require_equal(
        "functional",
        parsed.xc_functional.upper()
        if parsed.xc_functional is not None else None,
        base_config.functional.value.upper(),
    )

    scf = base_config.scf

    require_equal("scf_solver", parsed.scf_solver, scf.solver)
    require_equal("scf_guess", parsed.scf_guess, scf.scf_guess)
    require_float("eps_scf", parsed.eps_scf, scf.eps_scf)
    require_equal("max_scf", parsed.max_scf, scf.max_scf)

    if scf.solver == "OT":
        require_equal(
            "outer_scf_max",
            parsed.outer_scf_max,
            scf.outer_scf_max,
        )
        require_equal(
            "ot_minimizer",
            parsed.ot_minimizer,
            scf.ot_minimizer,
        )
        require_equal(
            "ot_preconditioner",
            parsed.ot_preconditioner,
            scf.ot_preconditioner,
        )
        require_float(
            "energy_gap",
            parsed.energy_gap,
            scf.energy_gap,
        )

    reference_basis = base_config.basis_potential

    if reference_basis is None:
        raise ValueError(
            "base-config.json has no basis/potential configuration."
        )

    if parameter == ConvergenceParameter.BASIS:
        if not isinstance(candidate.value, str):
            raise ValueError(
                "Basis convergence requires a string candidate value."
            )

        basis = candidate.basis_potential

        if basis is None:
            raise ValueError(
                "Basis candidate is missing manifest basis provenance."
            )

        from nsdw.workflows.convergence.cp2k_generation import (
            _validate_basis_candidate,
        )

        _validate_basis_candidate(
            base_config=base_config,
            candidate=basis,
        )
    else:
        if candidate.basis_potential is not None:
            raise ValueError(
                "Unexpected basis provenance in a non-basis campaign."
            )

        basis = reference_basis

    require_equal(
        "basis_set_file",
        parsed.basis_set_file,
        basis.basis_set_file,
    )
    require_equal(
        "potential_file",
        parsed.potential_file,
        basis.potential_file,
    )

    expected_kinds = {
        kind.element: (kind.basis_set, kind.potential)
        for kind in basis.kinds
    }

    actual_kinds = {
        kind.element: (kind.basis_set, kind.potential)
        for kind in parsed.kinds
    }

    require_equal("basis/potential kinds", actual_kinds, expected_kinds)


def inspect_cp2k_convergence_campaign(
    campaign_directory: str | Path,
) -> CP2KCampaignStatus:
    """Inspect candidates without executing CP2K or changing files."""
    campaign = Path(campaign_directory).expanduser().resolve()
    manifest = load_convergence_manifest(campaign / "manifest.json")

    if manifest.calculator != "cp2k":
        raise ValueError("Campaign manifest is not for CP2K.")

    from nsdw.calculators.cp2k.generation_models import CP2KInputConfig

    config_path = campaign / "base-config.json"

    base_config = None

    if config_path.is_file():
        base_config = CP2KInputConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )

    statuses = []

    for candidate in manifest.candidates:
        directory = campaign / candidate.directory
        input_path = directory / candidate.input_file

        try:
            directory = _within(campaign, directory)
            input_path = _within(directory, input_path)
        except ValueError as exc:
            statuses.append(
                CP2KCandidateStatus(
                    label=candidate.label,
                    state=CP2KCandidateState.INVALID,
                    reason=str(exc),
                    input_path=input_path,
                    output_path=None,
                )
            )
            continue

        if not input_path.is_file():
            statuses.append(
                CP2KCandidateStatus(
                    label=candidate.label,
                    state=CP2KCandidateState.INVALID,
                    reason="CP2K input file is missing.",
                    input_path=input_path,
                    output_path=None,
                )
            )
            continue

        derived = directory / input_path.with_suffix(".out").name
        legacy = directory / "calculation.out"

        # Validate output paths before checking their existence.
        # This also catches symlinks escaping the candidate directory.
        try:
            _within(directory, derived)
            _within(directory, legacy)
        except ValueError as exc:
            statuses.append(
                CP2KCandidateStatus(
                    label=candidate.label,
                    state=CP2KCandidateState.INVALID,
                    reason=str(exc),
                    input_path=input_path,
                    output_path=None,
                )
            )
            continue

        try:
            validate_cp2k_structure_evidence(
                project_root=campaign,
                campaign_directory=campaign,
                candidate_directory=directory,
                input_filename=candidate.input_file,
                coordinate_filename=candidate.coordinate_file,
            )

            parsed_input = _validate_candidate_input(
                candidate,
                manifest.parameter,
                input_path,
            )

            if base_config is None:
                raise ValueError(
                    "Cannot validate candidate without base-config.json."
                )

            _validate_fixed_configuration(
                parsed_input,
                candidate,
                manifest.parameter,
                base_config,
            )
        except Exception as exc:
            statuses.append(
                CP2KCandidateStatus(
                    label=candidate.label,
                    state=CP2KCandidateState.INVALID,
                    reason=f"Invalid CP2K candidate input: {exc}",
                    input_path=input_path,
                    output_path=None,
                )
            )
            continue

        existing = {
            path.resolve()
            for path in (derived, legacy)
            if path.is_file()
        }

        if len(existing) > 1:
            state = CP2KCandidateState.INVALID
            reason = "Ambiguous CP2K output files."
            output_path = None

        elif not existing:
            state = CP2KCandidateState.PENDING
            reason = "No CP2K output found; scheduler state unknown."
            output_path = None

        else:
            output_path = next(iter(existing))

            try:
                _within(directory, output_path)
                parsed_input = parse_cp2k_input(input_path)
                parsed_output = parse_cp2k_output(output_path)

                if (
                    parsed_output.normal_termination is not True
                    or parsed_output.scf.status
                    != CP2KSCFStatus.CONVERGED
                ):
                    state = CP2KCandidateState.INCOMPLETE
                    reason = "CP2K did not terminate successfully."

                else:
                    validate_cp2k_single_point_output(output_path)

                    for field in (
                        "project_name",
                        "run_type",
                        "charge",
                        "multiplicity",
                    ):
                        actual = getattr(parsed_input, field)
                        observed = getattr(parsed_output, field)

                        if (
                            actual is None
                            or observed is None
                            or actual != observed
                        ):
                            raise ValueError(
                                f"CP2K {field} identity mismatch."
                            )

                    state = CP2KCandidateState.COMPLETED
                    reason = "Valid completed CP2K single-point output."

            except Exception as exc:
                state = CP2KCandidateState.INVALID
                reason = str(exc)

        statuses.append(
            CP2KCandidateStatus(
                label=candidate.label,
                state=state,
                reason=reason,
                input_path=input_path,
                output_path=output_path,
            )
        )

    return CP2KCampaignStatus(
        campaign_directory=campaign,
        candidates=tuple(statuses),
    )
