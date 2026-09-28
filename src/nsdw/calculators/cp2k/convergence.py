"""CP2K-specific convergence diagnostics and assessment models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from nsdw.calculators.cp2k.models import (
    ParsedCP2KMultigrid,
)


class CP2KConvergenceMode(StrEnum):
    """Calculation mode used to gather convergence evidence."""

    FULL_SCF = "full_scf"
    GRID_DIAGNOSTIC = "grid_diagnostic"


@dataclass(frozen=True)
class CP2KMultigridAssessment:
    """
    Assessment of one CP2K MULTIGRID INFO block.

    Internal consistency does not imply numerical convergence.
    """

    available: bool
    internally_consistent: bool
    total_gridlevel_count: int | None
    grid_counts: tuple[int, ...]
    grid_fractions: tuple[float, ...]
    issues: tuple[str, ...]


def assess_cp2k_multigrid(
    multigrid: ParsedCP2KMultigrid | None,
) -> CP2KMultigridAssessment:
    """Validate multigrid count consistency and report distributions."""

    if multigrid is None:
        return CP2KMultigridAssessment(
            available=False,
            internally_consistent=False,
            total_gridlevel_count=None,
            grid_counts=(),
            grid_fractions=(),
            issues=("CP2K MULTIGRID INFO is unavailable.",),
        )

    issues: list[str] = []

    levels = multigrid.levels
    counts = tuple(level.count for level in levels)
    numbers = tuple(level.grid_number for level in levels)
    total = multigrid.total_gridlevel_count

    if not levels:
        issues.append("No multigrid levels were parsed.")

    if numbers != tuple(range(1, len(levels) + 1)):
        issues.append(
            "Multigrid levels are not consecutively numbered from 1."
        )

    if total is None:
        issues.append("Total gridlevel count is missing.")
    elif total == 0:
        issues.append("Total gridlevel count is zero.")
    elif sum(counts) != total:
        issues.append(
            "Sum of grid-level counts does not match reported total."
        )

    fractions = (
        tuple(count / total for count in counts)
        if total is not None and total > 0
        else ()
    )

    return CP2KMultigridAssessment(
        available=True,
        internally_consistent=not issues,
        total_gridlevel_count=total,
        grid_counts=counts,
        grid_fractions=fractions,
        issues=tuple(issues),
    )

@dataclass(frozen=True)
class CP2KCalculationValidity:
    """
    Validity of one CP2K calculation for convergence analysis.

    A valid GRID_DIAGNOSTIC calculation is not equivalent to
    a fully SCF-converged production calculation.
    """

    mode: CP2KConvergenceMode
    valid: bool
    issues: tuple[str, ...]


def assess_cp2k_calculation_validity(
    parsed: "ParsedCP2KResult",
    mode: CP2KConvergenceMode,
) -> CP2KCalculationValidity:
    """
    Check whether a CP2K calculation is usable in the given mode.

    FULL_SCF requires SCF convergence.

    GRID_DIAGNOSTIC permits unconverged SCF, provided the
    calculation terminated normally and contains usable
    energy and multigrid diagnostics.
    """

    from nsdw.calculators.cp2k.models import (
        CP2KRunType,
        CP2KSCFStatus,
    )

    issues: list[str] = []

    if parsed.run_type != CP2KRunType.ENERGY:
        issues.append(
            "Convergence analysis requires a CP2K ENERGY run."
        )

    if not parsed.normal_termination:
        issues.append(
            "CP2K did not terminate normally."
        )

    energy = parsed.energy.total_energy_hartree

    if energy is None:
        issues.append(
            "CP2K total energy is unavailable."
        )
    elif not math.isfinite(energy):
        issues.append(
            "CP2K total energy must be finite."
        )

    if mode == CP2KConvergenceMode.FULL_SCF:
        if parsed.scf.status != CP2KSCFStatus.CONVERGED:
            issues.append(
                "FULL_SCF mode requires converged SCF."
            )

    elif mode == CP2KConvergenceMode.GRID_DIAGNOSTIC:
        multigrid_assessment = assess_cp2k_multigrid(
            parsed.multigrid
        )

        if not multigrid_assessment.internally_consistent:
            issues.extend(
                multigrid_assessment.issues
            )

    else:
        raise ValueError(
            f"Unsupported CP2K convergence mode: {mode!r}"
        )

    return CP2KCalculationValidity(
        mode=mode,
        valid=not issues,
        issues=tuple(issues),
    )

@dataclass(frozen=True)
class CP2KInputConsistencyAssessment:
    """
    Consistency of parsed CP2K inputs in a convergence study.

    Consistency of parsed fields does not establish equivalence
    of all CP2K settings.
    """

    consistent: bool
    varied_parameter: str
    checked_fields: tuple[str, ...]
    issues: tuple[str, ...]


def assess_cp2k_input_consistency(
    inputs: list["ParsedCP2KInput"],
    varied_parameter: str,
) -> CP2KInputConsistencyAssessment:
    """
    Check consistency of parsed CP2K convergence-study inputs.

    Only the selected convergence parameter may vary.

    During basis-set convergence, individual basis assignments may
    change, but KIND identities, elements and pseudopotentials
    must remain fixed.

    Project names may differ between candidates.

    This assessment covers only settings exposed by the current
    CP2K input parser. It does not guarantee that all CP2K
    calculation settings are equivalent.
    """

    from nsdw.calculators.cp2k.models import (
        CP2KRunType,
    )

    from nsdw.workflows.convergence.models import (
        ConvergenceParameter,
    )

    parameter_fields = {
        ConvergenceParameter.BASIS: "kinds",
        ConvergenceParameter.CUTOFF: "cutoff_ry",
        ConvergenceParameter.RELATIVE_CUTOFF: (
            "relative_cutoff_ry"
        ),
        ConvergenceParameter.KPOINTS: "k_points",
    }

    try:
        parameter = ConvergenceParameter(
            varied_parameter
        )
    except ValueError as exc:
        raise ValueError(
            f"Unsupported convergence parameter: "
            f"{varied_parameter!r}"
        ) from exc

    varied_field = parameter_fields[parameter]

    checked_fields = (
        "run_type",
        "charge",
        "multiplicity",
        "xc_functional",
        "cutoff_ry",
        "relative_cutoff_ry",
        "eps_scf",
        "k_points",
        "kinds",
        "basis_set_file",
        "potential_file",
        "admm",
    )

    issues: list[str] = []

    if len(inputs) < 2:
        issues.append(
            "At least two CP2K inputs are required."
        )

    if not inputs:
        return CP2KInputConsistencyAssessment(
            consistent=False,
            varied_parameter=parameter.value,
            checked_fields=checked_fields,
            issues=tuple(issues),
        )

    reference = inputs[0]

    for index, parsed in enumerate(inputs):
        candidate_number = index + 1

        if parsed.run_type != CP2KRunType.ENERGY:
            issues.append(
                f"Candidate {candidate_number}: "
                "CP2K RUN_TYPE must be ENERGY."
            )

        for field_name in checked_fields:
            value = getattr(parsed, field_name)

            # Missing parsed settings cannot establish
            # scientific comparability.
            if value is None:
                issues.append(
                    f"Candidate {candidate_number}: "
                    f"{field_name} is missing."
                )
                continue

            # Basis convergence requires special handling
            # because KIND includes both basis and potential.
            if (
                field_name == "kinds"
                and parameter == ConvergenceParameter.BASIS
            ):
                reference_kinds = {
                    kind.kind: kind
                    for kind in reference.kinds
                }

                candidate_kinds = {
                    kind.kind: kind
                    for kind in parsed.kinds
                }

                if (
                    len(reference_kinds) != len(reference.kinds)
                    or len(candidate_kinds) != len(parsed.kinds)
                ):
                    issues.append(
                        f"Candidate {candidate_number}: "
                        "duplicate KIND names detected."
                    )

                if (
                    reference_kinds.keys()
                    != candidate_kinds.keys()
                ):
                    issues.append(
                        f"Candidate {candidate_number}: "
                        "KIND identities differ from "
                        "the reference input."
                    )

                for kind_name in sorted(
                    reference_kinds.keys()
                    & candidate_kinds.keys()
                ):
                    reference_kind = reference_kinds[
                        kind_name
                    ]

                    candidate_kind = candidate_kinds[
                        kind_name
                    ]

                    if (
                        reference_kind.element is None
                        or candidate_kind.element is None
                    ):
                        issues.append(
                            f"Candidate {candidate_number}: "
                            f"KIND {kind_name} element "
                            "is missing."
                        )

                    elif (
                        reference_kind.element
                        != candidate_kind.element
                    ):
                        issues.append(
                            f"Candidate {candidate_number}: "
                            f"KIND {kind_name} element differs."
                        )

                    if (
                        reference_kind.potential is None
                        or candidate_kind.potential is None
                    ):
                        issues.append(
                            f"Candidate {candidate_number}: "
                            f"KIND {kind_name} potential "
                            "is missing."
                        )

                    elif (
                        reference_kind.potential
                        != candidate_kind.potential
                    ):
                        issues.append(
                            f"Candidate {candidate_number}: "
                            f"KIND {kind_name} potential differs."
                        )

                    if (
                        reference_kind.basis_set is None
                        or candidate_kind.basis_set is None
                    ):
                        issues.append(
                            f"Candidate {candidate_number}: "
                            f"KIND {kind_name} basis_set "
                            "is missing."
                        )

                continue

            # The selected parameter is allowed to vary.
            if field_name == varied_field:
                continue

            reference_value = getattr(
                reference,
                field_name,
            )

            if value != reference_value:
                issues.append(
                    f"Candidate {candidate_number}: "
                    f"{field_name} differs from "
                    "the reference input."
                )

    # Confirm that the intended parameter actually varies.
    if parameter == ConvergenceParameter.BASIS:
        basis_assignments = []

        for parsed in inputs:
            assignments = tuple(
                sorted(
                    (
                        kind.kind,
                        kind.basis_set,
                    )
                    for kind in parsed.kinds
                )
            )

            basis_assignments.append(assignments)

        if len(set(basis_assignments)) < 2:
            issues.append(
                "The convergence parameter "
                "basis_set does not vary."
            )

    else:
        varied_values = [
            getattr(parsed, varied_field)
            for parsed in inputs
        ]

        if all(
            value is not None
            for value in varied_values
        ):
            if len(set(varied_values)) < 2:
                issues.append(
                    f"The convergence parameter "
                    f"{varied_field} does not vary."
                )

    return CP2KInputConsistencyAssessment(
        consistent=not issues,
        varied_parameter=parameter.value,
        checked_fields=checked_fields,
        issues=tuple(issues),
    )