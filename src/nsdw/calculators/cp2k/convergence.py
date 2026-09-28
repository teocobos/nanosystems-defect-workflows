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