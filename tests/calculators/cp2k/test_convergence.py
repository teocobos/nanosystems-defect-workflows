"""Tests for CP2K-specific convergence diagnostics."""

import pytest

from nsdw.calculators.cp2k.convergence import (
    CP2KConvergenceMode,
    assess_cp2k_multigrid,
)
from nsdw.calculators.cp2k.models import (
    ParsedCP2KMultigrid,
    ParsedCP2KMultigridLevel,
)
import math

from nsdw.calculators.cp2k.convergence import (
    assess_cp2k_calculation_validity,
)

from nsdw.calculators.cp2k.models import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KEnergy,
    ParsedCP2KResult,
    ParsedCP2KSCF,
)

def make_multigrid(
    counts=(65027435, 30531810, 17861146, 8080035),
    total=121500426,
):
    """Build multigrid data based on the real IGZO CP2K output."""

    return ParsedCP2KMultigrid(
        levels=tuple(
            ParsedCP2KMultigridLevel(
                grid_number=index,
                count=count,
                cutoff_au=cutoff,
            )
            for index, (count, cutoff) in enumerate(
                zip(
                    counts,
                    (200.0, 66.67, 22.22, 7.41),
                ),
                start=1,
            )
        ),
        total_gridlevel_count=total,
    )


def test_valid_igzo_multigrid():
    assessment = assess_cp2k_multigrid(make_multigrid())

    assert assessment.available
    assert assessment.internally_consistent
    assert assessment.grid_counts == (
        65027435,
        30531810,
        17861146,
        8080035,
    )
    assert sum(assessment.grid_fractions) == pytest.approx(1.0)
    assert assessment.issues == ()


def test_missing_multigrid():
    assessment = assess_cp2k_multigrid(None)

    assert not assessment.available
    assert not assessment.internally_consistent
    assert assessment.grid_counts == ()
    assert assessment.grid_fractions == ()
    assert assessment.issues


def test_inconsistent_total():
    assessment = assess_cp2k_multigrid(
        make_multigrid(total=100)
    )

    assert assessment.available
    assert not assessment.internally_consistent
    assert any(
        "does not match" in issue
        for issue in assessment.issues
    )


def test_zero_total():
    assessment = assess_cp2k_multigrid(
        make_multigrid(total=0)
    )

    assert not assessment.internally_consistent
    assert assessment.grid_fractions == ()


def test_missing_total():
    assessment = assess_cp2k_multigrid(
        make_multigrid(total=None)
    )

    assert not assessment.internally_consistent
    assert assessment.grid_fractions == ()


def test_empty_levels():
    assessment = assess_cp2k_multigrid(
        ParsedCP2KMultigrid(
            levels=(),
            total_gridlevel_count=0,
        )
    )

    assert not assessment.internally_consistent
    assert assessment.grid_counts == ()


def test_nonconsecutive_grid_numbers():
    multigrid = ParsedCP2KMultigrid(
        levels=(
            ParsedCP2KMultigridLevel(
                grid_number=1,
                count=50,
                cutoff_au=200.0,
            ),
            ParsedCP2KMultigridLevel(
                grid_number=3,
                count=50,
                cutoff_au=66.67,
            ),
        ),
        total_gridlevel_count=100,
    )

    assessment = assess_cp2k_multigrid(multigrid)

    assert not assessment.internally_consistent
    assert any(
        "consecutively numbered" in issue
        for issue in assessment.issues
    )


def test_convergence_modes():
    assert CP2KConvergenceMode.FULL_SCF == "full_scf"
    assert CP2KConvergenceMode.GRID_DIAGNOSTIC == "grid_diagnostic"

def make_cp2k_result(
    *,
    run_type=CP2KRunType.ENERGY,
    scf_status=CP2KSCFStatus.CONVERGED,
    normal_termination=True,
    energy=-100.0,
    multigrid=None,
):
    """Build a CP2K result for convergence-validity testing."""

    return ParsedCP2KResult(
        run_type=run_type,
        scf=ParsedCP2KSCF(
            status=scf_status,
        ),
        normal_termination=normal_termination,
        energy=ParsedCP2KEnergy(
            total_energy_hartree=energy,
        ),
        multigrid=multigrid,
    )


def test_full_scf_valid_calculation():
    parsed = make_cp2k_result()

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert assessment.valid
    assert assessment.issues == ()


def test_full_scf_rejects_unconverged_scf():
    parsed = make_cp2k_result(
        scf_status=CP2KSCFStatus.NOT_CONVERGED,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert not assessment.valid
    assert any("converged SCF" in issue for issue in assessment.issues)


def test_full_scf_rejects_unknown_scf():
    parsed = make_cp2k_result(
        scf_status=CP2KSCFStatus.UNKNOWN,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert not assessment.valid


def test_grid_diagnostic_accepts_unconverged_scf():
    parsed = make_cp2k_result(
        scf_status=CP2KSCFStatus.NOT_CONVERGED,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    )

    assert assessment.valid
    assert assessment.issues == ()


def test_grid_diagnostic_rejects_missing_multigrid():
    parsed = make_cp2k_result(
        multigrid=None,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    )

    assert not assessment.valid
    assert any("unavailable" in issue for issue in assessment.issues)


def test_grid_diagnostic_rejects_inconsistent_multigrid():
    parsed = make_cp2k_result(
        multigrid=make_multigrid(total=100),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    )

    assert not assessment.valid


@pytest.mark.parametrize(
    "mode",
    [
        CP2KConvergenceMode.FULL_SCF,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    ],
)
def test_both_modes_reject_abnormal_termination(mode):
    parsed = make_cp2k_result(
        normal_termination=False,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        mode,
    )

    assert not assessment.valid


@pytest.mark.parametrize(
    "mode",
    [
        CP2KConvergenceMode.FULL_SCF,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    ],
)
def test_both_modes_reject_missing_energy(mode):
    parsed = make_cp2k_result(
        energy=None,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        mode,
    )

    assert not assessment.valid


@pytest.mark.parametrize(
    "mode",
    [
        CP2KConvergenceMode.FULL_SCF,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    ],
)
def test_both_modes_reject_nonfinite_energy(mode):
    parsed = make_cp2k_result(
        energy=math.nan,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        mode,
    )

    assert not assessment.valid


def test_convergence_rejects_geometry_optimisation():
    parsed = make_cp2k_result(
        run_type=CP2KRunType.GEO_OPT,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert not assessment.valid
    assert any("ENERGY run" in issue for issue in assessment.issues)