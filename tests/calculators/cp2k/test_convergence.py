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
