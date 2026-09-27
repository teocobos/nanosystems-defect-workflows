"""Tests for CP2K parser intermediate models."""

import pytest
from pydantic import ValidationError

from nsdw.calculators.cp2k import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KEnergy,
    ParsedCP2KMultigrid,
    ParsedCP2KMultigridLevel,
    ParsedCP2KResult,
    ParsedCP2KSCF,
)


def test_minimal_cp2k_result():
    result = ParsedCP2KResult()

    assert result.cp2k_version is None
    assert result.run_type is None
    assert result.normal_termination is False
    assert result.scf.status == CP2KSCFStatus.UNKNOWN


def test_completed_energy_calculation():
    result = ParsedCP2KResult(
        cp2k_version="2026.2",
        run_type=CP2KRunType.ENERGY,
        project_name="sio2-bulk",
        charge=0,
        multiplicity=1,
        energy=ParsedCP2KEnergy(
            total_energy_hartree=-123.456789,
        ),
        scf=ParsedCP2KSCF(
            status=CP2KSCFStatus.CONVERGED,
            iterations=12,
        ),
        normal_termination=True,
    )

    assert result.cp2k_version == "2026.2"
    assert result.run_type == CP2KRunType.ENERGY
    assert result.energy.total_energy_hartree == -123.456789
    assert result.scf.iterations == 12
    assert result.normal_termination is True


def test_cp2k_result_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ParsedCP2KResult(
            cp2k_version="2026.2",
            unsupported_field="value",
        )


def test_multiplicity_must_be_positive():
    with pytest.raises(ValidationError):
        ParsedCP2KResult(
            multiplicity=0,
        )


def test_scf_iterations_cannot_be_negative():
    with pytest.raises(ValidationError):
        ParsedCP2KSCF(
            status=CP2KSCFStatus.CONVERGED,
            iterations=-1,
        )


def test_cp2k_result_is_immutable():
    result = ParsedCP2KResult(
        cp2k_version="2026.2",
    )

    with pytest.raises(ValidationError):
        result.cp2k_version = "2027.1"

def test_cp2k_multigrid_represents_reported_grid_levels():
    multigrid = ParsedCP2KMultigrid(
        levels=(
            ParsedCP2KMultigridLevel(
                grid_number=1,
                count=65027435,
                cutoff_au=200.00,
            ),
            ParsedCP2KMultigridLevel(
                grid_number=2,
                count=30531810,
                cutoff_au=66.67,
            ),
            ParsedCP2KMultigridLevel(
                grid_number=3,
                count=17861146,
                cutoff_au=22.22,
            ),
            ParsedCP2KMultigridLevel(
                grid_number=4,
                count=8080035,
                cutoff_au=7.41,
            ),
        ),
        total_gridlevel_count=121500426,
    )

    assert len(multigrid.levels) == 4
    assert multigrid.levels[0].grid_number == 1
    assert multigrid.levels[0].count == 65027435
    assert multigrid.levels[0].cutoff_au == pytest.approx(
        200.00
    )
    assert multigrid.levels[-1].grid_number == 4
    assert multigrid.total_gridlevel_count == 121500426


def test_cp2k_result_accepts_multigrid_information():
    multigrid = ParsedCP2KMultigrid(
        levels=(
            ParsedCP2KMultigridLevel(
                grid_number=1,
                count=65027435,
                cutoff_au=200.00,
            ),
        ),
        total_gridlevel_count=65027435,
    )

    result = ParsedCP2KResult(
        multigrid=multigrid,
    )

    assert result.multigrid == multigrid


def test_multigrid_level_requires_positive_grid_number():
    with pytest.raises(ValidationError):
        ParsedCP2KMultigridLevel(
            grid_number=0,
            count=100,
            cutoff_au=200.0,
        )


def test_multigrid_level_rejects_negative_count():
    with pytest.raises(ValidationError):
        ParsedCP2KMultigridLevel(
            grid_number=1,
            count=-1,
            cutoff_au=200.0,
        )


def test_multigrid_level_requires_positive_cutoff():
    with pytest.raises(ValidationError):
        ParsedCP2KMultigridLevel(
            grid_number=1,
            count=100,
            cutoff_au=0.0,
        )


def test_multigrid_rejects_negative_total_count():
    with pytest.raises(ValidationError):
        ParsedCP2KMultigrid(
            total_gridlevel_count=-1,
        )