"""Tests for CP2K parser intermediate models."""

import pytest
from pydantic import ValidationError

from nsdw.calculators.cp2k import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KEnergy,
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
