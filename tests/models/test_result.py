"""Tests for the top-level NSDW scientific result."""

import json

import pytest
from pydantic import ValidationError

from nsdw.models import (
    Backend,
    CalculationMetadata,
    CalculationStatus,
    CalculationType,
    EnergyResult,
    NSDWResult,
    Quantity,
)


def make_calculation() -> CalculationMetadata:
    return CalculationMetadata(
        id="igzo-vo-o001-q+2",
        type=CalculationType.DEFECT_RELAXATION,
        status=CalculationStatus.COMPLETED,
        backend=Backend.CP2K,
    )


def test_result_creation():
    result = NSDWResult(
        calculation=make_calculation(),
        energy=EnergyResult(
            total=Quantity(
                value=-1234.567,
                unit="eV",
            )
        ),
    )

    assert result.schema_version == "1.0.0"
    assert result.calculation.backend == Backend.CP2K
    assert result.energy is not None
    assert result.energy.total is not None
    assert result.energy.total.value == -1234.567


def test_result_serialises_to_json():
    result = NSDWResult(
        calculation=make_calculation(),
        energy=EnergyResult(
            total=Quantity(
                value=-1234.567,
                unit="eV",
            )
        ),
    )

    payload = json.loads(result.model_dump_json())

    assert payload["schema_version"] == "1.0.0"
    assert payload["calculation"]["backend"] == "cp2k"
    assert payload["energy"]["total"]["unit"] == "eV"


def test_result_round_trip():
    original = NSDWResult(
        calculation=make_calculation(),
        energy=EnergyResult(
            total=Quantity(
                value=-1234.567,
                unit="eV",
            )
        ),
    )

    payload = original.model_dump_json()
    restored = NSDWResult.model_validate_json(payload)

    assert restored == original


def test_result_rejects_unknown_schema_version():
    with pytest.raises(ValidationError):
        NSDWResult(
            schema_version="2.0.0",
            calculation=make_calculation(),
        )


def test_calculation_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        CalculationMetadata(
            id="test",
            type=CalculationType.SINGLE_POINT,
            status=CalculationStatus.COMPLETED,
            backend="quantum_espresso",
        )
