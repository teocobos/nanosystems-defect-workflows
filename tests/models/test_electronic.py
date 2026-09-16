"""Tests for NSDW electronic-structure result models."""

import pytest
from pydantic import ValidationError

from nsdw.models import (
    BandGapResult,
    ElectronicResult,
    EnergyReference,
    LocalisationResult,
    Quantity,
    ReferencedEnergy,
)


def test_electronic_result_creation():
    result = ElectronicResult(
        band_gap=BandGapResult(
            value=Quantity(value=3.21, unit="eV"),
            direct=False,
            method="PBE0-TC-LRC",
        ),
        vbm=ReferencedEnergy(
            value=Quantity(value=-5.42, unit="eV"),
            reference=EnergyReference.INTERNAL,
        ),
        cbm=ReferencedEnergy(
            value=Quantity(value=-2.21, unit="eV"),
            reference=EnergyReference.INTERNAL,
        ),
    )

    assert result.band_gap is not None
    assert result.band_gap.value.value == 3.21
    assert result.vbm is not None
    assert result.vbm.reference == EnergyReference.INTERNAL


def test_transition_energy_can_reference_vbm():
    energy = ReferencedEnergy(
        value=Quantity(value=1.35, unit="eV"),
        reference=EnergyReference.VBM,
    )

    assert energy.reference == EnergyReference.VBM


def test_localisation_result():
    localisation = LocalisationResult(
        carrier="electron",
        localised=True,
        site_index=17,
        species="In",
        ipr=0.42,
    )

    assert localisation.localised is True
    assert localisation.site_index == 17


def test_negative_site_index_is_rejected():
    with pytest.raises(ValidationError):
        LocalisationResult(
            carrier="electron",
            localised=True,
            site_index=-1,
        )