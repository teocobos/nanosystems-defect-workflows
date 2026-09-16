"""Tests for NSDW defect-physics result models."""

import pytest
from pydantic import ValidationError

from nsdw.models import (
    ChargeTransitionLevel,
    ChemicalPotential,
    DefectCorrection,
    DefectIdentity,
    DefectResult,
    DefectType,
    EnergyReference,
    FormationEnergyResult,
    Quantity,
    ReferencedEnergy,
)


def test_defect_identity():
    defect = DefectIdentity(
        name="V_O",
        type=DefectType.VACANCY,
        species="O",
        site_index=12,
        charge=2,
    )

    assert defect.name == "V_O"
    assert defect.charge == 2
    assert defect.type == DefectType.VACANCY


def test_formation_energy():
    result = FormationEnergyResult(
        value=Quantity(value=2.41, unit="eV"),
        fermi_level=ReferencedEnergy(
            value=Quantity(value=0.0, unit="eV"),
            reference=EnergyReference.VBM,
        ),
        chemical_potentials=(
            ChemicalPotential(
                species="O",
                value=Quantity(value=-4.92, unit="eV"),
                reference="O2",
            ),
        ),
        corrections=(
            DefectCorrection(
                method="freysoldt",
                value=Quantity(value=0.17, unit="eV"),
            ),
        ),
        source_calculation_ids=(
            "igzo-bulk",
            "igzo-vo-o001-q+2",
        ),
    )

    assert result.value.value == 2.41
    assert len(result.corrections) == 1


def test_formation_energy_rejects_internal_fermi_reference():
    with pytest.raises(ValidationError):
        FormationEnergyResult(
            value=Quantity(value=2.41, unit="eV"),
            fermi_level=ReferencedEnergy(
                value=Quantity(value=0.0, unit="eV"),
                reference=EnergyReference.INTERNAL,
            ),
            source_calculation_ids=(
                "bulk",
                "defect",
            ),
        )


def test_charge_transition_level():
    ctl = ChargeTransitionLevel(
        initial_charge=0,
        final_charge=-1,
        energy=ReferencedEnergy(
            value=Quantity(value=1.35, unit="eV"),
            reference=EnergyReference.VBM,
        ),
        correction_method="freysoldt",
        source_calculation_ids=(
            "igzo-vo-o001-q0",
            "igzo-vo-o001-q-1",
        ),
    )

    assert ctl.initial_charge == 0
    assert ctl.final_charge == -1
    assert ctl.energy.value.value == 1.35


def test_ctl_rejects_identical_charge_states():
    with pytest.raises(ValidationError):
        ChargeTransitionLevel(
            initial_charge=0,
            final_charge=0,
            energy=ReferencedEnergy(
                value=Quantity(value=1.35, unit="eV"),
                reference=EnergyReference.VBM,
            ),
            source_calculation_ids=(
                "calc-a",
                "calc-b",
            ),
        )


def test_ctl_requires_two_source_calculations():
    with pytest.raises(ValidationError):
        ChargeTransitionLevel(
            initial_charge=0,
            final_charge=-1,
            energy=ReferencedEnergy(
                value=Quantity(value=1.35, unit="eV"),
                reference=EnergyReference.VBM,
            ),
            source_calculation_ids=("calc-a",),
        )


def test_defect_result_contains_ctl():
    defect = DefectResult(
        identity=DefectIdentity(
            name="V_O",
            type=DefectType.VACANCY,
            species="O",
            site_index=12,
            charge=0,
        ),
        transition_levels=(
            ChargeTransitionLevel(
                initial_charge=0,
                final_charge=-1,
                energy=ReferencedEnergy(
                    value=Quantity(value=1.35, unit="eV"),
                    reference=EnergyReference.VBM,
                ),
                source_calculation_ids=(
                    "q0",
                    "q-1",
                ),
            ),
        ),
    )

    assert len(defect.transition_levels) == 1

def test_formation_energy_rejects_cbm_reference():
    with pytest.raises(ValidationError):
        FormationEnergyResult(
            value=Quantity(value=2.41, unit="eV"),
            fermi_level=ReferencedEnergy(
                value=Quantity(value=0.0, unit="eV"),
                reference=EnergyReference.CBM,
            ),
            source_calculation_ids=(
                "bulk",
                "defect",
            ),
        )


def test_ctl_rejects_cbm_reference():
    with pytest.raises(ValidationError):
        ChargeTransitionLevel(
            initial_charge=0,
            final_charge=-1,
            energy=ReferencedEnergy(
                value=Quantity(value=1.35, unit="eV"),
                reference=EnergyReference.CBM,
            ),
            source_calculation_ids=(
                "q0",
                "q-1",
            ),
        )
