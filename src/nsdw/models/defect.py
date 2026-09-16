"""Defect-physics result models for NSDW."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nsdw.models.electronic import EnergyReference, ReferencedEnergy
from nsdw.models.quantity import Quantity


class DefectType(StrEnum):
    """Supported point-defect classes."""

    VACANCY = "vacancy"
    INTERSTITIAL = "interstitial"
    SUBSTITUTION = "substitution"
    ANTISITE = "antisite"
    COMPLEX = "complex"


class DefectIdentity(BaseModel):
    """Calculator-independent identity of a defect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type: DefectType
    species: str | None = None
    site_index: int | None = Field(default=None, ge=0)
    charge: int
    multiplicity: int = Field(default=1, gt=0)


class ChemicalPotential(BaseModel):
    """Chemical potential used in defect thermodynamics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    species: str
    value: Quantity
    reference: str


class DefectCorrection(BaseModel):
    """Finite-size or electrostatic correction applied to a defect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    value: Quantity


class FormationEnergyResult(BaseModel):
    """Derived defect formation energy and its thermodynamic assumptions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Quantity
    fermi_level: ReferencedEnergy
    chemical_potentials: tuple[ChemicalPotential, ...] = ()
    corrections: tuple[DefectCorrection, ...] = ()
    source_calculation_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_fermi_reference(self) -> FormationEnergyResult:
        """Formation-energy Fermi levels must use the VBM reference."""
        if self.fermi_level.reference != EnergyReference.VBM:
            raise ValueError(
                "formation-energy Fermi level must be referenced to the VBM"
            )

        return self


class ChargeTransitionLevel(BaseModel):
    """Thermodynamic charge-transition level epsilon(q/q')."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_charge: int
    final_charge: int
    energy: ReferencedEnergy
    correction_method: str | None = None
    source_calculation_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_transition(self) -> ChargeTransitionLevel:
        """Validate the physical definition of a transition level."""
        if self.initial_charge == self.final_charge:
            raise ValueError(
                "charge-transition level requires different charge states"
            )

        if self.energy.reference != EnergyReference.VBM:
            raise ValueError(
                "charge-transition level must be referenced to the VBM"
            )

        if len(self.source_calculation_ids) < 2:
            raise ValueError(
                "charge-transition level requires at least two source calculations"
            )

        return self


class DefectResult(BaseModel):
    """Defect information associated with an NSDW scientific result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: DefectIdentity
    formation_energy: FormationEnergyResult | None = None
    transition_levels: tuple[ChargeTransitionLevel, ...] = ()
