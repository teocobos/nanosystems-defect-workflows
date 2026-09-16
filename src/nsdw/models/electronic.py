"""Electronic-structure results for NSDW."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from nsdw.models.quantity import Quantity


class EnergyReference(StrEnum):
    """Reference used for an electronic energy."""

    ABSOLUTE = "absolute"
    INTERNAL = "internal"
    VBM = "vbm"
    CBM = "cbm"
    FERMI_LEVEL = "fermi_level"
    VACUUM = "vacuum"


class ReferencedEnergy(BaseModel):
    """Energy whose physical reference is explicitly recorded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Quantity
    reference: EnergyReference


class BandGapResult(BaseModel):
    """Electronic band-gap result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Quantity
    direct: bool | None = None
    method: str | None = None


class LocalisationResult(BaseModel):
    """Description of a localised electronic carrier or state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    carrier: str
    localised: bool
    site_index: int | None = Field(default=None, ge=0)
    species: str | None = None
    ipr: float | None = Field(default=None, ge=0)


class ElectronicResult(BaseModel):
    """Calculator-independent electronic-structure information."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band_gap: BandGapResult | None = None

    vbm: ReferencedEnergy | None = None
    cbm: ReferencedEnergy | None = None
    fermi_level: ReferencedEnergy | None = None

    localisation: LocalisationResult | None = None
