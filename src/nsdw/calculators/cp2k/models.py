"""Typed intermediate models for parsed CP2K output."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CP2KRunType(StrEnum):
    """CP2K run types currently recognised by NSDW."""

    ENERGY = "ENERGY"
    ENERGY_FORCE = "ENERGY_FORCE"
    GEO_OPT = "GEO_OPT"
    CELL_OPT = "CELL_OPT"
    MD = "MD"
    BAND = "BAND"
    VIBRATIONAL_ANALYSIS = "VIBRATIONAL_ANALYSIS"


class CP2KSCFStatus(StrEnum):
    """Status of the final parsed SCF cycle."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    UNKNOWN = "unknown"


class ParsedCP2KEnergy(BaseModel):
    """Raw energy information extracted from CP2K."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    total_energy_hartree: float | None = None


class ParsedCP2KSCF(BaseModel):
    """Raw SCF information extracted from CP2K."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: CP2KSCFStatus = CP2KSCFStatus.UNKNOWN
    iterations: int | None = Field(default=None, ge=0)


class ParsedCP2KResult(BaseModel):
    """
    Calculator-specific representation of a parsed CP2K output.

    Values remain in the units and terminology reported by CP2K.
    Conversion into calculator-independent NSDW scientific models
    is the responsibility of the CP2K adapter.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    cp2k_version: str | None = None
    run_type: CP2KRunType | None = None

    project_name: str | None = None

    charge: int | None = None
    multiplicity: int | None = Field(
        default=None,
        gt=0,
    )

    energy: ParsedCP2KEnergy = ParsedCP2KEnergy()
    scf: ParsedCP2KSCF = ParsedCP2KSCF()

    normal_termination: bool = False

    warnings: tuple[str, ...] = ()
