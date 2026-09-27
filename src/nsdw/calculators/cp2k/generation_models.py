"""Configuration models for generating CP2K inputs."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CP2KXCFunctional(StrEnum):
    """Exchange-correlation functionals supported by CP2K generation."""

    PBE = "PBE"
    PBESOL = "PBEsol"


class CP2KCoordinateMode(StrEnum):
    """How atomic coordinates are supplied to CP2K."""

    EXTERNAL_XYZ = "external_xyz"
    EMBEDDED = "embedded"


class CP2KKindConfig(BaseModel):
    """Basis-set and pseudopotential assignment for one atomic species."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    element: str = Field(
        min_length=1,
    )

    basis_set: str = Field(
        min_length=1,
    )

    potential: str = Field(
        min_length=1,
    )


class CP2KBasisPotentialConfig(BaseModel):
    """CP2K basis and pseudopotential configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    basis_set_file: str = Field(
        min_length=1,
    )

    potential_file: str = Field(
        min_length=1,
    )

    kinds: tuple[CP2KKindConfig, ...] = ()


class CP2KSCFConfig(BaseModel):
    """SCF settings used when generating a CP2K input."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    scf_guess: str = "ATOMIC"

    eps_scf: float = Field(
        default=1.0e-6,
        gt=0,
    )

    max_scf: int = Field(
        default=100,
        gt=0,
    )

    outer_scf_max: int = Field(
        default=10,
        gt=0,
    )

    ot_minimizer: str = "CG"

    ot_preconditioner: str = "FULL_SINGLE_INVERSE"

    energy_gap: float = Field(
        default=0.001,
        gt=0,
    )


class CP2KInputConfig(BaseModel):
    """Scientific configuration for a generated CP2K input."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    project_name: str = Field(
        min_length=1,
    )

    functional: CP2KXCFunctional = CP2KXCFunctional.PBE

    run_type: str = "ENERGY_FORCE"

    charge: int = 0

    multiplicity: int = Field(
        default=1,
        gt=0,
    )

    cutoff_ry: float = Field(
        default=600.0,
        gt=0,
    )

    relative_cutoff_ry: float = Field(
        default=60.0,
        gt=0,
    )

    coordinate_mode: CP2KCoordinateMode = (
        CP2KCoordinateMode.EXTERNAL_XYZ
    )

    coordinate_file: str | None = None

    coordinate_file_format: str = "XYZ"

    k_points: tuple[int, int, int] | None = None

    scf: CP2KSCFConfig = Field(
        default_factory=CP2KSCFConfig,
    )

    basis_potential: CP2KBasisPotentialConfig | None = None