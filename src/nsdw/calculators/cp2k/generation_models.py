"""Configuration models for generating CP2K inputs."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    solver: str = Field(
        default="OT",
        pattern=r"^(OT|DIAGONALIZATION)$",
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



class CP2KKPointScheme(StrEnum):
    """Supported CP2K k-point sampling schemes."""

    NONE = "NONE"
    GAMMA = "GAMMA"
    MONKHORST_PACK = "MONKHORST-PACK"
    MACDONALD = "MACDONALD"
    GENERAL = "GENERAL"


class CP2KWeightedKPoint(BaseModel):
    """One explicitly specified weighted k-point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coordinates: tuple[float, float, float]
    weight: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_finite_values(self):
        from math import isfinite

        if not all(isfinite(value) for value in self.coordinates):
            raise ValueError("K-point coordinates must be finite.")

        if not isfinite(self.weight):
            raise ValueError("K-point weight must be finite.")

        return self


class CP2KKPointConfig(BaseModel):
    """Structured k-point sampling configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scheme: CP2KKPointScheme = CP2KKPointScheme.NONE

    mesh: tuple[int, int, int] | None = None

    shift: tuple[float, float, float] | None = None

    points: tuple[CP2KWeightedKPoint, ...] = ()

    @model_validator(mode="after")
    def validate_scheme(self):
        from math import isfinite

        if self.mesh is not None:
            if any(type(value) is not int or value <= 0 for value in self.mesh):
                raise ValueError("K-point mesh dimensions must be positive integers.")

        if self.shift is not None:
            if not all(isfinite(value) for value in self.shift):
                raise ValueError("K-point shifts must be finite.")

        for point in self.points:
            if not all(isfinite(value) for value in point.coordinates):
                raise ValueError("K-point coordinates must be finite.")
            if not isfinite(point.weight):
                raise ValueError("K-point weights must be finite.")

        if self.scheme in {
            CP2KKPointScheme.NONE,
            CP2KKPointScheme.GAMMA,
        }:
            if self.mesh is not None or self.shift is not None or self.points:
                raise ValueError(
                    "NONE and GAMMA schemes cannot define mesh, shift or points."
                )

        elif self.scheme == CP2KKPointScheme.MONKHORST_PACK:
            if self.mesh is None or self.shift is not None or self.points:
                raise ValueError(
                    "MONKHORST-PACK requires a mesh without shift or points."
                )

        elif self.scheme == CP2KKPointScheme.MACDONALD:
            if self.mesh is None or self.shift is None or self.points:
                raise ValueError(
                    "MACDONALD requires mesh and shift without explicit points."
                )

        elif self.scheme == CP2KKPointScheme.GENERAL:
            if not self.points or self.mesh is not None or self.shift is not None:
                raise ValueError(
                    "GENERAL requires explicit points without mesh or shift."
                )

        return self


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

    kpoint_sampling: CP2KKPointConfig | None = None

    @model_validator(mode="after")
    def validate_kpoint_configuration(self):
        if self.k_points is not None and self.kpoint_sampling is not None:
            raise ValueError(
                "Specify either legacy k_points or structured "
                "kpoint_sampling, not both."
            )

        return self

    scf: CP2KSCFConfig = Field(
        default_factory=CP2KSCFConfig,
    )

    basis_potential: CP2KBasisPotentialConfig | None = None