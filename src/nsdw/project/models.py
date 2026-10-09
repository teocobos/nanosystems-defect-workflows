"""Data models for NSDW material projects."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KSCFConfig,
    CP2KXCFunctional,
)


ProjectComponent = Literal[
    "cp2k",
    "vasp",
    "lammps",
    "mace",
]

COMPONENT_ORDER: tuple[ProjectComponent, ...] = (
    "cp2k",
    "vasp",
    "lammps",
    "mace",
)

MethodologyStatus = Literal[
    "candidate",
    "validated",
]


class CP2KMethodologyProvenance(BaseModel):
    """Provenance describing how a CP2K methodology was selected."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source: str = Field(
        min_length=1,
    )

    workflow: str = Field(
        min_length=1,
    )

    cutoff_report: str | None = None

    relative_cutoff_report: str | None = None

    kpoint_report: str | None = None
    final_cutoff_verification_report: str | None = None
    final_relative_cutoff_verification_report: str | None = None


class CP2KProductionMethodology(BaseModel):
    """Persistent scientific CP2K methodology for production work."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: MethodologyStatus = "candidate"

    functional: CP2KXCFunctional

    cutoff_ry: float = Field(
        gt=0,
    )

    relative_cutoff_ry: float = Field(
        gt=0,
    )

    k_points: tuple[int, int, int] | None = None

    scf: CP2KSCFConfig

    basis_potential: CP2KBasisPotentialConfig | None = None

    provenance: CP2KMethodologyProvenance


class ProjectMethodology(BaseModel):
    """Validated scientific methodologies associated with a project."""

    model_config = ConfigDict(
        extra="forbid",
    )

    cp2k: CP2KProductionMethodology | None = None


class ProjectConfig(BaseModel):
    """Top-level metadata describing an NSDW material project."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(
        default=1,
        ge=1,
    )

    name: str = Field(
        min_length=1,
    )

    material: str = Field(
        min_length=1,
    )

    nsdw_version: str = Field(
        min_length=1,
    )

    components: list[ProjectComponent] = Field(
        default_factory=list,
    )

    methodology: ProjectMethodology = Field(
        default_factory=ProjectMethodology,
    )

    @field_validator("components")
    @classmethod
    def normalise_components(
        cls,
        components: list[ProjectComponent],
    ) -> list[ProjectComponent]:
        """Remove duplicates and use the canonical component order."""

        selected = set(components)

        return [
            component
            for component in COMPONENT_ORDER
            if component in selected
        ]
