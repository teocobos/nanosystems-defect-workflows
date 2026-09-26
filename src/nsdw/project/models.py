"""Data models for NSDW material projects."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
