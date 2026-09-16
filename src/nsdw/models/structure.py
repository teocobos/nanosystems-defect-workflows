"""Structural information associated with an NSDW scientific result."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpaceGroup(BaseModel):
    """Crystallographic space-group information."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(ge=1, le=230)
    symbol: str

    @field_validator("symbol")
    @classmethod
    def symbol_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("space-group symbol must not be empty")

        return value


class StructureResult(BaseModel):
    """Calculator-independent structural metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    formula: str
    n_atoms: int = Field(gt=0)
    periodic: bool
    structure_hash: str

    volume: float | None = Field(default=None, gt=0)
    density: float | None = Field(default=None, gt=0)
    space_group: SpaceGroup | None = None

    @field_validator("formula", "structure_hash")
    @classmethod
    def strings_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value must not be empty")

        return value
