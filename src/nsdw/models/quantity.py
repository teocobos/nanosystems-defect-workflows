"""Unit-aware scientific quantities used by NSDW result models."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, field_validator


class Quantity(BaseModel):
    """A numerical scientific quantity with an explicit unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    unit: str

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, value: float) -> float:
        """Reject NaN and infinite values."""
        if not math.isfinite(value):
            raise ValueError("quantity value must be finite")
        return value

    @field_validator("unit")
    @classmethod
    def unit_must_not_be_empty(cls, unit: str) -> str:
        """Reject empty or whitespace-only units."""
        unit = unit.strip()

        if not unit:
            raise ValueError("quantity unit must not be empty")

        return unit
