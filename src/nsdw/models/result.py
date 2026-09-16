"""Top-level NSDW scientific result model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from nsdw.models.calculation import CalculationMetadata
from nsdw.models.electronic import ElectronicResult
from nsdw.models.quantity import Quantity
from nsdw.models.structure import StructureResult
from nsdw.models.defect import DefectResult


class EnergyResult(BaseModel):
    """Energy quantities associated with a calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: Quantity | None = None
    free: Quantity | None = None


class NSDWResult(BaseModel):
    """Calculator-independent scientific result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"

    calculation: CalculationMetadata
    structure: StructureResult | None = None
    energy: EnergyResult | None = None
    electronic: ElectronicResult | None = None
    defect: DefectResult | None = None