from pydantic import BaseModel, Field


class LatticeSummary(BaseModel):
    a: float = Field(gt=0)
    b: float = Field(gt=0)
    c: float = Field(gt=0)

    alpha: float
    beta: float
    gamma: float

    volume: float = Field(gt=0)


class StructureSummary(BaseModel):
    formula: str
    reduced_formula: str
    num_sites: int = Field(gt=0)
    density: float = Field(gt=0)

    lattice: LatticeSummary


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    value: str | None = None
    message: str | None = None


class StructureValidationResult(BaseModel):
    valid: bool
    minimum_distance: float | None = None
    ordered: bool
    checks: list[ValidationCheck]
    warnings: list[str] = []
    errors: list[str] = []
