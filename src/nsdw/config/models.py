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
