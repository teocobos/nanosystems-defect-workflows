from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    path: str
    format: str


class LatticeOutput(BaseModel):
    a_angstrom: float
    b_angstrom: float
    c_angstrom: float

    alpha_deg: float
    beta_deg: float
    gamma_deg: float

    volume_angstrom3: float


class StructureOutput(BaseModel):
    formula: str
    reduced_formula: str
    num_sites: int = Field(gt=0)
    density_g_cm3: float = Field(gt=0)

    lattice: LatticeOutput


class CheckOutput(BaseModel):
    name: str
    passed: bool
    value: str | None = None
    message: str | None = None


class ParserOutput(BaseModel):
    warnings: list[str] = Field(default_factory=list)


class ValidationOutput(BaseModel):
    valid: bool
    ordered: bool

    minimum_distance_angstrom: float | None = None
    minimum_distance_threshold_angstrom: float

    checks: list[CheckOutput] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class StructureValidationOutput(BaseModel):
    schema_version: str = "1.0"
    nsdw_version: str
    command: str = "structure.validate"

    source: SourceInfo
    structure: StructureOutput
    parser: ParserOutput
    validation: ValidationOutput