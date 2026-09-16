from pydantic import BaseModel, Field


# ============================================================================
# Shared output models
# ============================================================================


class SourceInfo(BaseModel):
    path: str
    format: str


class ParserOutput(BaseModel):
    warnings: list[str] = Field(default_factory=list)


# ============================================================================
# Structure validation output models
# ============================================================================


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


# ============================================================================
# Symmetry output models
# ============================================================================


class SymmetryInfoOutput(BaseModel):
    space_group_symbol: str
    space_group_number: int
    hall_symbol: str
    point_group: str
    crystal_system: str

    symprec_angstrom: float
    angle_tolerance_deg: float

    num_symmetry_operations: int


class InequivalentSiteOutput(BaseModel):
    site_id: str

    element: str

    # Canonical NSDW/Python zero-based index.
    representative_index: int

    # Human-facing one-based atom number.
    representative_atom_number: int

    multiplicity: int

    # Zero-based indices used internally by NSDW.
    equivalent_indices: list[int]

    # One-based atom numbers for human-facing output.
    equivalent_atom_numbers: list[int]

    fractional_coordinates: list[float]


class SiteAnalysisOutput(BaseModel):
    selected_element: str | None = None

    total_selected_sites: int
    num_inequivalent_sites: int

    inequivalent_sites: list[InequivalentSiteOutput] = Field(
        default_factory=list
    )


class StructureSymmetryOutput(BaseModel):
    schema_version: str = "1.0"
    nsdw_version: str
    command: str = "structure.symmetry"

    source: SourceInfo
    parser: ParserOutput

    symmetry: SymmetryInfoOutput
    site_analysis: SiteAnalysisOutput


# ============================================================================
# Supercell search output models
# ============================================================================


class SupercellCandidateOutput(BaseModel):
    scaling: list[int]

    num_atoms: int
    volume_angstrom3: float

    a_angstrom: float
    b_angstrom: float
    c_angstrom: float

    minimum_image_distance_angstrom: float
    anisotropy_ratio: float

    meets_min_atoms: bool
    meets_max_atoms: bool
    meets_min_image_distance: bool

    acceptable: bool


class SupercellConstraintsOutput(BaseModel):
    min_atoms: int
    max_atoms: int
    min_image_distance_angstrom: float

    max_scale: int
    image_range: int


class SupercellDiagnosticsOutput(BaseModel):
    image_distance_shortfall_angstrom: float | None = None
    atom_excess: int | None = None


class SupercellSearchOutput(BaseModel):
    schema_version: str = "1.0"
    nsdw_version: str
    command: str = "structure.supercell"

    source: SourceInfo
    parser: ParserOutput

    primitive_num_atoms: int

    search_method: str
    ranking_policy: str

    constraints: SupercellConstraintsOutput

    num_candidates_evaluated: int
    num_acceptable_candidates: int

    selected_candidate: SupercellCandidateOutput | None = None

    best_separation_within_atom_limits: (
        SupercellCandidateOutput | None
    ) = None

    smallest_meeting_image_distance: (
        SupercellCandidateOutput | None
    ) = None

    diagnostics: SupercellDiagnosticsOutput

    acceptable_candidates: list[
        SupercellCandidateOutput
    ] = Field(default_factory=list)

    candidates: list[
        SupercellCandidateOutput
    ] = Field(default_factory=list)


# ============================================================================
# Defect generation output models
# ============================================================================


class DefectSymmetryOutput(BaseModel):
    site_id: str

    primitive_site_index: int
    primitive_atom_number: int
    multiplicity: int

    primitive_fractional_coordinates: list[float]


class DefectSupercellOutput(BaseModel):
    scaling: list[int]

    pristine_num_atoms: int
    defect_num_atoms: int


class RemovedSiteOutput(BaseModel):
    supercell_site_index: int
    supercell_atom_number: int

    fractional_coordinates: list[float]


class DefectProvenanceOutput(BaseModel):
    parent_structure: str
    parent_structure_path: str
    parent_structure_sha256: str


class VacancyMetadataOutput(BaseModel):
    schema_version: str = "1.0"
    nsdw_version: str

    defect_id: str
    defect_type: str

    species: str
    charge_state: int

    symmetry: DefectSymmetryOutput
    supercell: DefectSupercellOutput
    removed_site: RemovedSiteOutput
    provenance: DefectProvenanceOutput

    structure_file: str


class DefectManifestEntryOutput(BaseModel):
    defect_id: str
    species: str
    charge_state: int

    symmetry_site_id: str
    multiplicity: int

    structure_file: str
    metadata_file: str


class DefectManifestOutput(BaseModel):
    schema_version: str = "1.0"
    nsdw_version: str
    command: str = "defects.generate-vacancies"

    source: SourceInfo
    parser: ParserOutput

    parent_structure_sha256: str

    species: str
    charge_state: int

    primitive_num_atoms: int

    supercell_scaling: list[int]
    pristine_supercell_num_atoms: int

    num_inequivalent_sites: int
    num_defects_generated: int

    pristine_structure_file: str

    defects: list[
        DefectManifestEntryOutput
    ] = Field(default_factory=list)