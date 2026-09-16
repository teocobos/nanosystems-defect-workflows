from pathlib import Path

from nsdw.config.models import (
    StructureSummary,
    StructureValidationResult,
)
from nsdw.structures.symmetry import SymmetryResult
from nsdw.structures.supercell import (
    SupercellCandidate,
    SupercellSearchResult,
)
from nsdw.structures.defects import (
    VacancyStructure,
)
from nsdw.output.models import (
    CheckOutput,
    InequivalentSiteOutput,
    LatticeOutput,
    ParserOutput,
    SiteAnalysisOutput,
    SourceInfo,
    StructureOutput,
    StructureSymmetryOutput,
    StructureValidationOutput,
    SymmetryInfoOutput,
    ValidationOutput,
    SupercellCandidateOutput,
    SupercellConstraintsOutput,
    SupercellDiagnosticsOutput,
    SupercellSearchOutput,
    DefectManifestEntryOutput,
    DefectManifestOutput,
    DefectProvenanceOutput,
    DefectSupercellOutput,
    DefectSymmetryOutput,
    RemovedSiteOutput,
    VacancyMetadataOutput,
)



OUTPUT_PRECISION = 8


def _round_float(
    value: float | None,
    digits: int = OUTPUT_PRECISION,
) -> float | None:
    """
    Round a floating-point value for deterministic exported output.

    Scientific calculations should continue to use the original
    full-precision values.
    """

    if value is None:
        return None

    return round(float(value), digits)


def build_structure_validation_output(
    *,
    source_path: str | Path,
    summary: StructureSummary,
    validation: StructureValidationResult,
    parser_warnings: list[str],
    nsdw_version: str,
    minimum_distance_threshold: float,
) -> StructureValidationOutput:
    """
    Build the canonical machine-readable result for structure validation.
    """

    path = Path(source_path).expanduser().resolve()

    return StructureValidationOutput(
        nsdw_version=nsdw_version,
        source=SourceInfo(
            path=str(path),
            format=path.suffix.lower().lstrip("."),
        ),
        structure=StructureOutput(
            formula=summary.formula,
            reduced_formula=summary.reduced_formula,
            num_sites=summary.num_sites,
            density_g_cm3=_round_float(summary.density),
            lattice=LatticeOutput(
                a_angstrom=_round_float(summary.lattice.a),
                b_angstrom=_round_float(summary.lattice.b),
                c_angstrom=_round_float(summary.lattice.c),
                alpha_deg=_round_float(summary.lattice.alpha),
                beta_deg=_round_float(summary.lattice.beta),
                gamma_deg=_round_float(summary.lattice.gamma),
                volume_angstrom3=_round_float(
                    summary.lattice.volume
                ),
            ),
        ),
        parser=ParserOutput(
            warnings=parser_warnings,
        ),
        validation=ValidationOutput(
            valid=validation.valid,
            ordered=validation.ordered,
            minimum_distance_angstrom=_round_float(
                validation.minimum_distance
            ),
            minimum_distance_threshold_angstrom=_round_float(
                minimum_distance_threshold
            ),
            checks=[
                CheckOutput(
                    name=check.name,
                    passed=check.passed,
                    value=check.value,
                    message=check.message,
                )
                for check in validation.checks
            ],
            warnings=validation.warnings,
            errors=validation.errors,
        ),
    )
def build_structure_symmetry_output(
    *,
    source_path: str | Path,
    symmetry: SymmetryResult,
    parser_warnings: list[str],
    nsdw_version: str,
    selected_element: str | None = None,
) -> StructureSymmetryOutput:
    """
    Build the canonical machine-readable symmetry result.

    Site IDs are assigned deterministically within each element,
    ordered by representative zero-based structure index.
    """

    path = Path(source_path).expanduser().resolve()

    if selected_element is not None:
        selected_element = selected_element.strip()

        matching_sites = [
            site
            for site in symmetry.inequivalent_sites
            if site.element.lower()
            == selected_element.lower()
        ]

    else:
        matching_sites = list(
            symmetry.inequivalent_sites
        )

    # Count all physical sites represented by the selected
    # inequivalent classes.
    total_selected_sites = sum(
        site.multiplicity
        for site in matching_sites
    )

    # Site IDs are numbered independently for each element.
    element_counters: dict[str, int] = {}

    output_sites = []

    for site in matching_sites:
        element_counters.setdefault(
            site.element,
            0,
        )

        element_counters[site.element] += 1

        site_id = (
            f"{site.element}"
            f"{element_counters[site.element]:03d}"
        )

        output_sites.append(
            InequivalentSiteOutput(
                site_id=site_id,
                element=site.element,
                representative_index=(
                    site.representative_index
                ),
                representative_atom_number=(
                    site.representative_index + 1
                ),
                multiplicity=site.multiplicity,
                equivalent_indices=(
                    site.equivalent_indices
                ),
                equivalent_atom_numbers=[
                    index + 1
                    for index
                    in site.equivalent_indices
                ],
                fractional_coordinates=[
                    _round_float(value)
                    for value
                    in site.fractional_coordinates
                ],
            )
        )

    return StructureSymmetryOutput(
        nsdw_version=nsdw_version,
        source=SourceInfo(
            path=str(path),
            format=path.suffix.lower().lstrip("."),
        ),
        parser=ParserOutput(
            warnings=parser_warnings,
        ),
        symmetry=SymmetryInfoOutput(
            space_group_symbol=(
                symmetry.space_group_symbol
            ),
            space_group_number=(
                symmetry.space_group_number
            ),
            hall_symbol=symmetry.hall_symbol,
            point_group=symmetry.point_group,
            crystal_system=symmetry.crystal_system,
            symprec_angstrom=_round_float(
                symmetry.symprec_angstrom
            ),
            angle_tolerance_deg=_round_float(
                symmetry.angle_tolerance_deg
            ),
            num_symmetry_operations=(
                symmetry.num_symmetry_operations
            ),
        ),
        site_analysis=SiteAnalysisOutput(
            selected_element=selected_element,
            total_selected_sites=(
                total_selected_sites
            ),
            num_inequivalent_sites=len(
                output_sites
            ),
            inequivalent_sites=output_sites,
        ),
    )
def _build_supercell_candidate_output(
    candidate: SupercellCandidate | None,
) -> SupercellCandidateOutput | None:
    """
    Convert a scientific supercell candidate into the
    canonical machine-readable output representation.
    """

    if candidate is None:
        return None

    return SupercellCandidateOutput(
        scaling=list(candidate.scaling),
        num_atoms=candidate.num_atoms,
        volume_angstrom3=_round_float(
            candidate.volume_angstrom3
        ),
        a_angstrom=_round_float(
            candidate.a_angstrom
        ),
        b_angstrom=_round_float(
            candidate.b_angstrom
        ),
        c_angstrom=_round_float(
            candidate.c_angstrom
        ),
        minimum_image_distance_angstrom=_round_float(
            candidate.minimum_image_distance_angstrom
        ),
        anisotropy_ratio=_round_float(
            candidate.anisotropy_ratio
        ),
        meets_min_atoms=candidate.meets_min_atoms,
        meets_max_atoms=candidate.meets_max_atoms,
        meets_min_image_distance=(
            candidate.meets_min_image_distance
        ),
        acceptable=candidate.acceptable,
    )


def build_supercell_search_output(
    *,
    source_path: str | Path,
    search: SupercellSearchResult,
    parser_warnings: list[str],
    nsdw_version: str,
) -> SupercellSearchOutput:
    """
    Build the canonical machine-readable supercell search result.
    """

    path = Path(
        source_path
    ).expanduser().resolve()

    best_within = (
        search.best_separation_within_atom_limits
    )

    smallest_meeting = (
        search.smallest_meeting_image_distance
    )

    image_shortfall = None

    if (
        search.selected_candidate is None
        and best_within is not None
        and not best_within.meets_min_image_distance
    ):
        image_shortfall = max(
            0.0,
            search.min_image_distance_angstrom
            - best_within.minimum_image_distance_angstrom,
        )

    atom_excess = None

    if (
        search.selected_candidate is None
        and smallest_meeting is not None
        and not smallest_meeting.meets_max_atoms
    ):
        atom_excess = max(
            0,
            smallest_meeting.num_atoms
            - search.max_atoms,
        )

    return SupercellSearchOutput(
        nsdw_version=nsdw_version,
        source=SourceInfo(
            path=str(path),
            format=path.suffix.lower().lstrip("."),
        ),
        parser=ParserOutput(
            warnings=parser_warnings,
        ),
        primitive_num_atoms=(
            search.primitive_num_atoms
        ),
        search_method=search.search_method,
        ranking_policy=search.ranking_policy,
        constraints=SupercellConstraintsOutput(
            min_atoms=search.min_atoms,
            max_atoms=search.max_atoms,
            min_image_distance_angstrom=_round_float(
                search.min_image_distance_angstrom
            ),
            max_scale=search.max_scale,
            image_range=search.image_range,
        ),
        num_candidates_evaluated=len(
            search.candidates
        ),
        num_acceptable_candidates=len(
            search.acceptable_candidates
        ),
        selected_candidate=(
            _build_supercell_candidate_output(
                search.selected_candidate
            )
        ),
        best_separation_within_atom_limits=(
            _build_supercell_candidate_output(
                best_within
            )
        ),
        smallest_meeting_image_distance=(
            _build_supercell_candidate_output(
                smallest_meeting
            )
        ),
        diagnostics=SupercellDiagnosticsOutput(
            image_distance_shortfall_angstrom=(
                _round_float(image_shortfall)
            ),
            atom_excess=atom_excess,
        ),
        acceptable_candidates=[
            _build_supercell_candidate_output(
                candidate
            )
            for candidate
            in search.acceptable_candidates
        ],
        candidates=[
            _build_supercell_candidate_output(
                candidate
            )
            for candidate
            in search.candidates
        ],
    )
def build_vacancy_metadata_output(
    *,
    vacancy: VacancyStructure,
    source_path: str | Path,
    parent_structure_sha256: str,
    structure_file: str,
    nsdw_version: str,
) -> VacancyMetadataOutput:
    """
    Build metadata for one generated vacancy structure.
    """

    path = Path(
        source_path
    ).expanduser().resolve()

    return VacancyMetadataOutput(
        nsdw_version=nsdw_version,
        defect_id=vacancy.defect_id,
        defect_type=vacancy.defect_type,
        species=vacancy.species,
        charge_state=vacancy.charge_state,
        symmetry=DefectSymmetryOutput(
            site_id=vacancy.symmetry_site_id,
            primitive_site_index=(
                vacancy.primitive_site_index
            ),
            primitive_atom_number=(
                vacancy.primitive_atom_number
            ),
            multiplicity=(
                vacancy.primitive_multiplicity
            ),
            primitive_fractional_coordinates=[
                _round_float(value)
                for value
                in vacancy.primitive_fractional_coordinates
            ],
        ),
        supercell=DefectSupercellOutput(
            scaling=list(
                vacancy.supercell_scaling
            ),
            pristine_num_atoms=(
                vacancy.pristine_supercell_num_atoms
            ),
            defect_num_atoms=(
                vacancy.defect_structure_num_atoms
            ),
        ),
        removed_site=RemovedSiteOutput(
            supercell_site_index=(
                vacancy.removed_supercell_site_index
            ),
            supercell_atom_number=(
                vacancy.removed_supercell_atom_number
            ),
            fractional_coordinates=[
                _round_float(value)
                for value
                in (
                    vacancy
                    .removed_supercell_fractional_coordinates
                )
            ],
        ),
        provenance=DefectProvenanceOutput(
            parent_structure=path.name,
            parent_structure_path=str(path),
            parent_structure_sha256=(
                parent_structure_sha256
            ),
        ),
        structure_file=structure_file,
    )


def build_defect_manifest_output(
    *,
    source_path: str | Path,
    parser_warnings: list[str],
    parent_structure_sha256: str,
    species: str,
    charge_state: int,
    primitive_num_atoms: int,
    supercell_scaling: tuple[int, int, int],
    pristine_supercell_num_atoms: int,
    num_inequivalent_sites: int,
    pristine_structure_file: str,
    defect_entries: list[
        DefectManifestEntryOutput
    ],
    nsdw_version: str,
) -> DefectManifestOutput:
    """
    Build the top-level vacancy-generation manifest.
    """

    path = Path(
        source_path
    ).expanduser().resolve()

    return DefectManifestOutput(
        nsdw_version=nsdw_version,
        source=SourceInfo(
            path=str(path),
            format=(
                path.suffix
                .lower()
                .lstrip(".")
            ),
        ),
        parser=ParserOutput(
            warnings=parser_warnings,
        ),
        parent_structure_sha256=(
            parent_structure_sha256
        ),
        species=species,
        charge_state=charge_state,
        primitive_num_atoms=(
            primitive_num_atoms
        ),
        supercell_scaling=list(
            supercell_scaling
        ),
        pristine_supercell_num_atoms=(
            pristine_supercell_num_atoms
        ),
        num_inequivalent_sites=(
            num_inequivalent_sites
        ),
        num_defects_generated=len(
            defect_entries
        ),
        pristine_structure_file=(
            pristine_structure_file
        ),
        defects=defect_entries,
    )