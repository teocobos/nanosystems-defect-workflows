from pathlib import Path

from nsdw.config.models import (
    StructureSummary,
    StructureValidationResult,
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
)
from nsdw.structures.symmetry import SymmetryResult

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