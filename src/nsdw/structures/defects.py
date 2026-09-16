from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure

from nsdw.structures.symmetry import (
    InequivalentSite,
    analyse_symmetry,
)


class DefectGenerationError(Exception):
    """Raised when NSDW cannot generate a requested defect."""


@dataclass
class VacancyStructure:
    """
    A single symmetry-inequivalent neutral vacancy structure.
    """

    defect_id: str
    defect_type: str
    species: str
    charge_state: int

    symmetry_site_id: str

    primitive_site_index: int
    primitive_atom_number: int
    primitive_multiplicity: int

    primitive_fractional_coordinates: tuple[
        float,
        float,
        float,
    ]

    supercell_scaling: tuple[int, int, int]

    pristine_supercell_num_atoms: int
    defect_structure_num_atoms: int

    removed_supercell_site_index: int
    removed_supercell_atom_number: int

    removed_supercell_fractional_coordinates: tuple[
        float,
        float,
        float,
    ]

    defect_structure: Structure


@dataclass
class VacancyGenerationResult:
    """
    Result of symmetry-inequivalent vacancy generation.
    """

    species: str
    charge_state: int

    primitive_num_atoms: int

    supercell_scaling: tuple[int, int, int]
    pristine_supercell_num_atoms: int

    num_inequivalent_sites: int

    pristine_supercell: Structure
    vacancies: list[VacancyStructure]


def _validate_scaling(
    scaling: tuple[int, int, int],
) -> None:
    """
    Validate a diagonal supercell scaling tuple.
    """

    if len(scaling) != 3:
        raise DefectGenerationError(
            "Supercell scaling must contain exactly "
            "three integers."
        )

    if any(
        not isinstance(value, int)
        for value in scaling
    ):
        raise DefectGenerationError(
            "Supercell scaling values must be integers."
        )

    if any(
        value < 1
        for value in scaling
    ):
        raise DefectGenerationError(
            "Supercell scaling values must be at least 1."
        )


def _site_species_symbol(
    structure: Structure,
    site_index: int,
) -> str:
    """
    Return the element symbol for an ordered structure site.
    """

    site = structure[site_index]

    if not site.is_ordered:
        raise DefectGenerationError(
            "Vacancy generation currently requires "
            "an ordered structure."
        )

    return site.specie.symbol


def _map_primitive_site_to_supercell(
    primitive_structure: Structure,
    supercell_structure: Structure,
    primitive_site_index: int,
    scaling: tuple[int, int, int],
    tolerance: float = 1e-6,
) -> int:
    """
    Map a primitive-cell site to its canonical image in a diagonal
    supercell.

    For diagonal scaling (na, nb, nc), a primitive fractional
    coordinate

        (x, y, z)

    maps to the canonical supercell fractional coordinate

        (x / na, y / nb, z / nc).

    The mapped atom is identified geometrically rather than by
    assuming pymatgen preserves a particular atom ordering.

    The canonical image corresponds to translation (0, 0, 0).
    """

    if tolerance <= 0.0:
        raise DefectGenerationError(
            "Mapping tolerance must be positive."
        )

    if (
        primitive_site_index < 0
        or primitive_site_index >= len(primitive_structure)
    ):
        raise DefectGenerationError(
            "Primitive site index is out of range."
        )

    species = _site_species_symbol(
        primitive_structure,
        primitive_site_index,
    )

    primitive_frac = np.mod(
        np.asarray(
            primitive_structure[
                primitive_site_index
            ].frac_coords,
            dtype=float,
        ),
        1.0,
    )

    scale = np.asarray(
        scaling,
        dtype=float,
    )

    target_frac = np.mod(
        primitive_frac / scale,
        1.0,
    )

    matching_indices: list[int] = []

    for index, site in enumerate(
        supercell_structure
    ):
        if not site.is_ordered:
            raise DefectGenerationError(
                "Vacancy generation currently requires "
                "an ordered structure."
            )

        if site.specie.symbol != species:
            continue

        candidate_frac = np.mod(
            np.asarray(
                site.frac_coords,
                dtype=float,
            ),
            1.0,
        )

        delta = (
            candidate_frac
            - target_frac
        )

        delta -= np.round(
            delta
        )

        cartesian_delta = (
            delta
            @ supercell_structure.lattice.matrix
        )

        distance = float(
            np.linalg.norm(
                cartesian_delta
            )
        )

        if distance <= tolerance:
            matching_indices.append(
                index
            )

    if len(matching_indices) == 0:
        raise DefectGenerationError(
            "Could not map primitive site "
            f"{primitive_site_index} to the supercell."
        )

    if len(matching_indices) > 1:
        raise DefectGenerationError(
            "Primitive-to-supercell site mapping was "
            "ambiguous."
        )

    return matching_indices[0]


def _make_site_id(
    species: str,
    number: int,
) -> str:
    """
    Generate a deterministic symmetry-site identifier.
    """

    return f"{species}{number:03d}"


def _prepare_inequivalent_sites(
    structure: Structure,
    *,
    species: str,
    symprec: float,
    angle_tolerance: float,
) -> list[tuple[str, InequivalentSite]]:
    """
    Analyse symmetry and return inequivalent sites for one element.
    """

    symmetry = analyse_symmetry(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    selected = [
        site
        for site in symmetry.inequivalent_sites
        if site.element == species
    ]

    if not selected:
        raise DefectGenerationError(
            f"No symmetry-inequivalent {species} "
            "sites were found."
        )

    selected.sort(
        key=lambda site: (
            site.representative_index,
            site.equivalent_indices,
        )
    )

    return [
        (
            _make_site_id(
                species,
                number,
            ),
            site,
        )
        for number, site in enumerate(
            selected,
            start=1,
        )
    ]


def generate_symmetry_inequivalent_vacancies(
    structure: Structure,
    *,
    species: str = "O",
    scaling: tuple[int, int, int] = (1, 1, 1),
    charge_state: int = 0,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    mapping_tolerance: float = 1e-6,
) -> VacancyGenerationResult:
    """
    Generate one vacancy structure for each symmetry-inequivalent
    site of the requested element.

    The current Phase 1 implementation:

    - requires an ordered periodic structure,
    - supports diagonal supercell scaling,
    - removes the canonical (0, 0, 0) supercell image of each
      primitive representative site,
    - records zero-based indices and one-based atom numbers,
    - creates neutral vacancies by default.

    The ``charge_state`` field is retained in the data model for
    provenance, but Phase 1A.5 is intended for neutral vacancies.
    """

    _validate_scaling(
        scaling
    )

    if len(structure) < 1:
        raise DefectGenerationError(
            "Structure contains no atomic sites."
        )

    if not structure.is_ordered:
        raise DefectGenerationError(
            "Vacancy generation currently requires "
            "an ordered structure."
        )

    if not species:
        raise DefectGenerationError(
            "Vacancy species must not be empty."
        )

    if charge_state != 0:
        raise DefectGenerationError(
            "Phase 1A.5 currently supports neutral "
            "vacancies only."
        )

    if symprec <= 0.0:
        raise DefectGenerationError(
            "symprec must be positive."
        )

    if mapping_tolerance <= 0.0:
        raise DefectGenerationError(
            "mapping_tolerance must be positive."
        )

    inequivalent_sites = (
        _prepare_inequivalent_sites(
            structure,
            species=species,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
    )

    pristine_supercell = (
        structure.copy()
    )

    pristine_supercell.make_supercell(
        list(scaling)
    )

    vacancies: list[
        VacancyStructure
    ] = []

    for (
        symmetry_site_id,
        inequivalent_site,
    ) in inequivalent_sites:
        primitive_index = (
            inequivalent_site
            .representative_index
        )

        primitive_atom_number = (
            primitive_index + 1
        )

        removed_index = (
            _map_primitive_site_to_supercell(
                primitive_structure=structure,
                supercell_structure=pristine_supercell,
                primitive_site_index=primitive_index,
                scaling=scaling,
                tolerance=mapping_tolerance,
            )
        )

        removed_site = (
            pristine_supercell[
                removed_index
            ]
        )

        removed_frac = tuple(
            float(value)
            for value
            in removed_site.frac_coords
        )

        primitive_frac = tuple(
            float(value)
            for value
            in structure[
                primitive_index
            ].frac_coords
        )

        defect_structure = (
            pristine_supercell.copy()
        )

        defect_structure.remove_sites(
            [removed_index]
        )

        defect_id = (
            f"V{species}_"
            f"{symmetry_site_id}"
        )

        vacancies.append(
            VacancyStructure(
                defect_id=defect_id,
                defect_type="vacancy",
                species=species,
                charge_state=charge_state,
                symmetry_site_id=(
                    symmetry_site_id
                ),
                primitive_site_index=(
                    primitive_index
                ),
                primitive_atom_number=(
                    primitive_atom_number
                ),
                primitive_multiplicity=(
                    inequivalent_site
                    .multiplicity
                ),
                primitive_fractional_coordinates=(
                    primitive_frac
                ),
                supercell_scaling=(
                    scaling
                ),
                pristine_supercell_num_atoms=len(
                    pristine_supercell
                ),
                defect_structure_num_atoms=len(
                    defect_structure
                ),
                removed_supercell_site_index=(
                    removed_index
                ),
                removed_supercell_atom_number=(
                    removed_index + 1
                ),
                removed_supercell_fractional_coordinates=(
                    removed_frac
                ),
                defect_structure=(
                    defect_structure
                ),
            )
        )

    return VacancyGenerationResult(
        species=species,
        charge_state=charge_state,
        primitive_num_atoms=len(
            structure
        ),
        supercell_scaling=scaling,
        pristine_supercell_num_atoms=len(
            pristine_supercell
        ),
        num_inequivalent_sites=len(
            inequivalent_sites
        ),
        pristine_supercell=(
            pristine_supercell
        ),
        vacancies=vacancies,
    )
