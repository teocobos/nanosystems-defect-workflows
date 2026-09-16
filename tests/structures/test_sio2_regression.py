from pathlib import Path

import pytest

from nsdw.structures.defects import (
    generate_symmetry_inequivalent_vacancies,
)
from nsdw.structures.parser import (
    load_structure,
)
from nsdw.structures.supercell import (
    search_supercells,
)
from nsdw.structures.symmetry import (
    analyse_symmetry,
)
from nsdw.structures.validator import (
    summarise_structure,
    validate_structure,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ALPHA_QUARTZ = (
    REPO_ROOT
    / "tests"
    / "data"
    / "sio2"
    / "alpha_quartz_cod1526860.cif"
)


def test_alpha_quartz_structure_summary():
    structure, _ = load_structure(
        ALPHA_QUARTZ
    )

    summary = summarise_structure(
        structure
    )

    assert len(structure) == 9

    assert (
        structure.composition.reduced_formula
        == "SiO2"
    )

    assert summary.lattice.volume == pytest.approx(
        112.3458,
        abs=1e-3,
    )


def test_alpha_quartz_is_structurally_valid():
    structure, _ = load_structure(
        ALPHA_QUARTZ
    )

    validation = validate_structure(
        structure
    )

    assert validation.valid


def test_alpha_quartz_symmetry():
    structure, _ = load_structure(
        ALPHA_QUARTZ
    )

    symmetry = analyse_symmetry(
        structure
    )

    assert (
        symmetry.space_group_number
        == 152
    )

    assert (
        symmetry.space_group_symbol
        == "P3_121"
    )

    assert (
        symmetry.crystal_system
        == "trigonal"
    )

    assert (
        symmetry.point_group
        == "32"
    )

    assert (
        symmetry.num_symmetry_operations
        == 6
    )


def test_alpha_quartz_has_one_oxygen_class():
    structure, _ = load_structure(
        ALPHA_QUARTZ
    )

    symmetry = analyse_symmetry(
        structure
    )

    oxygen_sites = [
        site
        for site
        in symmetry.inequivalent_sites
        if site.element == "O"
    ]

    assert len(
        oxygen_sites
    ) == 1

    oxygen = oxygen_sites[0]

    assert (
        oxygen.representative_index
        == 3
    )

    assert (
        oxygen.multiplicity
        == 6
    )

    assert (
        oxygen.equivalent_indices
        == [3, 4, 5, 6, 7, 8]
    )


def test_alpha_quartz_supercell_selection():
    structure, _ = load_structure(
        ALPHA_QUARTZ
    )

    result = search_supercells(
        structure,
        min_atoms=50,
        max_atoms=250,
        min_image_distance=10.0,
        max_scale=4,
        image_range=2,
    )

    assert (
        result.selected_candidate
        is not None
    )

    selected = (
        result.selected_candidate
    )

    assert (
        selected.scaling
        == (3, 3, 2)
    )

    assert (
        selected.num_atoms
        == 162
    )

    assert (
        selected.minimum_image_distance_angstrom == pytest.approx(
            10.7976,
            abs=1e-4,
        )
    )


def test_alpha_quartz_generates_one_oxygen_vacancy():
    structure, _ = load_structure(
        ALPHA_QUARTZ
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(3, 3, 2),
        )
    )

    assert (
        result.pristine_supercell_num_atoms
        == 162
    )

    assert (
        result.num_inequivalent_sites
        == 1
    )

    assert len(
        result.vacancies
    ) == 1

    vacancy = (
        result.vacancies[0]
    )

    assert (
        vacancy.defect_id
        == "VO_O001"
    )

    assert (
        vacancy.primitive_site_index
        == 3
    )

    assert (
        vacancy.primitive_atom_number
        == 4
    )

    assert (
        vacancy.primitive_multiplicity
        == 6
    )

    assert (
        vacancy.defect_structure_num_atoms
        == 161
    )


def test_alpha_quartz_vacancy_removes_only_oxygen():
    structure, _ = load_structure(
        ALPHA_QUARTZ
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(3, 3, 2),
        )
    )

    vacancy = (
        result.vacancies[0]
    )

    pristine = (
        result.pristine_supercell
    )

    defect = (
        vacancy.defect_structure
    )

    pristine_si = (
        pristine.composition["Si"]
    )

    defect_si = (
        defect.composition["Si"]
    )

    pristine_o = (
        pristine.composition["O"]
    )

    defect_o = (
        defect.composition["O"]
    )

    assert (
        defect_si
        == pristine_si
    )

    assert (
        defect_o
        == pristine_o - 1
    )
