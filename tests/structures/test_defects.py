from pathlib import Path

import pytest

from pymatgen.core import (
    Lattice,
    Structure,
)

from nsdw.structures.defects import (
    DefectGenerationError,
    generate_symmetry_inequivalent_vacancies,
)
from nsdw.structures.parser import (
    load_structure,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

IGZO_ORDERED = (
    REPO_ROOT
    / "tests"
    / "data"
    / "igzo"
    / "igzo_crystal_ordered_003.cif"
)


def test_simple_cubic_single_vacancy_class():
    structure = Structure(
        Lattice.cubic(5.0),
        ["O"],
        [[0.0, 0.0, 0.0]],
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(2, 2, 2),
        )
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

    assert vacancy.defect_id == "VO_O001"

    assert (
        vacancy.symmetry_site_id
        == "O001"
    )

    assert (
        vacancy.pristine_supercell_num_atoms
        == 8
    )

    assert (
        vacancy.defect_structure_num_atoms
        == 7
    )

    assert (
        len(vacancy.defect_structure)
        == 7
    )


def test_igzo_has_four_oxygen_vacancy_classes():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(1, 1, 1),
        )
    )

    assert (
        result.num_inequivalent_sites
        == 4
    )

    assert len(
        result.vacancies
    ) == 4

    assert [
        vacancy.symmetry_site_id
        for vacancy in result.vacancies
    ] == [
        "O001",
        "O002",
        "O003",
        "O004",
    ]

    assert [
        vacancy.defect_id
        for vacancy in result.vacancies
    ] == [
        "VO_O001",
        "VO_O002",
        "VO_O003",
        "VO_O004",
    ]


def test_igzo_primitive_site_indices():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
        )
    )

    assert [
        vacancy.primitive_site_index
        for vacancy in result.vacancies
    ] == [
        9,
        10,
        15,
        16,
    ]

    assert [
        vacancy.primitive_atom_number
        for vacancy in result.vacancies
    ] == [
        10,
        11,
        16,
        17,
    ]

    assert [
        vacancy.primitive_multiplicity
        for vacancy in result.vacancies
    ] == [
        3,
        3,
        3,
        3,
    ]


def test_igzo_4x4x1_vacancies_have_correct_atom_counts():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(4, 4, 1),
        )
    )

    assert (
        result.pristine_supercell_num_atoms
        == 336
    )

    assert len(
        result.vacancies
    ) == 4

    for vacancy in result.vacancies:
        assert (
            vacancy.pristine_supercell_num_atoms
            == 336
        )

        assert (
            vacancy.defect_structure_num_atoms
            == 335
        )

        assert (
            len(vacancy.defect_structure)
            == 335
        )


def test_removed_atom_is_oxygen():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(4, 4, 1),
        )
    )

    pristine = (
        result.pristine_supercell
    )

    for vacancy in result.vacancies:
        removed_index = (
            vacancy
            .removed_supercell_site_index
        )

        assert (
            pristine[
                removed_index
            ].specie.symbol
            == "O"
        )


def test_each_igzo_vacancy_removes_one_oxygen():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(4, 4, 1),
        )
    )

    pristine_oxygen = sum(
        1
        for site
        in result.pristine_supercell
        if site.specie.symbol == "O"
    )

    for vacancy in result.vacancies:
        defect_oxygen = sum(
            1
            for site
            in vacancy.defect_structure
            if site.specie.symbol == "O"
        )

        assert (
            defect_oxygen
            == pristine_oxygen - 1
        )


def test_mapping_is_deterministic():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    first = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(4, 4, 1),
        )
    )

    second = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(4, 4, 1),
        )
    )

    assert [
        vacancy.removed_supercell_site_index
        for vacancy in first.vacancies
    ] == [
        vacancy.removed_supercell_site_index
        for vacancy in second.vacancies
    ]


def test_missing_species_rejected():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(
        DefectGenerationError,
        match="No symmetry-inequivalent",
    ):
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
        )


def test_invalid_scaling_rejected():
    structure = Structure(
        Lattice.cubic(5.0),
        ["O"],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(
        DefectGenerationError,
        match="at least 1",
    ):
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(2, 0, 2),
        )


def test_charged_vacancy_rejected_in_phase_1():
    structure = Structure(
        Lattice.cubic(5.0),
        ["O"],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(
        DefectGenerationError,
        match="neutral",
    ):
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            charge_state=2,
        )
def test_igzo_supercell_mapping_preserves_representative_sites():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    scaling = (4, 4, 1)

    result = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=scaling,
        )
    )

    for vacancy in result.vacancies:
        primitive_site = structure[
            vacancy.primitive_site_index
        ]

        expected = (
            primitive_site.frac_coords[0] / scaling[0],
            primitive_site.frac_coords[1] / scaling[1],
            primitive_site.frac_coords[2] / scaling[2],
        )

        actual = (
            vacancy
            .removed_supercell_fractional_coordinates
        )

        assert actual == pytest.approx(
            expected,
            abs=1e-8,
        )