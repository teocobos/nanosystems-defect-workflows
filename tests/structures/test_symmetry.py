from pathlib import Path

import pytest

from pymatgen.core import Lattice, Structure

from nsdw.output.builders import (
    build_structure_symmetry_output,
)
from nsdw.structures.parser import load_structure
from nsdw.structures.symmetry import (
    SymmetryAnalysisError,
    analyse_symmetry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

IGZO_ORDERED = (
    REPO_ROOT
    / "tests"
    / "data"
    / "igzo"
    / "igzo_crystal_ordered_003.cif"
)


def test_simple_cubic_symmetry():
    structure = Structure(
        Lattice.cubic(5.43),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    result = analyse_symmetry(structure)

    assert result.space_group_symbol == "Pm-3m"
    assert result.space_group_number == 221
    assert result.crystal_system == "cubic"
    assert result.num_symmetry_operations > 0
    assert len(result.inequivalent_sites) == 1


def test_ordered_igzo_symmetry():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    result = analyse_symmetry(
        structure,
        symprec=1e-3,
    )

    assert result.space_group_symbol == "R3m"
    assert result.space_group_number == 160
    assert result.crystal_system == "trigonal"
    assert result.point_group == "3m"
    assert result.num_symmetry_operations == 18


def test_ordered_igzo_oxygen_classes():
    structure, parser_warnings = load_structure(
        IGZO_ORDERED
    )

    symmetry = analyse_symmetry(
        structure,
        symprec=1e-3,
    )

    output = build_structure_symmetry_output(
        source_path=IGZO_ORDERED,
        symmetry=symmetry,
        parser_warnings=parser_warnings,
        nsdw_version="0.1.0",
        selected_element="O",
    )

    sites = output.site_analysis

    assert sites.selected_element == "O"
    assert sites.total_selected_sites == 12
    assert sites.num_inequivalent_sites == 4

    assert [
        site.site_id
        for site in sites.inequivalent_sites
    ] == [
        "O001",
        "O002",
        "O003",
        "O004",
    ]

    assert [
        site.representative_index
        for site in sites.inequivalent_sites
    ] == [
        9,
        10,
        15,
        16,
    ]

    assert [
        site.equivalent_indices
        for site in sites.inequivalent_sites
    ] == [
        [9, 11, 13],
        [10, 12, 14],
        [15, 17, 19],
        [16, 18, 20],
    ]

    assert [
        site.representative_atom_number
        for site in sites.inequivalent_sites
    ] == [
        10,
        11,
        16,
        17,
    ]


def test_disordered_structure_rejected():
    structure = Structure(
        Lattice.cubic(5.0),
        [
            {"Si": 0.5, "Ge": 0.5},
        ],
        [
            [0.0, 0.0, 0.0],
        ],
    )

    with pytest.raises(
        SymmetryAnalysisError,
        match="fully ordered",
    ):
        analyse_symmetry(structure)
