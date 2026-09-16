from pathlib import Path

import pytest

from pymatgen.core import Lattice, Structure

from nsdw.output.builders import (
    build_supercell_search_output,
)
from nsdw.structures.parser import load_structure
from nsdw.structures.supercell import (
    SupercellSearchError,
    calculate_anisotropy_ratio,
    get_minimum_lattice_translation,
    search_supercells,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

IGZO_ORDERED = (
    REPO_ROOT
    / "tests"
    / "data"
    / "igzo"
    / "igzo_crystal_ordered_003.cif"
)


def test_minimum_translation_cubic():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    distance = get_minimum_lattice_translation(
        structure
    )

    assert distance == pytest.approx(
        5.0,
        abs=1e-8,
    )


def test_minimum_translation_hexagonal():
    structure = Structure(
        Lattice.hexagonal(
            a=3.0,
            c=10.0,
        ),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    distance = get_minimum_lattice_translation(
        structure
    )

    assert distance == pytest.approx(
        3.0,
        abs=1e-8,
    )


def test_anisotropy_cubic():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    ratio = calculate_anisotropy_ratio(
        structure
    )

    assert ratio == pytest.approx(
        1.0,
        abs=1e-8,
    )


def test_supercell_search_finds_candidate():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "Si"],
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
    )

    result = search_supercells(
        structure,
        min_atoms=8,
        max_atoms=64,
        min_image_distance=9.0,
        max_scale=3,
    )

    assert result.selected_candidate is not None

    candidate = result.selected_candidate

    assert candidate.acceptable is True
    assert candidate.num_atoms >= 8
    assert candidate.num_atoms <= 64

    assert (
        candidate.minimum_image_distance_angstrom
        >= 9.0
    )


def test_supercell_search_no_candidate():
    structure = Structure(
        Lattice.cubic(2.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    result = search_supercells(
        structure,
        min_atoms=1,
        max_atoms=8,
        min_image_distance=20.0,
        max_scale=2,
    )

    assert result.selected_candidate is None
    assert len(result.candidates) == 8


def test_invalid_atom_limits_rejected():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(
        SupercellSearchError,
        match="max_atoms",
    ):
        search_supercells(
            structure,
            min_atoms=100,
            max_atoms=50,
        )


def test_igzo_default_constraints_explain_tradeoff():
    structure, parser_warnings = load_structure(
        IGZO_ORDERED
    )

    search = search_supercells(
        structure,
        min_atoms=50,
        max_atoms=250,
        min_image_distance=10.0,
        max_scale=4,
    )

    assert search.selected_candidate is None

    best = (
        search.best_separation_within_atom_limits
    )

    assert best is not None
    assert best.scaling == (3, 3, 1)
    assert best.num_atoms == 189

    assert (
        best.minimum_image_distance_angstrom
        == pytest.approx(
            9.897,
            abs=1e-6,
        )
    )

    meeting = (
        search.smallest_meeting_image_distance
    )

    assert meeting is not None
    assert meeting.scaling == (4, 4, 1)
    assert meeting.num_atoms == 336

    output = build_supercell_search_output(
        source_path=IGZO_ORDERED,
        search=search,
        parser_warnings=parser_warnings,
        nsdw_version="0.1.0",
    )

    assert output.selected_candidate is None

    assert (
        output.diagnostics
        .image_distance_shortfall_angstrom
        == pytest.approx(
            0.103,
            abs=1e-6,
        )
    )

    assert output.diagnostics.atom_excess == 86


def test_igzo_relaxed_atom_ceiling_finds_candidate():
    structure, _ = load_structure(
        IGZO_ORDERED
    )

    result = search_supercells(
        structure,
        min_atoms=50,
        max_atoms=400,
        min_image_distance=10.0,
        max_scale=4,
    )

    assert result.selected_candidate is not None

    candidate = result.selected_candidate

    assert candidate.acceptable is True
    assert candidate.scaling == (4, 4, 1)
    assert candidate.num_atoms == 336