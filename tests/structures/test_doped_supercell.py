"""Tests for doped-backed supercell selection."""

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.structures.doped_supercell import select_doped_supercell
from nsdw.structures.supercell_selection import SupercellSelectionError


pytest.importorskip("doped")


def test_doped_selects_valid_supercell():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    selection = select_doped_supercell(
        structure,
        min_atoms=8,
        max_atoms=64,
        min_image_distance=9.9,
    )

    assert selection.engine == "doped"
    assert 8 <= selection.num_atoms <= 64
    assert selection.minimum_image_distance_angstrom >= 9.9
    assert selection.determinant == selection.num_atoms


def test_doped_rejects_invalid_atom_limits():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(
        SupercellSelectionError,
        match="min_atoms",
    ):
        select_doped_supercell(
            structure,
            min_atoms=100,
            max_atoms=50,
        )


def test_doped_enforces_maximum_atoms():
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(
        SupercellSelectionError,
        match="exceeding max_atoms",
    ):
        select_doped_supercell(
            structure,
            min_atoms=8,
            max_atoms=8,
            min_image_distance=15.0,
        )


def test_doped_igzo_non_diagonal_supercell():
    """Validate the reference 252-atom IGZO supercell."""
    from pathlib import Path

    import pytest
    from pymatgen.core import Structure

    cif_path = (
        Path(__file__).parents[1]
        / "data"
        / "igzo"
        / "igzo_crystal_ordered_003.cif"
    )

    structure = Structure.from_file(str(cif_path))

    result = select_doped_supercell(
        structure,
        min_atoms=50,
        max_atoms=252,
        min_image_distance=10.0,
        ideal_threshold=0.0,
    )

    assert result.engine == "doped"
    assert result.num_atoms == 252
    assert result.determinant == 12

    assert result.transformation_matrix == (
        (4, 2, 0),
        (2, 4, 0),
        (0, 0, 1),
    )

    assert result.minimum_image_distance_angstrom == pytest.approx(
        11.428071228339451,
        abs=1e-4,
    )