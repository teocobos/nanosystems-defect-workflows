"""Tests for backend-independent supercell selections."""

import pytest

from pymatgen.core import Lattice, Structure

from nsdw.structures.supercell_selection import (
    SelectedSupercell,
    SupercellSelectionError,
)


def make_structure() -> Structure:
    return Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0.0, 0.0, 0.0]],
    )


def test_manual_diagonal_supercell():
    primitive = make_structure()

    supercell = primitive.copy()
    supercell.make_supercell([2, 2, 2])

    selection = SelectedSupercell(
        engine="manual",
        transformation_matrix=(
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
        ),
        structure=supercell,
        primitive_num_atoms=1,
        num_atoms=8,
        minimum_image_distance_angstrom=10.0,
    )

    assert selection.determinant == 8
    assert selection.num_atoms == 8


def test_non_diagonal_supercell():
    primitive = make_structure()

    matrix = (
        (2, 1, 0),
        (0, 2, 0),
        (0, 0, 1),
    )

    supercell = primitive.copy()
    supercell.make_supercell(matrix)

    selection = SelectedSupercell(
        engine="doped",
        transformation_matrix=matrix,
        structure=supercell,
        primitive_num_atoms=1,
        num_atoms=4,
        minimum_image_distance_angstrom=5.0,
    )

    assert selection.determinant == 4
    assert selection.num_atoms == 4


def test_singular_matrix_rejected():
    with pytest.raises(
        SupercellSelectionError,
        match="nonsingular",
    ):
        SelectedSupercell(
            engine="manual",
            transformation_matrix=(
                (1, 0, 0),
                (0, 0, 0),
                (0, 0, 1),
            ),
            structure=make_structure(),
            primitive_num_atoms=1,
            num_atoms=1,
            minimum_image_distance_angstrom=5.0,
        )


def test_incorrect_atom_count_rejected():
    with pytest.raises(
        SupercellSelectionError,
        match="atom count",
    ):
        SelectedSupercell(
            engine="manual",
            transformation_matrix=(
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),
            ),
            structure=make_structure(),
            primitive_num_atoms=1,
            num_atoms=1,
            minimum_image_distance_angstrom=5.0,
        )
