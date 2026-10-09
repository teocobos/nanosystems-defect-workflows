"""Tests for unified supercell selection."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.structures.select_supercell import select_supercell
from nsdw.structures.supercell_selection import (
    SupercellSelectionError,
)


@pytest.fixture
def silicon():
    """Small cubic test structure."""
    return Structure(
        Lattice.cubic(5.43),
        ["Si"],
        [[0, 0, 0]],
    )


def test_manual_diagonal_matrix(silicon):
    result = select_supercell(
        silicon,
        engine="manual",
        matrix=[
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ],
    )

    assert result.engine == "manual"
    assert result.num_atoms == 8
    assert result.determinant == 8
    assert result.minimum_image_distance_angstrom == pytest.approx(
        10.86
    )


def test_manual_non_diagonal_matrix(silicon):
    result = select_supercell(
        silicon,
        engine="manual",
        matrix=[
            [2, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
        ],
    )

    assert result.engine == "manual"
    assert result.num_atoms == 4
    assert result.determinant == 4


def test_manual_requires_matrix(silicon):
    with pytest.raises(
        SupercellSelectionError,
        match="requires a transformation matrix",
    ):
        select_supercell(
            silicon,
            engine="manual",
        )


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0], [0, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_manual_rejects_invalid_matrices(silicon, matrix):
    with pytest.raises(SupercellSelectionError):
        select_supercell(
            silicon,
            engine="manual",
            matrix=matrix,
        )


def test_native_engine(silicon):
    result = select_supercell(
        silicon,
        engine="native",
        min_atoms=8,
        max_atoms=8,
        min_image_distance=10.0,
        max_scale=2,
    )

    assert result.engine == "native"
    assert result.num_atoms == 8
    assert result.determinant == 8


def test_native_rejects_manual_matrix(silicon):
    with pytest.raises(SupercellSelectionError):
        select_supercell(
            silicon,
            engine="native",
            matrix=[
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ],
        )


def test_native_reports_no_valid_candidate(silicon):
    with pytest.raises(
        SupercellSelectionError,
        match="no supercell satisfying",
    ):
        select_supercell(
            silicon,
            engine="native",
            min_atoms=100,
            max_atoms=200,
            min_image_distance=10.0,
            max_scale=2,
        )


def test_doped_engine(silicon):
    pytest.importorskip("doped")

    result = select_supercell(
        silicon,
        engine="doped",
        min_atoms=8,
        max_atoms=64,
        min_image_distance=10.0,
    )

    assert result.engine == "doped"
    assert result.num_atoms >= 8
    assert result.num_atoms <= 64
    assert result.minimum_image_distance_angstrom >= 10.0


def test_doped_rejects_manual_matrix(silicon):
    with pytest.raises(SupercellSelectionError):
        select_supercell(
            silicon,
            engine="doped",
            matrix=[
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ],
        )


def test_rejects_unknown_engine(silicon):
    with pytest.raises(
        SupercellSelectionError,
        match="Unsupported supercell engine",
    ):
        select_supercell(
            silicon,
            engine="unknown",
        )
