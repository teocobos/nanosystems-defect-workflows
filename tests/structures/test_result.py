"""Tests for calculator-independent structure results."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.structures.result import (
    build_structure_result,
    calculate_structure_hash,
)


def _make_structure() -> Structure:
    """Return a small ordered periodic test structure."""

    return Structure(
        lattice=Lattice.cubic(5.0),
        species=["Si", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
    )


def test_build_structure_result_contains_expected_metadata():
    structure = _make_structure()

    result = build_structure_result(structure)

    assert result.formula == structure.composition.formula
    assert result.n_atoms == 2
    assert result.periodic is True
    assert result.volume == pytest.approx(structure.volume)
    assert result.density == pytest.approx(structure.density)
    assert len(result.structure_hash) == 64


def test_structure_hash_is_deterministic():
    structure = _make_structure()

    first = calculate_structure_hash(structure)
    second = calculate_structure_hash(structure.copy())

    assert first == second


def test_structure_hash_changes_when_coordinates_change():
    first = _make_structure()
    second = _make_structure()

    second.translate_sites(
        [1],
        [0.01, 0.0, 0.0],
        frac_coords=True,
        to_unit_cell=False,
    )

    assert (
        calculate_structure_hash(first)
        != calculate_structure_hash(second)
    )


def test_structure_hash_changes_when_lattice_changes():
    first = _make_structure()

    second = Structure(
        lattice=Lattice.cubic(5.1),
        species=["Si", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
    )

    assert (
        calculate_structure_hash(first)
        != calculate_structure_hash(second)
    )


def test_structure_hash_ignores_insignificant_float_noise():
    first = _make_structure()
    second = _make_structure()

    second.translate_sites(
        [1],
        [1e-12, 0.0, 0.0],
        frac_coords=True,
        to_unit_cell=False,
    )

    assert (
        calculate_structure_hash(first)
        == calculate_structure_hash(second)
    )


def test_structure_hash_preserves_site_order():
    first = _make_structure()

    second = Structure(
        lattice=Lattice.cubic(5.0),
        species=["O", "Si"],
        coords=[
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
        ],
    )

    assert (
        calculate_structure_hash(first)
        != calculate_structure_hash(second)
    )


def test_structure_hash_rejects_disordered_structure():
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=[
            {"Si": 0.5, "Ge": 0.5},
        ],
        coords=[
            [0.0, 0.0, 0.0],
        ],
    )

    with pytest.raises(
        ValueError,
        match="disordered structure",
    ):
        calculate_structure_hash(structure)


def test_structure_hash_is_valid_sha256_hex():
    structure = _make_structure()

    digest = calculate_structure_hash(structure)

    assert len(digest) == 64
    assert all(
        character in "0123456789abcdef"
        for character in digest
    )
