import pytest

from pymatgen.core import Lattice, Structure

from nsdw.structures.validator import (
    get_minimum_distance,
    summarise_structure,
    validate_structure,
)


def test_valid_ordered_structure():
    """
    A simple physically sensible ordered structure should validate.
    """

    structure = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
    )

    result = validate_structure(structure)

    assert result.valid is True
    assert result.ordered is True
    assert result.errors == []
    assert result.minimum_distance is not None
    assert result.minimum_distance > 0.5


def test_unphysically_close_atoms_fail():
    """
    Sites separated by less than the validation threshold must fail.
    """

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "Si"],
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ],
    )

    result = validate_structure(
        structure,
        min_distance=0.5,
    )

    assert result.valid is False
    assert result.minimum_distance == pytest.approx(
        0.05,
        abs=1e-8,
    )

    assert any(
        "Minimum interatomic distance" in error
        for error in result.errors
    )


def test_custom_minimum_distance_threshold():
    """
    Changing the threshold should alter the validation decision.
    """

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "Si"],
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
    )

    result_loose = validate_structure(
        structure,
        min_distance=0.4,
    )

    result_strict = validate_structure(
        structure,
        min_distance=0.6,
    )

    assert result_loose.valid is True
    assert result_strict.valid is False


def test_minimum_distance():
    """
    Minimum periodic distance should be calculated correctly.
    """

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "Si"],
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
    )

    distance = get_minimum_distance(structure)

    assert distance == pytest.approx(
        0.5,
        abs=1e-8,
    )


def test_structure_summary():
    """
    Structure summary should expose basic structural metadata.
    """

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "Si"],
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
    )

    summary = summarise_structure(structure)

    assert summary.num_sites == 2
    assert summary.lattice.a == pytest.approx(5.0)
    assert summary.lattice.b == pytest.approx(5.0)
    assert summary.lattice.c == pytest.approx(5.0)
    assert summary.lattice.volume == pytest.approx(125.0)
