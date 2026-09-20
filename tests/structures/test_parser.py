from pathlib import Path

import pytest

from pymatgen.core import Lattice, Structure

from nsdw.structures.parser import (
    StructureParseError,
    load_structure,
)


def test_missing_file():
    """
    Missing structure files should fail cleanly.
    """

    with pytest.raises(FileNotFoundError):
        load_structure("this_file_does_not_exist.cif")


def test_unsupported_format(tmp_path: Path):
    """
    Unsupported structure formats should be rejected.
    """

    file = tmp_path / "structure.txt"
    file.write_text("not a structure")

    with pytest.raises(StructureParseError):
        load_structure(file)


def test_plain_xyz_rejected(tmp_path: Path):
    """
    Plain XYZ must not be interpreted as periodic without lattice data.
    """

    file = tmp_path / "structure.xyz"

    file.write_text(
        "2\n"
        "example\n"
        "Si 0.0 0.0 0.0\n"
        "Si 1.0 1.0 1.0\n"
    )

    with pytest.raises(
        StructureParseError,
        match="lattice metadata",
    ):
        load_structure(file)


def test_cif_parsing(tmp_path: Path):
    """
    A valid CIF should return a periodic pymatgen Structure.
    """

    original = Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
    )

    file = tmp_path / "silicon.cif"

    original.to(filename=file)

    parsed, parser_warnings = load_structure(file)

    assert len(parsed) == 2
    assert parsed.lattice.a == pytest.approx(5.43)
    assert isinstance(parser_warnings, list)

def test_xyz_with_lattice_parameters(tmp_path: Path):
    """XYZ coordinates plus six lattice parameters form a periodic structure."""

    file = tmp_path / "structure.xyz"
    file.write_text(
        "3\n"
        "example SiO2\n"
        "Si 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "O 0.0 1.0 0.0\n",
        encoding="utf-8",
    )

    structure, parser_warnings = load_structure(
        file,
        lattice_parameters=(10.0, 11.0, 12.0, 90.0, 90.0, 120.0),
    )

    # Check the atomic structure.
    assert len(structure) == 3

    # Check the lattice lengths (angstroms).
    assert structure.lattice.abc == pytest.approx(
        (10.0, 11.0, 12.0)
    )

    # Check the lattice angles (degrees).
    assert structure.lattice.angles == pytest.approx(
        (90.0, 90.0, 120.0)
    )

    # Check the atomic species and their ordering.
    assert [str(site.specie) for site in structure] == [
        "Si", "O", "O"
    ]

    # Check that the original Cartesian coordinates are preserved.
    expected_coordinates = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    for actual, expected in zip(
        structure.cart_coords,
        expected_coordinates,
        strict=True,
    ):
        assert actual == pytest.approx(expected)

    # Check that the parser returns its warnings in the expected format.
    assert isinstance(parser_warnings, list)

@pytest.mark.parametrize(
    "lattice_parameters",
    [
        (0.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        (10.0, -1.0, 10.0, 90.0, 90.0, 90.0),
        (10.0, 10.0, 10.0, 0.0, 90.0, 90.0),
        (10.0, 10.0, 10.0, 90.0, 90.0, 180.0),
        (10.0, 10.0, 10.0, 10.0, 10.0, 170.0),
    ],
)
def test_xyz_rejects_invalid_lattice(tmp_path: Path, lattice_parameters):
    """Invalid or degenerate lattice parameters must be rejected."""

    file = tmp_path / "structure.xyz"
    file.write_text(
        "1\n"
        "example\n"
        "Si 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(StructureParseError):
        load_structure(file, lattice_parameters=lattice_parameters)