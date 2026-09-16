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
