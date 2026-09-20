"""Load periodic atomic structures from supported file formats."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

from pymatgen.core import Lattice, Structure


class StructureParseError(Exception):
    """Raised when NSDW cannot parse a structure file."""


SUPPORTED_FORMATS = {".cif", ".xyz"}


def _lattice_from_parameters(
    parameters: tuple[float, float, float, float, float, float],
) -> Lattice:
    """Construct a non-degenerate lattice from lengths and angles."""

    if len(parameters) != 6:
        raise StructureParseError(
            "Lattice parameters must contain "
            "(a, b, c, alpha, beta, gamma)."
        )

    try:
        a, b, c, alpha, beta, gamma = (
            float(value) for value in parameters
        )
    except (TypeError, ValueError) as exc:
        raise StructureParseError(
            "Lattice parameters must be numeric."
        ) from exc

    if not all(math.isfinite(value) for value in (
        a, b, c, alpha, beta, gamma
    )):
        raise StructureParseError(
            "Lattice parameters must be finite."
        )

    if any(length <= 0 for length in (a, b, c)):
        raise StructureParseError(
            "Lattice lengths must be positive."
        )

    if any(not 0 < angle < 180 for angle in (
        alpha, beta, gamma
    )):
        raise StructureParseError(
            "Lattice angles must be between 0 and 180 degrees."
        )

    try:
        lattice = Lattice.from_parameters(
            a, b, c, alpha, beta, gamma
        )
    except (ValueError, ArithmeticError) as exc:
        raise StructureParseError(
            f"Invalid lattice parameters: {exc}"
        ) from exc

    if (
        not math.isfinite(lattice.volume)
        or lattice.volume <= 1e-8
    ):
        raise StructureParseError(
            "Lattice parameters produce a degenerate cell."
        )

    return lattice


def _load_xyz(
    path: Path,
    lattice: Lattice,
) -> Structure:
    """Load standard XYZ Cartesian coordinates into a periodic cell."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()

        if not lines:
            raise ValueError("XYZ file is empty.")

        n_atoms = int(lines[0].strip())

        if n_atoms <= 0:
            raise ValueError("XYZ atom count must be positive.")

        if len(lines) < n_atoms + 2:
            raise ValueError(
                "XYZ file has fewer coordinate lines than declared."
            )

        if any(line.strip() for line in lines[n_atoms + 2:]):
            raise ValueError(
                "XYZ file contains unexpected trailing content."
            )

        species = []
        coordinates = []

        for line in lines[2:n_atoms + 2]:
            fields = line.split()

            if len(fields) != 4:
                raise ValueError(
                    "Each XYZ atom line must contain "
                    "an element and three Cartesian coordinates."
                )

            element = fields[0]
            xyz = [float(value) for value in fields[1:]]

            if not all(math.isfinite(value) for value in xyz):
                raise ValueError(
                    "XYZ coordinates must be finite."
                )

            species.append(element)
            coordinates.append(xyz)

        return Structure(
            lattice,
            species,
            coordinates,
            coords_are_cartesian=True,
            to_unit_cell=False,
        )

    except (ValueError, TypeError, IndexError) as exc:
        raise StructureParseError(
            f"Failed to parse XYZ file '{path}': {exc}"
        ) from exc


def load_structure(
    path: str | Path,
    lattice_parameters: (
        tuple[float, float, float, float, float, float] | None
    ) = None,
) -> tuple[Structure, list[str]]:
    """
    Load a periodic structure from CIF or XYZ.

    Standard XYZ requires explicit lattice parameters:
    (a, b, c, alpha, beta, gamma), in angstroms and degrees.

    Returns the pymatgen Structure and captured parser warnings.
    """

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Structure file not found: {path}"
        )

    if not path.is_file():
        raise StructureParseError(
            f"Path is not a file: {path}"
        )

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise StructureParseError(
            f"Unsupported structure format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if suffix == ".xyz":
        if lattice_parameters is None:
            raise StructureParseError(
                "XYZ support requires lattice metadata. "
                "Supply lattice_parameters="
                "(a, b, c, alpha, beta, gamma)."
            )

        lattice = _lattice_from_parameters(lattice_parameters)

        return _load_xyz(path, lattice), []

    if lattice_parameters is not None:
        raise StructureParseError(
            "Explicit lattice parameters are only supported "
            "for XYZ input; CIF already contains its lattice."
        )

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            structure = Structure.from_file(path)

    except Exception as exc:
        raise StructureParseError(
            f"Failed to parse structure file '{path}': {exc}"
        ) from exc

    if len(structure) == 0:
        raise StructureParseError(
            "Parsed structure contains no atomic sites."
        )

    parser_warnings = []

    for warning in caught_warnings:
        message = str(warning.message).strip()

        if message and message not in parser_warnings:
            parser_warnings.append(message)

    return structure, parser_warnings