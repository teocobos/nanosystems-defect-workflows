from pathlib import Path
import warnings

from pymatgen.core import Structure


class StructureParseError(Exception):
    """Raised when NSDW cannot parse a structure file."""


SUPPORTED_FORMATS = {".cif", ".xyz"}


def load_structure(
    path: str | Path,
) -> tuple[Structure, list[str]]:
    """
    Load a periodic structure from a supported structure file.

    Returns
    -------
    structure
        Parsed pymatgen Structure.

    parser_warnings
        Warnings emitted by the underlying parser. These are captured
        rather than printed directly so NSDW can report them in a
        controlled and reproducible way.
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

    # Standard XYZ does not contain a periodic lattice.
    # NSDW must never invent one silently.
    if suffix == ".xyz":
        raise StructureParseError(
            "XYZ support requires lattice metadata and will be "
            "implemented in a later Phase 1A step."
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