from pathlib import Path

from pymatgen.core import Structure


class StructureParseError(Exception):
    """Raised when NSDW cannot parse a structure file."""


SUPPORTED_FORMATS = {".cif", ".xyz"}


def load_structure(path: str | Path) -> Structure:
    """
    Load a periodic structure from a supported structure file.

    Parameters
    ----------
    path
        Path to the structure file.

    Returns
    -------
    pymatgen.core.Structure
        Parsed periodic structure.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    StructureParseError
        If the format is unsupported or the structure cannot be parsed.
    """

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    if not path.is_file():
        raise StructureParseError(f"Path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise StructureParseError(
            f"Unsupported structure format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if suffix == ".xyz":
        raise StructureParseError(
            "XYZ support requires lattice metadata and will be "
            "implemented in the next Phase 1A step."
        )

    try:
        structure = Structure.from_file(path)
    except Exception as exc:
        raise StructureParseError(
            f"Failed to parse structure file '{path}': {exc}"
        ) from exc

    if len(structure) == 0:
        raise StructureParseError("Parsed structure contains no atomic sites.")

    return structure
