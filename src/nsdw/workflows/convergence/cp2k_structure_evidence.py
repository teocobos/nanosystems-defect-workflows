"""Independent geometry and input-dependency evidence checks."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path

from pymatgen.core import Structure

from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


COORD_TOLERANCE_ANGSTROM = 1e-6
CELL_TOLERANCE_ANGSTROM = 1e-8


def _reject(message: str) -> None:
    raise ConvergenceEvidenceValidationError(message)


def _contained_file(root: Path, path: Path) -> Path:
    resolved = path.resolve()

    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        _reject(f"Evidence path escapes project root: {path}")

    if not resolved.is_file():
        _reject(f"Missing structure evidence file: {path}")

    return resolved


def _clean_lines(text: str) -> list[str]:
    result = []

    for original in text.splitlines():
        line = re.split(r"[!#]", original, maxsplit=1)[0].strip()

        if line:
            result.append(line)

    return result


def _section(text: str, name: str) -> list[str]:
    """Extract exactly one explicitly terminated CP2K section."""
    stack = []
    matches = []
    active = None

    for line in _clean_lines(text):
        if not line.startswith("&"):
            if active is not None:
                active.append(line)
            continue

        parts = line.split()
        marker = parts[0].upper()

        if marker == "&END":
            closing = parts[1].upper() if len(parts) > 1 else None

            if not stack:
                _reject("Unexpected CP2K section terminator.")

            if closing is not None and closing != stack[-1]:
                _reject("Mismatched CP2K section terminator.")

            ending = stack.pop()

            if ending == name and not stack:
                _reject("Unexpected top-level CP2K section.")

            if ending == name and active is not None:
                matches.append(active)
                active = None

            continue

        if marker.startswith("&END"):
            _reject("Unsupported CP2K section terminator.")

        section_name = marker[1:]

        if section_name == name:
            if active is not None:
                _reject(f"Nested or repeated {name} section.")
            active = []

        elif active is not None:
            _reject(
                f"Nested section inside {name} is not supported."
            )

        stack.append(section_name)

    if stack:
        _reject("Unterminated CP2K section.")

    if len(matches) != 1:
        _reject(f"Expected exactly one {name} section.")

    return matches[0]


def _keyword(lines: list[str], name: str) -> str:
    matches = [
        line.split(None, 1)[1]
        for line in lines
        if line.split()[0].upper() == name
        and len(line.split(None, 1)) == 2
    ]

    if len(matches) != 1:
        _reject(f"Expected exactly one {name} keyword.")

    return matches[0]


def _vector(value: str, name: str) -> tuple[float, ...]:
    parts = value.split()

    if len(parts) != 3:
        _reject(f"Invalid {name} cell vector.")

    try:
        values = tuple(float(item) for item in parts)
    except ValueError:
        _reject(f"Invalid numeric {name} cell vector.")

    if not all(math.isfinite(item) for item in values):
        _reject(f"Non-finite {name} cell vector.")

    return values


def _verify_xyz(
    xyz_path: Path,
    structure: Structure,
) -> None:
    lines = xyz_path.read_text(encoding="utf-8").splitlines()

    if len(lines) < 2:
        _reject("Malformed XYZ coordinate file.")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        _reject("Invalid XYZ atom count.")

    if n_atoms != len(structure):
        _reject("XYZ atom count differs from canonical structure.")

    if len(lines) != n_atoms + 2:
        _reject("XYZ contains missing or unexpected records.")

    for index, (line, site) in enumerate(
        zip(lines[2:], structure, strict=True)
    ):
        parts = line.split()

        if len(parts) != 4:
            _reject(f"Invalid XYZ atom record {index}.")

        if parts[0] != site.specie.symbol:
            _reject(f"XYZ species mismatch at atom {index}.")

        try:
            coordinates = tuple(float(x) for x in parts[1:])
        except ValueError:
            _reject(f"Invalid XYZ coordinates at atom {index}.")

        if not all(math.isfinite(x) for x in coordinates):
            _reject(f"Non-finite XYZ coordinates at atom {index}.")

        for actual, expected in zip(
            coordinates,
            site.coords,
            strict=True,
        ):
            if abs(actual - float(expected)) > COORD_TOLERANCE_ANGSTROM:
                _reject(
                    f"XYZ coordinate mismatch at atom {index}."
                )


def validate_cp2k_structure_evidence(
    *,
    project_root: str | Path,
    campaign_directory: str | Path,
    candidate_directory: str | Path,
    input_filename: str,
    coordinate_filename: str,
) -> None:
    """Check canonical geometry, candidate XYZ, cell and input reference."""
    root = Path(project_root).expanduser().resolve()
    campaign = Path(campaign_directory).resolve()
    directory = Path(candidate_directory).resolve()

    for location in (campaign, directory):
        try:
            location.relative_to(root)
        except ValueError:
            _reject("Structure evidence directory escapes project.")

    try:
        directory.relative_to(campaign)
    except ValueError:
        _reject("Candidate directory escapes campaign.")

    structure_path = _contained_file(
        root, campaign / "structure.json"
    )
    input_path = _contained_file(
        root, directory / input_filename
    )
    xyz_path = _contained_file(
        root, directory / coordinate_filename
    )

    if input_path.parent != directory or xyz_path.parent != directory:
        _reject("Candidate evidence file escapes candidate directory.")

    try:
        structure = Structure.from_file(structure_path)
    except Exception as exc:
        raise ConvergenceEvidenceValidationError(
            "Cannot load canonical periodic structure."
        ) from exc

    if not structure.is_ordered or len(structure) == 0:
        _reject("Canonical structure must be ordered and nonempty.")

    text = input_path.read_text(encoding="utf-8")

    # Fail closed: preprocessing can change effective CP2K settings.
    if re.search(r"(?im)^\s*@\s*\w+", text):
        _reject("CP2K preprocessing directives require resolution.")

    if re.search(r"\$\{?[\w]", text):
        _reject("Unresolved CP2K variable substitution.")

    topology = _section(text, "TOPOLOGY")
    cell = _section(text, "CELL")

    coordinate_reference = _keyword(
        topology, "COORD_FILE_NAME"
    )

    try:
        reference_tokens = shlex.split(coordinate_reference)
    except ValueError:
        _reject("Invalid CP2K coordinate reference.")

    if (
        len(reference_tokens) != 1
        or reference_tokens[0] != coordinate_filename
    ):
        _reject(
            "CP2K coordinate reference differs from manifest."
        )

    if _keyword(topology, "COORD_FILE_FORMAT").upper() != "XYZ":
        _reject("CP2K coordinate format must be XYZ.")

    if _keyword(cell, "PERIODIC").upper() != "XYZ":
        _reject("CP2K cell must be periodic in XYZ.")

    for index, name in enumerate(("A", "B", "C")):
        actual = _vector(_keyword(cell, name), name)
        expected = structure.lattice.matrix[index]

        for a, b in zip(actual, expected, strict=True):
            if abs(a - float(b)) > CELL_TOLERANCE_ANGSTROM:
                _reject(f"CP2K {name} cell vector mismatch.")

    _verify_xyz(xyz_path, structure)
