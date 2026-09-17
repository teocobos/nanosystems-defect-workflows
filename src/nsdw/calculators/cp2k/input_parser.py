"""Parser for scientifically relevant CP2K input settings."""

from __future__ import annotations

import re
from pathlib import Path

from nsdw.calculators.cp2k.models import (
    CP2KRunType,
    ParsedCP2KInput,
    ParsedCP2KKind,
)


class CP2KInputParseError(RuntimeError):
    """Raised when a CP2K input cannot be parsed."""


def _keyword(
    text: str,
    keyword: str,
) -> str | None:
    """Return the first value associated with a CP2K keyword."""

    pattern = re.compile(
        rf"^\s*{re.escape(keyword)}\s+(.+?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    match = pattern.search(text)

    if match is None:
        return None

    return match.group(1).strip()


def _section(
    text: str,
    name: str,
) -> str | None:
    """Return the contents of the first simple CP2K section."""

    pattern = re.compile(
        rf"&{re.escape(name)}\b[^\n]*\n"
        rf"(.*?)"
        rf"&END\s+{re.escape(name)}\b",
        re.DOTALL | re.IGNORECASE,
    )

    match = pattern.search(text)

    if match is None:
        return None

    return match.group(1)


def _parse_run_type(
    value: str | None,
    warnings: list[str],
) -> CP2KRunType | None:
    if value is None:
        return None

    raw = value.split()[0].upper()

    try:
        return CP2KRunType(raw)

    except ValueError:
        warnings.append(
            f"Unsupported CP2K run type: {raw}"
        )
        return None


def _parse_float(
    value: str | None,
) -> float | None:
    if value is None:
        return None

    return float(
        value.split()[0]
    )


def _parse_int(
    value: str | None,
) -> int | None:
    if value is None:
        return None

    return int(
        value.split()[0]
    )


def _parse_k_points(
    text: str,
) -> tuple[int, int, int] | None:
    section = _section(
        text,
        "KPOINTS",
    )

    if section is None:
        return None

    scheme = _keyword(
        section,
        "SCHEME",
    )

    if scheme is None:
        return None

    parts = scheme.split()

    if (
        len(parts) >= 4
        and parts[0].upper()
        in {
            "MONKHORST-PACK",
            "MONKHORST_PACK",
        }
    ):
        return (
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
        )

    return None


def _parse_xc_functional(
    text: str,
) -> str | None:
    match = re.search(
        r"&XC_FUNCTIONAL(?:\s+([^\s&]+))?",
        text,
        re.IGNORECASE,
    )

    if match is None:
        return None

    return (
        match.group(1)
        if match.group(1)
        else None
    )


def _parse_kinds(
    text: str,
) -> tuple[ParsedCP2KKind, ...]:
    """Parse species-specific CP2K KIND assignments."""

    pattern = re.compile(
        r"&KIND\s+(\S+)\s*\n"
        r"(.*?)"
        r"&END\s+KIND\b",
        re.DOTALL | re.IGNORECASE,
    )

    kinds: list[ParsedCP2KKind] = []

    for match in pattern.finditer(text):
        kind_name = match.group(1)
        body = match.group(2)

        kinds.append(
            ParsedCP2KKind(
                kind=kind_name,
                element=_keyword(
                    body,
                    "ELEMENT",
                ),
                basis_set=_keyword(
                    body,
                    "BASIS_SET",
                ),
                potential=_keyword(
                    body,
                    "POTENTIAL",
                ),
            )
        )

    return tuple(kinds)


def parse_cp2k_input_text(
    text: str,
) -> ParsedCP2KInput:
    """Parse scientifically relevant settings from CP2K input text."""

    if not text.strip():
        raise CP2KInputParseError(
            "CP2K input is empty"
        )

    warnings: list[str] = []

    mgrid = _section(
        text,
        "MGRID",
    )

    scf = _section(
        text,
        "SCF",
    )

    return ParsedCP2KInput(
        project_name=_keyword(
            text,
            "PROJECT",
        ),
        run_type=_parse_run_type(
            _keyword(
                text,
                "RUN_TYPE",
            ),
            warnings,
        ),
        charge=_parse_int(
            _keyword(
                text,
                "CHARGE",
            )
        ),
        multiplicity=_parse_int(
            _keyword(
                text,
                "MULTIPLICITY",
            )
        ),
        xc_functional=_parse_xc_functional(
            text
        ),
        cutoff_ry=_parse_float(
            _keyword(
                mgrid or "",
                "CUTOFF",
            )
        ),
        relative_cutoff_ry=_parse_float(
            _keyword(
                mgrid or "",
                "REL_CUTOFF",
            )
        ),
        eps_scf=_parse_float(
            _keyword(
                scf or "",
                "EPS_SCF",
            )
        ),
        k_points=_parse_k_points(
            text
        ),
        kinds=_parse_kinds(
            text
        ),
        basis_set_file=_keyword(
            text,
            "BASIS_SET_FILE_NAME",
        ),
        potential_file=_keyword(
            text,
            "POTENTIAL_FILE_NAME",
        ),
        admm=(
            "&AUXILIARY_DENSITY_MATRIX_METHOD"
            in text.upper()
        ),
        warnings=tuple(warnings),
    )


def parse_cp2k_input(
    path: str | Path,
) -> ParsedCP2KInput:
    """Parse a CP2K input file."""

    input_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CP2K input file not found: {input_path}"
        )

    text = input_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return parse_cp2k_input_text(
        text
    )
