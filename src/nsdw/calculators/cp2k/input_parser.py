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
    """Parse the exchange-correlation functional from CP2K input."""

    section = _section(
        text,
        "XC_FUNCTIONAL",
    )

    # Handle named compact forms such as:
    #
    # &XC_FUNCTIONAL PBE
    # &END XC_FUNCTIONAL
    named_match = re.search(
        r"&XC_FUNCTIONAL\s+([^\s&]+)",
        text,
        re.IGNORECASE,
    )

    if named_match is not None:
        return named_match.group(1)

    if section is None:
        return None

    upper_section = section.upper()

    # Explicit PBEsol form:
    #
    # &XC_FUNCTIONAL
    #   &GGA_X_PBE_SOL
    #   &END GGA_X_PBE_SOL
    #   &GGA_C_PBE_SOL
    #   &END GGA_C_PBE_SOL
    # &END XC_FUNCTIONAL
    if (
        "&GGA_X_PBE_SOL" in upper_section
        and "&GGA_C_PBE_SOL" in upper_section
    ):
        return "PBEsol"

    # Explicit PBE exchange/correlation form.
    if (
        "&GGA_X_PBE" in upper_section
        and "&GGA_C_PBE" in upper_section
        and "&GGA_X_PBE_SOL" not in upper_section
        and "&GGA_C_PBE_SOL" not in upper_section
    ):
        return "PBE"

    return None


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




def _parse_scf_solver(text: str) -> str | None:
    """Detect an explicitly configured SCF solver."""
    stack: list[str] = []
    solvers: list[str] = []

    for original in text.splitlines():
        line = original.split("!", 1)[0].split("#", 1)[0].strip()

        if not line.startswith("&"):
            continue

        parts = line.split()
        marker = parts[0].upper()

        if marker == "&END" or marker.startswith("&END_"):
            if stack:
                stack.pop()
            continue

        if marker.startswith("&END"):
            if stack:
                stack.pop()
            continue

        section = marker[1:]

        if section in {"OT", "DIAGONALIZATION"}:
            if stack and stack[-1] == "SCF":
                solvers.append(section)

        stack.append(section)

    if len(solvers) > 1:
        raise CP2KInputParseError(
            "Conflicting or repeated SCF solver sections."
        )

    return solvers[0] if solvers else None


def _scf_keyword_values(text: str) -> dict[str, str]:
    """Extract SCF keywords using their exact nested section paths.

    This is a deliberately restricted parser for generated CP2K
    SCF inputs, not a general CP2K preprocessing implementation.
    """
    stack: list[str] = []
    values: dict[str, str] = {}

    fields = {
        ("SCF",): {
            "SCF_GUESS",
            "EPS_SCF",
            "MAX_SCF",
        },
        ("SCF", "OUTER_SCF"): {
            "MAX_SCF",
        },
        ("SCF", "OT"): {
            "MINIMIZER",
            "PRECONDITIONER",
            "ENERGY_GAP",
        },
    }

    for original in text.splitlines():
        line = original.split("!", 1)[0].split("#", 1)[0].strip()

        if not line:
            continue

        if line.startswith("&"):
            parts = line.split()
            marker = parts[0].upper()

            if marker.startswith("&END"):
                if not stack:
                    raise CP2KInputParseError(
                        "Unexpected CP2K section terminator."
                    )

                expected = stack[-1]
                explicit = (
                    parts[1].upper()
                    if marker == "&END" and len(parts) > 1
                    else marker.removeprefix("&END_")
                    if marker.startswith("&END_")
                    else None
                )

                if explicit and explicit != expected:
                    raise CP2KInputParseError(
                        "Mismatched CP2K section terminator."
                    )

                stack.pop()
            else:
                stack.append(marker[1:])

            continue

        parts = line.split(None, 1)

        if len(parts) != 2:
            continue

        keyword, value = parts
        keyword = keyword.upper()

        # Match the trailing SCF path, regardless of the
        # enclosing FORCE_EVAL/DFT hierarchy.
        for suffix, accepted in fields.items():
            if (
                len(stack) >= len(suffix)
                and tuple(stack[-len(suffix):]) == suffix
                and keyword in accepted
            ):
                key = "/".join((*suffix, keyword))

                if key in values:
                    raise CP2KInputParseError(
                        f"Repeated SCF setting: {key}"
                    )

                values[key] = value.strip()
                break

    if stack:
        raise CP2KInputParseError(
            "Unclosed CP2K section in input."
        )

    return values


def parse_cp2k_input_text(
    text: str,
) -> ParsedCP2KInput:
    """Parse scientifically relevant settings from CP2K input text."""

    if not text.strip():
        raise CP2KInputParseError(
            "CP2K input is empty"
        )

    warnings: list[str] = []
    scf_values = _scf_keyword_values(text)

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
        scf_guess=scf_values.get("SCF/SCF_GUESS"),
        max_scf=_parse_int(
            scf_values.get("SCF/MAX_SCF")
        ),
        outer_scf_max=_parse_int(
            scf_values.get("SCF/OUTER_SCF/MAX_SCF")
        ),
        ot_minimizer=scf_values.get("SCF/OT/MINIMIZER"),
        ot_preconditioner=scf_values.get(
            "SCF/OT/PRECONDITIONER"
        ),
        energy_gap=_parse_float(
            scf_values.get("SCF/OT/ENERGY_GAP")
        ),
        k_points=_parse_k_points(
            text
        ),
        scf_solver=_parse_scf_solver(text),
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
