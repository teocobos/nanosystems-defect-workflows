"""Parser for CP2K text output."""

from __future__ import annotations

import re
from pathlib import Path

from nsdw.calculators.cp2k.models import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KEnergy,
    ParsedCP2KResult,
    ParsedCP2KSCF,
)


class CP2KParseError(RuntimeError):
    """Raised when a CP2K output cannot be parsed."""


_VERSION_RE = re.compile(
    r"CP2K\|\s+version string:\s+CP2K version\s+(\S+)"
)

_PROJECT_RE = re.compile(
    r"GLOBAL\|\s+Project name\s+(\S+)"
)

_RUN_TYPE_RE = re.compile(
    r"GLOBAL\|\s+Run type\s+(\S+)"
)

_CHARGE_RE = re.compile(
    r"DFT\|\s+Charge\s+(-?\d+)"
)

_MULTIPLICITY_RE = re.compile(
    r"DFT\|\s+Multiplicity\s+(\d+)"
)

_ENERGY_RE = re.compile(
    r"ENERGY\|.*?energy\s+"
    r"\[(?:a\.u\.|hartree)\]\s*:?\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)

_SCF_CONVERGED_RE = re.compile(
    r"SCF run converged in\s+(\d+)\s+steps"
)

_SCF_NOT_CONVERGED_RE = re.compile(
    r"SCF run NOT converged",
    re.IGNORECASE,
)


def _last_match(
    pattern: re.Pattern[str],
    text: str,
) -> re.Match[str] | None:
    """Return the final match for a repeated CP2K quantity."""

    matches = list(pattern.finditer(text))

    if not matches:
        return None

    return matches[-1]


def parse_cp2k_text(text: str) -> ParsedCP2KResult:
    """Parse CP2K output text into a typed intermediate result."""

    if not text.strip():
        raise CP2KParseError(
            "CP2K output is empty"
        )

    warnings: list[str] = []

    version_match = _VERSION_RE.search(text)
    project_match = _PROJECT_RE.search(text)
    run_type_match = _RUN_TYPE_RE.search(text)
    charge_match = _CHARGE_RE.search(text)
    multiplicity_match = _MULTIPLICITY_RE.search(text)

    energy_match = _last_match(
        _ENERGY_RE,
        text,
    )

    converged_match = _last_match(
        _SCF_CONVERGED_RE,
        text,
    )

    not_converged_match = _last_match(
        _SCF_NOT_CONVERGED_RE,
        text,
    )

    cp2k_version = (
        version_match.group(1)
        if version_match
        else None
    )

    project_name = (
        project_match.group(1)
        if project_match
        else None
    )

    charge = (
        int(charge_match.group(1))
        if charge_match
        else None
    )

    multiplicity = (
        int(multiplicity_match.group(1))
        if multiplicity_match
        else None
    )

    run_type = None

    if run_type_match:
        raw_run_type = run_type_match.group(1)

        try:
            run_type = CP2KRunType(
                raw_run_type
            )
        except ValueError:
            warnings.append(
                "Unsupported CP2K run type: "
                f"{raw_run_type}"
            )

    total_energy = (
        float(energy_match.group(1))
        if energy_match
        else None
    )

    if converged_match:
        scf_status = CP2KSCFStatus.CONVERGED
        scf_iterations = int(
            converged_match.group(1)
        )

    elif not_converged_match:
        scf_status = CP2KSCFStatus.NOT_CONVERGED
        scf_iterations = None

    else:
        scf_status = CP2KSCFStatus.UNKNOWN
        scf_iterations = None

    normal_termination = (
        "PROGRAM ENDED AT" in text
    )

    if total_energy is None:
        warnings.append(
            "No total FORCE_EVAL energy found"
        )

    if scf_status == CP2KSCFStatus.UNKNOWN:
        warnings.append(
            "Final SCF status could not be determined"
        )

    if not normal_termination:
        warnings.append(
            "Normal CP2K termination marker not found"
        )

    return ParsedCP2KResult(
        cp2k_version=cp2k_version,
        run_type=run_type,
        project_name=project_name,
        charge=charge,
        multiplicity=multiplicity,
        energy=ParsedCP2KEnergy(
            total_energy_hartree=total_energy,
        ),
        scf=ParsedCP2KSCF(
            status=scf_status,
            iterations=scf_iterations,
        ),
        normal_termination=normal_termination,
        warnings=tuple(warnings),
    )


def parse_cp2k_output(
    path: str | Path,
) -> ParsedCP2KResult:
    """Parse a CP2K output file."""

    output_path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not output_path.is_file():
        raise FileNotFoundError(
            f"CP2K output file not found: "
            f"{output_path}"
        )

    text = output_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return parse_cp2k_text(text)
