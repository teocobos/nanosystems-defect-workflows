"""Strict CP2K single-point output provenance validation."""

import math
import re
from pathlib import Path

from nsdw.calculators.cp2k.parser import parse_cp2k_output
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


_SCF_SUCCESS = re.compile(
    r"SCF run converged in\s+\d+\s+steps",
    re.IGNORECASE,
)

_SCF_FAILURE = re.compile(
    r"SCF run NOT converged",
    re.IGNORECASE,
)

_ENERGY = re.compile(
    r"^\s*ENERGY\|\s*Total FORCE_EVAL\b.*?"
    r"energy\s*(?:\[[^\]]+\])?\s*:?\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[EeDd][-+]?\d+)?)\s*$",
    re.IGNORECASE,
)

_TERMINATION = re.compile(
    r"PROGRAM ENDED AT",
    re.IGNORECASE,
)


def validate_cp2k_single_point_output(
    path: str | Path,
) -> float:
    """Return the verified final energy in Hartree.

    Fail closed unless the output contains exactly one successful
    SCF cycle, one FORCE_EVAL energy and one normal termination,
    occurring in that order.

    This deliberately does not support multistep calculations.
    """
    output_path = Path(path).expanduser().resolve()

    if not output_path.is_file():
        raise ConvergenceEvidenceValidationError(
            f"CP2K output is missing: {output_path}"
        )

    lines = output_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    events = []
    for line_number, line in enumerate(lines, start=1):
        if _SCF_SUCCESS.search(line):
            events.append((line_number, "converged", None))
        elif _SCF_FAILURE.search(line):
            events.append((line_number, "failed", None))

        energy_match = _ENERGY.search(line)
        if energy_match:
            value = float(
                energy_match.group(1).replace("D", "E").replace("d", "e")
            )
            events.append((line_number, "energy", value))

        if _TERMINATION.search(line):
            events.append((line_number, "termination", None))

    event_types = [kind for _, kind, _ in events]

    if event_types != [
        "converged",
        "energy",
        "termination",
    ]:
        raise ConvergenceEvidenceValidationError(
            "Ambiguous CP2K single-point event sequence: "
            f"{event_types!r}."
        )

    verified_energy = events[1][2]

    if verified_energy is None or not math.isfinite(
        verified_energy
    ):
        raise ConvergenceEvidenceValidationError(
            "Final CP2K energy is not finite."
        )

    try:
        parsed = parse_cp2k_output(output_path)
    except Exception as exc:
        raise ConvergenceEvidenceValidationError(
            "Cannot parse CP2K single-point output."
        ) from exc

    if (
        parsed.run_type is None
        or parsed.run_type.value != "ENERGY"
    ):
        raise ConvergenceEvidenceValidationError(
            "Convergence evidence requires CP2K ENERGY run type."
        )

    parsed_energy = parsed.energy.total_energy_hartree

    if (
        parsed_energy is None
        or not math.isfinite(parsed_energy)
        or not math.isclose(
            parsed_energy,
            verified_energy,
            rel_tol=1e-12,
            abs_tol=1e-10,
        )
    ):
        raise ConvergenceEvidenceValidationError(
            "Parsed energy disagrees with final SCF energy evidence."
        )

    return verified_energy
