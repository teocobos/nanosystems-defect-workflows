
"""Validate NSDW HPC profiles against AiiDA configuration."""

from __future__ import annotations

from dataclasses import dataclass

from nsdw.execution.profiles.models import HPCProfile


@dataclass(frozen=True)
class HPCProfileValidationResult:
    """Result of validating an HPC execution profile."""

    name: str
    valid: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def validate_hpc_profile(
    profile: HPCProfile,
) -> HPCProfileValidationResult:
    """Validate a stored AiiDA HPC profile without submitting jobs."""

    checks: list[str] = []
    errors: list[str] = []

    try:
        from aiida import load_profile
        from aiida.orm import load_code, load_computer
    except ImportError:
        return HPCProfileValidationResult(
            name=profile.name,
            valid=False,
            checks=(),
            errors=("AiiDA is not installed.",),
        )

    try:
        load_profile(profile.aiida_profile)
        checks.append("AiiDA profile loaded")
    except Exception as exc:
        errors.append(f"Cannot load AiiDA profile: {exc}")
        return HPCProfileValidationResult(
            name=profile.name,
            valid=False,
            checks=tuple(checks),
            errors=tuple(errors),
        )

    computer = None
    code = None

    try:
        computer = load_computer(profile.computer)
        checks.append("AiiDA computer found")
    except Exception as exc:
        errors.append(f"Cannot load computer: {exc}")

    try:
        code = load_code(profile.code)
        checks.append("AiiDA code found")
    except Exception as exc:
        errors.append(f"Cannot load code: {exc}")

    if computer is not None and code is not None:
        code_computer = getattr(code, "computer", None)

        if code_computer is None:
            errors.append(
                "Selected code has no associated computer."
            )
        elif code_computer.uuid != computer.uuid:
            errors.append(
                "Selected code belongs to a different computer."
            )
        else:
            checks.append("Code and computer match")

    return HPCProfileValidationResult(
        name=profile.name,
        valid=not errors,
        checks=tuple(checks),
        errors=tuple(errors),
    )

