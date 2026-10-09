"""Independent validation of CP2K convergence evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)


class ConvergenceEvidenceValidationError(ValueError):
    """Convergence evidence is missing or inconsistent."""


def _contained_path(
    *,
    project_root: Path,
    path: Path,
) -> Path:
    """Resolve an evidence path and enforce project containment."""
    root = project_root.expanduser().resolve()
    resolved = path.expanduser().resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConvergenceEvidenceValidationError(
            f"Evidence path escapes the NSDW project: {path}"
        ) from exc

    if not resolved.is_file():
        raise ConvergenceEvidenceValidationError(
            f"Evidence file does not exist: {resolved}"
        )

    return resolved


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise ConvergenceEvidenceValidationError(
            f"{name} must be numeric."
        )

    result = float(value)

    if not math.isfinite(result):
        raise ConvergenceEvidenceValidationError(
            f"{name} must be finite."
        )

    return result


def _canonical_value(value: Any) -> Any:
    """Normalize report and manifest candidate values."""
    if hasattr(value, "unit") and hasattr(value, "value"):
        return {
            "value": float(value.value),
            "unit": value.unit,
        }

    if isinstance(value, tuple):
        return list(value)

    return value


def validate_convergence_evidence(
    *,
    project_root: str | Path,
    report_path: str | Path,
    expected_parameter: str,
    expected_value: float | tuple[int, int, int],
    expected_structure_hash: str | None = None,
    expected_n_atoms: int | None = None,
) -> dict[str, Any]:
    """Validate a report against its corresponding campaign manifest.

    This verifies report and manifest consistency, not the underlying
    CP2K input files or completed calculation outputs.
    """
    root = Path(project_root).expanduser().resolve()

    requested_report = Path(report_path).expanduser()

    if not requested_report.is_absolute():
        requested_report = root / requested_report

    report_file = _contained_path(
        project_root=root,
        path=requested_report,
    )

    manifest_file = _contained_path(
        project_root=root,
        path=report_file.parent / "manifest.json",
    )

    try:
        report = json.loads(
            report_file.read_text(encoding="utf-8")
        )
        manifest = load_convergence_manifest(manifest_file)
    except (OSError, ValueError) as exc:
        raise ConvergenceEvidenceValidationError(
            "Cannot load convergence report or manifest."
        ) from exc

    if not isinstance(report, dict):
        raise ConvergenceEvidenceValidationError(
            "Convergence report must be a JSON object."
        )

    if manifest.calculator != "cp2k":
        raise ConvergenceEvidenceValidationError(
            "Manifest does not describe a CP2K campaign."
        )

    if (
        report.get("parameter") != expected_parameter
        or manifest.parameter.value != expected_parameter
    ):
        raise ConvergenceEvidenceValidationError(
            "Convergence parameter mismatch."
        )

    tolerance = _numeric(
        report.get("energy_tolerance_ev_per_atom"),
        "energy tolerance",
    )

    if tolerance <= 0:
        raise ConvergenceEvidenceValidationError(
            "Energy tolerance must be positive."
        )

    if report.get("energy_tolerance_satisfied") is not True:
        raise ConvergenceEvidenceValidationError(
            "Convergence tolerance was not satisfied."
        )

    if not isinstance(report.get("structure_hash"), str):
        raise ConvergenceEvidenceValidationError(
            "Missing structure hash."
        )

    n_atoms = report.get("n_atoms")

    if type(n_atoms) is not int or n_atoms <= 0:
        raise ConvergenceEvidenceValidationError(
            "Invalid atom count."
        )

    if (
        expected_structure_hash is not None
        and report["structure_hash"] != expected_structure_hash
    ):
        raise ConvergenceEvidenceValidationError(
            "Structure hash mismatch."
        )

    if (
        expected_n_atoms is not None
        and n_atoms != expected_n_atoms
    ):
        raise ConvergenceEvidenceValidationError(
            "Atom count mismatch."
        )

    candidates = report.get("candidates")

    if not isinstance(candidates, list):
        raise ConvergenceEvidenceValidationError(
            "Report candidates must be a list."
        )

    if len(candidates) != len(manifest.candidates):
        raise ConvergenceEvidenceValidationError(
            "Report and manifest candidate counts differ."
        )

    selected_label = report.get("selected_candidate_label")

    if not isinstance(selected_label, str) or not selected_label:
        raise ConvergenceEvidenceValidationError(
            "Missing selected candidate."
        )

    if len({c.label for c in manifest.candidates}) != len(
        manifest.candidates
    ):
        raise ConvergenceEvidenceValidationError(
            "Duplicate manifest candidate labels."
        )

    report_labels = []

    for recorded, generated in zip(
        candidates,
        manifest.candidates,
        strict=True,
    ):
        if not isinstance(recorded, dict):
            raise ConvergenceEvidenceValidationError(
                "Invalid report candidate."
            )

        if (
            recorded.get("label") != generated.label
            or recorded.get("order") != generated.order
            or recorded.get("value")
            != _canonical_value(generated.value)
        ):
            raise ConvergenceEvidenceValidationError(
                "Report candidate does not match manifest."
            )

        report_labels.append(recorded["label"])

        _numeric(
            recorded.get("total_energy_ev"),
            "candidate total energy",
        )
        _numeric(
            recorded.get("energy_ev_per_atom"),
            "candidate energy per atom",
        )
        _numeric(
            recorded.get("reference_energy_difference_ev_per_atom"),
            "reference energy difference",
        )

    if len(set(report_labels)) != len(report_labels):
        raise ConvergenceEvidenceValidationError(
            "Duplicate report candidate labels."
        )

    selected = [
        candidate
        for candidate in candidates
        if candidate["label"] == selected_label
        and candidate.get("selected") is True
    ]

    if (
        len(selected) != 1
        or sum(c.get("selected") is True for c in candidates) != 1
    ):
        raise ConvergenceEvidenceValidationError(
            "Selected candidate is missing or inconsistent."
        )

    candidate = selected[0]
    value = candidate["value"]

    if expected_parameter == "kpoints":
        expected = list(expected_value)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(v) is not int or v <= 0 for v in value)
            or value != expected
        ):
            raise ConvergenceEvidenceValidationError(
                "Selected k-point mesh mismatch."
            )
    else:
        if (
            not isinstance(value, dict)
            or value.get("unit") != "Ry"
            or _numeric(value.get("value"), "selected value")
            != expected_value
        ):
            raise ConvergenceEvidenceValidationError(
                "Selected numerical parameter mismatch."
            )

    difference = _numeric(
        candidate.get(
            "maximum_higher_cost_energy_difference_ev_per_atom"
        ),
        "selected tail energy difference",
    )

    if (
        candidate.get("tail_stable") is not True
        or difference < 0
        or difference > tolerance
    ):
        raise ConvergenceEvidenceValidationError(
            "Selected candidate is not numerically stable."
        )


    # Independently verify the complete numerical convergence report.
    manifest_tolerance = _numeric(
        manifest.criterion.energy_tolerance_ev_per_atom,
        "manifest energy tolerance",
    )

    if tolerance != manifest_tolerance:
        raise ConvergenceEvidenceValidationError(
            "Report tolerance differs from manifest criterion."
        )

    if report.get("reference_candidate_label") != candidates[-1]["label"]:
        raise ConvergenceEvidenceValidationError(
            "Incorrect reference candidate."
        )

    energies = [
        _numeric(row["energy_ev_per_atom"], "energy per atom")
        for row in candidates
    ]

    for row, energy in zip(candidates, energies, strict=True):
        total = _numeric(row["total_energy_ev"], "total energy")

        if not math.isclose(
            total,
            energy * n_atoms,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ConvergenceEvidenceValidationError(
                "Total and per-atom energies disagree."
            )

    stable_labels = []

    for index, row in enumerate(candidates):
        adjacent = (
            abs(energies[index] - energies[index + 1])
            if index + 1 < len(candidates)
            else None
        )

        reference = abs(energies[index] - energies[-1])

        tail = (
            max(
                abs(energies[index] - higher)
                for higher in energies[index + 1:]
            )
            if index + 1 < len(candidates)
            else None
        )

        def require_difference(field, expected):
            recorded = row.get(field)

            if expected is None:
                if recorded is not None:
                    raise ConvergenceEvidenceValidationError(
                        f"{field} must be null for reference candidate."
                    )
                return

            actual = _numeric(recorded, field)

            if not math.isclose(
                actual,
                expected,
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise ConvergenceEvidenceValidationError(
                    f"Incorrect reported {field}."
                )

        require_difference(
            "adjacent_energy_difference_ev_per_atom",
            adjacent,
        )
        require_difference(
            "reference_energy_difference_ev_per_atom",
            reference,
        )
        require_difference(
            "maximum_higher_cost_energy_difference_ev_per_atom",
            tail,
        )

        stable = tail <= tolerance if tail is not None else None

        if row.get("tail_stable") is not stable:
            raise ConvergenceEvidenceValidationError(
                "Reported tail stability disagrees with energies."
            )

        if stable is True:
            stable_labels.append(row["label"])

    expected_selection = (
        stable_labels[0] if stable_labels else None
    )

    if report.get("selected_candidate_label") != expected_selection:
        raise ConvergenceEvidenceValidationError(
            "Selected candidate is not the least expensive stable candidate."
        )

    if report.get("energy_tolerance_satisfied") is not bool(
        stable_labels
    ):
        raise ConvergenceEvidenceValidationError(
            "Overall convergence decision disagrees with energies."
        )

    return report
