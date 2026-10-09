"""Regression tests for CP2K convergence evidence validation."""

import json
from copy import deepcopy

import pytest

from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
    validate_convergence_evidence,
)
from nsdw.workflows.convergence.manifest import (
    ConvergenceManifestCandidate,
    ConvergenceStudyManifest,
    write_convergence_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCriterion,
    ConvergenceParameter,
)


def _campaign(
    root,
    *,
    parameter=ConvergenceParameter.CUTOFF,
):
    directory = root / "campaign"
    directory.mkdir(parents=True, exist_ok=True)

    if parameter == ConvergenceParameter.KPOINTS:
        values = ((1, 1, 1), (2, 2, 2), (3, 3, 3))
        labels = ("1x1x1", "2x2x2", "3x3x3")
        expected = (2, 2, 2)
    else:
        values = (
            Quantity(value=400.0, unit="Ry"),
            Quantity(value=560.0, unit="Ry"),
            Quantity(value=700.0, unit="Ry"),
        )
        labels = ("400-Ry", "560-Ry", "700-Ry")
        expected = 560.0

    manifest = ConvergenceStudyManifest(
        calculator="cp2k",
        parameter=parameter,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=0.001
        ),
        candidates=tuple(
            ConvergenceManifestCandidate(
                label=label,
                order=index,
                value=value,
                directory=label,
                input_file="calculation.inp",
                coordinate_file="structure.xyz",
            )
            for index, (label, value) in enumerate(
                zip(labels, values, strict=True)
            )
        ),
    )

    write_convergence_manifest(
        manifest=manifest,
        path=directory / "manifest.json",
    )

    candidates = []

    for index, (label, value) in enumerate(
        zip(labels, values, strict=True)
    ):
        selected = index == 1

        candidates.append({
            "label": label,
            "order": index,
            "value": (
                value.model_dump(mode="json")
                if isinstance(value, Quantity)
                else list(value)
            ),
            "total_energy_ev": -100.0 + index * 0.0001,
            "energy_ev_per_atom": -10.0 + index * 0.00001,
            "adjacent_energy_difference_ev_per_atom": (
                None if index == 0 else 0.00001
            ),
            "reference_energy_difference_ev_per_atom": (
                0.00002 if index != 2 else 0.0
            ),
            "maximum_higher_cost_energy_difference_ev_per_atom": (
                0.0002 if index != 2 else None
            ),
            "tail_stable": (
                True if index != 2 else None
            ),
            "selected": selected,
        })


    # Derive all reported energy differences from the actual energies.
    energies = [-10.0, -9.99999, -9.99998]

    for index, row in enumerate(candidates):
        energy = energies[index]
        row["energy_ev_per_atom"] = energy
        row["total_energy_ev"] = energy * 10

        row["adjacent_energy_difference_ev_per_atom"] = (
            abs(energy - energies[index + 1])
            if index + 1 < len(energies)
            else None
        )

        row["reference_energy_difference_ev_per_atom"] = abs(
            energy - energies[-1]
        )

        tail = (
            max(abs(energy - higher) for higher in energies[index + 1:])
            if index + 1 < len(energies)
            else None
        )

        row["maximum_higher_cost_energy_difference_ev_per_atom"] = tail
        row["tail_stable"] = (
            tail <= 0.001 if tail is not None else None
        )

    # Candidate zero is also stable, so make it unstable by
    # increasing its energy separation from the converged tail.
    energies[0] = -10.01
    candidates[0]["energy_ev_per_atom"] = energies[0]
    candidates[0]["total_energy_ev"] = energies[0] * 10

    for index, row in enumerate(candidates):
        energy = energies[index]
        row["adjacent_energy_difference_ev_per_atom"] = (
            abs(energy - energies[index + 1])
            if index + 1 < len(energies)
            else None
        )
        row["reference_energy_difference_ev_per_atom"] = abs(
            energy - energies[-1]
        )
        tail = (
            max(abs(energy - higher) for higher in energies[index + 1:])
            if index + 1 < len(energies)
            else None
        )
        row["maximum_higher_cost_energy_difference_ev_per_atom"] = tail
        row["tail_stable"] = (
            tail <= 0.001 if tail is not None else None
        )

    report = {
        "parameter": parameter.value,
        "energy_tolerance_ev_per_atom": 0.001,
        "structure_hash": "a" * 64,
        "n_atoms": 10,
        "reference_candidate_label": labels[-1],
        "selected_candidate_label": labels[1],
        "energy_tolerance_satisfied": True,
        "candidates": candidates,
    }

    report_path = directory / "convergence-report.json"

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    return report_path, report, expected


def _validate(root, report_path, expected, parameter="cutoff"):
    return validate_convergence_evidence(
        project_root=root,
        report_path=report_path,
        expected_parameter=parameter,
        expected_value=expected,
        expected_structure_hash="a" * 64,
        expected_n_atoms=10,
    )


def test_valid_cutoff_evidence(tmp_path):
    path, report, expected = _campaign(tmp_path)

    result = _validate(tmp_path, path, expected)

    assert result == report


def test_valid_kpoint_evidence(tmp_path):
    path, report, expected = _campaign(
        tmp_path,
        parameter=ConvergenceParameter.KPOINTS,
    )

    result = _validate(
        tmp_path, path, expected, parameter="kpoints"
    )

    assert result == report


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(parameter="relative_cutoff"),
        lambda r: r.update(
            energy_tolerance_satisfied=False
        ),
        lambda r: r.update(
            energy_tolerance_ev_per_atom=-0.001
        ),
        lambda r: r.update(
            structure_hash="b" * 64
        ),
        lambda r: r.update(n_atoms=11),
        lambda r: r.update(
            selected_candidate_label="missing"
        ),
        lambda r: r["candidates"][1].update(
            value={"value": 600.0, "unit": "Ry"}
        ),
        lambda r: r["candidates"][1].update(
            tail_stable=False
        ),
        lambda r: r["candidates"][1].update(
            maximum_higher_cost_energy_difference_ev_per_atom=0.01
        ),
        lambda r: r["candidates"][1].update(
            total_energy_ev=float("nan")
        ),
        lambda r: r["candidates"][1].update(
            order=99
        ),
    ],
)
def test_rejects_invalid_report(tmp_path, mutation):
    path, report, expected = _campaign(tmp_path)

    corrupted = deepcopy(report)
    mutation(corrupted)

    path.write_text(
        json.dumps(corrupted) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError
    ):
        _validate(tmp_path, path, expected)


def test_rejects_missing_report(tmp_path):
    path, _, expected = _campaign(tmp_path)
    path.unlink()

    with pytest.raises(
        ConvergenceEvidenceValidationError
    ):
        _validate(tmp_path, path, expected)


def test_rejects_path_outside_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    path, _, expected = _campaign(outside)

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="escapes",
    ):
        _validate(project, path, expected)


def test_rejects_symlink_escape(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    path, _, expected = _campaign(outside)

    link = project / "external-report.json"
    link.symlink_to(path)

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="escapes",
    ):
        _validate(project, link, expected)


def test_rejects_manifest_candidate_mismatch(tmp_path):
    path, _, expected = _campaign(tmp_path)

    manifest_path = path.parent / "manifest.json"
    data = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    data["candidates"][1]["label"] = "incorrect"

    manifest_path.write_text(
        json.dumps(data) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError
    ):
        _validate(tmp_path, path, expected)


def test_rejects_tolerance_different_from_manifest(tmp_path):
    path, report, expected = _campaign(tmp_path)
    report["energy_tolerance_ev_per_atom"] = 0.1
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="tolerance differs",
    ):
        _validate(tmp_path, path, expected)


def test_rejects_falsified_energy_difference(tmp_path):
    path, report, expected = _campaign(tmp_path)
    report["candidates"][1][
        "reference_energy_difference_ev_per_atom"
    ] = 0.0005
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="reference_energy_difference",
    ):
        _validate(tmp_path, path, expected)


def test_rejects_falsified_atom_normalization(tmp_path):
    path, report, expected = _campaign(tmp_path)
    report["candidates"][1]["total_energy_ev"] = -123.0
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="per-atom energies",
    ):
        _validate(tmp_path, path, expected)


def test_rejects_falsified_reference_label(tmp_path):
    path, report, expected = _campaign(tmp_path)
    report["reference_candidate_label"] = "400-Ry"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="reference candidate",
    ):
        _validate(tmp_path, path, expected)
