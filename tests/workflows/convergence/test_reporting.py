from __future__ import annotations

import pytest

from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.analyser import (
    analyse_convergence_against_reference,
    analyse_convergence_energy,
    analyse_convergence_tail_stability,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceCriterion,
    ConvergenceObservation,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)
from nsdw.workflows.convergence.reporting import (
    ConvergenceReport,
    build_convergence_report,
)


STRUCTURE_HASH = "a" * 64


def _cutoff_study() -> ConvergenceStudyDefinition:
    return ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400_ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="500_ry",
                order=1,
                value=Quantity(value=500.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="600_ry",
                order=2,
                value=Quantity(value=600.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="700_ry",
                order=3,
                value=Quantity(value=700.0, unit="Ry"),
            ),
        ],
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=0.001,
        ),
    )


def _observations(
    study: ConvergenceStudyDefinition,
) -> list[ConvergenceObservation]:
    energies_ev_per_atom = (
        -10.0200,
        -10.0006,
        -10.0004,
        -10.0002,
    )
    n_atoms = 10

    return [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy * n_atoms,
            n_atoms=n_atoms,
            structure_hash=STRUCTURE_HASH,
        )
        for candidate, energy in zip(
            study.candidates,
            energies_ev_per_atom,
            strict=True,
        )
    ]


def _build_report():
    study = _cutoff_study()
    observations = _observations(study)

    return build_convergence_report(
        study=study,
        observations=observations,
        adjacent_analysis=analyse_convergence_energy(
            study,
            observations,
        ),
        reference_analysis=analyse_convergence_against_reference(
            study,
            observations,
        ),
        tail_analysis=analyse_convergence_tail_stability(
            study,
            observations,
        ),
    )


def test_report_contains_study_summary():
    report = _build_report()

    assert report.parameter == ConvergenceParameter.CUTOFF
    assert report.energy_tolerance_ev_per_atom == pytest.approx(0.001)
    assert report.structure_hash == STRUCTURE_HASH
    assert report.n_atoms == 10
    assert report.reference_candidate_label == "700_ry"
    assert report.selected_candidate_label == "500_ry"
    assert report.energy_tolerance_satisfied is True


def test_report_contains_one_row_per_candidate():
    report = _build_report()

    assert len(report.candidates) == 4
    assert [row.label for row in report.candidates] == [
        "400_ry",
        "500_ry",
        "600_ry",
        "700_ry",
    ]
    assert [row.order for row in report.candidates] == [0, 1, 2, 3]


def test_report_preserves_structured_candidate_values():
    report = _build_report()

    assert report.candidates[0].value == Quantity(
        value=400.0,
        unit="Ry",
    )
    assert report.candidates[-1].value == Quantity(
        value=700.0,
        unit="Ry",
    )


def test_report_contains_energy_data():
    report = _build_report()

    assert report.candidates[0].total_energy_ev == pytest.approx(
        -100.200
    )
    assert report.candidates[0].energy_ev_per_atom == pytest.approx(
        -10.0200
    )
    assert report.candidates[3].energy_ev_per_atom == pytest.approx(
        -10.0002
    )


def test_report_distinguishes_adjacent_reference_and_tail_differences():
    report = _build_report()

    candidate = report.candidates[1]

    assert candidate.adjacent_energy_difference_ev_per_atom == pytest.approx(
        0.0002
    )
    assert candidate.reference_energy_difference_ev_per_atom == pytest.approx(
        0.0004
    )
    assert (
        candidate.maximum_higher_cost_energy_difference_ev_per_atom
        == pytest.approx(0.0004)
    )
    assert candidate.tail_stable is True


def test_report_final_candidate_has_no_tail_stability_decision():
    report = _build_report()

    final_candidate = report.candidates[-1]

    assert (
        final_candidate.adjacent_energy_difference_ev_per_atom
        is None
    )
    assert final_candidate.reference_energy_difference_ev_per_atom == pytest.approx(
        0.0
    )
    assert (
        final_candidate.maximum_higher_cost_energy_difference_ev_per_atom
        is None
    )
    assert final_candidate.tail_stable is None


def test_report_marks_selected_candidate():
    report = _build_report()

    selected = [
        row.label
        for row in report.candidates
        if row.selected
    ]

    assert selected == ["500_ry"]

def test_report_serialises_to_json_compatible_dict():
    report = _build_report()

    data = report.to_dict()

    assert data["parameter"] == "cutoff"
    assert data["energy_tolerance_ev_per_atom"] == pytest.approx(
        0.001
    )
    assert data["structure_hash"] == STRUCTURE_HASH
    assert data["n_atoms"] == 10
    assert data["reference_candidate_label"] == "700_ry"
    assert data["selected_candidate_label"] == "500_ry"
    assert data["energy_tolerance_satisfied"] is True

    assert data["candidates"][0]["value"] == {
        "value": 400.0,
        "unit": "Ry",
    }
    assert data["candidates"][1]["tail_stable"] is True
    assert data["candidates"][1]["selected"] is True


def test_report_writes_json(tmp_path):
    report = _build_report()
    output_path = tmp_path / "convergence-report.json"

    report.write_json(output_path)

    assert output_path.exists()

    import json

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["parameter"] == "cutoff"
    assert data["selected_candidate_label"] == "500_ry"
    assert len(data["candidates"]) == 4


def test_report_writes_csv(tmp_path):
    report = _build_report()
    output_path = tmp_path / "convergence-report.csv"

    report.write_csv(output_path)

    assert output_path.exists()

    import csv

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["label"] == "400_ry"
    assert rows[0]["value"] == "400.0"
    assert rows[0]["value_unit"] == "Ry"
    assert rows[1]["selected"] == "True"
    assert rows[1]["tail_stable"] == "True"


def test_csv_preserves_missing_final_tail_values(tmp_path):
    report = _build_report()
    output_path = tmp_path / "convergence-report.csv"

    report.write_csv(output_path)

    import csv

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    final_row = rows[-1]

    assert final_row["label"] == "700_ry"
    assert (
        final_row[
            "maximum_higher_cost_energy_difference_ev_per_atom"
        ]
        == ""
    )
    assert final_row["tail_stable"] == ""


def test_json_export_does_not_change_scientific_decision(tmp_path):
    report = _build_report()
    output_path = tmp_path / "convergence-report.json"

    report.write_json(output_path)

    assert report.selected_candidate_label == "500_ry"
    assert report.energy_tolerance_satisfied is True

def _basis_study() -> ConvergenceStudyDefinition:
    return ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.BASIS,
        candidates=[
            ConvergenceCandidate(
                label="DZVP",
                order=0,
                value="DZVP",
            ),
            ConvergenceCandidate(
                label="TZVP",
                order=1,
                value="TZVP",
            ),
            ConvergenceCandidate(
                label="TZV2P",
                order=2,
                value="TZV2P",
            ),
        ],
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=0.001,
        ),
    )


def _kpoints_study() -> ConvergenceStudyDefinition:
    return ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.KPOINTS,
        candidates=[
            ConvergenceCandidate(
                label="2x2x1",
                order=0,
                value=(2, 2, 1),
            ),
            ConvergenceCandidate(
                label="4x4x2",
                order=1,
                value=(4, 4, 2),
            ),
            ConvergenceCandidate(
                label="6x6x3",
                order=2,
                value=(6, 6, 3),
            ),
        ],
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=0.001,
        ),
    )


def _generic_report(
    study: ConvergenceStudyDefinition,
) -> ConvergenceReport:
    energies_ev_per_atom = (
        -10.0100,
        -10.0004,
        -10.0002,
    )
    n_atoms = 10

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy * n_atoms,
            n_atoms=n_atoms,
            structure_hash=STRUCTURE_HASH,
        )
        for candidate, energy in zip(
            study.candidates,
            energies_ev_per_atom,
            strict=True,
        )
    ]

    return build_convergence_report(
        study=study,
        observations=observations,
        adjacent_analysis=analyse_convergence_energy(
            study,
            observations,
        ),
        reference_analysis=analyse_convergence_against_reference(
            study,
            observations,
        ),
        tail_analysis=analyse_convergence_tail_stability(
            study,
            observations,
        ),
    )


def test_basis_report_preserves_categorical_values():
    report = _generic_report(_basis_study())

    assert report.parameter == ConvergenceParameter.BASIS
    assert [row.value for row in report.candidates] == [
        "DZVP",
        "TZVP",
        "TZV2P",
    ]
    assert report.selected_candidate_label == "TZVP"


def test_basis_json_preserves_categorical_values():
    report = _generic_report(_basis_study())

    data = report.to_dict()

    assert data["parameter"] == "basis"
    assert data["candidates"][1]["value"] == "TZVP"


def test_kpoints_report_preserves_mesh_values():
    report = _generic_report(_kpoints_study())

    assert report.parameter == ConvergenceParameter.KPOINTS
    assert report.candidates[1].value == (4, 4, 2)
    assert report.selected_candidate_label == "4x4x2"


def test_kpoints_json_serialises_mesh_as_list():
    report = _generic_report(_kpoints_study())

    data = report.to_dict()

    assert data["parameter"] == "kpoints"
    assert data["candidates"][1]["value"] == [4, 4, 2]


def test_kpoints_csv_serialises_mesh_readably(tmp_path):
    report = _generic_report(_kpoints_study())
    output_path = tmp_path / "kpoints.csv"

    report.write_csv(output_path)

    import csv

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows[1]["value"] == "4x4x2"
    assert rows[1]["value_unit"] == ""

def test_report_writes_convergence_plot(tmp_path):
    report = _build_report()
    output_path = tmp_path / "convergence.png"

    report.write_plot(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_report_writes_svg_convergence_plot(tmp_path):
    report = _build_report()
    output_path = tmp_path / "convergence.svg"

    report.write_plot(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_supports_basis_candidates(tmp_path):
    report = _generic_report(_basis_study())
    output_path = tmp_path / "basis-convergence.png"

    report.write_plot(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_supports_kpoint_candidates(tmp_path):
    report = _generic_report(_kpoints_study())
    output_path = tmp_path / "kpoints-convergence.png"

    report.write_plot(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_rejects_unsupported_output_format(tmp_path):
    report = _build_report()
    output_path = tmp_path / "convergence.txt"

    with pytest.raises(
        ValueError,
        match="Unsupported convergence plot format",
    ):
        report.write_plot(output_path)