from nsdw.calculators.cp2k.generation_models import CP2KBasisPotentialConfig
"""Tests for methodology provenance-path validation."""

from pathlib import Path

import pytest

from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
)
from nsdw.calculators.cp2k.generation_models import CP2KSCFConfig
from nsdw.workflows.convergence.cp2k_methodology_evidence import (
    resolve_methodology_evidence_paths,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "project"
    workflow = root / "workflows" / "convergence"
    workflow.mkdir(parents=True)

    candidate = workflow / "methodology-candidate.yaml"
    candidate.write_text("status: candidate\n", encoding="utf-8")

    for stage in ("cutoff", "relative_cutoff"):
        directory = workflow / stage
        directory.mkdir()
        (directory / "convergence-report.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

    return root, candidate


def methodology(
    *,
    k_points=None,
    cutoff="cutoff/convergence-report.json",
    relative="relative_cutoff/convergence-report.json",
    kpoint=None,
    final_cutoff=None,
    final_relative=None,
    solver="OT",
):
    return CP2KProductionMethodology(
        status="candidate",
        functional="PBE",
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        k_points=k_points,
        scf=CP2KSCFConfig(solver=solver),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(),
        ),
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
            cutoff_report=cutoff,
            relative_cutoff_report=relative,
            kpoint_report=kpoint,
            final_cutoff_verification_report=final_cutoff,
            final_relative_cutoff_verification_report=final_relative,
        ),
    )


def validate(setup, value):
    root, candidate = setup
    return resolve_methodology_evidence_paths(
        project_root=root,
        candidate_path=candidate,
        methodology=value,
    )


def test_accepts_gamma_only_report_paths(setup):
    resolved = validate(setup, methodology())
    assert set(resolved) == {
        "cutoff_report",
        "relative_cutoff_report",
    }


@pytest.mark.parametrize(
    ("cutoff", "relative"),
    [
        (None, "relative_cutoff/convergence-report.json"),
        ("cutoff/convergence-report.json", None),
        ("missing/convergence-report.json",
         "relative_cutoff/convergence-report.json"),
        ("../cutoff/convergence-report.json",
         "relative_cutoff/convergence-report.json"),
        ("/tmp/convergence-report.json",
         "relative_cutoff/convergence-report.json"),
        ("cutoff/convergence-report.json",
         "cutoff/convergence-report.json"),
    ],
)
def test_rejects_missing_unsafe_or_duplicate_paths(
    setup,
    cutoff,
    relative,
):
    with pytest.raises(ConvergenceEvidenceValidationError):
        validate(
            setup,
            methodology(cutoff=cutoff, relative=relative),
        )


def test_rejects_kpoints_without_verification(setup):
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Required methodology report",
    ):
        validate(
            setup,
            methodology(
                k_points=(2, 2, 2),
                solver="DIAGONALIZATION",
            ),
        )


def test_rejects_gamma_only_with_kpoint_report(setup):
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Gamma-only",
    ):
        validate(
            setup,
            methodology(
                kpoint="kpoints/convergence-report.json",
            ),
        )


def test_rejects_kpoints_with_ot_solver(setup):
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="diagonalisation",
    ):
        validate(
            setup,
            methodology(
                k_points=(2, 2, 2),
                solver="OT",
            ),
        )


def test_rejects_symlink_escape(setup, tmp_path):
    root, candidate = setup

    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "convergence-report.json"
    target.write_text("{}\n", encoding="utf-8")

    link = candidate.parent / "escape"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ConvergenceEvidenceValidationError):
        validate(
            setup,
            methodology(
                cutoff="escape/convergence-report.json",
            ),
        )


def test_rejects_candidate_outside_project(setup, tmp_path):
    root, candidate = setup
    external = tmp_path / "external.yaml"
    external.write_text("status: candidate\n", encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="escapes the project",
    ):
        resolve_methodology_evidence_paths(
            project_root=root,
            candidate_path=external,
            methodology=methodology(),
        )


def test_methodology_validation_rejects_non_candidate_status(setup):
    from nsdw.workflows.convergence.cp2k_methodology_evidence import (
        validate_cp2k_methodology_evidence,
    )

    root, candidate = setup
    value = methodology().model_copy(update={"status": "validated"})

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="candidate status",
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate,
            methodology=value,
        )


def test_methodology_validation_rejects_placeholder_reports(setup):
    from nsdw.workflows.convergence.cp2k_methodology_evidence import (
        validate_cp2k_methodology_evidence,
    )

    root, candidate = setup

    with pytest.raises(ConvergenceEvidenceValidationError):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate,
            methodology=methodology(),
        )


def test_tail_stability_accepts_nonselected_stable_value():
    from nsdw.workflows.convergence.cp2k_methodology_evidence import (
        _require_tail_stable_value,
    )

    report = {
        "energy_tolerance_ev_per_atom": 0.01,
        "candidates": [
            {
                "value": {"value": 400.0, "unit": "Ry"},
                "energy_ev_per_atom": -10.0,
                "tail_stable": True,
            },
            {
                "value": {"value": 500.0, "unit": "Ry"},
                "energy_ev_per_atom": -10.001,
                "tail_stable": True,
            },
            {
                "value": {"value": 600.0, "unit": "Ry"},
                "energy_ev_per_atom": -10.002,
                "tail_stable": None,
            },
        ],
    }

    _require_tail_stable_value(
        report=report,
        parameter="cutoff",
        proposed_value=500.0,
    )


@pytest.mark.parametrize(
    "proposed",
    [300.0, 600.0],
)
def test_tail_stability_rejects_missing_or_reference_value(proposed):
    from nsdw.workflows.convergence.cp2k_methodology_evidence import (
        _require_tail_stable_value,
    )

    report = {
        "energy_tolerance_ev_per_atom": 0.01,
        "candidates": [
            {
                "value": {"value": 400.0, "unit": "Ry"},
                "energy_ev_per_atom": -10.0,
                "tail_stable": True,
            },
            {
                "value": {"value": 600.0, "unit": "Ry"},
                "energy_ev_per_atom": -10.001,
                "tail_stable": None,
            },
        ],
    }

    with pytest.raises(ConvergenceEvidenceValidationError):
        _require_tail_stable_value(
            report=report,
            parameter="cutoff",
            proposed_value=proposed,
        )


def test_tail_stability_rejects_unstable_proposed_value():
    from nsdw.workflows.convergence.cp2k_methodology_evidence import (
        _require_tail_stable_value,
    )

    report = {
        "energy_tolerance_ev_per_atom": 0.001,
        "candidates": [
            {
                "value": {"value": 400.0, "unit": "Ry"},
                "energy_ev_per_atom": -10.0,
                "tail_stable": False,
            },
            {
                "value": {"value": 600.0, "unit": "Ry"},
                "energy_ev_per_atom": -10.1,
                "tail_stable": None,
            },
        ],
    }

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="not tail-stable",
    ):
        _require_tail_stable_value(
            report=report,
            parameter="cutoff",
            proposed_value=400.0,
        )


def test_cross_campaign_rejects_wrong_fixed_cutoff(monkeypatch):
    from types import SimpleNamespace

    from nsdw.workflows.convergence import (
        cp2k_methodology_evidence as module,
    )

    def settings(**changes):
        values = {
            "run_type": "ENERGY",
            "charge": 0,
            "multiplicity": 1,
            "xc_functional": "PBE",
            "eps_scf": 1e-6,
            "kinds": (),
            "basis_set_file": "BASIS_MOLOPT",
            "potential_file": "GTH_POTENTIALS",
            "admm": False,
            "k_points": None,
            "scf_solver": "OT",
            "scf_guess": "ATOMIC",
            "max_scf": 100,
            "outer_scf_max": 10,
            "ot_minimizer": "CG",
            "ot_preconditioner": "FULL_SINGLE_INVERSE",
            "energy_gap": 0.001,
            "cutoff_ry": 560.0,
            "relative_cutoff_ry": 40.0,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    campaigns = {
        "cutoff_report": [settings(cutoff_ry=400.0)],
        "relative_cutoff_report": [settings(cutoff_ry=500.0)],
    }

    monkeypatch.setattr(
        module,
        "_campaign_inputs",
        lambda path: campaigns[path],
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="wrong fixed cutoff",
    ):
        module._validate_cross_campaign_settings(
            paths={key: key for key in campaigns},
            methodology=methodology(),
        )


def test_cross_campaign_rejects_changed_charge(monkeypatch):
    from types import SimpleNamespace

    from nsdw.workflows.convergence import (
        cp2k_methodology_evidence as module,
    )

    def settings(charge):
        return SimpleNamespace(
            run_type="ENERGY",
            charge=charge,
            multiplicity=1,
            xc_functional="PBE",
            eps_scf=1e-6,
            kinds=(),
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            admm=False,
            k_points=None,
            scf_solver="OT",
            scf_guess="ATOMIC",
            max_scf=100,
            outer_scf_max=10,
            ot_minimizer="CG",
            ot_preconditioner="FULL_SINGLE_INVERSE",
            energy_gap=0.001,
            cutoff_ry=560.0,
            relative_cutoff_ry=40.0,
        )

    campaigns = {
        "cutoff_report": [settings(0)],
        "relative_cutoff_report": [settings(1)],
    }

    monkeypatch.setattr(
        module,
        "_campaign_inputs",
        lambda path: campaigns[path],
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="charge mismatch",
    ):
        module._validate_cross_campaign_settings(
            paths={key: key for key in campaigns},
            methodology=methodology(),
        )
