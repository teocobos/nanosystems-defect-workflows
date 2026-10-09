"""Adversarial tests for CP2K production SCF provenance."""

from types import SimpleNamespace

import pytest

from nsdw.calculators.cp2k.generation_models import CP2KSCFConfig
from nsdw.workflows.convergence.cp2k_methodology_evidence import (
    ConvergenceEvidenceValidationError,
    _scf_evidence_signature,
    _validate_scf_identity,
)


def _methodology(solver="OT", **changes):
    values = {
        "solver": solver,
        "scf_guess": "ATOMIC",
        "eps_scf": 1e-6,
        "max_scf": 100,
        "outer_scf_max": 10,
        "ot_minimizer": "CG",
        "ot_preconditioner": "FULL_SINGLE_INVERSE",
        "energy_gap": 0.001,
    }
    values.update(changes)

    return SimpleNamespace(scf=CP2KSCFConfig(**values))


def _input(solver="OT", **changes):
    values = {
        "scf_solver": solver,
        "scf_guess": "ATOMIC",
        "eps_scf": 1e-6,
        "max_scf": 100,
        "outer_scf_max": 10 if solver == "OT" else None,
        "ot_minimizer": "CG" if solver == "OT" else None,
        "ot_preconditioner": (
            "FULL_SINGLE_INVERSE" if solver == "OT" else None
        ),
        "energy_gap": 0.001 if solver == "OT" else None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("scf_guess", "RESTART"),
        ("eps_scf", 1e-5),
        ("max_scf", 50),
        ("outer_scf_max", 5),
        ("ot_minimizer", "DIIS"),
        ("ot_preconditioner", "FULL_ALL"),
        ("energy_gap", 0.005),
    ],
)
def test_rejects_modified_ot_setting(field, changed):
    evidence = _input(**{field: changed})

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Production SCF",
    ):
        _validate_scf_identity(
            parsed_input=evidence,
            methodology=_methodology(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "scf_guess",
        "eps_scf",
        "max_scf",
        "outer_scf_max",
        "ot_minimizer",
        "ot_preconditioner",
        "energy_gap",
    ],
)
def test_rejects_missing_ot_setting(field):
    evidence = _input(**{field: None})

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Production SCF",
    ):
        _validate_scf_identity(
            parsed_input=evidence,
            methodology=_methodology(),
        )


def test_rejects_solver_mismatch():
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="solver",
    ):
        _validate_scf_identity(
            parsed_input=_input(solver="OT"),
            methodology=_methodology(solver="DIAGONALIZATION"),
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("scf_guess", "RESTART"),
        ("eps_scf", 1e-5),
        ("max_scf", 50),
    ],
)
def test_rejects_modified_diagonalization_setting(field, changed):
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Production SCF",
    ):
        _validate_scf_identity(
            parsed_input=_input(
                solver="DIAGONALIZATION",
                **{field: changed},
            ),
            methodology=_methodology(solver="DIAGONALIZATION"),
        )


def test_accepts_matching_ot_evidence():
    _validate_scf_identity(
        parsed_input=_input(),
        methodology=_methodology(),
    )


def test_accepts_matching_diagonalization_evidence():
    _validate_scf_identity(
        parsed_input=_input(solver="DIAGONALIZATION"),
        methodology=_methodology(solver="DIAGONALIZATION"),
    )


def test_diagonalization_ignores_inactive_ot_parameters():
    evidence = _input(
        solver="DIAGONALIZATION",
        outer_scf_max=None,
        ot_minimizer=None,
        ot_preconditioner=None,
        energy_gap=None,
    )

    _validate_scf_identity(
        parsed_input=evidence,
        methodology=_methodology(
            solver="DIAGONALIZATION",
            outer_scf_max=99,
            energy_gap=0.01,
        ),
    )


def test_initial_ot_signature_detects_changed_minimizer():
    original = _input()
    modified = _input(ot_minimizer="DIIS")

    assert (
        _scf_evidence_signature(original)
        != _scf_evidence_signature(modified)
    )


def test_initial_ot_signature_detects_changed_outer_limit():
    original = _input()
    modified = _input(outer_scf_max=20)

    assert (
        _scf_evidence_signature(original)
        != _scf_evidence_signature(modified)
    )
