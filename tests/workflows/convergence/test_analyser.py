"""Tests for calculator-independent convergence energy analysis."""

from __future__ import annotations

import pytest

from nsdw.workflows.convergence.analyser import (
    ConvergenceAnalysisError,
    analyse_convergence_energy,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceCriterion,
    ConvergenceObservation,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)


STRUCTURE_HASH = "a" * 64


def _study(
    *,
    tolerance: float = 1.0e-3,
) -> ConvergenceStudyDefinition:
    """Return a simple three-candidate basis study."""

    return ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.BASIS,
        candidates=[
            ConvergenceCandidate(
                label="dzvp",
                order=0,
            ),
            ConvergenceCandidate(
                label="tzvp",
                order=1,
            ),
            ConvergenceCandidate(
                label="tzv2p",
                order=2,
            ),
        ],
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=tolerance
        ),
    )


def _observation(
    label: str,
    order: int,
    energy_ev_per_atom: float,
    *,
    n_atoms: int = 84,
    structure_hash: str = STRUCTURE_HASH,
) -> ConvergenceObservation:
    """Build an observation from a desired energy per atom."""

    return ConvergenceObservation(
        candidate=ConvergenceCandidate(
            label=label,
            order=order,
        ),
        total_energy_ev=(
            energy_ev_per_atom * n_atoms
        ),
        n_atoms=n_atoms,
        structure_hash=structure_hash,
    )


def test_energy_analysis_selects_least_expensive_candidate_within_tolerance():
    study = _study(
        tolerance=1.0e-3,
    )

    observations = [
        _observation(
            "dzvp",
            0,
            -10.0000,
        ),
        _observation(
            "tzvp",
            1,
            -10.0100,
        ),
        _observation(
            "tzv2p",
            2,
            -10.0105,
        ),
    ]

    analysis = analyse_convergence_energy(
        study,
        observations,
    )

    assert analysis.energy_tolerance_satisfied is True
    assert analysis.selected_candidate_label == "tzvp"

    assert len(analysis.comparisons) == 2

    assert (
        analysis.comparisons[0]
        .energy_difference_ev_per_atom
        == pytest.approx(0.0100)
    )

    assert (
        analysis.comparisons[0]
        .within_energy_tolerance
        is False
    )

    assert (
        analysis.comparisons[1]
        .energy_difference_ev_per_atom
        == pytest.approx(0.0005)
    )

    assert (
        analysis.comparisons[1]
        .within_energy_tolerance
        is True
    )


def test_energy_analysis_reports_unsatisfied_tolerance():
    study = _study(
        tolerance=1.0e-3,
    )

    observations = [
        _observation(
            "dzvp",
            0,
            -10.000,
        ),
        _observation(
            "tzvp",
            1,
            -10.010,
        ),
        _observation(
            "tzv2p",
            2,
            -10.020,
        ),
    ]

    analysis = analyse_convergence_energy(
        study,
        observations,
    )

    assert analysis.energy_tolerance_satisfied is False
    assert analysis.selected_candidate_label is None


def test_energy_analysis_rejects_mismatched_structure_hashes():
    study = _study()

    observations = [
        _observation(
            "dzvp",
            0,
            -10.000,
        ),
        _observation(
            "tzvp",
            1,
            -10.010,
            structure_hash="b" * 64,
        ),
        _observation(
            "tzv2p",
            2,
            -10.011,
        ),
    ]

    with pytest.raises(
        ConvergenceAnalysisError,
        match="same computational structure",
    ):
        analyse_convergence_energy(
            study,
            observations,
        )


def test_energy_analysis_rejects_mismatched_atom_counts():
    study = _study()

    observations = [
        _observation(
            "dzvp",
            0,
            -10.000,
        ),
        _observation(
            "tzvp",
            1,
            -10.010,
            n_atoms=42,
        ),
        _observation(
            "tzv2p",
            2,
            -10.011,
        ),
    ]

    with pytest.raises(
        ConvergenceAnalysisError,
        match="same atom count",
    ):
        analyse_convergence_energy(
            study,
            observations,
        )


def test_energy_analysis_rejects_candidate_mismatch():
    study = _study()

    observations = [
        _observation(
            "dzvp",
            0,
            -10.000,
        ),
        _observation(
            "wrong",
            1,
            -10.010,
        ),
        _observation(
            "tzv2p",
            2,
            -10.011,
        ),
    ]

    with pytest.raises(
        ConvergenceAnalysisError,
        match="candidate definitions or ordering",
    ):
        analyse_convergence_energy(
            study,
            observations,
        )


def test_energy_analysis_rejects_missing_observation():
    study = _study()

    observations = [
        _observation(
            "dzvp",
            0,
            -10.000,
        ),
        _observation(
            "tzvp",
            1,
            -10.010,
        ),
    ]

    with pytest.raises(
        ConvergenceAnalysisError,
        match="number of convergence observations",
    ):
        analyse_convergence_energy(
            study,
            observations,
        )
