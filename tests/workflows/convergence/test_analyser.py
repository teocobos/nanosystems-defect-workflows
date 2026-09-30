"""Tests for calculator-independent convergence energy analysis."""

from __future__ import annotations

import pytest

from nsdw.workflows.convergence.analyser import (
    ConvergenceAnalysisError,
    analyse_convergence_energy,
    analyse_convergence_against_reference,
    analyse_convergence_tail_stability,
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

def _cutoff_study(
    *,
    tolerance: float = 1.0e-3,
) -> ConvergenceStudyDefinition:
    """Construct a five-candidate cutoff convergence study."""

    return ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label=f"{cutoff}_ry",
                order=index,
            )
            for index, cutoff in enumerate(
                (400, 450, 500, 550, 600)
            )
        ],
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=tolerance
        ),
    )


def _cutoff_observations(
    energies: tuple[float, ...],
) -> list[ConvergenceObservation]:
    """Build cutoff observations using the same 84-atom structure."""

    return [
        _observation(
            label=f"{cutoff}_ry",
            order=index,
            energy_ev_per_atom=energy,
        )
        for index, (cutoff, energy) in enumerate(
            zip(
                (400, 450, 500, 550, 600),
                energies,
            )
        )
    ]


def test_reference_analysis_avoids_premature_adjacent_selection():
    """A small adjacent difference must not imply convergence."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.0000,
            -10.0004,
            -10.0080,
            -10.0082,
            -10.0083,
        )
    )

    adjacent = analyse_convergence_energy(
        study,
        observations,
    )

    reference = analyse_convergence_against_reference(
        study,
        observations,
    )

    assert adjacent.selected_candidate_label == "400_ry"
    assert reference.reference_candidate_label == "600_ry"
    assert reference.selected_candidate_label == "500_ry"

    assert reference.energy_tolerance_satisfied


def test_reference_analysis_reports_unsatisfied_tolerance():
    """No non-reference candidate satisfies the tolerance."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.000,
            -10.010,
            -10.020,
            -10.030,
            -10.040,
        )
    )

    analysis = analyse_convergence_against_reference(
        study,
        observations,
    )

    assert analysis.selected_candidate_label is None
    assert not analysis.energy_tolerance_satisfied
    assert len(analysis.comparisons) == 4


def test_reference_analysis_excludes_reference_from_selection():
    """The reference cannot validate itself."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.000,
            -10.010,
            -10.020,
            -10.030,
            -10.040,
        )
    )

    analysis = analyse_convergence_against_reference(
        study,
        observations,
    )

    assert all(
        comparison.candidate_label != "600_ry"
        for comparison in analysis.comparisons
    )


def test_reference_analysis_rejects_structure_mismatch():
    """Reference analysis must preserve structure validation."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.000,
            -10.005,
            -10.008,
            -10.009,
            -10.010,
        )
    )

    observations[2] = _observation(
        "500_ry",
        2,
        -10.008,
        structure_hash="b" * 64,
    )

    with pytest.raises(
        ConvergenceAnalysisError,
        match="same computational structure",
    ):
        analyse_convergence_against_reference(
            study,
            observations,
        )


def test_reference_analysis_reports_energy_differences():
    """Each comparison uses the final candidate as reference."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.0000,
            -10.0004,
            -10.0080,
            -10.0082,
            -10.0083,
        )
    )

    analysis = analyse_convergence_against_reference(
        study,
        observations,
    )

    differences = [
        comparison.energy_difference_ev_per_atom
        for comparison in analysis.comparisons
    ]

    assert differences == pytest.approx(
        [0.0083, 0.0079, 0.0003, 0.0001]
    )

def test_tail_stability_requires_agreement_with_all_higher_cost_candidates():
    """Reference agreement alone must not imply tail stability."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.0200,
            -10.0100,
            -10.0000,
            -10.0050,
            -10.0002,
        )
    )

    reference = analyse_convergence_against_reference(
        study,
        observations,
    )

    tail = analyse_convergence_tail_stability(
        study,
        observations,
    )

    assert reference.selected_candidate_label == "500_ry"
    assert tail.selected_candidate_label is None
    assert not tail.energy_tolerance_satisfied


def test_tail_stability_selects_least_expensive_stable_candidate():
    """Select the first candidate stable against the complete tail."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.0200,
            -10.0100,
            -10.0000,
            -10.0004,
            -10.0002,
        )
    )

    analysis = analyse_convergence_tail_stability(
        study,
        observations,
    )

    assert analysis.selected_candidate_label == "500_ry"
    assert analysis.energy_tolerance_satisfied


def test_tail_stability_reports_maximum_higher_cost_difference():
    """Each candidate records its worst difference across the tail."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.0200,
            -10.0100,
            -10.0000,
            -10.0004,
            -10.0002,
        )
    )

    analysis = analyse_convergence_tail_stability(
        study,
        observations,
    )

    differences = [
        comparison.maximum_energy_difference_ev_per_atom
        for comparison in analysis.comparisons
    ]

    assert differences == pytest.approx(
        [0.0200, 0.0100, 0.0004, 0.0002]
    )

    assert [
        comparison.within_energy_tolerance
        for comparison in analysis.comparisons
    ] == [False, False, True, True]


def test_tail_stability_excludes_final_candidate_from_selection():
    """The final candidate cannot establish stability by itself."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.000,
            -10.010,
            -10.020,
            -10.030,
            -10.040,
        )
    )

    analysis = analyse_convergence_tail_stability(
        study,
        observations,
    )

    assert len(analysis.comparisons) == 4
    assert all(
        comparison.candidate_label != "600_ry"
        for comparison in analysis.comparisons
    )


def test_tail_stability_reuses_generic_study_validation():
    """Tail analysis must reject mismatched computational structures."""

    study = _cutoff_study()

    observations = _cutoff_observations(
        (
            -10.0000,
            -10.0050,
            -10.0080,
            -10.0082,
            -10.0083,
        )
    )

    observations[2] = _observation(
        "500_ry",
        2,
        -10.0080,
        structure_hash="b" * 64,
    )

    with pytest.raises(
        ConvergenceAnalysisError,
        match="same computational structure",
    ):
        analyse_convergence_tail_stability(
            study,
            observations,
        )