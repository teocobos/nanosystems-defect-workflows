"""Calculator-independent convergence analysis."""

from __future__ import annotations

from dataclasses import dataclass

from nsdw.workflows.convergence.models import (
    ConvergenceObservation,
    ConvergenceStudyDefinition,
)


class ConvergenceAnalysisError(RuntimeError):
    """Raised when a convergence study cannot be analysed safely."""


@dataclass(frozen=True)
class ConvergenceComparison:
    """
    Energy comparison between successive convergence candidates.

    ``within_energy_tolerance`` records only the generic energy
    criterion. Calculator-specific workflows may require additional
    diagnostics before declaring a numerical parameter converged.
    """

    candidate_label: str
    next_candidate_label: str
    energy_difference_ev_per_atom: float
    within_energy_tolerance: bool


@dataclass(frozen=True)
class ConvergenceAnalysis:
    """
    Generic energy analysis for a convergence study.

    ``selected_candidate_label`` is the least expensive candidate
    satisfying the configured energy criterion. It does not by itself
    imply that calculator-specific convergence requirements have been
    satisfied.
    """

    selected_candidate_label: str | None
    comparisons: tuple[ConvergenceComparison, ...]
    energy_tolerance_satisfied: bool


def analyse_convergence_energy(
    study: ConvergenceStudyDefinition,
    observations: list[ConvergenceObservation],
) -> ConvergenceAnalysis:
    """
    Analyse successive energy changes in a convergence study.

    Energy differences are evaluated per atom between successive
    candidates. The least expensive candidate satisfying the configured
    energy tolerance is identified.

    This is a calculator-independent energy analysis. Calculator-
    specific workflows may impose additional convergence requirements.

    All observations must describe the same computational structure.
    """

    if len(observations) != len(study.candidates):
        raise ConvergenceAnalysisError(
            "The number of convergence observations must match "
            "the number of study candidates."
        )

    expected_candidates = [
        (candidate.label, candidate.order)
        for candidate in study.candidates
    ]

    observed_candidates = [
        (
            observation.candidate.label,
            observation.candidate.order,
        )
        for observation in observations
    ]

    if observed_candidates != expected_candidates:
        raise ConvergenceAnalysisError(
            "Convergence observations do not match the study "
            "candidate definitions or ordering."
        )

    structure_hashes = {
        observation.structure_hash
        for observation in observations
    }

    if len(structure_hashes) != 1:
        raise ConvergenceAnalysisError(
            "Convergence observations must use the same "
            "computational structure."
        )

    atom_counts = {
        observation.n_atoms
        for observation in observations
    }

    if len(atom_counts) != 1:
        raise ConvergenceAnalysisError(
            "Convergence observations must use the same atom count."
        )

    tolerance = (
        study.criterion.energy_tolerance_ev_per_atom
    )

    comparisons: list[ConvergenceComparison] = []

    for current, next_observation in zip(
        observations,
        observations[1:],
    ):
        difference = abs(
            next_observation.energy_ev_per_atom
            - current.energy_ev_per_atom
        )

        comparisons.append(
            ConvergenceComparison(
                candidate_label=current.candidate.label,
                next_candidate_label=(
                    next_observation.candidate.label
                ),
                energy_difference_ev_per_atom=difference,
                within_energy_tolerance=(
                    difference <= tolerance
                ),
            )
        )

    selected_candidate_label = None

    for comparison in comparisons:
        if comparison.within_energy_tolerance:
            selected_candidate_label = (
                comparison.candidate_label
            )
            break

    return ConvergenceAnalysis(
        selected_candidate_label=selected_candidate_label,
        comparisons=tuple(comparisons),
        energy_tolerance_satisfied=(
            selected_candidate_label is not None
        ),
    )

@dataclass(frozen=True)
class ConvergenceReferenceComparison:
    """Energy difference between a candidate and the reference."""

    candidate_label: str
    reference_candidate_label: str
    energy_difference_ev_per_atom: float
    within_energy_tolerance: bool


@dataclass(frozen=True)
class ConvergenceReferenceAnalysis:
    """
    Energy convergence relative to the highest-cost candidate.

    The reference is the final candidate in the study.

    A selected candidate satisfies the energy tolerance against
    that reference. This is an energy-only assessment, not proof
    of complete numerical convergence.
    """

    reference_candidate_label: str
    selected_candidate_label: str | None
    comparisons: tuple[ConvergenceReferenceComparison, ...]
    energy_tolerance_satisfied: bool


def analyse_convergence_against_reference(
    study: ConvergenceStudyDefinition,
    observations: list[ConvergenceObservation],
) -> ConvergenceReferenceAnalysis:
    """
    Compare each candidate's energy against the highest-cost reference.

    Reuse the generic analyser to validate candidate ordering,
    structure hashes and atom counts.

    The reference itself is excluded from selection because its
    zero energy difference is true by definition and provides no
    independent convergence evidence.
    """

    analyse_convergence_energy(
        study,
        observations,
    )

    reference = observations[-1]
    reference_energy = reference.energy_ev_per_atom

    tolerance = (
        study.criterion.energy_tolerance_ev_per_atom
    )

    comparisons: list[ConvergenceReferenceComparison] = []

    for observation in observations[:-1]:
        difference = abs(
            observation.energy_ev_per_atom
            - reference_energy
        )

        comparisons.append(
            ConvergenceReferenceComparison(
                candidate_label=observation.candidate.label,
                reference_candidate_label=reference.candidate.label,
                energy_difference_ev_per_atom=difference,
                within_energy_tolerance=(
                    difference <= tolerance
                ),
            )
        )

    selected_candidate_label = None

    for comparison in comparisons:
        if comparison.within_energy_tolerance:
            selected_candidate_label = comparison.candidate_label
            break

    return ConvergenceReferenceAnalysis(
        reference_candidate_label=reference.candidate.label,
        selected_candidate_label=selected_candidate_label,
        comparisons=tuple(comparisons),
        energy_tolerance_satisfied=(
            selected_candidate_label is not None
        ),
    )