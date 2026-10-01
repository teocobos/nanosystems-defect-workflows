"""Adapters from NSDW scientific results to convergence observations."""

from __future__ import annotations

from nsdw.models.result import NSDWResult
from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceObservation,
)


class ConvergenceObservationError(ValueError):
    """Raised when an NSDW result cannot form a convergence observation."""


def convergence_observation_from_result(
    *,
    candidate: ConvergenceCandidate,
    result: NSDWResult,
) -> ConvergenceObservation:
    """Convert an NSDW scientific result into a convergence observation."""

    if result.structure is None:
        raise ConvergenceObservationError(
            "Convergence analysis requires structural metadata."
        )

    if result.energy is None:
        raise ConvergenceObservationError(
            "Convergence analysis requires energy data."
        )

    if result.energy.total is None:
        raise ConvergenceObservationError(
            "Convergence analysis requires a total energy."
        )

    if result.energy.total.unit != "eV":
        raise ConvergenceObservationError(
            "Convergence analysis requires total energy in eV, "
            f"got {result.energy.total.unit!r}."
        )

    return ConvergenceObservation(
        candidate=candidate,
        total_energy_ev=result.energy.total.value,
        n_atoms=result.structure.n_atoms,
        structure_hash=result.structure.structure_hash,
    )


def collect_convergence_observations(
    *,
    candidates: tuple[ConvergenceCandidate, ...],
    results: tuple[NSDWResult, ...],
) -> tuple[ConvergenceObservation, ...]:
    """Convert ordered convergence results into observations."""

    if len(candidates) != len(results):
        raise ConvergenceObservationError(
            "Convergence candidate and result counts must match: "
            f"got {len(candidates)} candidates and "
            f"{len(results)} results."
        )

    observations: list[ConvergenceObservation] = []

    for candidate, result in zip(
        candidates,
        results,
        strict=True,
    ):
        if result.calculation.id != candidate.label:
            raise ConvergenceObservationError(
                f"Result calculation ID {result.calculation.id!r} "
                f"does not match candidate {candidate.label!r}."
            )

        observations.append(
            convergence_observation_from_result(
                candidate=candidate,
                result=result,
            )
        )

    return tuple(observations)