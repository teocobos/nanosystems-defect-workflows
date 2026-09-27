"""Calculator-independent models for convergence studies."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class ConvergenceParameter(StrEnum):
    """Numerical parameter varied during a convergence study."""

    BASIS = "basis"
    CUTOFF = "cutoff"
    RELATIVE_CUTOFF = "relative_cutoff"
    KPOINTS = "kpoints"


class ConvergenceCriterion(BaseModel):
    """
    Criterion used to decide whether a parameter is converged.

    Energy convergence is measured per atom so calculations using
    structures with different atom counts cannot be compared
    accidentally on raw total-energy differences.
    """

    energy_tolerance_ev_per_atom: Annotated[
        float,
        Field(gt=0.0),
    ] = 1.0e-3


class ConvergenceCandidate(BaseModel):
    """
    One candidate setting in a convergence study.

    ``label`` is the stable human-readable identifier used in reports
    and directory names. ``order`` represents increasing computational
    expense within the study.
    """

    label: str = Field(min_length=1)
    order: Annotated[int, Field(ge=0)]


class ConvergenceObservation(BaseModel):
    """
    Scientific result for one convergence candidate.

    Total energy is stored in eV. Energy per atom is derived from the
    total energy and atom count rather than supplied independently.
    ``structure_hash`` identifies the exact computational geometry used
    for the calculation.
    """

    candidate: ConvergenceCandidate
    total_energy_ev: Annotated[
        float,
        Field(allow_inf_nan=False),
    ]
    n_atoms: Annotated[int, Field(gt=0)]
    structure_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @property
    def energy_ev_per_atom(self) -> float:
        """Return the total energy normalised by atom count."""

        return self.total_energy_ev / self.n_atoms

class ConvergenceStudyDefinition(BaseModel):
    """
    Definition of a calculator-independent convergence study.

    Candidates must be supplied in increasing computational-cost order.
    The analyser can therefore select the least expensive candidate
    satisfying the convergence criterion rather than simply selecting
    the lowest total energy.
    """

    parameter: ConvergenceParameter
    candidates: list[ConvergenceCandidate] = Field(min_length=2)
    criterion: ConvergenceCriterion = Field(
        default_factory=ConvergenceCriterion
    )

    @model_validator(mode="after")
    def validate_candidate_order(
        self,
    ) -> "ConvergenceStudyDefinition":
        """
        Require candidates to be supplied in strictly increasing
        computational-cost order.
        """

        orders = [
            candidate.order
            for candidate in self.candidates
        ]

        if orders != sorted(orders):
            raise ValueError(
                "Convergence candidates must be supplied in "
                "increasing computational-cost order."
            )

        if len(set(orders)) != len(orders):
            raise ValueError(
                "Convergence candidate orders must be unique."
            )

        return self