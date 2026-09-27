"""Tests for calculator-independent convergence models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceCriterion,
    ConvergenceObservation,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)


def _candidates() -> list[ConvergenceCandidate]:
    """Return a simple ordered candidate set."""

    return [
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
    ]


def test_observation_derives_energy_per_atom():
    observation = ConvergenceObservation(
        candidate=ConvergenceCandidate(
            label="tzvp",
            order=1,
        ),
        total_energy_ev=-840.0,
        n_atoms=84,
        structure_hash="a" * 64,
    )

    assert observation.energy_ev_per_atom == pytest.approx(
        -10.0
    )


def test_observation_requires_positive_atom_count():
    with pytest.raises(ValidationError):
        ConvergenceObservation(
            candidate=ConvergenceCandidate(
                label="tzvp",
                order=1,
            ),
            total_energy_ev=-840.0,
            n_atoms=0,
            structure_hash="a" * 64,
        )


def test_observation_requires_valid_structure_hash():
    with pytest.raises(ValidationError):
        ConvergenceObservation(
            candidate=ConvergenceCandidate(
                label="tzvp",
                order=1,
            ),
            total_energy_ev=-840.0,
            n_atoms=84,
            structure_hash="not-a-sha256",
        )


def test_observation_rejects_uppercase_structure_hash():
    with pytest.raises(ValidationError):
        ConvergenceObservation(
            candidate=ConvergenceCandidate(
                label="tzvp",
                order=1,
            ),
            total_energy_ev=-840.0,
            n_atoms=84,
            structure_hash="A" * 64,
        )


@pytest.mark.parametrize(
    "invalid_energy",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_observation_rejects_non_finite_energy(
    invalid_energy,
):
    with pytest.raises(ValidationError):
        ConvergenceObservation(
            candidate=ConvergenceCandidate(
                label="tzvp",
                order=1,
            ),
            total_energy_ev=invalid_energy,
            n_atoms=84,
            structure_hash="a" * 64,
        )


def test_convergence_parameter_values():
    assert ConvergenceParameter.BASIS == "basis"
    assert ConvergenceParameter.CUTOFF == "cutoff"
    assert (
        ConvergenceParameter.RELATIVE_CUTOFF
        == "relative_cutoff"
    )
    assert ConvergenceParameter.KPOINTS == "kpoints"


def test_default_energy_tolerance():
    criterion = ConvergenceCriterion()

    assert (
        criterion.energy_tolerance_ev_per_atom
        == pytest.approx(1.0e-3)
    )


def test_energy_tolerance_must_be_positive():
    with pytest.raises(ValidationError):
        ConvergenceCriterion(
            energy_tolerance_ev_per_atom=0.0
        )

    with pytest.raises(ValidationError):
        ConvergenceCriterion(
            energy_tolerance_ev_per_atom=-1.0e-3
        )


def test_candidate_requires_non_negative_order():
    with pytest.raises(ValidationError):
        ConvergenceCandidate(
            label="invalid",
            order=-1,
        )


def test_candidate_requires_non_empty_label():
    with pytest.raises(ValidationError):
        ConvergenceCandidate(
            label="",
            order=0,
        )


def test_study_requires_at_least_two_candidates():
    with pytest.raises(ValidationError):
        ConvergenceStudyDefinition(
            parameter=ConvergenceParameter.BASIS,
            candidates=[
                ConvergenceCandidate(
                    label="dzvp",
                    order=0,
                )
            ],
        )


def test_study_accepts_ordered_candidates():
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.BASIS,
        candidates=_candidates(),
    )

    assert study.parameter == ConvergenceParameter.BASIS
    assert len(study.candidates) == 3
    assert study.candidates[0].label == "dzvp"
    assert study.candidates[-1].label == "tzv2p"


def test_study_rejects_decreasing_candidate_order():
    with pytest.raises(
        ValidationError,
        match="increasing computational-cost order",
    ):
        ConvergenceStudyDefinition(
            parameter=ConvergenceParameter.BASIS,
            candidates=[
                ConvergenceCandidate(
                    label="dzvp",
                    order=0,
                ),
                ConvergenceCandidate(
                    label="tzv2p",
                    order=2,
                ),
                ConvergenceCandidate(
                    label="tzvp",
                    order=1,
                ),
            ],
        )


def test_study_rejects_duplicate_candidate_order():
    with pytest.raises(
        ValidationError,
        match="orders must be unique",
    ):
        ConvergenceStudyDefinition(
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
                    order=1,
                ),
            ],
        )


def test_study_accepts_custom_tolerance():
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400",
                order=0,
            ),
            ConvergenceCandidate(
                label="500",
                order=1,
            ),
        ],
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=5.0e-4
        ),
    )

    assert (
        study.criterion.energy_tolerance_ev_per_atom
        == pytest.approx(5.0e-4)
    )