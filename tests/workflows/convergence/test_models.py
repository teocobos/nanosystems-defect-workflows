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
from nsdw.models.quantity import Quantity

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

def test_candidate_accepts_quantity_value():
    candidate = ConvergenceCandidate(
        label="600Ry",
        order=1,
        value=Quantity(
            value=600.0,
            unit="Ry",
        ),
    )

    assert isinstance(candidate.value, Quantity)
    assert candidate.value.value == pytest.approx(600.0)
    assert candidate.value.unit == "Ry"


def test_candidate_accepts_basis_value():
    candidate = ConvergenceCandidate(
        label="tzvp",
        order=1,
        value="TZVP-MOLOPT-SR-GTH",
    )

    assert candidate.value == "TZVP-MOLOPT-SR-GTH"


def test_candidate_accepts_kpoint_mesh_value():
    candidate = ConvergenceCandidate(
        label="2x2x1",
        order=1,
        value=(2, 2, 1),
    )

    assert candidate.value == (2, 2, 1)


def test_candidate_value_remains_optional():
    candidate = ConvergenceCandidate(
        label="legacy",
        order=0,
    )

    assert candidate.value is None


def test_cutoff_study_accepts_quantity_values_in_ry():
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400Ry",
                order=0,
                value=Quantity(
                    value=400.0,
                    unit="Ry",
                ),
            ),
            ConvergenceCandidate(
                label="600Ry",
                order=1,
                value=Quantity(
                    value=600.0,
                    unit="Ry",
                ),
            ),
        ],
    )

    assert study.candidates[0].value == Quantity(
        value=400.0,
        unit="Ry",
    )


def test_cutoff_study_rejects_string_candidate_value():
    with pytest.raises(
        ValidationError,
        match="CUTOFF candidates must use Quantity values",
    ):
        ConvergenceStudyDefinition(
            parameter=ConvergenceParameter.CUTOFF,
            candidates=[
                ConvergenceCandidate(
                    label="400Ry",
                    order=0,
                    value="400Ry",
                ),
                ConvergenceCandidate(
                    label="600Ry",
                    order=1,
                    value="600Ry",
                ),
            ],
        )


def test_cutoff_study_rejects_non_ry_quantity():
    with pytest.raises(
        ValidationError,
        match="CUTOFF candidate quantities must use Ry",
    ):
        ConvergenceStudyDefinition(
            parameter=ConvergenceParameter.CUTOFF,
            candidates=[
                ConvergenceCandidate(
                    label="400eV",
                    order=0,
                    value=Quantity(
                        value=400.0,
                        unit="eV",
                    ),
                ),
                ConvergenceCandidate(
                    label="600eV",
                    order=1,
                    value=Quantity(
                        value=600.0,
                        unit="eV",
                    ),
                ),
            ],
        )


def test_relative_cutoff_study_accepts_quantity_values_in_ry():
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="40Ry",
                order=0,
                value=Quantity(
                    value=40.0,
                    unit="Ry",
                ),
            ),
            ConvergenceCandidate(
                label="60Ry",
                order=1,
                value=Quantity(
                    value=60.0,
                    unit="Ry",
                ),
            ),
        ],
    )

    assert isinstance(study.candidates[0].value, Quantity)


def test_basis_study_rejects_quantity_candidate_value():
    with pytest.raises(
        ValidationError,
        match="BASIS candidates must use string values",
    ):
        ConvergenceStudyDefinition(
            parameter=ConvergenceParameter.BASIS,
            candidates=[
                ConvergenceCandidate(
                    label="dzvp",
                    order=0,
                    value=Quantity(
                        value=1.0,
                        unit="Ry",
                    ),
                ),
                ConvergenceCandidate(
                    label="tzvp",
                    order=1,
                    value=Quantity(
                        value=2.0,
                        unit="Ry",
                    ),
                ),
            ],
        )


def test_kpoints_study_accepts_mesh_values():
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.KPOINTS,
        candidates=[
            ConvergenceCandidate(
                label="1x1x1",
                order=0,
                value=(1, 1, 1),
            ),
            ConvergenceCandidate(
                label="2x2x1",
                order=1,
                value=(2, 2, 1),
            ),
        ],
    )

    assert study.candidates[-1].value == (2, 2, 1)


def test_kpoints_study_rejects_string_candidate_value():
    with pytest.raises(
        ValidationError,
        match="KPOINTS candidates must use three-integer meshes",
    ):
        ConvergenceStudyDefinition(
            parameter=ConvergenceParameter.KPOINTS,
            candidates=[
                ConvergenceCandidate(
                    label="1x1x1",
                    order=0,
                    value="1x1x1",
                ),
                ConvergenceCandidate(
                    label="2x2x1",
                    order=1,
                    value="2x2x1",
                ),
            ],
        )