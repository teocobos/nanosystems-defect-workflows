"""Portable recovery identity tests."""

import pytest

from nsdw.workflows.convergence.recovery_identity import (
    execution_attempt_identity,
    scientific_recovery_identity,
)


def scientific_identity():
    return scientific_recovery_identity(
        original_campaign_uuid="campaign-268-uuid",
        structure_uuid="structure-227-uuid",
        code_input_identity="cp2k-pbe-input-v1",
        candidate_parameters={
            "40-Ry": {"CUTOFF": 400, "REL_CUTOFF": 40},
            "50-Ry": {"CUTOFF": 400, "REL_CUTOFF": 50},
        },
    )


def test_scientific_identity_is_deterministic():
    assert scientific_identity() == scientific_identity()
    assert len(scientific_identity()) == 64


def test_scientific_identity_ignores_dictionary_order():
    first = scientific_recovery_identity(
        original_campaign_uuid="campaign",
        structure_uuid="structure",
        code_input_identity="cp2k-pbe",
        candidate_parameters={
            "40-Ry": {"CUTOFF": 400, "REL_CUTOFF": 40},
            "50-Ry": {"CUTOFF": 400, "REL_CUTOFF": 50},
        },
    )

    second = scientific_recovery_identity(
        original_campaign_uuid="campaign",
        structure_uuid="structure",
        code_input_identity="cp2k-pbe",
        candidate_parameters={
            "50-Ry": {"REL_CUTOFF": 50, "CUTOFF": 400},
            "40-Ry": {"REL_CUTOFF": 40, "CUTOFF": 400},
        },
    )

    assert first == second


def test_scientific_identity_changes_with_inputs():
    changed = scientific_recovery_identity(
        original_campaign_uuid="campaign-268-uuid",
        structure_uuid="structure-227-uuid",
        code_input_identity="cp2k-pbe-input-v1",
        candidate_parameters={
            "40-Ry": {"CUTOFF": 400, "REL_CUTOFF": 45},
            "50-Ry": {"CUTOFF": 400, "REL_CUTOFF": 50},
        },
    )

    assert changed != scientific_identity()


def test_execution_identity_is_backend_specific():
    scientific = scientific_identity()

    lumi = execution_attempt_identity(
        scientific_identity=scientific,
        backend="aiida",
        computer_uuid="lumi-computer-uuid",
        resources={"walltime": 1800, "queue": "small"},
        attempt_number=1,
    )

    cirrus = execution_attempt_identity(
        scientific_identity=scientific,
        backend="aiida",
        computer_uuid="cirrus-computer-uuid",
        resources={"walltime": 1800, "queue": "standard"},
        attempt_number=1,
    )

    isambard = execution_attempt_identity(
        scientific_identity=scientific,
        backend="aiida",
        computer_uuid="isambard-computer-uuid",
        resources={"walltime": 1800, "gpus": 1},
        attempt_number=1,
    )

    assert len({lumi, cirrus, isambard}) == 3


def test_attempt_number_changes_execution_identity():
    kwargs = dict(
        scientific_identity=scientific_identity(),
        backend="aiida",
        computer_uuid="lumi-computer-uuid",
        resources={"walltime": 1800},
    )

    assert execution_attempt_identity(
        **kwargs, attempt_number=1
    ) != execution_attempt_identity(
        **kwargs, attempt_number=2
    )


def test_invalid_attempt_number_rejected():
    with pytest.raises(ValueError):
        execution_attempt_identity(
            scientific_identity=scientific_identity(),
            backend="aiida",
            computer_uuid="computer",
            resources={},
            attempt_number=0,
        )


def test_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        scientific_recovery_identity(
            original_campaign_uuid="campaign",
            structure_uuid="structure",
            code_input_identity="cp2k",
            candidate_parameters={"40-Ry": {"energy": float("nan")}},
        )


def test_scientific_identity_rejects_empty_candidates():
    with pytest.raises(ValueError):
        scientific_recovery_identity(
            original_campaign_uuid="campaign",
            structure_uuid="structure",
            code_input_identity="cp2k",
            candidate_parameters={},
        )
