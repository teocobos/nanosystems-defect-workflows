"""Execution fingerprint verification without AiiDA or HPC."""

import pytest

from nsdw.workflows.convergence.recovery_identity import (
    execution_attempt_identity,
    verify_execution_attempt_identity,
)


def make_attempt(
    *,
    backend="aiida",
    computer_uuid="computer-a",
    resources=None,
    attempt_number=1,
):
    if resources is None:
        resources = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 1800,
        }

    return execution_attempt_identity(
        scientific_identity="scientific-example",
        backend=backend,
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=attempt_number,
    )


def verify(recorded, **changes):
    values = {
        "scientific_identity": "scientific-example",
        "backend": "aiida",
        "computer_uuid": "computer-a",
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 1800,
        },
        "attempt_number": 1,
    }
    values.update(changes)

    return verify_execution_attempt_identity(
        recorded_identity=recorded,
        **values,
    )


def test_matching_execution_fingerprint():
    recorded = make_attempt()
    assert verify(recorded) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "another-engine"},
        {"computer_uuid": "computer-b"},
        {"attempt_number": 2},
        {"scientific_identity": "different-science"},
        {"resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 600,
        }},
    ],
)
def test_execution_fingerprint_mismatch(changes):
    with pytest.raises(ValueError, match="Execution identity mismatch"):
        verify(make_attempt(), **changes)


def test_invalid_attempt_number_is_rejected():
    with pytest.raises(ValueError, match="Attempt number"):
        verify(make_attempt(), attempt_number=0)


def test_invalid_recorded_fingerprint_is_rejected():
    with pytest.raises(ValueError, match="recorded identity"):
        verify("not-a-sha256-fingerprint")


def test_dictionary_order_does_not_change_identity():
    recorded = make_attempt()

    reordered = {
        "max_wallclock_seconds": 1800,
        "num_mpiprocs_per_machine": 1,
        "num_machines": 1,
    }

    assert verify(recorded, resources=reordered) is True
