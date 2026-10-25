"""Fingerprint validation against durable recovery reservations."""

from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_identity import (
    execution_attempt_identity,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)
from nsdw.workflows.convergence.recovery_reserved_identity import (
    validate_reserved_execution_identity,
)


def make_case(tmp_path, attempt_number=2):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    computer_uuid = str(uuid4())
    resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 1800,
        "queue_name": "small",
        "account": "project-example",
    }

    science = "science-example"

    fingerprint = execution_attempt_identity(
        scientific_identity=science,
        backend="aiida",
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=attempt_number,
    )

    store.reserve(
        RecoveryRecord(
            science,
            fingerprint,
            attempt_number=attempt_number,
        )
    )

    metadata = RecoveryExecutionMetadata(
        scientific_identity=science,
        execution_identity=fingerprint,
        engine="aiida",
        computer_uuid=computer_uuid,
    )

    return store, metadata, resources


def test_matching_reservation(tmp_path):
    store, metadata, resources = make_case(tmp_path)

    assert validate_reserved_execution_identity(
        store,
        metadata,
        resources=resources,
    ) is True


def test_missing_reservation_rejected(tmp_path):
    store, metadata, resources = make_case(tmp_path)

    wrong = RecoveryExecutionMetadata(
        scientific_identity="missing-science",
        execution_identity=metadata.execution_identity,
        engine="aiida",
        computer_uuid=metadata.computer_uuid,
    )

    with pytest.raises(ValueError, match="reservation"):
        validate_reserved_execution_identity(
            store,
            wrong,
            resources=resources,
        )


def test_unknown_attempt_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(RecoveryRecord("science", "execution"))

    metadata = RecoveryExecutionMetadata(
        scientific_identity="science",
        execution_identity="execution",
        engine="aiida",
        computer_uuid=str(uuid4()),
    )

    with pytest.raises(ValueError, match="attempt number"):
        validate_reserved_execution_identity(
            store,
            metadata,
            resources={},
        )


def test_wrong_resources_rejected(tmp_path):
    store, metadata, resources = make_case(tmp_path)

    changed = dict(resources)
    changed["max_wallclock_seconds"] = 600

    with pytest.raises(ValueError, match="Execution identity mismatch"):
        validate_reserved_execution_identity(
            store,
            metadata,
            resources=changed,
        )


def test_wrong_execution_identity_rejected(tmp_path):
    store, metadata, resources = make_case(tmp_path)

    wrong = RecoveryExecutionMetadata(
        scientific_identity=metadata.scientific_identity,
        execution_identity="different-execution",
        engine="aiida",
        computer_uuid=metadata.computer_uuid,
    )

    with pytest.raises(ValueError, match="execution identity"):
        validate_reserved_execution_identity(
            store,
            wrong,
            resources=resources,
        )
