"""Read-only AiiDA reservation provenance boundary tests."""

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_reserved_provenance import (
    validate_aiida_reserved_provenance,
)
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


def make_case(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    computer_uuid = str(uuid4())
    process_uuid = str(uuid4())
    parent_uuid = str(uuid4())
    resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 1800,
        "queue_name": "small",
        "account": "project-example",
    }

    science = "science-example"
    identity = execution_attempt_identity(
        scientific_identity=science,
        backend="aiida",
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=2,
    )

    reserved = store.reserve(
        RecoveryRecord(
            science, identity, attempt_number=2
        )
    )
    submitting = store.begin_submission(reserved)
    store.bind_parent_process(submitting, parent_uuid)
    store.bind_process(submitting, process_uuid)

    metadata = RecoveryExecutionMetadata(
        scientific_identity=science,
        execution_identity=identity,
        engine="aiida",
        computer_uuid=computer_uuid,
        process_uuid=process_uuid,
    )

    process = SimpleNamespace(
        inputs=SimpleNamespace(
            resources=SimpleNamespace(
                get_dict=lambda: dict(resources)
            )
        )
    )

    return store, metadata, process


def validate(store, metadata, process):
    tracking = store.get_tracking(metadata.scientific_identity)

    parent = SimpleNamespace(
        uuid=(
            tracking["parent_process_uuid"]
            if tracking is not None
            else None
        )
    )
    with patch(
        "nsdw.workflows.convergence."
        "aiida_reserved_provenance.validate_tracking_identity",
        return_value=process,
    ):
        return validate_aiida_reserved_provenance(
            store,
            metadata,
            expected_recovery_parent=parent,
            original_campaign=object(),
        )


def test_matching_bound_process(tmp_path):
    store, metadata, process = make_case(tmp_path)
    assert validate(store, metadata, process) is True


def test_missing_metadata_process_uuid(tmp_path):
    store, metadata, process = make_case(tmp_path)

    metadata = RecoveryExecutionMetadata(
        scientific_identity=metadata.scientific_identity,
        execution_identity=metadata.execution_identity,
        engine="aiida",
        computer_uuid=metadata.computer_uuid,
    )

    with pytest.raises(ValueError, match="Missing process UUID"):
        validate(store, metadata, process)


def test_mismatched_process_uuid(tmp_path):
    store, metadata, process = make_case(tmp_path)

    metadata = RecoveryExecutionMetadata(
        scientific_identity=metadata.scientific_identity,
        execution_identity=metadata.execution_identity,
        engine="aiida",
        computer_uuid=metadata.computer_uuid,
        process_uuid=str(uuid4()),
    )

    with pytest.raises(ValueError, match="Reserved process UUID mismatch"):
        validate(store, metadata, process)


def test_changed_process_resources(tmp_path):
    store, metadata, process = make_case(tmp_path)

    process.inputs.resources.get_dict = lambda: {
        "num_machines": 2
    }

    with pytest.raises(ValueError, match="Execution identity mismatch"):
        validate(store, metadata, process)


def test_missing_reservation(tmp_path):
    store, metadata, process = make_case(tmp_path)

    metadata = RecoveryExecutionMetadata(
        scientific_identity="missing",
        execution_identity=metadata.execution_identity,
        engine="aiida",
        computer_uuid=metadata.computer_uuid,
        process_uuid=metadata.process_uuid,
    )

    with pytest.raises(ValueError, match="reservation"):
        validate(store, metadata, process)



def test_unbound_reservation_process(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    computer_uuid = str(uuid4())
    process_uuid = str(uuid4())
    resources = {"max_wallclock_seconds": 1800}

    identity = execution_attempt_identity(
        scientific_identity="science",
        backend="aiida",
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=2,
    )

    store.reserve(
        RecoveryRecord("science", identity, attempt_number=2)
    )

    metadata = RecoveryExecutionMetadata(
        scientific_identity="science",
        execution_identity=identity,
        engine="aiida",
        computer_uuid=computer_uuid,
        process_uuid=process_uuid,
    )

    with pytest.raises(ValueError, match="unbound"):
        validate(store, metadata, object())


def test_unknown_attempt_number(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    process_uuid = str(uuid4())

    reserved = store.reserve(
        RecoveryRecord("science", "execution")
    )
    submitting = store.begin_submission(reserved)
    store.bind_process(submitting, process_uuid)

    metadata = RecoveryExecutionMetadata(
        scientific_identity="science",
        execution_identity="execution",
        engine="aiida",
        computer_uuid=str(uuid4()),
        process_uuid=process_uuid,
    )

    with pytest.raises(ValueError, match="Unknown recovery attempt"):
        validate(store, metadata, object())


def test_wrong_computer_uuid(tmp_path):
    store, metadata, process = make_case(tmp_path)

    wrong = RecoveryExecutionMetadata(
        scientific_identity=metadata.scientific_identity,
        execution_identity=metadata.execution_identity,
        engine="aiida",
        computer_uuid=str(uuid4()),
        process_uuid=metadata.process_uuid,
    )

    with pytest.raises(ValueError, match="Execution identity mismatch"):
        validate(store, wrong, process)


def test_aiida_provenance_rejection_propagates(tmp_path):
    store, metadata, _ = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence."
        "aiida_reserved_provenance.validate_tracking_identity",
        side_effect=ValueError(
            "Imported recovery child provenance rejected."
        ),
    ):
        with pytest.raises(ValueError, match="Imported"):
            validate_aiida_reserved_provenance(
                store,
                metadata,
                expected_recovery_parent=SimpleNamespace(
                    uuid=store.get_tracking(
                        metadata.scientific_identity
                    )["parent_process_uuid"]
                ),
                original_campaign=object(),
            )



def test_prepared_reservation_without_binding_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    computer_uuid = str(uuid4())
    resources = {"max_wallclock_seconds": 1800}

    identity = execution_attempt_identity(
        scientific_identity="science",
        backend="aiida",
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=2,
    )

    store.reserve(
        RecoveryRecord("science", identity, attempt_number=2)
    )

    metadata = RecoveryExecutionMetadata(
        scientific_identity="science",
        execution_identity=identity,
        engine="aiida",
        computer_uuid=computer_uuid,
        process_uuid=str(uuid4()),
    )

    with pytest.raises(ValueError, match="unbound"):
        validate(store, metadata, object())


def test_reconciliation_state_rejected(tmp_path):
    from nsdw.workflows.convergence.recovery_state import (
        RecoveryState,
    )

    store, metadata, process = make_case(tmp_path)
    record = store.get(metadata.scientific_identity)

    store.advance(
        record,
        RecoveryState.RECONCILIATION_REQUIRED,
    )

    with pytest.raises(ValueError, match="lifecycle state"):
        validate(store, metadata, process)



def test_bound_process_without_submission_rejected(tmp_path):
    store, metadata, process = make_case(tmp_path)

    tracking = store.get_tracking(metadata.scientific_identity)
    assert tracking["process_uuid"] == metadata.process_uuid
    assert tracking["submission_started"] is True

    inconsistent = dict(tracking)
    inconsistent["submission_started"] = False

    with patch.object(
        store,
        "get_tracking",
        return_value=inconsistent,
    ):
        with pytest.raises(
            ValueError,
            match="submission has not started",
        ):
            validate(store, metadata, process)



def test_direct_validator_rejects_missing_parent_binding(tmp_path):
    store, metadata, process = make_case(tmp_path)

    with store._connect() as connection:
        connection.execute(
            """
            UPDATE recovery_reservations
            SET parent_process_uuid = NULL
            WHERE scientific_identity = ?
            """,
            (metadata.scientific_identity,),
        )

    with pytest.raises(ValueError, match="not bound"):
        validate(store, metadata, process)


def test_direct_validator_rejects_wrong_parent(tmp_path):
    store, metadata, process = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence."
        "aiida_reserved_provenance.validate_tracking_identity",
        return_value=process,
    ):
        with pytest.raises(ValueError, match="does not match"):
            validate_aiida_reserved_provenance(
                store,
                metadata,
                expected_recovery_parent=SimpleNamespace(
                    uuid=str(uuid4())
                ),
                original_campaign=object(),
            )


def test_direct_validator_rejects_parent_child_collision(tmp_path):
    store, metadata, process = make_case(tmp_path)

    with store._connect() as connection:
        connection.execute(
            """
            UPDATE recovery_reservations
            SET parent_process_uuid = process_uuid
            WHERE scientific_identity = ?
            """,
            (metadata.scientific_identity,),
        )

    with pytest.raises(ValueError, match="identical UUIDs"):
        validate(store, metadata, process)
