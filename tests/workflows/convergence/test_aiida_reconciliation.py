from unittest.mock import patch
from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_reconciliation import (
    reconcile_aiida_reservation,
)
from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision as Decision,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


def make_case(tmp_path, *, bound=True):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    process_uuid = str(uuid4())
    parent_uuid = str(uuid4())

    record = store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=2,
        )
    )

    if bound:
        record = store.begin_submission(record)
        store.bind_parent_process(record, parent_uuid)
        store.bind_process(record, process_uuid)

    metadata = RecoveryExecutionMetadata(
        scientific_identity="science",
        execution_identity="execution",
        engine="aiida",
        computer_uuid=str(uuid4()),
        process_uuid=process_uuid if bound else None,
    )

    return store, metadata


def reconcile(store, metadata):
    class FakeWorkChainNode:
        pass

    parent = FakeWorkChainNode()
    parent.uuid = store.get_tracking(
        metadata.scientific_identity
    )["parent_process_uuid"]
    campaign = FakeWorkChainNode()

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "WorkChainNode",
        FakeWorkChainNode,
    ):
        return reconcile_aiida_reservation(
            store,
            metadata,
            expected_recovery_parent=parent,
            original_campaign=campaign,
        )


def test_unsubmitted_reservation(tmp_path):
    store, metadata = make_case(tmp_path, bound=False)

    result = reconcile(store, metadata)

    assert result.decision == Decision.NOT_SUBMITTED


def test_verified_bound_process(tmp_path):
    store, metadata = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        return_value=True,
    ):
        result = reconcile(store, metadata)

    assert result.decision == Decision.VERIFIED


def test_failed_provenance(tmp_path):
    store, metadata = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        return_value=False,
    ):
        result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_provenance_exception(tmp_path):
    store, metadata = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        side_effect=ValueError("Invalid AiiDA provenance"),
    ):
        result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_process_identity_mismatch(tmp_path):
    store, metadata = make_case(tmp_path)

    altered = RecoveryExecutionMetadata(
        scientific_identity=metadata.scientific_identity,
        execution_identity=metadata.execution_identity,
        engine="aiida",
        computer_uuid=metadata.computer_uuid,
        process_uuid=str(uuid4()),
    )

    result = reconcile(store, altered)

    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_missing_process_binding(tmp_path):
    store, metadata = make_case(tmp_path, bound=False)
    record = store.begin_submission(store.get("science"))

    result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_reconciliation_does_not_modify_database(tmp_path):
    store, metadata = make_case(tmp_path)
    before = store.get_tracking("science")

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        return_value=True,
    ):
        reconcile(store, metadata)

    assert store.get_tracking("science") == before



def test_reservation_changed_during_verification(tmp_path):
    from nsdw.workflows.convergence.recovery_state import (
        RecoveryState,
    )

    store, metadata = make_case(tmp_path)

    def change_reservation(*args, **kwargs):
        record = store.get(metadata.scientific_identity)
        store.advance(
            record,
            RecoveryState.RECONCILIATION_REQUIRED,
        )
        return True

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        side_effect=change_reservation,
    ):
        result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "changed during" in result.reason



def test_invalid_recovery_parent_rejected(tmp_path):
    store, metadata = make_case(tmp_path)

    result = reconcile_aiida_reservation(
        store,
        metadata,
        expected_recovery_parent=None,
        original_campaign=None,
    )

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "recovery parent" in result.reason


def test_invalid_original_campaign_rejected(tmp_path):
    store, metadata = make_case(tmp_path)

    class FakeWorkChainNode:
        pass

    parent = FakeWorkChainNode()
    parent.uuid = store.get_tracking(
        metadata.scientific_identity
    )["parent_process_uuid"]

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "WorkChainNode",
        FakeWorkChainNode,
    ):
        result = reconcile_aiida_reservation(
            store,
            metadata,
            expected_recovery_parent=parent,
            original_campaign=None,
        )

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "Original campaign" in result.reason


def test_unexpected_provenance_exception_fails_closed(tmp_path):
    store, metadata = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        side_effect=RuntimeError("Unexpected AiiDA backend failure"),
    ):
        result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "could not be verified" in result.reason



def test_stale_snapshot_rejected(tmp_path):
    store, metadata = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        return_value=True,
    ), patch.object(
        store,
        "matches_snapshot",
        return_value=False,
    ):
        result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "snapshot" in result.reason


def test_matching_snapshot_verified(tmp_path):
    store, metadata = make_case(tmp_path)

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "validate_aiida_reserved_provenance",
        return_value=True,
    ):
        result = reconcile(store, metadata)

    assert result.decision == Decision.VERIFIED



def test_missing_parent_binding_fails_closed(tmp_path):
    store, metadata = make_case(tmp_path)

    # Simulate a legacy reservation with no parent UUID.
    with store._connect() as connection:
        connection.execute(
            """
            UPDATE recovery_reservations
            SET parent_process_uuid = NULL
            WHERE scientific_identity = ?
            """,
            (metadata.scientific_identity,),
        )

    result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "not bound" in result.reason


def test_conflicting_parent_binding_fails_closed(tmp_path):
    store, metadata = make_case(tmp_path)

    class FakeWorkChainNode:
        pass

    parent = FakeWorkChainNode()
    parent.uuid = str(uuid4())
    campaign = FakeWorkChainNode()

    with patch(
        "nsdw.workflows.convergence.aiida_reconciliation."
        "WorkChainNode",
        FakeWorkChainNode,
    ):
        result = reconcile_aiida_reservation(
            store,
            metadata,
            expected_recovery_parent=parent,
            original_campaign=campaign,
        )

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "does not match" in result.reason


def test_identical_parent_child_uuid_fails_closed(tmp_path):
    store, metadata = make_case(tmp_path)

    # Simulate corrupted persisted metadata.
    with store._connect() as connection:
        connection.execute(
            """
            UPDATE recovery_reservations
            SET parent_process_uuid = process_uuid
            WHERE scientific_identity = ?
            """,
            (metadata.scientific_identity,),
        )

    result = reconcile(store, metadata)

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert "identical UUIDs" in result.reason
