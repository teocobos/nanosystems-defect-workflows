"""AiiDA crash-reconciliation integration regression tests.

Exercise persisted SQLite recovery state across simulated restarts.
No live AiiDA processes or HPC jobs are submitted.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_reconciliation import (
    reconcile_aiida_reservation,
)
from nsdw.workflows.convergence.aiida_recovery_lookup import (
    lookup_recovery_process,
)
from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision as Decision,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationConflict,
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)
from nsdw.workflows.convergence.recovery_submission_boundary import (
    execute_claimed_submission,
)


class FakeWorkChainNode:
    """Minimal stand-in for AiiDA WorkChainNode type checks."""


def make_case(tmp_path, *, bind_process=True):
    database = tmp_path / "recovery.sqlite"
    store = ReservationStore(database)

    reserved = store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=2,
        )
    )

    claimed = store.claim_submission(
        reserved,
        expected_version=0,
    )

    parent_uuid = str(uuid4())
    child_uuid = str(uuid4())

    if bind_process:
        store.bind_parent_process(claimed, parent_uuid)
        store.bind_process(claimed, child_uuid)

    metadata = RecoveryExecutionMetadata(
        scientific_identity="science",
        execution_identity="execution",
        engine="aiida",
        computer_uuid=str(uuid4()),
        process_uuid=child_uuid if bind_process else None,
    )

    return database, reserved, metadata, parent_uuid, child_uuid


def reconcile(store, metadata, parent_uuid, *, provenance=True):
    parent = FakeWorkChainNode()
    parent.uuid = parent_uuid
    campaign = FakeWorkChainNode()

    with (
        patch(
            "nsdw.workflows.convergence.aiida_reconciliation."
            "WorkChainNode",
            FakeWorkChainNode,
        ),
        patch(
            "nsdw.workflows.convergence.aiida_reconciliation."
            "validate_aiida_reserved_provenance",
            return_value=provenance,
        ),
    ):
        return reconcile_aiida_reservation(
            store,
            metadata,
            expected_recovery_parent=parent,
            original_campaign=campaign,
        )


def test_verified_recovery_after_restart_is_read_only(tmp_path):
    database, reserved, metadata, parent_uuid, child_uuid = (
        make_case(tmp_path)
    )

    restarted = ReservationStore(database)
    before = restarted.get_tracking("science")

    result = reconcile(
        restarted,
        metadata,
        parent_uuid,
        provenance=True,
    )

    assert result.decision == Decision.VERIFIED
    assert restarted.get_tracking("science") == before
    assert before["parent_process_uuid"] == parent_uuid
    assert before["process_uuid"] == child_uuid

    callbacks = []

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            restarted,
            reserved,
            expected_version=0,
            submit_callback=lambda: callbacks.append("submitted"),
        )

    assert callbacks == []


def test_interrupted_submission_after_restart_remains_blocked(tmp_path):
    database, reserved, metadata, _, _ = make_case(
        tmp_path,
        bind_process=False,
    )

    restarted = ReservationStore(database)
    before = restarted.get_tracking("science")

    result = reconcile(
        restarted,
        metadata,
        str(uuid4()),
    )

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert restarted.get_tracking("science") == before

    callbacks = []

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            restarted,
            reserved,
            expected_version=0,
            submit_callback=lambda: callbacks.append("submitted"),
        )

    assert callbacks == []


def test_conflicting_parent_after_restart_fails_closed(tmp_path):
    database, reserved, metadata, parent_uuid, child_uuid = (
        make_case(tmp_path)
    )

    restarted = ReservationStore(database)
    before = restarted.get_tracking("science")

    wrong_parent_uuid = str(uuid4())

    assert wrong_parent_uuid not in (parent_uuid, child_uuid)

    result = reconcile(
        restarted,
        metadata,
        wrong_parent_uuid,
    )

    assert result.decision == Decision.RECONCILIATION_REQUIRED
    assert restarted.get_tracking("science") == before

    assert before["parent_process_uuid"] == parent_uuid
    assert before["process_uuid"] == child_uuid

    callbacks = []

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            restarted,
            reserved,
            expected_version=0,
            submit_callback=lambda: callbacks.append("submitted"),
        )

    assert callbacks == []
