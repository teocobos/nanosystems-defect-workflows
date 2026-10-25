"""Crash-recovery integration regression tests.

These tests exercise the real SQLite reservation store across
simulated process restarts without submitting AiiDA or HPC jobs.
"""

from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationConflict,
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState,
)
from nsdw.workflows.convergence.recovery_submission_boundary import (
    execute_claimed_submission,
)


def make_reserved(database):
    store = ReservationStore(database)
    record = store.reserve(
        RecoveryRecord(
            scientific_identity="science",
            execution_identity="execution",
            attempt_number=1,
        )
    )
    return store, record


def test_restart_before_submission_preserves_reservation(tmp_path):
    database = tmp_path / "recovery.sqlite"

    _, reserved = make_reserved(database)

    restarted = ReservationStore(database)
    recovered = restarted.get("science")

    assert recovered == reserved
    assert recovered.state is RecoveryState.RESERVED

    tracking = restarted.get_tracking("science")
    assert tracking["submission_started"] is False

    claimed = restarted.claim_submission(
        recovered,
        expected_version=tracking["version"],
    )

    assert claimed.state is RecoveryState.SUBMITTING

    with pytest.raises(ReservationConflict):
        restarted.claim_submission(
            recovered,
            expected_version=tracking["version"],
        )


def test_restart_after_submission_claim_blocks_callback(tmp_path):
    database = tmp_path / "recovery.sqlite"

    store, reserved = make_reserved(database)
    store.claim_submission(reserved, expected_version=0)

    restarted = ReservationStore(database)
    tracking = restarted.get_tracking("science")

    assert tracking["submission_started"] is True
    assert tracking["process_uuid"] is None
    assert tracking["requires_reconciliation"] is True

    callbacks = []

    def forbidden_callback():
        callbacks.append("submitted")
        return str(uuid4())

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            restarted,
            reserved,
            expected_version=0,
            submit_callback=forbidden_callback,
        )

    assert callbacks == []


def test_restart_after_process_binding_preserves_identity(tmp_path):
    database = tmp_path / "recovery.sqlite"

    store, reserved = make_reserved(database)
    process_uuid = str(uuid4())

    calls = []

    def submit_callback():
        calls.append("submitted")
        return process_uuid

    result = execute_claimed_submission(
        store,
        reserved,
        expected_version=0,
        submit_callback=submit_callback,
    )

    assert result == process_uuid

    restarted = ReservationStore(database)
    tracking = restarted.get_tracking("science")

    assert tracking["process_uuid"] == process_uuid
    assert tracking["submission_started"] is True

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            restarted,
            reserved,
            expected_version=0,
            submit_callback=submit_callback,
        )

    assert calls == ["submitted"]


def test_parent_child_identity_survives_restart(tmp_path):
    database = tmp_path / "recovery.sqlite"

    store, reserved = make_reserved(database)
    claimed = store.claim_submission(
        reserved,
        expected_version=0,
    )

    parent_uuid = str(uuid4())
    child_uuid = str(uuid4())

    assert parent_uuid != child_uuid

    store.bind_parent_process(claimed, parent_uuid)
    store.bind_process(claimed, child_uuid)

    restarted = ReservationStore(database)
    tracking = restarted.get_tracking("science")

    assert tracking["parent_process_uuid"] == parent_uuid
    assert tracking["process_uuid"] == child_uuid
    assert tracking["parent_process_uuid"] != tracking["process_uuid"]

    with pytest.raises(ReservationConflict):
        restarted.claim_submission(
            reserved,
            expected_version=0,
        )
