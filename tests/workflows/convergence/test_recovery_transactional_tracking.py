"""Transactional recovery process-tracking tests."""

import sqlite3
from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationConflict,
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState as S,
)


def make_record():
    return RecoveryRecord("scientific-id", "execution-id")


def test_existing_database_migrates_without_data_loss(tmp_path):
    path = tmp_path / "legacy.sqlite"

    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE recovery_reservations (
                scientific_identity TEXT PRIMARY KEY,
                execution_identity TEXT NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0
            )
        """)
        connection.execute(
            """
            INSERT INTO recovery_reservations
            VALUES (?, ?, ?, 0)
            """,
            ("scientific-id", "execution-id", "reserved"),
        )

    store = ReservationStore(path)

    assert store.get("scientific-id").state is S.RESERVED
    tracking = store.get_tracking("scientific-id")
    assert tracking["submission_started"] is False
    assert tracking["process_uuid"] is None

    # Migration must also be safe when repeated.
    assert ReservationStore(path).get("scientific-id") is not None


def test_submission_intent_is_durable(tmp_path):
    path = tmp_path / "recovery.sqlite"
    store = ReservationStore(path)
    reserved = store.reserve(make_record())

    submitting = store.begin_submission(reserved)

    reopened = ReservationStore(path)
    tracking = reopened.get_tracking("scientific-id")

    assert submitting.state is S.SUBMITTING
    assert tracking["submission_started"] is True
    assert tracking["requires_reconciliation"] is True


def test_process_binding_persists_after_restart(tmp_path):
    path = tmp_path / "recovery.sqlite"
    store = ReservationStore(path)

    submitting = store.begin_submission(
        store.reserve(make_record())
    )

    process_uuid = str(uuid4())
    store.bind_process(submitting, process_uuid)

    reopened = ReservationStore(path)
    tracking = reopened.get_tracking("scientific-id")

    assert tracking["process_uuid"] == process_uuid
    assert tracking["requires_reconciliation"] is False


def test_process_binding_cannot_be_replaced(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    submitting = store.begin_submission(
        store.reserve(make_record())
    )

    first = str(uuid4())
    store.bind_process(submitting, first)

    with pytest.raises(ReservationConflict):
        store.bind_process(submitting, str(uuid4()))

    assert store.get_tracking("scientific-id")["process_uuid"] == first


def test_duplicate_submission_intent_is_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    reserved = store.reserve(make_record())

    store.begin_submission(reserved)

    with pytest.raises(ReservationConflict):
        store.begin_submission(reserved)


def test_invalid_process_uuid_does_not_modify_database(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    submitting = store.begin_submission(
        store.reserve(make_record())
    )

    with pytest.raises(ValueError):
        store.bind_process(submitting, "invalid")

    assert store.get_tracking("scientific-id")["process_uuid"] is None


def test_wrong_attempt_cannot_bind_process(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.begin_submission(store.reserve(make_record()))

    impostor = RecoveryRecord(
        "scientific-id", "different-attempt", S.SUBMITTING
    )

    with pytest.raises(ReservationConflict):
        store.bind_process(impostor, str(uuid4()))


def test_tracking_of_unknown_recovery(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    assert store.get_tracking("unknown") is None


def test_existing_reservation_contract_remains_valid(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    reserved = store.reserve(make_record())

    assert store.get("scientific-id") == reserved

    updated = store.advance(reserved, S.RECONCILIATION_REQUIRED)
    assert updated.state is S.RECONCILIATION_REQUIRED
