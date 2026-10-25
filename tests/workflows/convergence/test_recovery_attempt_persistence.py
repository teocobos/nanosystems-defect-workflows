"""Persistent recovery attempt-number safety tests."""

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


def test_attempt_number_survives_restart(tmp_path):
    path = tmp_path / "recovery.sqlite"
    store = ReservationStore(path)

    record = RecoveryRecord(
        "science", "execution", attempt_number=2
    )
    reserved = store.reserve(record)

    reopened = ReservationStore(path)

    assert reopened.get("science") == reserved
    assert reopened.get_tracking("science")["attempt_number"] == 2


def test_unknown_attempt_remains_unknown(tmp_path):
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
            "INSERT INTO recovery_reservations VALUES (?, ?, ?, 0)",
            ("science", "execution", "reserved"),
        )

    store = ReservationStore(path)

    assert store.get("science").attempt_number is None
    assert store.get_tracking("science")["attempt_number"] is None

    assert ReservationStore(path).get(
        "science"
    ).attempt_number is None


def test_wrong_attempt_cannot_advance(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    reserved = store.reserve(
        RecoveryRecord("science", "execution", attempt_number=2)
    )

    impostor = RecoveryRecord(
        "science", "execution", S.RESERVED, attempt_number=3
    )

    with pytest.raises(ReservationConflict):
        store.advance(impostor, S.RECONCILIATION_REQUIRED)

    assert store.get("science") == reserved


def test_wrong_attempt_cannot_begin_submission(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(
        RecoveryRecord("science", "execution", attempt_number=2)
    )

    impostor = RecoveryRecord(
        "science", "execution", S.RESERVED, attempt_number=3
    )

    with pytest.raises(ReservationConflict):
        store.begin_submission(impostor)


def test_wrong_attempt_cannot_bind_process(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    submitting = store.begin_submission(
        store.reserve(
            RecoveryRecord(
                "science", "execution", attempt_number=2
            )
        )
    )

    impostor = RecoveryRecord(
        "science", "execution", S.SUBMITTING, attempt_number=3
    )

    with pytest.raises(ReservationConflict):
        store.bind_process(impostor, str(uuid4()))

    assert store.get_tracking("science")["process_uuid"] is None
    assert submitting.attempt_number == 2
