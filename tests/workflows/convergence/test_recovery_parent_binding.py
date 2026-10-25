
import sqlite3

import pytest

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord


PARENT_UUID = "11111111-1111-4111-8111-111111111111"
CHILD_UUID = "22222222-2222-4222-8222-222222222222"


def make_claimed(tmp_path):
    database = tmp_path / "recovery.sqlite"
    store = ReservationStore(database)

    record = store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=1,
        )
    )

    claimed = store.claim_submission(
        record,
        expected_version=0,
    )

    return store, claimed, database


def test_new_database_contains_parent_column(tmp_path):
    _, _, database = make_claimed(tmp_path)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(recovery_reservations)"
            )
        }

    assert "parent_process_uuid" in columns


def test_parent_and_child_are_distinct(tmp_path):
    store, claimed, database = make_claimed(tmp_path)

    store.bind_parent_process(claimed, PARENT_UUID)
    store.bind_process(claimed, CHILD_UUID)

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT parent_process_uuid, process_uuid
            FROM recovery_reservations
            WHERE scientific_identity = ?
            """,
            ("science",),
        ).fetchone()

    assert row == (PARENT_UUID, CHILD_UUID)


def test_duplicate_parent_binding_rejected(tmp_path):
    store, claimed, _ = make_claimed(tmp_path)

    store.bind_parent_process(claimed, PARENT_UUID)

    with pytest.raises(RuntimeError):
        store.bind_parent_process(claimed, CHILD_UUID)


def test_invalid_parent_uuid_rejected(tmp_path):
    store, claimed, _ = make_claimed(tmp_path)

    with pytest.raises(ValueError):
        store.bind_parent_process(claimed, "not-a-uuid")


def test_unclaimed_parent_binding_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    record = store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=1,
        )
    )

    with pytest.raises(ValueError):
        store.bind_parent_process(record, PARENT_UUID)



def test_parent_first_rejects_same_child_uuid(tmp_path):
    store, claimed, _ = make_claimed(tmp_path)

    store.bind_parent_process(claimed, PARENT_UUID)

    with pytest.raises(RuntimeError):
        store.bind_process(claimed, PARENT_UUID)


def test_child_first_rejects_same_parent_uuid(tmp_path):
    store, claimed, _ = make_claimed(tmp_path)

    store.bind_process(claimed, CHILD_UUID)

    with pytest.raises(RuntimeError):
        store.bind_parent_process(claimed, CHILD_UUID)
