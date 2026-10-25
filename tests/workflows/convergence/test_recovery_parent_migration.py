
"""Backward compatibility tests for recovery reservation databases."""

import sqlite3

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)


CHILD_UUID = "22222222-2222-4222-8222-222222222222"


def test_existing_reservation_survives_parent_column_migration(tmp_path):
    database = tmp_path / "legacy-recovery.sqlite"

    # Simulate the schema before parent_process_uuid was introduced.
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE recovery_reservations (
                scientific_identity TEXT PRIMARY KEY,
                execution_identity TEXT NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                submission_started INTEGER NOT NULL DEFAULT 0,
                process_uuid TEXT,
                attempt_number INTEGER
            )
            """
        )

        connection.execute(
            """
            INSERT INTO recovery_reservations (
                scientific_identity,
                execution_identity,
                state,
                version,
                submission_started,
                process_uuid,
                attempt_number
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-science",
                "legacy-execution",
                "submitted",
                3,
                1,
                CHILD_UUID,
                2,
            ),
        )

    # Initialising the current store should migrate the legacy schema.
    store = ReservationStore(database)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(recovery_reservations)"
            )
        }

        row = connection.execute(
            """
            SELECT scientific_identity,
                   execution_identity,
                   state,
                   version,
                   submission_started,
                   process_uuid,
                   attempt_number,
                   parent_process_uuid
            FROM recovery_reservations
            WHERE scientific_identity = ?
            """,
            ("legacy-science",),
        ).fetchone()

    assert "parent_process_uuid" in columns

    assert row == (
        "legacy-science",
        "legacy-execution",
        "submitted",
        3,
        1,
        CHILD_UUID,
        2,
        None,
    )

    tracking = store.get_tracking("legacy-science")

    assert tracking["process_uuid"] == CHILD_UUID
    assert tracking["attempt_number"] == 2
    assert tracking["version"] == 3


def test_parent_column_migration_is_idempotent(tmp_path):
    database = tmp_path / "legacy-recovery.sqlite"

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE recovery_reservations (
                scientific_identity TEXT PRIMARY KEY,
                execution_identity TEXT NOT NULL,
                state TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                submission_started INTEGER NOT NULL DEFAULT 0,
                process_uuid TEXT,
                attempt_number INTEGER
            )
            """
        )

    ReservationStore(database)
    ReservationStore(database)

    with sqlite3.connect(database) as connection:
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(recovery_reservations)"
            )
        ]

    assert columns.count("parent_process_uuid") == 1
