"""Local SQLite prototype for atomic recovery reservations.

Not intended for shared HPC filesystems or multi-host deployment.
No AiiDA submission is performed here.
"""

import sqlite3
from pathlib import Path

from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState,
    transition,
)


class ReservationConflict(RuntimeError):
    """A scientific recovery identity has already been reserved."""


class ReservationStore:
    """Persistent local reservation store with transactional updates."""

    def __init__(self, database_path):
        self.path = Path(database_path)
        if str(self.path) == ":memory:":
            raise ValueError("Use a persistent local database path.")
        if not self.path.parent.is_dir():
            raise ValueError("Database parent directory does not exist.")
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self):
        """Create or migrate the local reservation schema atomically."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS recovery_reservations (
                        scientific_identity TEXT PRIMARY KEY,
                        execution_identity TEXT NOT NULL,
                        state TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 0,
                        submission_started INTEGER NOT NULL DEFAULT 0,
                        process_uuid TEXT,
                        parent_process_uuid TEXT,
                        attempt_number INTEGER
                    )
                """)

                columns = {
                    row[1] for row in connection.execute(
                        "PRAGMA table_info(recovery_reservations)"
                    )
                }

                if "submission_started" not in columns:
                    connection.execute("""
                        ALTER TABLE recovery_reservations
                        ADD COLUMN submission_started
                        INTEGER NOT NULL DEFAULT 0
                    """)

                if "process_uuid" not in columns:
                    connection.execute("""
                        ALTER TABLE recovery_reservations
                        ADD COLUMN process_uuid TEXT
                    """)


                if "parent_process_uuid" not in columns:
                    connection.execute(
                        "ALTER TABLE recovery_reservations "
                        "ADD COLUMN parent_process_uuid TEXT"
                    )

                if "attempt_number" not in columns:
                    connection.execute(
                        "ALTER TABLE recovery_reservations "
                        "ADD COLUMN attempt_number INTEGER"
                    )

                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def reserve(self, record):
        """Atomically reserve a previously unclaimed scientific identity."""
        if not isinstance(record, RecoveryRecord):
            raise TypeError("Expected RecoveryRecord.")
        if record.state is not RecoveryState.PREPARED:
            raise ValueError("Only prepared records may be reserved.")

        reserved = transition(record, RecoveryState.RESERVED)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO recovery_reservations
                    (scientific_identity, execution_identity, state,
                     version, attempt_number)
                    VALUES (?, ?, ?, 0, ?)
                    """,
                    (
                        reserved.scientific_identity,
                        reserved.execution_identity,
                        reserved.state.value,
                        reserved.attempt_number,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ReservationConflict(
                    "Scientific recovery already reserved."
                ) from exc
            except BaseException:
                connection.rollback()
                raise

        return reserved

    def get(self, scientific_identity):
        """Read the currently stored reservation."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT scientific_identity, execution_identity,
                       state, attempt_number
                FROM recovery_reservations
                WHERE scientific_identity = ?
                """,
                (scientific_identity,),
            ).fetchone()

        if row is None:
            return None

        return RecoveryRecord(
            scientific_identity=row[0],
            execution_identity=row[1],
            state=RecoveryState(row[2]),
            attempt_number=row[3],
        )

    def advance(self, record, target):
        """Atomically update only the expected state and attempt."""
        if not isinstance(record, RecoveryRecord):
            raise TypeError("Expected RecoveryRecord.")

        updated = transition(record, target)

        # Submission intent must be durably recorded by
        # begin_submission(), never by the generic state updater.
        if target is RecoveryState.SUBMITTING:
            raise ValueError(
                "Use begin_submission() to enter SUBMITTING."
            )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE recovery_reservations
                    SET state = ?, version = version + 1
                    WHERE scientific_identity = ?
                      AND execution_identity = ?
                      AND attempt_number IS ?
                      AND state = ?
                      AND (
                          ? != ?
                          OR (
                              submission_started = 1
                              AND process_uuid IS NOT NULL
                          )
                      )
                    """,
                    (
                        updated.state.value,
                        record.scientific_identity,
                        record.execution_identity,
                        record.attempt_number,
                        record.state.value,
                        updated.state.value,
                        RecoveryState.SUBMITTED.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ReservationConflict(
                        "Reservation missing, stale or owned "
                        "by another execution attempt."
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        return updated

    def begin_submission(self, record, *, expected_version=None):
        """Atomically record intent before invoking a submission API.

        A crash after this transaction must be reconciled; it must
        not automatically cause another submission.
        """
        if not isinstance(record, RecoveryRecord):
            raise TypeError("Expected RecoveryRecord.")

        if record.state is not RecoveryState.RESERVED:
            raise ValueError("Submission requires RESERVED state.")

        if expected_version is not None and (
            type(expected_version) is not int or expected_version < 0
        ):
            raise ValueError(
                "Expected version must be a nonnegative integer."
            )

        updated = transition(record, RecoveryState.SUBMITTING)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE recovery_reservations
                    SET state = ?, submission_started = 1,
                        version = version + 1
                    WHERE scientific_identity = ?
                      AND execution_identity = ?
                      AND attempt_number IS ?
                      AND state = ?
                      AND submission_started = 0
                      AND process_uuid IS NULL
                      AND (? IS NULL OR version = ?)
                    """,
                    (
                        updated.state.value,
                        record.scientific_identity,
                        record.execution_identity,
                        record.attempt_number,
                        record.state.value,
                        expected_version,
                        expected_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ReservationConflict(
                        "Submission already started or reservation stale."
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        return updated

    def claim_submission(self, record, *, expected_version: int):
        """Atomically claim submission using a mandatory version token.

        This records durable submission intent. It does not call AiiDA
        or submit an HPC calculation.
        """
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError(
                "Expected version must be a nonnegative integer."
            )

        return self.begin_submission(
            record,
            expected_version=expected_version,
        )

    def bind_process(self, record, process_uuid):
        """Atomically bind a process UUID to a submitting attempt."""
        from uuid import UUID

        if not isinstance(record, RecoveryRecord):
            raise TypeError("Expected RecoveryRecord.")
        if record.state is not RecoveryState.SUBMITTING:
            raise ValueError("Process binding requires SUBMITTING state.")
        if not isinstance(process_uuid, str):
            raise TypeError("Process UUID must be a string.")

        try:
            parsed = UUID(process_uuid)
        except (ValueError, AttributeError) as exc:
            raise ValueError("Invalid process UUID.") from exc

        if str(parsed) != process_uuid:
            raise ValueError("Process UUID must be canonical.")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE recovery_reservations
                    SET process_uuid = ?, version = version + 1
                    WHERE scientific_identity = ?
                      AND execution_identity = ?
                      AND attempt_number IS ?
                      AND state = ?
                      AND submission_started = 1
                      AND process_uuid IS NULL
                      AND (
                          parent_process_uuid IS NULL
                          OR parent_process_uuid != ?
                      )
                    """,
                    (
                        process_uuid,
                        record.scientific_identity,
                        record.execution_identity,
                        record.attempt_number,
                        record.state.value,
                        process_uuid,
                    ),
                )

                if cursor.rowcount != 1:
                    raise ReservationConflict(
                        "Process already bound or reservation stale."
                    )

                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        return process_uuid


    def bind_parent_process(self, record, parent_process_uuid):
        """Atomically bind a recovery-parent UUID to a claimed attempt."""
        from uuid import UUID

        if not isinstance(record, RecoveryRecord):
            raise TypeError("Expected RecoveryRecord.")

        if record.state is not RecoveryState.SUBMITTING:
            raise ValueError(
                "Parent binding requires SUBMITTING state."
            )

        if not isinstance(parent_process_uuid, str):
            raise TypeError("Parent process UUID must be a string.")

        try:
            parsed = UUID(parent_process_uuid)
        except (ValueError, AttributeError) as exc:
            raise ValueError("Invalid parent process UUID.") from exc

        if str(parsed) != parent_process_uuid:
            raise ValueError("Parent process UUID must be canonical.")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                cursor = connection.execute(
                    """
                    UPDATE recovery_reservations
                    SET parent_process_uuid = ?,
                        version = version + 1
                    WHERE scientific_identity = ?
                      AND execution_identity = ?
                      AND attempt_number IS ?
                      AND state = ?
                      AND submission_started = 1
                      AND parent_process_uuid IS NULL
                      AND (
                          process_uuid IS NULL
                          OR process_uuid != ?
                      )
                    """,
                    (
                        parent_process_uuid,
                        record.scientific_identity,
                        record.execution_identity,
                        record.attempt_number,
                        RecoveryState.SUBMITTING.value,
                        parent_process_uuid,
                    ),
                )

                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "Parent binding rejected: stale or "
                        "conflicting recovery attempt."
                    )

                connection.commit()

            except Exception:
                connection.rollback()
                raise

        return parent_process_uuid

    def get_tracking(self, scientific_identity):
        """Return durable submission metadata for reconciliation."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT execution_identity, state,
                       submission_started, process_uuid,
                       attempt_number, version, parent_process_uuid
                FROM recovery_reservations
                WHERE scientific_identity = ?
                """,
                (scientific_identity,),
            ).fetchone()

        if row is None:
            return None

        return {
            "scientific_identity": scientific_identity,
            "execution_identity": row[0],
            "state": row[1],
            "submission_started": bool(row[2]),
            "process_uuid": row[3],
            "attempt_number": row[4],
            "version": row[5],
            "parent_process_uuid": row[6],
            "requires_reconciliation": (
                bool(row[2])
                and row[3] is None
                and row[1] not in (
                    RecoveryState.COMPLETED.value,
                    RecoveryState.FAILED.value,
                )
            ),
        }


    def matches_snapshot(
        self,
        *,
        scientific_identity: str,
        execution_identity: str,
        process_uuid: str,
        attempt_number: int,
        version: int,
    ) -> bool:
        """Check a persisted attempt snapshot in one read transaction.

        This method is read-only and does not authorise execution.
        """
        if not isinstance(scientific_identity, str) or not scientific_identity:
            raise ValueError("Scientific identity must be nonempty.")

        if not isinstance(execution_identity, str) or not execution_identity:
            raise ValueError("Execution identity must be nonempty.")

        if not isinstance(process_uuid, str) or not process_uuid:
            raise ValueError("Process UUID must be nonempty.")

        if type(attempt_number) is not int or attempt_number <= 0:
            raise ValueError("Attempt number must be a positive integer.")

        if type(version) is not int or version < 0:
            raise ValueError("Version must be a nonnegative integer.")

        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM recovery_reservations
                    WHERE scientific_identity = ?
                      AND execution_identity = ?
                      AND process_uuid = ?
                      AND attempt_number = ?
                      AND version = ?
                      AND submission_started = 1
                      AND state IN (
                          'submitting',
                          'submitted',
                          'running',
                          'completed',
                          'failed'
                      )
                    """,
                    (
                        scientific_identity,
                        execution_identity,
                        process_uuid,
                        attempt_number,
                        version,
                    ),
                ).fetchone()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        return row is not None

    def get_correlation_id(self, scientific_identity: str) -> str:
        """Derive the correlation ID from a persisted recovery attempt.

        Read-only. Does not reserve, submit or reconcile a process.
        """
        from nsdw.workflows.convergence.recovery_correlation import (
            recovery_correlation_id,
        )

        if not isinstance(scientific_identity, str) or not scientific_identity:
            raise ValueError("Invalid scientific identity.")

        tracking = self.get_tracking(scientific_identity)

        if tracking is None:
            raise ValueError("Recovery reservation not found.")

        attempt_number = tracking["attempt_number"]

        if type(attempt_number) is not int or attempt_number < 1:
            raise ValueError(
                "Cannot derive correlation ID without a valid attempt number."
            )

        return recovery_correlation_id(
            tracking["scientific_identity"],
            tracking["execution_identity"],
            attempt_number,
        )
