"""Safety requirements for durable submission transitions."""

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


def prepared():
    return RecoveryRecord("science", "attempt")


def test_generic_advance_cannot_start_submission(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    reserved = store.reserve(prepared())

    with pytest.raises((ReservationConflict, ValueError)):
        store.advance(reserved, S.SUBMITTING)

    tracking = store.get_tracking("science")
    assert tracking["state"] == S.RESERVED.value
    assert tracking["submission_started"] is False


def test_submitted_requires_bound_process(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    submitting = store.begin_submission(
        store.reserve(prepared())
    )

    with pytest.raises((ReservationConflict, ValueError)):
        store.advance(submitting, S.SUBMITTED)

    assert store.get_tracking("science")["process_uuid"] is None


def test_submitted_after_process_binding(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    submitting = store.begin_submission(
        store.reserve(prepared())
    )

    process_uuid = str(uuid4())
    store.bind_process(submitting, process_uuid)

    submitted = store.advance(submitting, S.SUBMITTED)

    assert submitted.state is S.SUBMITTED
    assert store.get_tracking("science")["process_uuid"] == process_uuid


def test_uncertain_submission_remains_blocked(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    submitting = store.begin_submission(
        store.reserve(prepared())
    )

    reconciled = store.advance(
        submitting, S.RECONCILIATION_REQUIRED
    )

    assert reconciled.state is S.RECONCILIATION_REQUIRED

    with pytest.raises((ReservationConflict, ValueError)):
        store.advance(reconciled, S.SUBMITTED)
