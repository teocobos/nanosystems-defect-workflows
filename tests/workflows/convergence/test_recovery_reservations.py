"""Atomic local reservation tests."""

import pytest

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationConflict,
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState as S,
)


def make_record(execution="attempt-one"):
    return RecoveryRecord(
        scientific_identity="scientific-recovery",
        execution_identity=execution,
    )


def test_reservation_persists_across_instances(tmp_path):
    path = tmp_path / "recovery.sqlite"
    first = ReservationStore(path)
    reserved = first.reserve(make_record())

    second = ReservationStore(path)
    assert second.get(reserved.scientific_identity) == reserved


def test_duplicate_reservation_is_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(make_record())

    with pytest.raises(ReservationConflict):
        store.reserve(make_record())


def test_different_execution_attempt_cannot_claim_same_science(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(make_record("lumi-attempt"))

    with pytest.raises(ReservationConflict):
        store.reserve(make_record("cirrus-attempt"))


def test_distinct_scientific_recoveries_can_coexist(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(make_record())

    other = RecoveryRecord("another-science", "another-attempt")
    assert store.reserve(other).state is S.RESERVED


def test_atomic_state_progression(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    current = store.reserve(make_record())

    from uuid import uuid4

    current = store.begin_submission(current)
    store.bind_process(current, str(uuid4()))

    for target in (
        S.SUBMITTED, S.RUNNING, S.COMPLETED
    ):
        current = store.advance(current, target)

    assert store.get(current.scientific_identity).state is S.COMPLETED


def test_stale_update_is_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    original = store.reserve(make_record())

    store.begin_submission(original)

    with pytest.raises(ReservationConflict):
        store.advance(original, S.RECONCILIATION_REQUIRED)


def test_wrong_execution_attempt_cannot_advance(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(make_record("original"))

    impostor = RecoveryRecord(
        "scientific-recovery",
        "different-attempt",
        S.RESERVED,
    )

    with pytest.raises(ReservationConflict):
        store.begin_submission(impostor)


def test_uncertain_submission_blocks_retries(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    current = store.reserve(make_record())
    current = store.begin_submission(current)
    current = store.advance(current, S.RECONCILIATION_REQUIRED)

    assert store.get(current.scientific_identity).state is (
        S.RECONCILIATION_REQUIRED
    )

    with pytest.raises(ReservationConflict):
        store.reserve(make_record("new-attempt"))

    with pytest.raises(ValueError):
        store.advance(current, S.SUBMITTING)


def test_invalid_state_transition_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    current = store.reserve(make_record())

    with pytest.raises(ValueError):
        store.advance(current, S.COMPLETED)

    assert store.get(current.scientific_identity).state is S.RESERVED


def test_database_parent_must_exist(tmp_path):
    with pytest.raises(ValueError):
        ReservationStore(tmp_path / "missing" / "recovery.sqlite")
