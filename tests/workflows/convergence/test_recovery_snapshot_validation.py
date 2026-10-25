from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState,
)


def make_bound_store(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    record = store.reserve(
        RecoveryRecord("science", "execution", attempt_number=2)
    )
    record = store.begin_submission(record)
    process_uuid = str(uuid4())
    store.bind_process(record, process_uuid)
    return store, record, process_uuid


def check(store, process_uuid, *, version=2, attempt_number=2):
    return store.matches_snapshot(
        scientific_identity="science",
        execution_identity="execution",
        process_uuid=process_uuid,
        attempt_number=attempt_number,
        version=version,
    )


def test_matching_snapshot(tmp_path):
    store, _, process_uuid = make_bound_store(tmp_path)
    assert check(store, process_uuid) is True


def test_stale_version_rejected(tmp_path):
    store, record, process_uuid = make_bound_store(tmp_path)
    store.advance(record, RecoveryState.SUBMITTED)
    assert check(store, process_uuid) is False


def test_current_version_accepted(tmp_path):
    store, record, process_uuid = make_bound_store(tmp_path)
    store.advance(record, RecoveryState.SUBMITTED)
    assert check(store, process_uuid, version=3) is True


def test_wrong_process_rejected(tmp_path):
    store, _, _ = make_bound_store(tmp_path)
    assert check(store, str(uuid4())) is False


def test_wrong_attempt_rejected(tmp_path):
    store, _, process_uuid = make_bound_store(tmp_path)
    assert check(store, process_uuid, attempt_number=3) is False


def test_missing_reservation_rejected(tmp_path):
    store, _, process_uuid = make_bound_store(tmp_path)
    assert store.matches_snapshot(
        scientific_identity="unknown",
        execution_identity="execution",
        process_uuid=process_uuid,
        attempt_number=2,
        version=2,
    ) is False


def test_reconciliation_state_rejected(tmp_path):
    store, record, process_uuid = make_bound_store(tmp_path)
    store.advance(record, RecoveryState.RECONCILIATION_REQUIRED)
    assert check(store, process_uuid, version=3) is False


def test_snapshot_check_is_read_only(tmp_path):
    store, _, process_uuid = make_bound_store(tmp_path)
    before = store.get_tracking("science")
    assert check(store, process_uuid) is True
    assert store.get_tracking("science") == before


@pytest.mark.parametrize(
    "attempt,version",
    [(True, 2), (2, True), (0, 2), (2, -1)],
)
def test_invalid_numeric_inputs(tmp_path, attempt, version):
    store, _, process_uuid = make_bound_store(tmp_path)
    with pytest.raises(ValueError):
        check(
            store,
            process_uuid,
            attempt_number=attempt,
            version=version,
        )
