from uuid import uuid4

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState,
)


def make_store(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    record = store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=2,
        )
    )
    return store, record


def test_initial_version_is_zero(tmp_path):
    store, _ = make_store(tmp_path)

    assert store.get_tracking("science")["version"] == 0


def test_submission_increments_version(tmp_path):
    store, record = make_store(tmp_path)

    store.begin_submission(record)

    assert store.get_tracking("science")["version"] == 1


def test_process_binding_increments_version(tmp_path):
    store, record = make_store(tmp_path)

    record = store.begin_submission(record)
    store.bind_process(record, str(uuid4()))

    assert store.get_tracking("science")["version"] == 2


def test_state_transition_increments_version(tmp_path):
    store, record = make_store(tmp_path)

    record = store.begin_submission(record)
    store.bind_process(record, str(uuid4()))
    store.advance(record, RecoveryState.SUBMITTED)

    assert store.get_tracking("science")["version"] == 3


def test_tracking_read_does_not_increment_version(tmp_path):
    store, _ = make_store(tmp_path)

    before = store.get_tracking("science")
    after = store.get_tracking("science")

    assert before == after
    assert after["version"] == 0


def test_unknown_reservation_has_no_version(tmp_path):
    store, _ = make_store(tmp_path)

    assert store.get_tracking("unknown") is None
