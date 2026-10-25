import pytest

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationConflict,
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState,
)


def make_case(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    record = store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=2,
        )
    )
    return store, record


def test_matching_version_begins_submission(tmp_path):
    store, record = make_case(tmp_path)

    updated = store.begin_submission(record, expected_version=0)

    assert updated.state is RecoveryState.SUBMITTING

    tracking = store.get_tracking("science")
    assert tracking["state"] == "submitting"
    assert tracking["submission_started"] is True
    assert tracking["version"] == 1


def test_stale_version_rejected_without_mutation(tmp_path):
    store, record = make_case(tmp_path)

    before = store.get_tracking("science")

    with pytest.raises(ReservationConflict):
        store.begin_submission(record, expected_version=1)

    assert store.get_tracking("science") == before


def test_duplicate_submission_rejected(tmp_path):
    store, record = make_case(tmp_path)

    store.begin_submission(record, expected_version=0)

    with pytest.raises(ReservationConflict):
        store.begin_submission(record, expected_version=0)

    assert store.get_tracking("science")["version"] == 1


def test_legacy_call_remains_compatible(tmp_path):
    store, record = make_case(tmp_path)

    updated = store.begin_submission(record)

    assert updated.state is RecoveryState.SUBMITTING


@pytest.mark.parametrize("version", [-1, True, 1.5, "0"])
def test_invalid_expected_version_rejected(tmp_path, version):
    store, record = make_case(tmp_path)

    with pytest.raises(ValueError):
        store.begin_submission(record, expected_version=version)

    assert store.get_tracking("science")["version"] == 0
