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


def test_strict_claim_succeeds(tmp_path):
    store, record = make_case(tmp_path)

    updated = store.claim_submission(
        record,
        expected_version=0,
    )

    assert updated.state is RecoveryState.SUBMITTING
    assert store.get_tracking("science")["submission_started"] is True
    assert store.get_tracking("science")["version"] == 1


def test_strict_claim_rejects_stale_version(tmp_path):
    store, record = make_case(tmp_path)

    with pytest.raises(ReservationConflict):
        store.claim_submission(
            record,
            expected_version=1,
        )

    assert store.get_tracking("science")["version"] == 0


def test_strict_claim_rejects_duplicate(tmp_path):
    store, record = make_case(tmp_path)

    store.claim_submission(record, expected_version=0)

    with pytest.raises(ReservationConflict):
        store.claim_submission(record, expected_version=0)


def test_strict_claim_requires_version(tmp_path):
    store, record = make_case(tmp_path)

    with pytest.raises(TypeError):
        store.claim_submission(record)


@pytest.mark.parametrize(
    "version",
    [None, True, -1, 1.5, "0"],
)
def test_strict_claim_rejects_invalid_version(tmp_path, version):
    store, record = make_case(tmp_path)

    with pytest.raises(ValueError):
        store.claim_submission(
            record,
            expected_version=version,
        )

    assert store.get_tracking("science")["version"] == 0
