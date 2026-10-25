
import pytest

from nsdw.workflows.convergence.recovery_correlation import (
    recovery_correlation_id,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord


def test_correlation_matches_persisted_reservation(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=3,
        )
    )

    assert store.get_correlation_id("science") == (
        recovery_correlation_id("science", "execution", 3)
    )


def test_correlation_survives_store_reopening(tmp_path):
    path = tmp_path / "recovery.sqlite"
    store = ReservationStore(path)

    store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=1,
        )
    )

    original = store.get_correlation_id("science")
    reopened = ReservationStore(path)

    assert reopened.get_correlation_id("science") == original


def test_missing_reservation_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    with pytest.raises(ValueError):
        store.get_correlation_id("missing")


def test_missing_attempt_number_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    store.reserve(
        RecoveryRecord(
            "science",
            "execution",
        )
    )

    with pytest.raises(ValueError):
        store.get_correlation_id("science")


def test_invalid_identity_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    with pytest.raises(ValueError):
        store.get_correlation_id("")
