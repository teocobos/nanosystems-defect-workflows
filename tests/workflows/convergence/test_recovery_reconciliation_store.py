from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision as Decision,
)
from nsdw.workflows.convergence.recovery_reconciliation_store import (
    reconcile_reservation,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord


def make_store(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=2,
        )
    )
    return store


def test_reserved_without_submission(tmp_path):
    store = make_store(tmp_path)

    result = reconcile_reservation(store, "science")

    assert result.decision == Decision.NOT_SUBMITTED


def test_submission_started_without_process(tmp_path):
    store = make_store(tmp_path)
    store.begin_submission(store.get("science"))

    result = reconcile_reservation(store, "science")

    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_bound_process_without_provenance(tmp_path):
    store = make_store(tmp_path)
    record = store.begin_submission(store.get("science"))
    store.bind_process(record, str(uuid4()))

    result = reconcile_reservation(store, "science")

    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_bound_process_with_explicit_verification(tmp_path):
    store = make_store(tmp_path)
    record = store.begin_submission(store.get("science"))
    store.bind_process(record, str(uuid4()))

    result = reconcile_reservation(
        store,
        "science",
        provenance_verified=True,
    )

    assert result.decision == Decision.VERIFIED


def test_unknown_reservation(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        reconcile_reservation(store, "unknown")


def test_invalid_verification_flag(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(TypeError):
        reconcile_reservation(
            store,
            "science",
            provenance_verified=1,
        )


def test_reconciliation_is_read_only(tmp_path):
    store = make_store(tmp_path)
    before = store.get_tracking("science")

    reconcile_reservation(store, "science")

    after = store.get_tracking("science")
    assert after == before
