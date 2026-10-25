from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_inspection import (
    inspect_recovery,
)
from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
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


def test_reserved_attempt_not_submitted(tmp_path):
    store, record = make_case(tmp_path)

    result = inspect_recovery(store, "science")

    assert result.decision is ReconciliationDecision.NOT_SUBMITTED
    assert result.requires_manual_reconciliation is False
    assert result.process_uuid is None


def test_interrupted_submission_requires_reconciliation(tmp_path):
    store, record = make_case(tmp_path)
    store.claim_submission(record, expected_version=0)

    result = inspect_recovery(store, "science")

    assert (
        result.decision
        is ReconciliationDecision.RECONCILIATION_REQUIRED
    )
    assert result.requires_manual_reconciliation is True
    assert result.process_uuid is None


def test_bound_process_not_automatically_verified(tmp_path):
    store, record = make_case(tmp_path)
    claimed = store.claim_submission(record, expected_version=0)

    process_uuid = str(uuid4())
    store.bind_process(claimed, process_uuid)

    result = inspect_recovery(store, "science")

    assert result.process_uuid == process_uuid
    assert (
        result.decision
        is ReconciliationDecision.RECONCILIATION_REQUIRED
    )


def test_missing_reservation_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    with pytest.raises(ValueError):
        inspect_recovery(store, "missing")


def test_invalid_store_rejected():
    with pytest.raises(TypeError):
        inspect_recovery(None, "science")
