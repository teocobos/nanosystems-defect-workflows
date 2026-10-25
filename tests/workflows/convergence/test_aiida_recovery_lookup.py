
from dataclasses import dataclass

import pytest

from nsdw.workflows.convergence.aiida_recovery_discovery import (
    DiscoveryStatus,
)
from nsdw.workflows.convergence.aiida_recovery_lookup import (
    lookup_recovery_process,
)
from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord


@dataclass
class FakeDiscovery:
    status: DiscoveryStatus
    process_uuid: str | None = None


def make_store(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    record = store.reserve(
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=1,
        )
    )
    return store, record


def discover(status, uuid=None):
    return lambda correlation: FakeDiscovery(status, uuid)


def test_unstarted_reservation_is_not_submitted(tmp_path):
    store, record = make_store(tmp_path)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.NOT_FOUND),
    )

    assert result.decision is ReconciliationDecision.NOT_SUBMITTED


def test_interrupted_submission_remains_blocked(tmp_path):
    store, record = make_store(tmp_path)
    store.claim_submission(record, expected_version=0)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.NOT_FOUND),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED


def test_discovered_process_is_not_automatically_verified(tmp_path):
    store, record = make_store(tmp_path)
    store.claim_submission(record, expected_version=0)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.FOUND, "process-uuid"),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED
    assert result.discovered_uuid == "process-uuid"


def test_ambiguous_discovery_is_blocked(tmp_path):
    store, record = make_store(tmp_path)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.AMBIGUOUS),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED


def test_conflicting_persisted_uuid_is_blocked(tmp_path):
    store, record = make_store(tmp_path)
    claimed = store.claim_submission(record, expected_version=0)
    store.bind_process(claimed, "11111111-1111-4111-8111-111111111111")

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.FOUND, "22222222-2222-4222-8222-222222222222"),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED
    assert "not been bound" in result.reason


def test_missing_reservation_rejected(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    with pytest.raises(ValueError):
        lookup_recovery_process(
            store,
            "missing",
            discovery=discover(DiscoveryStatus.NOT_FOUND),
        )


def test_invalid_store_rejected():
    with pytest.raises(TypeError):
        lookup_recovery_process(
            None,
            "science",
            discovery=discover(DiscoveryStatus.NOT_FOUND),
        )



PARENT_UUID = "33333333-3333-4333-8333-333333333333"
CHILD_UUID = "44444444-4444-4444-8444-444444444444"
OTHER_PARENT_UUID = "55555555-5555-4555-8555-555555555555"


def test_matching_bound_parent_still_requires_provenance(tmp_path):
    store, record = make_store(tmp_path)
    claimed = store.claim_submission(record, expected_version=0)
    store.bind_parent_process(claimed, PARENT_UUID)
    store.bind_process(claimed, CHILD_UUID)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.FOUND, PARENT_UUID),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED
    assert result.discovered_uuid == PARENT_UUID
    assert "provenance verification" in result.reason


def test_conflicting_parent_uuid_is_rejected(tmp_path):
    store, record = make_store(tmp_path)
    claimed = store.claim_submission(record, expected_version=0)
    store.bind_parent_process(claimed, PARENT_UUID)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.FOUND, OTHER_PARENT_UUID),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED
    assert "conflicts" in result.reason


def test_parent_child_uuid_confusion_is_rejected(tmp_path):
    store, record = make_store(tmp_path)
    claimed = store.claim_submission(record, expected_version=0)
    store.bind_parent_process(claimed, PARENT_UUID)
    store.bind_process(claimed, CHILD_UUID)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.FOUND, CHILD_UUID),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED
    assert "equals retry-child" in result.reason


def test_discovered_unbound_parent_requires_reconciliation(tmp_path):
    store, record = make_store(tmp_path)
    store.claim_submission(record, expected_version=0)

    result = lookup_recovery_process(
        store,
        "science",
        discovery=discover(DiscoveryStatus.FOUND, PARENT_UUID),
    )

    assert result.decision is ReconciliationDecision.RECONCILIATION_REQUIRED
    assert "not been bound" in result.reason


def test_tracking_exposes_distinct_parent_and_child(tmp_path):
    store, record = make_store(tmp_path)
    claimed = store.claim_submission(record, expected_version=0)
    store.bind_parent_process(claimed, PARENT_UUID)
    store.bind_process(claimed, CHILD_UUID)

    tracking = store.get_tracking("science")

    assert tracking["parent_process_uuid"] == PARENT_UUID
    assert tracking["process_uuid"] == CHILD_UUID
