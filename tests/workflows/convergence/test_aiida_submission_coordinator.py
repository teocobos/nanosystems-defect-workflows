"""Tests for fail-closed AiiDA submission coordination."""

from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_submission_coordinator import (
    submit_reserved_aiida_child,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
)


@pytest.fixture
def reservation(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")

    record = RecoveryRecord(
        scientific_identity=f"science-{uuid4()}",
        execution_identity=f"execution-{uuid4()}",
    )

    reserved = store.reserve(record)

    return store, reserved


def test_coordinator_binds_parent_and_child(reservation):
    store, record = reservation

    parent_uuid = str(uuid4())
    child_uuid = str(uuid4())
    calls = []

    result = submit_reserved_aiida_child(
        store,
        record,
        expected_version=store.get_tracking(
            record.scientific_identity
        )["version"],
        parent_process_uuid=parent_uuid,
        submit_callback=lambda: (
            calls.append("submitted") or child_uuid
        ),
    )

    tracking = store.get_tracking(record.scientific_identity)

    assert result == child_uuid
    assert calls == ["submitted"]
    assert tracking["parent_process_uuid"] == parent_uuid
    assert tracking["process_uuid"] == child_uuid
    assert tracking["submission_started"] is True


def test_coordinator_does_not_repeat_ambiguous_submission(
    reservation,
):
    store, record = reservation

    calls = []

    def failing_submit():
        calls.append("submitted")
        raise RuntimeError("Simulated submission interruption")

    with pytest.raises(RuntimeError):
        submit_reserved_aiida_child(
            store,
            record,
            expected_version=store.get_tracking(
                record.scientific_identity
            )["version"],
            parent_process_uuid=str(uuid4()),
            submit_callback=failing_submit,
        )

    assert len(calls) == 1

    tracking = store.get_tracking(record.scientific_identity)
    assert tracking["submission_started"] is True

    with pytest.raises(Exception):
        submit_reserved_aiida_child(
            store,
            record,
            expected_version=tracking["version"],
            parent_process_uuid=tracking["parent_process_uuid"],
            submit_callback=failing_submit,
        )

    assert len(calls) == 1


def test_coordinator_rejects_wrong_parent(reservation):
    store, record = reservation

    original_parent = str(uuid4())

    claimed = store.claim_submission(
        record,
        expected_version=store.get_tracking(
            record.scientific_identity
        )["version"],
    )

    store.bind_parent_process(claimed, original_parent)

    calls = []

    with pytest.raises(ValueError, match="parent UUID differs"):
        submit_reserved_aiida_child(
            store,
            record,
            expected_version=store.get_tracking(
                record.scientific_identity
            )["version"],
            parent_process_uuid=str(uuid4()),
            submit_callback=lambda: calls.append("submitted"),
        )

    assert calls == []
