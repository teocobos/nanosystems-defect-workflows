from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationConflict,
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord
from nsdw.workflows.convergence.recovery_submission_boundary import (
    execute_claimed_submission,
)


def make_case(tmp_path):
    store = ReservationStore(tmp_path / "recovery.sqlite")
    record = store.reserve(
        RecoveryRecord("science", "execution", attempt_number=2)
    )
    return store, record


def test_callback_runs_only_after_claim(tmp_path):
    store, record = make_case(tmp_path)
    process_uuid = str(uuid4())

    def callback():
        tracking = store.get_tracking("science")
        assert tracking["state"] == "submitting"
        assert tracking["submission_started"] is True
        return process_uuid

    result = execute_claimed_submission(
        store,
        record,
        expected_version=0,
        submit_callback=callback,
    )

    assert result == process_uuid
    assert store.get_tracking("science")["process_uuid"] == process_uuid


def test_callback_failure_blocks_resubmission(tmp_path):
    store, record = make_case(tmp_path)
    calls = []

    def failing_callback():
        calls.append("called")
        raise RuntimeError("Simulated submission failure")

    with pytest.raises(RuntimeError):
        execute_claimed_submission(
            store,
            record,
            expected_version=0,
            submit_callback=failing_callback,
        )

    tracking = store.get_tracking("science")
    assert tracking["submission_started"] is True
    assert tracking["process_uuid"] is None

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            store,
            record,
            expected_version=0,
            submit_callback=failing_callback,
        )

    assert calls == ["called"]


def test_stale_claim_never_invokes_callback(tmp_path):
    store, record = make_case(tmp_path)
    calls = []

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            store,
            record,
            expected_version=99,
            submit_callback=lambda: calls.append("called"),
        )

    assert calls == []
    assert store.get_tracking("science")["submission_started"] is False


def test_invalid_callback_never_claims(tmp_path):
    store, record = make_case(tmp_path)

    with pytest.raises(TypeError):
        execute_claimed_submission(
            store,
            record,
            expected_version=0,
            submit_callback=None,
        )

    assert store.get_tracking("science")["state"] == "reserved"


def test_invalid_uuid_blocks_second_callback(tmp_path):
    store, record = make_case(tmp_path)
    calls = []

    def callback():
        calls.append("called")
        return "invalid-uuid"

    with pytest.raises(ValueError):
        execute_claimed_submission(
            store,
            record,
            expected_version=0,
            submit_callback=callback,
        )

    with pytest.raises(ReservationConflict):
        execute_claimed_submission(
            store,
            record,
            expected_version=0,
            submit_callback=callback,
        )

    assert calls == ["called"]
    assert store.get_tracking("science")["process_uuid"] is None
