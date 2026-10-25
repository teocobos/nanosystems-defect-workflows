"""Tests for the portable recovery lifecycle."""

import pytest

from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState as S,
    requires_reconciliation,
    transition,
)


def record():
    return RecoveryRecord(
        scientific_identity="scientific-hash",
        execution_identity="execution-hash",
    )


def test_normal_lifecycle():
    current = record()
    for target in (
        S.RESERVED,
        S.SUBMITTING,
        S.SUBMITTED,
        S.RUNNING,
        S.COMPLETED,
    ):
        current = transition(current, target)

    assert current.state is S.COMPLETED


def test_original_record_is_immutable():
    original = record()
    updated = transition(original, S.RESERVED)

    assert original.state is S.PREPARED
    assert updated.state is S.RESERVED


def test_cannot_submit_without_reservation():
    with pytest.raises(ValueError):
        transition(record(), S.SUBMITTING)


def test_cannot_skip_submission_state():
    reserved = transition(record(), S.RESERVED)
    with pytest.raises(ValueError):
        transition(reserved, S.SUBMITTED)


def test_duplicate_transition_rejected():
    reserved = transition(record(), S.RESERVED)
    with pytest.raises(ValueError):
        transition(reserved, S.RESERVED)


def test_uncertain_submission_requires_reconciliation():
    current = transition(record(), S.RESERVED)
    current = transition(current, S.SUBMITTING)
    current = transition(current, S.RECONCILIATION_REQUIRED)

    assert requires_reconciliation(current)

    with pytest.raises(ValueError):
        transition(current, S.SUBMITTING)


def test_completed_attempt_cannot_restart():
    current = record()
    for target in (
        S.RESERVED, S.SUBMITTING, S.SUBMITTED, S.COMPLETED
    ):
        current = transition(current, target)

    with pytest.raises(ValueError):
        transition(current, S.RESERVED)


def test_failed_attempt_cannot_restart():
    current = record()
    for target in (
        S.RESERVED, S.SUBMITTING, S.SUBMITTED, S.FAILED
    ):
        current = transition(current, target)

    with pytest.raises(ValueError):
        transition(current, S.SUBMITTING)


def test_invalid_identity_rejected():
    with pytest.raises(ValueError):
        RecoveryRecord("", "execution")

    with pytest.raises(ValueError):
        RecoveryRecord("scientific", "")


def test_invalid_state_rejected():
    with pytest.raises(TypeError):
        RecoveryRecord("scientific", "execution", "running")


def test_invalid_transition_target_rejected():
    with pytest.raises(TypeError):
        transition(record(), "reserved")


def test_submitted_attempt_can_finish_without_running_observation():
    current = record()
    for target in (
        S.RESERVED, S.SUBMITTING, S.SUBMITTED, S.COMPLETED
    ):
        current = transition(current, target)

    assert current.state is S.COMPLETED
