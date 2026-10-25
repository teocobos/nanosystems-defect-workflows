"""Recovery attempt-number validation and compatibility."""

import pytest

from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState,
    transition,
)


def test_legacy_record_has_unknown_attempt():
    record = RecoveryRecord("science", "execution")
    assert record.attempt_number is None


def test_explicit_attempt_is_preserved():
    record = RecoveryRecord(
        "science",
        "execution",
        attempt_number=2,
    )

    reserved = transition(record, RecoveryState.RESERVED)

    assert reserved.attempt_number == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_invalid_attempt_number_rejected(value):
    with pytest.raises(ValueError, match="Attempt number"):
        RecoveryRecord(
            "science",
            "execution",
            attempt_number=value,
        )
