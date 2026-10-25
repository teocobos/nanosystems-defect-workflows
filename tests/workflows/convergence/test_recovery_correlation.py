
import pytest

from nsdw.workflows.convergence.recovery_correlation import (
    recovery_correlation_id,
)


def test_correlation_is_deterministic():
    first = recovery_correlation_id("science", "execution", 1)
    second = recovery_correlation_id("science", "execution", 1)
    assert first == second
    assert first.startswith("nsdw-recovery-v1-")


def test_correlation_changes_with_attempt():
    first = recovery_correlation_id("science", "execution", 1)
    second = recovery_correlation_id("science", "execution", 2)
    assert first != second


def test_correlation_changes_with_execution():
    first = recovery_correlation_id("science", "execution-a", 1)
    second = recovery_correlation_id("science", "execution-b", 1)
    assert first != second


def test_correlation_changes_with_science():
    first = recovery_correlation_id("science-a", "execution", 1)
    second = recovery_correlation_id("science-b", "execution", 1)
    assert first != second


@pytest.mark.parametrize(
    "science,execution,attempt",
    [
        ("", "execution", 1),
        ("science", "", 1),
        ("science", "execution", 0),
        ("science", "execution", -1),
        ("science", "execution", True),
        (None, "execution", 1),
    ],
)
def test_invalid_correlation_inputs_rejected(
    science, execution, attempt
):
    with pytest.raises(ValueError):
        recovery_correlation_id(science, execution, attempt)
