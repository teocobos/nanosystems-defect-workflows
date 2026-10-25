"""Backend-independent recovery lifecycle rules."""

from dataclasses import dataclass, replace
from enum import Enum


class RecoveryState(str, Enum):
    PREPARED = "prepared"
    RESERVED = "reserved"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RECONCILIATION_REQUIRED = "reconciliation_required"


_ALLOWED = {
    RecoveryState.PREPARED: {RecoveryState.RESERVED},
    RecoveryState.RESERVED: {
        RecoveryState.SUBMITTING,
        RecoveryState.RECONCILIATION_REQUIRED,
    },
    RecoveryState.SUBMITTING: {
        RecoveryState.SUBMITTED,
        RecoveryState.RECONCILIATION_REQUIRED,
    },
    RecoveryState.SUBMITTED: {
        RecoveryState.RUNNING,
        RecoveryState.COMPLETED,
        RecoveryState.FAILED,
        RecoveryState.RECONCILIATION_REQUIRED,
    },
    RecoveryState.RUNNING: {
        RecoveryState.COMPLETED,
        RecoveryState.FAILED,
        RecoveryState.RECONCILIATION_REQUIRED,
    },
    RecoveryState.COMPLETED: set(),
    RecoveryState.FAILED: set(),
    RecoveryState.RECONCILIATION_REQUIRED: set(),
}


@dataclass(frozen=True)
class RecoveryRecord:
    """Immutable snapshot, not a persistent database reservation."""

    scientific_identity: str
    execution_identity: str
    state: RecoveryState = RecoveryState.PREPARED
    attempt_number: int | None = None

    def __post_init__(self):
        if not isinstance(self.scientific_identity, str):
            raise TypeError("Scientific identity must be a string.")
        if not self.scientific_identity.strip():
            raise ValueError("Scientific identity cannot be empty.")
        if not isinstance(self.execution_identity, str):
            raise TypeError("Execution identity must be a string.")
        if not self.execution_identity.strip():
            raise ValueError("Execution identity cannot be empty.")
        if not isinstance(self.state, RecoveryState):
            raise TypeError("State must be a RecoveryState.")
        if self.attempt_number is not None:
            if (
                type(self.attempt_number) is not int
                or self.attempt_number < 1
            ):
                raise ValueError(
                    "Attempt number must be a positive integer."
                )


def transition(
    record: RecoveryRecord,
    target: RecoveryState,
) -> RecoveryRecord:
    """Return a new record only for an allowed transition."""
    if not isinstance(record, RecoveryRecord):
        raise TypeError("Expected a RecoveryRecord.")
    if not isinstance(target, RecoveryState):
        raise TypeError("Expected a RecoveryState.")

    if target not in _ALLOWED[record.state]:
        raise ValueError(
            f"Invalid recovery transition: "
            f"{record.state.value} -> {target.value}"
        )

    return replace(record, state=target)


def requires_reconciliation(record: RecoveryRecord) -> bool:
    """Identify an unresolved, potentially submitted attempt."""
    if not isinstance(record, RecoveryRecord):
        raise TypeError("Expected a RecoveryRecord.")
    return record.state == RecoveryState.RECONCILIATION_REQUIRED
