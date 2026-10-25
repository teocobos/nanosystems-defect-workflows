"""Backend-independent recovery reservation contract."""

from typing import Protocol, runtime_checkable

from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
    RecoveryState,
)


@runtime_checkable
class RecoveryReservationStore(Protocol):
    """Contract for persistent, atomic recovery storage."""

    def reserve(self, record: RecoveryRecord) -> RecoveryRecord:
        """Claim an unreserved scientific recovery identity."""
        ...

    def get(self, scientific_identity: str) -> RecoveryRecord | None:
        """Retrieve the current persistent recovery snapshot."""
        ...

    def advance(
        self,
        record: RecoveryRecord,
        target: RecoveryState,
    ) -> RecoveryRecord:
        """Atomically apply a validated state transition."""
        ...
