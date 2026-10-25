"""Read-only inspection of persisted recovery submissions.

This module never invokes AiiDA submission or SLURM.
"""

from dataclasses import dataclass

from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision,
)
from nsdw.workflows.convergence.recovery_reconciliation_store import (
    reconcile_reservation,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)


@dataclass(frozen=True)
class RecoveryInspection:
    scientific_identity: str
    execution_identity: str
    state: str
    version: int
    process_uuid: str | None
    decision: ReconciliationDecision
    reason: str
    requires_manual_reconciliation: bool


def inspect_recovery(
    store: ReservationStore,
    scientific_identity: str,
) -> RecoveryInspection:
    """Inspect a persisted attempt without authorising execution."""

    if not isinstance(store, ReservationStore):
        raise TypeError("Expected ReservationStore.")

    if not isinstance(scientific_identity, str) or not scientific_identity:
        raise ValueError("Scientific identity must be nonempty.")

    tracking = store.get_tracking(scientific_identity)

    if tracking is None:
        raise ValueError("Recovery reservation not found.")

    result = reconcile_reservation(
        store,
        scientific_identity,
        provenance_verified=False,
    )

    return RecoveryInspection(
        scientific_identity=scientific_identity,
        execution_identity=tracking["execution_identity"],
        state=tracking["state"],
        version=tracking["version"],
        process_uuid=tracking["process_uuid"],
        decision=result.decision,
        reason=result.reason,
        requires_manual_reconciliation=(
            result.decision
            == ReconciliationDecision.RECONCILIATION_REQUIRED
        ),
    )
