"""Read-only reconciliation of persisted recovery reservations."""

from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationResult,
    decide_reconciliation,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)


def reconcile_reservation(
    store: ReservationStore,
    scientific_identity: str,
    *,
    provenance_verified: bool = False,
) -> ReconciliationResult:
    """Classify a stored attempt without modifying it."""

    if not isinstance(store, ReservationStore):
        raise TypeError("Expected a ReservationStore.")

    if not isinstance(scientific_identity, str) or not scientific_identity:
        raise ValueError("Scientific identity must be nonempty.")

    if type(provenance_verified) is not bool:
        raise TypeError("provenance_verified must be bool.")

    tracking = store.get_tracking(scientific_identity)

    if tracking is None:
        raise ValueError("Recovery reservation not found.")

    if tracking["scientific_identity"] != scientific_identity:
        raise ValueError("Stored scientific identity mismatch.")

    return decide_reconciliation(
        submission_started=tracking["submission_started"],
        process_uuid=tracking["process_uuid"],
        provenance_verified=provenance_verified,
        state=tracking["state"],
    )
