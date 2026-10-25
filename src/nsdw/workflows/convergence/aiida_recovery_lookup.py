
"""Read-only reconciliation lookup across SQLite and AiiDA."""

from dataclasses import dataclass

from nsdw.workflows.convergence.aiida_recovery_discovery import (
    DiscoveryStatus,
    discover_recovery_parent,
)
from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)


@dataclass(frozen=True)
class RecoveryLookup:
    decision: ReconciliationDecision
    discovered_uuid: str | None
    reason: str


def lookup_recovery_process(
    store: ReservationStore,
    scientific_identity: str,
    *,
    discovery=discover_recovery_parent,
) -> RecoveryLookup:
    """Inspect one reserved attempt without changing execution state.

    A discovered process is not considered provenance-verified.
    """

    if not isinstance(store, ReservationStore):
        raise TypeError("Expected ReservationStore.")

    tracking = store.get_tracking(scientific_identity)

    if tracking is None:
        raise ValueError("Recovery reservation not found.")

    correlation_id = store.get_correlation_id(scientific_identity)
    result = discovery(correlation_id)

    # Detect a reservation change across the external discovery query.
    if store.get_tracking(scientific_identity) != tracking:
        return RecoveryLookup(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            None,
            "Reservation changed during discovery.",
        )

    if result.status is DiscoveryStatus.AMBIGUOUS:
        return RecoveryLookup(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            None,
            "Multiple correlated AiiDA processes found.",
        )

    if result.status is DiscoveryStatus.FOUND:
        parent_uuid = result.process_uuid

        if not isinstance(parent_uuid, str) or not parent_uuid:
            return RecoveryLookup(
                ReconciliationDecision.RECONCILIATION_REQUIRED,
                None,
                "Discovery returned an invalid parent UUID.",
            )

        if parent_uuid == tracking["process_uuid"]:
            return RecoveryLookup(
                ReconciliationDecision.RECONCILIATION_REQUIRED,
                parent_uuid,
                "Discovered parent UUID equals retry-child UUID.",
            )

        persisted_parent = tracking["parent_process_uuid"]

        if persisted_parent is None:
            return RecoveryLookup(
                ReconciliationDecision.RECONCILIATION_REQUIRED,
                parent_uuid,
                "Discovered parent has not been bound to the reservation.",
            )

        if persisted_parent != parent_uuid:
            return RecoveryLookup(
                ReconciliationDecision.RECONCILIATION_REQUIRED,
                parent_uuid,
                "Discovered parent conflicts with persisted parent UUID.",
            )

        return RecoveryLookup(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            parent_uuid,
            "Matching recovery parent requires provenance verification.",
        )

    if (
        result.status is DiscoveryStatus.NOT_FOUND
        and tracking["submission_started"] is False
        and tracking["process_uuid"] is None
        and tracking["parent_process_uuid"] is None
        and tracking["state"] in {"prepared", "reserved"}
    ):
        return RecoveryLookup(
            ReconciliationDecision.NOT_SUBMITTED,
            None,
            "Submission has not started.",
        )

    return RecoveryLookup(
        ReconciliationDecision.RECONCILIATION_REQUIRED,
        None,
        "Submission state cannot establish absence of a process.",
    )
