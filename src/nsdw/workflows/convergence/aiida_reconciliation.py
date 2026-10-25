"""Read-only AiiDA recovery reconciliation."""

from aiida.orm import WorkChainNode

from nsdw.workflows.convergence.aiida_reserved_provenance import (
    validate_aiida_reserved_provenance,
)
from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision,
    ReconciliationResult,
)
from nsdw.workflows.convergence.recovery_reconciliation_store import (
    reconcile_reservation,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


def reconcile_aiida_reservation(
    store: ReservationStore,
    metadata: RecoveryExecutionMetadata,
    *,
    expected_recovery_parent: WorkChainNode,
    original_campaign: WorkChainNode,
) -> ReconciliationResult:
    """Verify a bound AiiDA process without taking execution actions."""

    if not isinstance(store, ReservationStore):
        raise TypeError("Expected a ReservationStore.")

    if not isinstance(metadata, RecoveryExecutionMetadata):
        raise TypeError("Expected RecoveryExecutionMetadata.")

    tracking = store.get_tracking(metadata.scientific_identity)

    if tracking is None:
        raise ValueError("Recovery reservation not found.")

    if (
        tracking["execution_identity"] != metadata.execution_identity
        or tracking["process_uuid"] != metadata.process_uuid
    ):
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Persisted execution or process identity mismatch.",
        )

    initial = reconcile_reservation(
        store,
        metadata.scientific_identity,
    )

    if initial.decision == ReconciliationDecision.NOT_SUBMITTED:
        return initial

    if (
        tracking["process_uuid"] is None
        or tracking["submission_started"] is not True
        or tracking["state"] == "reconciliation_required"
    ):
        return initial

    if not isinstance(expected_recovery_parent, WorkChainNode):
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Expected recovery parent is not an AiiDA WorkChainNode.",
        )

    if not isinstance(original_campaign, WorkChainNode):
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Original campaign is not an AiiDA WorkChainNode.",
        )

    persisted_parent_uuid = tracking["parent_process_uuid"]

    if persisted_parent_uuid is None:
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Recovery parent UUID is not bound to the reservation.",
        )

    supplied_parent_uuid = str(
        getattr(expected_recovery_parent, "uuid", "")
    )

    if supplied_parent_uuid != persisted_parent_uuid:
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Recovery parent UUID does not match the reservation.",
        )

    if supplied_parent_uuid == metadata.process_uuid:
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Recovery parent and retry child have identical UUIDs.",
        )

    try:
        verified = validate_aiida_reserved_provenance(
            store,
            metadata,
            expected_recovery_parent=expected_recovery_parent,
            original_campaign=original_campaign,
        )
    except Exception:
        # A failed or interrupted provenance lookup must never
        # authorise execution or establish a verified process.
        verified = False

    if verified is not True:
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "AiiDA process provenance could not be verified.",
        )

    current_tracking = store.get_tracking(
        metadata.scientific_identity
    )

    if current_tracking != tracking:
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Reservation changed during AiiDA provenance verification.",
        )

    if not store.matches_snapshot(
        scientific_identity=metadata.scientific_identity,
        execution_identity=metadata.execution_identity,
        process_uuid=metadata.process_uuid,
        attempt_number=tracking["attempt_number"],
        version=tracking["version"],
    ):
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Reservation snapshot is no longer valid.",
        )

    return reconcile_reservation(
        store,
        metadata.scientific_identity,
        provenance_verified=True,
    )
