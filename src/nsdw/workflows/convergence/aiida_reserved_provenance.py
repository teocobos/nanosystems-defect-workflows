"""Read-only binding of AiiDA retry provenance to a reservation."""

from collections.abc import Callable

from aiida.orm import WorkChainNode, load_node

from nsdw.workflows.convergence.aiida_recovery_identity import (
    validate_tracking_identity,
)
from nsdw.workflows.convergence.recovery_reserved_identity import (
    validate_reserved_execution_identity,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


def validate_aiida_reserved_provenance(
    store: ReservationStore,
    metadata: RecoveryExecutionMetadata,
    *,
    expected_recovery_parent: WorkChainNode,
    original_campaign: WorkChainNode,
    process_loader: Callable = load_node,
) -> bool:
    """Validate a reserved attempt against genuine AiiDA provenance.

    Read-only. Does not submit jobs, reconcile missing processes,
    or certify scientific convergence.
    """
    if not isinstance(store, ReservationStore):
        raise TypeError("Expected ReservationStore.")

    if not isinstance(metadata, RecoveryExecutionMetadata):
        raise TypeError("Expected RecoveryExecutionMetadata.")

    tracking = store.get_tracking(metadata.scientific_identity)

    if tracking is None:
        raise ValueError("Missing recovery reservation.")

    if tracking["execution_identity"] != metadata.execution_identity:
        raise ValueError("Recovery execution identity mismatch.")

    if tracking["attempt_number"] is None:
        raise ValueError("Unknown recovery attempt number.")

    if metadata.process_uuid is None:
        raise ValueError(
            "Missing process UUID; reconciliation required."
        )

    if tracking["process_uuid"] is None:
        raise ValueError(
            "Reservation process UUID is unbound; "
            "reconciliation required."
        )

    if tracking["submission_started"] is not True:
        raise ValueError(
            "Recovery reservation submission has not started."
        )

    allowed_states = {
        "submitting",
        "submitted",
        "running",
        "completed",
        "failed",
    }

    if tracking["state"] not in allowed_states:
        raise ValueError(
            "Recovery reservation lifecycle state is not eligible "
            "for process verification."
        )

    if tracking["process_uuid"] != metadata.process_uuid:
        raise ValueError("Reserved process UUID mismatch.")

    persisted_parent_uuid = tracking["parent_process_uuid"]

    if persisted_parent_uuid is None:
        raise ValueError(
            "Recovery parent UUID is not bound to the reservation."
        )

    supplied_parent_uuid = str(
        getattr(expected_recovery_parent, "uuid", "")
    )

    if supplied_parent_uuid != persisted_parent_uuid:
        raise ValueError(
            "Recovery parent UUID does not match the reservation."
        )

    if supplied_parent_uuid == metadata.process_uuid:
        raise ValueError(
            "Recovery parent and retry child have identical UUIDs."
        )

    process = validate_tracking_identity(
        metadata,
        process_loader=process_loader,
        expected_recovery_parent=expected_recovery_parent,
        original_campaign=original_campaign,
    )

    # These inputs have already been checked against the
    # provenance-reconstructed retry builder.
    resources = process.inputs.resources.get_dict()

    return validate_reserved_execution_identity(
        store,
        metadata,
        resources=resources,
    )
