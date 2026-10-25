"""Read-only execution identity validation against reservations.

This adapter verifies consistency between a durable reservation,
recovery metadata and caller-supplied execution resources.

It does not authenticate AiiDA process provenance or authorise
submission.
"""

from nsdw.workflows.convergence.aiida_recovery_identity import (
    validate_aiida_execution_fingerprint,
)
from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


def validate_reserved_execution_identity(
    store: ReservationStore,
    metadata: RecoveryExecutionMetadata,
    *,
    resources: dict,
) -> bool:
    """Verify an execution fingerprint using a stored attempt number."""
    if not isinstance(store, ReservationStore):
        raise TypeError("Expected ReservationStore.")

    if not isinstance(metadata, RecoveryExecutionMetadata):
        raise TypeError("Expected RecoveryExecutionMetadata.")

    if not isinstance(resources, dict):
        raise TypeError("Execution resources must be a dictionary.")

    tracking = store.get_tracking(metadata.scientific_identity)

    if tracking is None:
        raise ValueError("Missing recovery reservation.")

    if tracking["scientific_identity"] != metadata.scientific_identity:
        raise ValueError("Recovery scientific identity mismatch.")

    if tracking["execution_identity"] != metadata.execution_identity:
        raise ValueError("Recovery execution identity mismatch.")

    attempt_number = tracking["attempt_number"]

    if attempt_number is None:
        raise ValueError(
            "Recovery reservation has an unknown attempt number."
        )

    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("Invalid stored recovery attempt number.")

    return validate_aiida_execution_fingerprint(
        metadata,
        computer_uuid=metadata.computer_uuid,
        resources=resources,
        attempt_number=attempt_number,
    )
