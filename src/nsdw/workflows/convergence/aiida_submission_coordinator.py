"""Fail-closed coordination of reserved AiiDA submissions.

The callback must submit through the appropriate AiiDA execution
context. This module does not create or infer CALL_WORK links.
"""

from collections.abc import Callable
from uuid import UUID

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import (
    RecoveryRecord,
)


def submit_reserved_aiida_child(
    store: ReservationStore,
    record: RecoveryRecord,
    *,
    expected_version: int,
    parent_process_uuid: str,
    submit_callback: Callable[[], str],
) -> str:
    """Bind the parent before claiming and submitting a child.

    The callback is responsible for producing a genuine AiiDA child
    of the specified parent. This function does not establish
    AiiDA provenance itself.

    A callback failure must not trigger automatic resubmission.
    """
    if not isinstance(store, ReservationStore):
        raise TypeError("Expected ReservationStore.")

    if not isinstance(record, RecoveryRecord):
        raise TypeError("Expected RecoveryRecord.")

    if not callable(submit_callback):
        raise TypeError("Submission callback must be callable.")

    try:
        parent_uuid = str(UUID(str(parent_process_uuid)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(
            "Invalid recovery parent process UUID."
        ) from exc

    tracking = store.get_tracking(record.scientific_identity)

    if tracking is None:
        raise ValueError("Recovery reservation does not exist.")

    if tracking["version"] != expected_version:
        raise ValueError("Recovery reservation version changed.")

    # Validate the persisted parent before claiming anything.
    existing_parent = tracking["parent_process_uuid"]

    if existing_parent is not None and existing_parent != parent_uuid:
        raise ValueError(
            "Recovery parent UUID differs from persisted parent."
        )

    # The atomic claim must precede parent binding.
    # After this point, failures remain ambiguous and fail closed.
    claimed = store.claim_submission(
        record,
        expected_version=expected_version,
    )

    store.bind_parent_process(claimed, parent_uuid)

    process_uuid = submit_callback()

    store.bind_process(claimed, process_uuid)

    return process_uuid
