"""Explicit recovery submission boundary.

No AiiDA submission implementation is provided here.
"""

from collections.abc import Callable

from nsdw.workflows.convergence.recovery_reservations import (
    ReservationStore,
)
from nsdw.workflows.convergence.recovery_state import RecoveryRecord


def execute_claimed_submission(
    store: ReservationStore,
    record: RecoveryRecord,
    *,
    expected_version: int,
    submit_callback: Callable[[], str],
) -> str:
    """Claim a recovery attempt before invoking an injected callback.

    A failed callback leaves durable submission intent in place.
    Retrying this function cannot reclaim the same reservation.
    """
    if not isinstance(store, ReservationStore):
        raise TypeError("Expected ReservationStore.")

    if not isinstance(record, RecoveryRecord):
        raise TypeError("Expected RecoveryRecord.")

    if not callable(submit_callback):
        raise TypeError("Submission callback must be callable.")

    claimed = store.claim_submission(
        record,
        expected_version=expected_version,
    )

    process_uuid = submit_callback()

    store.bind_process(claimed, process_uuid)

    return process_uuid
