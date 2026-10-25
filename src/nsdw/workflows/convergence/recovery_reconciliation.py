"""Read-only recovery reconciliation decisions.

This module never submits, cancels, or modifies AiiDA processes.
"""

from dataclasses import dataclass
from enum import Enum


class ReconciliationDecision(str, Enum):
    NOT_SUBMITTED = "not_submitted"
    VERIFIED = "verified"
    RECONCILIATION_REQUIRED = "reconciliation_required"


@dataclass(frozen=True)
class ReconciliationResult:
    decision: ReconciliationDecision
    reason: str


def decide_reconciliation(
    *,
    submission_started: bool,
    process_uuid: str | None,
    provenance_verified: bool = False,
    state: str | None = None,
) -> ReconciliationResult:
    """Classify evidence without taking an execution action."""

    if type(submission_started) is not bool:
        raise TypeError("submission_started must be bool.")

    if type(provenance_verified) is not bool:
        raise TypeError("provenance_verified must be bool.")

    if process_uuid is not None:
        if not isinstance(process_uuid, str) or not process_uuid.strip():
            raise ValueError("process_uuid must be a nonempty string.")

    if state is not None and not isinstance(state, str):
        raise TypeError("state must be a string or None.")

    if provenance_verified and process_uuid is None:
        raise ValueError(
            "Provenance cannot be verified without a process UUID."
        )

    if state == "reconciliation_required":
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Reservation already requires reconciliation.",
        )

    if not submission_started:
        if process_uuid is None and not provenance_verified:
            if state in (None, "prepared", "reserved"):
                return ReconciliationResult(
                    ReconciliationDecision.NOT_SUBMITTED,
                    "No submission has started and no process is bound.",
                )

        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Reservation contains inconsistent submission evidence.",
        )

    if process_uuid is None:
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Submission started without a bound process UUID.",
        )

    if not provenance_verified:
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Bound process has not passed provenance validation.",
        )

    if state not in (
        None,
        "submitting",
        "submitted",
        "running",
        "completed",
        "failed",
    ):
        return ReconciliationResult(
            ReconciliationDecision.RECONCILIATION_REQUIRED,
            "Reservation lifecycle state is inconsistent.",
        )

    return ReconciliationResult(
        ReconciliationDecision.VERIFIED,
        "Bound process passed provenance validation.",
    )
