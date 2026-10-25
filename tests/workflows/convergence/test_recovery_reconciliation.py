import pytest

from nsdw.workflows.convergence.recovery_reconciliation import (
    ReconciliationDecision as Decision,
    decide_reconciliation,
)


def test_prepared_without_submission():
    result = decide_reconciliation(
        submission_started=False,
        process_uuid=None,
        state="prepared",
    )
    assert result.decision == Decision.NOT_SUBMITTED


def test_reserved_without_submission():
    result = decide_reconciliation(
        submission_started=False,
        process_uuid=None,
        state="reserved",
    )
    assert result.decision == Decision.NOT_SUBMITTED


def test_submission_started_without_process():
    result = decide_reconciliation(
        submission_started=True,
        process_uuid=None,
        state="submitting",
    )
    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_bound_process_without_verified_provenance():
    result = decide_reconciliation(
        submission_started=True,
        process_uuid="process-uuid",
        state="submitting",
    )
    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_verified_process():
    result = decide_reconciliation(
        submission_started=True,
        process_uuid="process-uuid",
        provenance_verified=True,
        state="submitted",
    )
    assert result.decision == Decision.VERIFIED


def test_completed_process_still_requires_provenance():
    result = decide_reconciliation(
        submission_started=True,
        process_uuid="process-uuid",
        state="completed",
    )
    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_failed_process_can_have_verified_identity():
    result = decide_reconciliation(
        submission_started=True,
        process_uuid="process-uuid",
        provenance_verified=True,
        state="failed",
    )
    assert result.decision == Decision.VERIFIED


def test_reconciliation_state_remains_blocked():
    result = decide_reconciliation(
        submission_started=True,
        process_uuid="process-uuid",
        provenance_verified=True,
        state="reconciliation_required",
    )
    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_unstarted_reservation_with_bound_process():
    result = decide_reconciliation(
        submission_started=False,
        process_uuid="process-uuid",
        state="reserved",
    )
    assert result.decision == Decision.RECONCILIATION_REQUIRED


def test_verified_provenance_without_uuid_rejected():
    with pytest.raises(ValueError, match="without a process UUID"):
        decide_reconciliation(
            submission_started=True,
            process_uuid=None,
            provenance_verified=True,
        )


def test_invalid_submission_flag_rejected():
    with pytest.raises(TypeError):
        decide_reconciliation(
            submission_started=1,
            process_uuid=None,
        )


def test_invalid_provenance_flag_rejected():
    with pytest.raises(TypeError):
        decide_reconciliation(
            submission_started=True,
            process_uuid="process-uuid",
            provenance_verified=1,
        )


def test_unknown_state_fails_closed():
    result = decide_reconciliation(
        submission_started=True,
        process_uuid="process-uuid",
        provenance_verified=True,
        state="unexpected",
    )
    assert result.decision == Decision.RECONCILIATION_REQUIRED
