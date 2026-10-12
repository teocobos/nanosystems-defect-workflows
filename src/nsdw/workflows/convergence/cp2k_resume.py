"""Read-only resumption planning for CP2K convergence campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from nsdw.workflows.convergence.cp2k_status import (
    CP2KCampaignStatus,
    CP2KCandidateState,
    inspect_cp2k_convergence_campaign,
)


class CP2KResumeAction(StrEnum):
    """Safe actions available before scheduler reconciliation."""

    PRESERVE = "preserve"
    MONITOR = "monitor"
    BLOCK = "block"


@dataclass(frozen=True)
class CP2KCandidateResumePlan:
    label: str
    scientific_state: CP2KCandidateState
    action: CP2KResumeAction
    reason: str


@dataclass(frozen=True)
class CP2KCampaignResumePlan:
    campaign_directory: Path
    candidates: tuple[CP2KCandidateResumePlan, ...]

    @property
    def all_completed(self) -> bool:
        return bool(self.candidates) and all(
            candidate.action == CP2KResumeAction.PRESERVE
            for candidate in self.candidates
        )

    @property
    def requires_attention(self) -> bool:
        return any(
            candidate.action == CP2KResumeAction.BLOCK
            for candidate in self.candidates
        )


def plan_cp2k_convergence_resume(
    campaign_directory: str | Path,
) -> CP2KCampaignResumePlan:
    """Inspect a campaign and produce a plan without side effects."""

    status: CP2KCampaignStatus = (
        inspect_cp2k_convergence_campaign(campaign_directory)
    )

    candidates = []

    for candidate in status.candidates:
        if candidate.state == CP2KCandidateState.COMPLETED:
            action = CP2KResumeAction.PRESERVE
            reason = (
                "Scientifically validated CP2K output exists; "
                "preserve without resubmission."
            )

        elif candidate.state == CP2KCandidateState.PENDING:
            action = CP2KResumeAction.BLOCK
            reason = (
                "No CP2K output found. Scheduler state and "
                "submission history are not yet reconciled."
            )

        elif candidate.state == CP2KCandidateState.INCOMPLETE:
            action = CP2KResumeAction.BLOCK
            reason = (
                "Existing CP2K output is incomplete; "
                "manual recovery decision required."
            )

        else:
            action = CP2KResumeAction.BLOCK
            reason = (
                "Candidate scientific evidence is invalid: "
                f"{candidate.reason}"
            )

        candidates.append(
            CP2KCandidateResumePlan(
                label=candidate.label,
                scientific_state=candidate.state,
                action=action,
                reason=reason,
            )
        )

    return CP2KCampaignResumePlan(
        campaign_directory=status.campaign_directory,
        candidates=tuple(candidates),
    )


def plan_cp2k_convergence_slurm_resume(
    campaign_directory: str | Path,
    *,
    target: str,
    executor,
) -> CP2KCampaignResumePlan:
    """Read-only reconciliation of CP2K evidence and recorded SLURM jobs."""
    from nsdw.execution.slurm import SlurmJobState
    from nsdw.execution.submission_records import (
        SubmissionRecordError,
        load_slurm_submission_record,
    )
    from nsdw.workflows.convergence.manifest import (
        load_convergence_manifest,
    )

    campaign = Path(campaign_directory).expanduser().resolve()
    status = inspect_cp2k_convergence_campaign(campaign)
    manifest = load_convergence_manifest(campaign / "manifest.json")

    if len(status.candidates) != len(manifest.candidates):
        raise ValueError("Manifest and status candidate counts differ.")

    plans = []

    for candidate, entry in zip(
        status.candidates, manifest.candidates, strict=True
    ):
        if candidate.label != entry.label:
            raise ValueError("Manifest and status candidate labels differ.")

        if candidate.state == CP2KCandidateState.COMPLETED:
            action = CP2KResumeAction.PRESERVE
            reason = "Scientifically validated CP2K output exists."

        elif candidate.state != CP2KCandidateState.PENDING:
            action = CP2KResumeAction.BLOCK
            reason = f"Scientific state is {candidate.state.value}."

        else:
            action = CP2KResumeAction.BLOCK
            reason = "Scheduler evidence is not yet reconciled."

            from nsdw.execution.submission_claims import (
                SubmissionClaimError,
                inspect_submission_claim,
            )
            from nsdw.execution.submission_records import sha256_file

            candidate_directory = (campaign / entry.directory).resolve()
            input_path = (candidate_directory / entry.input_file).resolve()

            try:
                if not candidate_directory.is_relative_to(campaign):
                    raise SubmissionClaimError(
                        "Candidate directory escapes campaign."
                    )

                if not input_path.is_relative_to(candidate_directory):
                    raise SubmissionClaimError(
                        "Candidate input escapes directory."
                    )

                claim = inspect_submission_claim(
                    candidate_directory=candidate_directory,
                )

                if claim is not None:
                    if claim.calculation_id != entry.label:
                        raise SubmissionClaimError(
                            "Claim calculation ID mismatch."
                        )

                    if claim.target != target:
                        raise SubmissionClaimError(
                            "Claim HPC target mismatch."
                        )

                    if claim.input_sha256 != sha256_file(input_path):
                        raise SubmissionClaimError(
                            "Claim input SHA-256 mismatch."
                        )

                record = load_slurm_submission_record(
                    record_path=Path("submission.json"),
                    candidate_directory=candidate_directory,
                    calculation_id=entry.label,
                    target=target,
                    input_file=Path(entry.input_file),
                )

                if claim is not None and record is not None:
                    if claim.input_sha256 != record.input_sha256:
                        raise SubmissionClaimError(
                            "Claim and submission record disagree."
                        )

            except (SubmissionClaimError, SubmissionRecordError, OSError) as exc:
                record = None
                reason = f"Untrusted submission evidence: {exc}"
            else:
                if record is None:
                    reason = (
                        "Submission claim exists without a job record; "
                        "manual reconciliation required."
                        if claim is not None
                        else
                        "No submission record; absence does not prove "
                        "that no SLURM job exists."
                    )
                    reason = (
                        "No submission record; absence does not prove "
                        "that no SLURM job exists."
                    )
                else:
                    try:
                        scheduler = executor.query_status(record.job_id)
                    except Exception as exc:
                        reason = (
                            "SLURM status unavailable; submission blocked: "
                            f"{exc}"
                        )
                    else:
                        if (
                            scheduler is not None
                            and scheduler.job_id == record.job_id
                            and scheduler.state in {
                                SlurmJobState.PENDING,
                                SlurmJobState.RUNNING,
                                SlurmJobState.COMPLETING,
                            }
                        ):
                            action = CP2KResumeAction.MONITOR
                            reason = (
                                f"Recorded SLURM job {record.job_id} "
                                f"is {scheduler.state.value}."
                            )
                        else:
                            reason = (
                                f"Recorded SLURM job {record.job_id} "
                                "is not confirmed active; manual "
                                "reconciliation required."
                            )

        plans.append(
            CP2KCandidateResumePlan(
                label=candidate.label,
                scientific_state=candidate.state,
                action=action,
                reason=reason,
            )
        )

    return CP2KCampaignResumePlan(
        campaign_directory=campaign,
        candidates=tuple(plans),
    )
