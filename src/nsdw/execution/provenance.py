"""Execution provenance helpers for NSDW."""

from __future__ import annotations

import shlex

from nsdw.execution.slurm import (
    SlurmJob,
    SlurmStatusResult,
    SlurmSubmissionResult,
)
from nsdw.models.provenance import (
    ExecutionPlatform,
    ExecutionProvenance,
    SchedulerType,
)


def build_slurm_execution_provenance(
    *,
    submission: SlurmSubmissionResult,
    status: SlurmStatusResult,
    job: SlurmJob,
    platform: ExecutionPlatform,
) -> ExecutionProvenance:
    """Build NSDW execution provenance from a SLURM job."""

    if submission.job_id != status.job_id:
        raise ValueError(
            "SLURM submission/status job-ID mismatch: "
            f"submission={submission.job_id!r}, "
            f"status={status.job_id!r}"
        )

    return ExecutionProvenance(
        platform=platform,
        scheduler=SchedulerType.SLURM,
        host=None,
        job_id=submission.job_id,
        command=shlex.join(job.command),
        started_at=None,
        completed_at=None,
    )
