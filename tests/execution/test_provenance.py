from pathlib import Path

import pytest

from nsdw.execution.provenance import (
    build_slurm_execution_provenance,
)
from nsdw.execution.slurm import (
    SlurmJob,
    SlurmJobState,
    SlurmResources,
    SlurmStatusResult,
    SlurmSubmissionResult,
)
from nsdw.models.provenance import (
    ExecutionPlatform,
    SchedulerType,
)


def _submission() -> SlurmSubmissionResult:
    return SlurmSubmissionResult(
        calculation_id="igzo_sp",
        job_id="123456",
        host="login01",
        submitted_at="2026-09-17T15:00:00+01:00",
        script_path=Path("/work/test/job.sh"),
    )


def _status() -> SlurmStatusResult:
    return SlurmStatusResult(
        job_id="123456",
        state=SlurmJobState.COMPLETED,
        raw_state="COMPLETED",
        exit_code="0:0",
    )


def _job() -> SlurmJob:
    return SlurmJob(
        name="igzo_sp",
        calculation_id="igzo_sp",
        working_directory=Path("/work/test"),
        command=(
            "srun",
            "--hint=nomultithread",
            "--distribution=block:block",
            "cp2k.psmp",
            "-i",
            "igzo.inp",
            "-o",
            "igzo.out",
        ),
        resources=SlurmResources(
            nodes=1,
            tasks_per_node=128,
            cpus_per_task=1,
        ),
    )


def test_build_slurm_execution_provenance():
    provenance = build_slurm_execution_provenance(
        submission=_submission(),
        status=_status(),
        job=_job(),
        platform=ExecutionPlatform.ARCHER2,
    )

    assert provenance.platform == ExecutionPlatform.ARCHER2
    assert provenance.scheduler == SchedulerType.SLURM
    assert provenance.job_id == "123456"
    assert provenance.host is None
    assert provenance.started_at is None
    assert provenance.completed_at is None
    assert provenance.command == (
        "srun --hint=nomultithread "
        "--distribution=block:block "
        "cp2k.psmp -i igzo.inp -o igzo.out"
    )


def test_slurm_provenance_rejects_job_id_mismatch():
    status = SlurmStatusResult(
        job_id="999999",
        state=SlurmJobState.COMPLETED,
        raw_state="COMPLETED",
        exit_code="0:0",
    )

    with pytest.raises(
        ValueError,
        match="job-ID mismatch",
    ):
        build_slurm_execution_provenance(
            submission=_submission(),
            status=status,
            job=_job(),
            platform=ExecutionPlatform.ARCHER2,
        )
