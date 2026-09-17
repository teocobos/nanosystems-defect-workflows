from pathlib import Path

import pytest

from nsdw.execution.slurm import (
    SlurmExecutionError,
    SlurmExecutor,
    SlurmJob,
    SlurmResources,
    SlurmSubmissionResult,
)


def _job(tmp_path: Path) -> SlurmJob:
    return SlurmJob(
        name="test-job",
        calculation_id="test_001",
        working_directory=tmp_path,
        command=("echo", "hello"),
        resources=SlurmResources(),
    )


def _make_fake_sbatch(
    tmp_path: Path,
    body: str,
) -> Path:
    path = tmp_path / "sbatch"

    path.write_text(
        "#!/bin/bash\n" + body,
        encoding="utf-8",
    )

    path.chmod(0o755)

    return path


def test_slurm_executor_submits_job(tmp_path):
    sbatch = _make_fake_sbatch(
        tmp_path,
        'echo "Submitted batch job 123456"\n',
    )

    executor = SlurmExecutor(
        sbatch_executable=str(sbatch),
    )

    result = executor.submit(
        _job(tmp_path),
        script_file=Path("job.sh"),
    )

    assert isinstance(
        result,
        SlurmSubmissionResult,
    )

    assert result.calculation_id == "test_001"
    assert result.job_id == "123456"
    assert result.host
    assert result.submitted_at

    assert result.script_path == (
        tmp_path / "job.sh"
    ).resolve()

    script = (
        tmp_path / "job.sh"
    ).read_text()

    assert "#SBATCH --job-name=test-job" in script
    assert "echo hello" in script


def test_slurm_executor_submission_failure(tmp_path):
    sbatch = _make_fake_sbatch(
        tmp_path,
        'echo "allocation unavailable" >&2\n'
        "exit 1\n",
    )

    executor = SlurmExecutor(
        sbatch_executable=str(sbatch),
    )

    with pytest.raises(
        SlurmExecutionError,
        match="allocation unavailable",
    ):
        executor.submit(_job(tmp_path))


def test_slurm_executor_rejects_bad_output(tmp_path):
    sbatch = _make_fake_sbatch(
        tmp_path,
        'echo "unexpected output"\n',
    )

    executor = SlurmExecutor(
        sbatch_executable=str(sbatch),
    )

    with pytest.raises(
        SlurmExecutionError,
        match="Could not parse SLURM job ID",
    ):
        executor.submit(_job(tmp_path))


def test_slurm_executor_missing_sbatch(tmp_path):
    executor = SlurmExecutor(
        sbatch_executable=str(
            tmp_path / "does-not-exist"
        ),
    )

    with pytest.raises(
        SlurmExecutionError,
        match="sbatch executable was not found",
    ):
        executor.submit(_job(tmp_path))


def test_slurm_executor_missing_workdir(tmp_path):
    job = SlurmJob(
        name="test",
        calculation_id="test",
        working_directory=(
            tmp_path / "missing-directory"
        ),
        command=("echo", "hello"),
        resources=SlurmResources(),
    )

    executor = SlurmExecutor()

    with pytest.raises(
        SlurmExecutionError,
        match="working directory does not exist",
    ):
        executor.submit(job)
def test_slurm_query_running_job(tmp_path):
    squeue = _make_fake_sbatch(
        tmp_path,
        'echo "RUNNING"\n',
    )

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
    )

    result = executor.query_active("123456")

    assert result is not None
    assert result.job_id == "123456"
    assert result.state.value == "running"
    assert result.raw_state == "RUNNING"


def test_slurm_query_pending_job(tmp_path):
    squeue = _make_fake_sbatch(
        tmp_path,
        'echo "PENDING"\n',
    )

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
    )

    result = executor.query_active("123456")

    assert result is not None
    assert result.state.value == "pending"


def test_slurm_query_job_not_active(tmp_path):
    squeue = _make_fake_sbatch(
        tmp_path,
        "exit 0\n",
    )

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
    )

    assert executor.query_active("123456") is None


def test_slurm_query_unknown_state(tmp_path):
    squeue = _make_fake_sbatch(
        tmp_path,
        'echo "SOME_NEW_STATE"\n',
    )

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
    )

    result = executor.query_active("123456")

    assert result is not None
    assert result.state.value == "unknown"
    assert result.raw_state == "SOME_NEW_STATE"


def test_slurm_query_failure(tmp_path):
    squeue = _make_fake_sbatch(
        tmp_path,
        'echo "scheduler unavailable" >&2\n'
        "exit 1\n",
    )

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
    )

    with pytest.raises(
        SlurmExecutionError,
        match="scheduler unavailable",
    ):
        executor.query_active("123456")