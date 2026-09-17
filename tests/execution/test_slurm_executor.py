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

def test_slurm_accounting_completed_job(tmp_path):
    sacct = _make_fake_sbatch(
        tmp_path,
        'echo "123456|COMPLETED|0:0"\n',
    )

    executor = SlurmExecutor(
        sacct_executable=str(sacct),
    )

    result = executor.query_accounting("123456")

    assert result is not None
    assert result.job_id == "123456"
    assert result.state.value == "completed"
    assert result.raw_state == "COMPLETED"
    assert result.exit_code == "0:0"


def test_slurm_accounting_failed_job(tmp_path):
    sacct = _make_fake_sbatch(
        tmp_path,
        'echo "123456|FAILED|1:0"\n',
    )

    executor = SlurmExecutor(
        sacct_executable=str(sacct),
    )

    result = executor.query_accounting("123456")

    assert result is not None
    assert result.state.value == "failed"
    assert result.exit_code == "1:0"


def test_slurm_accounting_selects_parent_job(tmp_path):
    sacct = _make_fake_sbatch(
        tmp_path,
        'echo "123456.batch|COMPLETED|0:0"\n'
        'echo "123456.extern|COMPLETED|0:0"\n'
        'echo "123456|COMPLETED|0:0"\n',
    )

    executor = SlurmExecutor(
        sacct_executable=str(sacct),
    )

    result = executor.query_accounting("123456")

    assert result is not None
    assert result.job_id == "123456"
    assert result.state.value == "completed"
    assert result.raw_state == "COMPLETED"


def test_slurm_accounting_no_record(tmp_path):
    sacct = _make_fake_sbatch(
        tmp_path,
        "exit 0\n",
    )

    executor = SlurmExecutor(
        sacct_executable=str(sacct),
    )

    assert executor.query_accounting("123456") is None


def test_slurm_accounting_failure(tmp_path):
    sacct = _make_fake_sbatch(
        tmp_path,
        'echo "accounting unavailable" >&2\n'
        "exit 1\n",
    )

    executor = SlurmExecutor(
        sacct_executable=str(sacct),
    )

    with pytest.raises(
        SlurmExecutionError,
        match="accounting unavailable",
    ):
        executor.query_accounting("123456")


def test_slurm_accounting_missing_sacct():
    executor = SlurmExecutor(
        sacct_executable="/definitely/missing/sacct",
    )

    with pytest.raises(
        SlurmExecutionError,
        match="sacct executable was not found",
    ):
        executor.query_accounting("123456")

def test_slurm_query_status_prefers_active_job(tmp_path):
    squeue = _make_fake_sbatch(
        tmp_path,
        'echo "RUNNING"\n',
    )

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
        sacct_executable="/definitely/missing/sacct",
    )

    result = executor.query_status("123456")

    assert result is not None
    assert result.job_id == "123456"
    assert result.state.value == "running"
    assert result.raw_state == "RUNNING"
    assert result.exit_code is None


def test_slurm_query_status_falls_back_to_accounting(
    tmp_path,
):
    squeue = _make_fake_sbatch(
        tmp_path,
        "exit 0\n",
    )

    sacct = tmp_path / "fake-sacct"
    sacct.write_text(
        '#!/bin/bash\n'
        'echo "123456|COMPLETED|0:0"\n',
        encoding="utf-8",
    )
    sacct.chmod(0o755)

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
        sacct_executable=str(sacct),
    )

    result = executor.query_status("123456")

    assert result is not None
    assert result.job_id == "123456"
    assert result.state.value == "completed"
    assert result.raw_state == "COMPLETED"
    assert result.exit_code == "0:0"


def test_slurm_query_status_returns_none_when_unknown(
    tmp_path,
):
    squeue = _make_fake_sbatch(
        tmp_path,
        "exit 0\n",
    )

    sacct = tmp_path / "fake-sacct"
    sacct.write_text(
        "#!/bin/bash\n"
        "exit 0\n",
        encoding="utf-8",
    )
    sacct.chmod(0o755)

    executor = SlurmExecutor(
        squeue_executable=str(squeue),
        sacct_executable=str(sacct),
    )

    assert executor.query_status("123456") is None