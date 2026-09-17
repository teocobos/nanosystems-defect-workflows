from pathlib import Path
from unittest.mock import Mock, patch

import shutil
import pytest

from nsdw.execution.slurm import (
    SlurmJob,
    SlurmJobState,
    SlurmResources,
    SlurmStatusResult,
    SlurmSubmissionResult,
)
from nsdw.workflows.hpc_single_point import (
    HPCSinglePointSubmission,
    HPCSinglePointWorkflowError,
    collect_cp2k_single_point_archer2,
    run_cp2k_single_point_archer2,
    submit_cp2k_single_point_archer2,
    wait_for_cp2k_single_point,
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


@patch(
    "nsdw.workflows.hpc_single_point."
    "build_archer2_cp2k_job"
)
def test_submit_cp2k_single_point_archer2(
    mock_build_job,
):
    executor = Mock()
    job = SlurmJob(
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
            account="e05",
            partition="standard",
            qos="standard",
        ),
    )

    mock_build_job.return_value = job

    submission = _submission()
    executor.submit.return_value = submission

    result = submit_cp2k_single_point_archer2(
        calculation_id="igzo_sp",
        working_directory=Path("/work/test"),
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        executor=executor,
    )

    assert result.job == job
    assert result.submission == submission

    mock_build_job.assert_called_once_with(
        calculation_id="igzo_sp",
        working_directory=Path("/work/test"),
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        walltime="01:00:00",
        module="cp2k",
        executable="cp2k.psmp",
    )

    executor.submit.assert_called_once_with(job)

@patch(
    "nsdw.workflows.hpc_single_point."
    "build_archer2_cp2k_job"
)
def test_submit_cp2k_single_point_archer2_custom_module(
    mock_build_job,
):
    executor = Mock()

    job = SlurmJob(
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
            account="e05",
            partition="standard",
            qos="standard",
        ),
    )

    mock_build_job.return_value = job
    executor.submit.return_value = _submission()

    submit_cp2k_single_point_archer2(
        calculation_id="igzo_sp",
        working_directory=Path("/work/test"),
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        module="cp2k/cp2k-2025.2",
        executable="cp2k.psmp",
        executor=executor,
    )

    mock_build_job.assert_called_once_with(
        calculation_id="igzo_sp",
        working_directory=Path("/work/test"),
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        walltime="01:00:00",
        module="cp2k/cp2k-2025.2",
        executable="cp2k.psmp",
    )

@patch(
    "nsdw.workflows.hpc_single_point."
    "wait_for_slurm_job"
)
def test_wait_for_cp2k_single_point_completed(
    mock_wait,
):
    executor = Mock()
    submission = _submission()

    status = SlurmStatusResult(
        job_id="123456",
        state=SlurmJobState.COMPLETED,
        raw_state="COMPLETED",
        exit_code="0:0",
    )

    mock_wait.return_value = status

    result = wait_for_cp2k_single_point(
        submission=submission,
        executor=executor,
    )

    assert result == status

    mock_wait.assert_called_once_with(
        executor,
        "123456",
        config=None,
    )


@patch(
    "nsdw.workflows.hpc_single_point."
    "wait_for_slurm_job"
)
def test_wait_for_cp2k_single_point_failed(
    mock_wait,
):
    executor = Mock()
    submission = _submission()

    mock_wait.return_value = SlurmStatusResult(
        job_id="123456",
        state=SlurmJobState.FAILED,
        raw_state="FAILED",
        exit_code="1:0",
    )

    with pytest.raises(
        HPCSinglePointWorkflowError,
        match="state=failed",
    ):
        wait_for_cp2k_single_point(
            submission=submission,
            executor=executor,
        )

def test_collect_cp2k_single_point_archer2(tmp_path):
    fixture_directory = (
        Path("tests/calculators/cp2k/fixtures")
        .resolve()
    )

    input_file = Path("igzo_ordered_003_sp.inp")
    output_file = Path("igzo_ordered_003_sp_real.out")

    shutil.copy(
        fixture_directory / input_file,
        tmp_path / input_file,
    )
    shutil.copy(
        fixture_directory / output_file,
        tmp_path / output_file,
    )

    job = SlurmJob(
        name="igzo_sp",
        calculation_id="igzo_sp",
        working_directory=tmp_path,
        command=(
            "srun",
            "--hint=nomultithread",
            "--distribution=block:block",
            "cp2k.psmp",
            "-i",
            str(input_file),
            "-o",
            str(output_file),
        ),
        resources=SlurmResources(
            nodes=1,
            tasks_per_node=128,
            cpus_per_task=1,
            account="e05",
            partition="standard",
            qos="standard",
        ),
    )

    submission = SlurmSubmissionResult(
        calculation_id="igzo_sp",
        job_id="123456",
        host="login01",
        submitted_at="2026-09-17T15:00:00+01:00",
        script_path=tmp_path / "job.sh",
    )

    hpc_submission = HPCSinglePointSubmission(
        job=job,
        submission=submission,
    )

    status = SlurmStatusResult(
        job_id="123456",
        state=SlurmJobState.COMPLETED,
        raw_state="COMPLETED",
        exit_code="0:0",
    )

    result = collect_cp2k_single_point_archer2(
        hpc_submission=hpc_submission,
        status=status,
        input_file=input_file,
        output_file=output_file,
        result_file="result.json",
    )

    assert result.calculation.id == "igzo_sp"

    provenance = result.provenance.execution

    assert provenance.platform == ExecutionPlatform.ARCHER2
    assert provenance.scheduler == SchedulerType.SLURM
    assert provenance.job_id == "123456"
    assert provenance.host is None

    assert provenance.command == (
        "srun --hint=nomultithread "
        "--distribution=block:block "
        "cp2k.psmp -i igzo_ordered_003_sp.inp "
        "-o igzo_ordered_003_sp_real.out"
    )

    assert len(result.provenance.input_files) == 1
    assert len(result.provenance.output_files) == 1

    result_path = tmp_path / "result.json"

    assert result_path.is_file()

    persisted = result_path.read_text(
        encoding="utf-8",
    )

    assert '"job_id": "123456"' in persisted
    assert '"platform": "archer2"' in persisted
    assert '"scheduler": "slurm"' in persisted

@patch(
    "nsdw.workflows.hpc_single_point."
    "collect_cp2k_single_point_archer2"
)
@patch(
    "nsdw.workflows.hpc_single_point."
    "wait_for_cp2k_single_point"
)
@patch(
    "nsdw.workflows.hpc_single_point."
    "submit_cp2k_single_point_archer2"
)
def test_run_cp2k_single_point_archer2(
    mock_submit,
    mock_wait,
    mock_collect,
    tmp_path,
):
    executor = Mock()

    job = SlurmJob(
        name="igzo_sp",
        calculation_id="igzo_sp",
        working_directory=tmp_path,
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
            account="e05",
            partition="standard",
            qos="standard",
        ),
    )

    submission = SlurmSubmissionResult(
        calculation_id="igzo_sp",
        job_id="123456",
        host="login01",
        submitted_at="2026-09-17T15:00:00+01:00",
        script_path=tmp_path / "job.sh",
    )

    hpc_submission = HPCSinglePointSubmission(
        job=job,
        submission=submission,
    )

    status = SlurmStatusResult(
        job_id="123456",
        state=SlurmJobState.COMPLETED,
        raw_state="COMPLETED",
        exit_code="0:0",
    )

    expected_result = Mock()

    mock_submit.return_value = hpc_submission
    mock_wait.return_value = status
    mock_collect.return_value = expected_result

    result = run_cp2k_single_point_archer2(
        calculation_id="igzo_sp",
        working_directory=tmp_path,
        input_file="igzo.inp",
        output_file="igzo.out",
        account="e05",
        executor=executor,
    )

    assert result is expected_result

    mock_submit.assert_called_once_with(
        calculation_id="igzo_sp",
        working_directory=tmp_path.resolve(),
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        walltime="01:00:00",
        module="cp2k",
        executable="cp2k.psmp",
        executor=executor,
    )

    mock_wait.assert_called_once_with(
        submission=submission,
        executor=executor,
        monitor_config=None,
    )

    mock_collect.assert_called_once_with(
        hpc_submission=hpc_submission,
        status=status,
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        result_file="result.json",
    )

@patch(
    "nsdw.workflows.hpc_single_point."
    "collect_cp2k_single_point_archer2"
)
@patch(
    "nsdw.workflows.hpc_single_point."
    "wait_for_cp2k_single_point"
)
@patch(
    "nsdw.workflows.hpc_single_point."
    "submit_cp2k_single_point_archer2"
)
def test_run_cp2k_single_point_archer2_custom_module(
    mock_submit,
    mock_wait,
    mock_collect,
    tmp_path,
):
    executor = Mock()

    job = SlurmJob(
        name="igzo_sp",
        calculation_id="igzo_sp",
        working_directory=tmp_path,
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
            account="e05",
            partition="standard",
            qos="standard",
        ),
    )

    submission = SlurmSubmissionResult(
        calculation_id="igzo_sp",
        job_id="123456",
        host="login01",
        submitted_at="2026-09-17T15:00:00+01:00",
        script_path=tmp_path / "job.sh",
    )

    hpc_submission = HPCSinglePointSubmission(
        job=job,
        submission=submission,
    )

    status = SlurmStatusResult(
        job_id="123456",
        state=SlurmJobState.COMPLETED,
        raw_state="COMPLETED",
        exit_code="0:0",
    )

    mock_submit.return_value = hpc_submission
    mock_wait.return_value = status
    mock_collect.return_value = Mock()

    run_cp2k_single_point_archer2(
        calculation_id="igzo_sp",
        working_directory=tmp_path,
        input_file="igzo.inp",
        output_file="igzo.out",
        account="e05",
        module="cp2k/cp2k-2025.2",
        executable="cp2k.psmp",
        executor=executor,
    )

    mock_submit.assert_called_once_with(
        calculation_id="igzo_sp",
        working_directory=tmp_path.resolve(),
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        walltime="01:00:00",
        module="cp2k/cp2k-2025.2",
        executable="cp2k.psmp",
        executor=executor,
    )