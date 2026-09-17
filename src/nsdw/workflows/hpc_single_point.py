from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from nsdw.calculators.cp2k import (
    adapt_cp2k_result,
    build_archer2_cp2k_job,
    parse_cp2k_input,
    parse_cp2k_output,
)
from nsdw.execution.monitor import (
    SlurmMonitorConfig,
    wait_for_slurm_job,
)
from nsdw.execution.slurm import (
    SlurmExecutor,
    SlurmJob,
    SlurmJobState,
    SlurmStatusResult,
    SlurmSubmissionResult,
)
from nsdw.models.provenance import ExecutionPlatform

from nsdw.models.result import NSDWResult

from nsdw.execution.provenance import (
    build_slurm_execution_provenance,
)

class HPCSinglePointWorkflowError(RuntimeError):
    """Raised when an HPC single-point workflow fails."""

class HPCSinglePointSubmission(BaseModel):
    """A submitted HPC single-point calculation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    job: SlurmJob
    submission: SlurmSubmissionResult

def submit_cp2k_single_point_archer2(
    *,
    calculation_id: str,
    working_directory: Path,
    input_file: Path,
    output_file: Path,
    account: str,
    nodes: int = 1,
    tasks_per_node: int = 128,
    cpus_per_task: int = 1,
    walltime: str = "01:00:00",
    executor: SlurmExecutor | None = None,
) -> HPCSinglePointSubmission:
    """Build and submit a CP2K single-point job to ARCHER2."""

    slurm_executor = executor or SlurmExecutor()

    job = build_archer2_cp2k_job(
        calculation_id=calculation_id,
        working_directory=working_directory,
        input_file=input_file,
        output_file=output_file,
        account=account,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        walltime=walltime,
    )

    submission = slurm_executor.submit(job)

    return HPCSinglePointSubmission(
        job=job,
        submission=submission,
    )

def wait_for_cp2k_single_point(
    *,
    submission: SlurmSubmissionResult,
    executor: SlurmExecutor,
    monitor_config: SlurmMonitorConfig | None = None,
):
    """Wait for a submitted CP2K SLURM job to finish."""

    status = wait_for_slurm_job(
        executor,
        submission.job_id,
        config=monitor_config,
    )

    if status.state != SlurmJobState.COMPLETED:
        raise HPCSinglePointWorkflowError(
            "CP2K SLURM job did not complete successfully: "
            f"job_id={submission.job_id}, "
            f"state={status.state.value}, "
            f"raw_state={status.raw_state}, "
            f"exit_code={status.exit_code}"
        )

    return status

def collect_cp2k_single_point_archer2(
    *,
    hpc_submission: HPCSinglePointSubmission,
    status: SlurmStatusResult,
    input_file: str | Path,
    output_file: str | Path,
    result_file: str | Path = "result.json",
) -> NSDWResult:
    """Parse and collect a completed ARCHER2 CP2K calculation."""

    working_directory = (
        hpc_submission.job.working_directory
        .expanduser()
        .resolve()
    )

    calculation_id = hpc_submission.job.calculation_id

    input_file = Path(input_file)
    output_file = Path(output_file)
    result_file = Path(result_file)

    input_path = (
        input_file
        if input_file.is_absolute()
        else working_directory / input_file
    )

    output_path = (
        output_file
        if output_file.is_absolute()
        else working_directory / output_file
    )

    parsed_input = parse_cp2k_input(
        input_path
    )

    parsed_output = parse_cp2k_output(
        output_path
    )

    execution_provenance = build_slurm_execution_provenance(
        submission=hpc_submission.submission,
        status=status,
        job=hpc_submission.job,
        platform=ExecutionPlatform.ARCHER2,
    )

    result = adapt_cp2k_result(
        parsed_output,
        input_settings=parsed_input,
        input_path=input_path,
        output_path=output_path,
        calculation_id=calculation_id,
        execution_provenance=execution_provenance,
    )

    destination = (
        result_file
        if result_file.is_absolute()
        else working_directory / result_file
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        result.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    return result

def run_cp2k_single_point_archer2(
    *,
    calculation_id: str,
    working_directory: str | Path,
    input_file: str | Path,
    output_file: str | Path,
    account: str,
    nodes: int = 1,
    tasks_per_node: int = 128,
    cpus_per_task: int = 1,
    walltime: str = "01:00:00",
    result_file: str | Path = "result.json",
    executor: SlurmExecutor | None = None,
    monitor_config: SlurmMonitorConfig | None = None,
) -> NSDWResult:
    """Submit, monitor, and collect an ARCHER2 CP2K calculation."""

    slurm_executor = executor or SlurmExecutor()

    working_directory = (
        Path(working_directory)
        .expanduser()
        .resolve()
    )

    input_file = Path(input_file)
    output_file = Path(output_file)

    hpc_submission = submit_cp2k_single_point_archer2(
        calculation_id=calculation_id,
        working_directory=working_directory,
        input_file=input_file,
        output_file=output_file,
        account=account,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        walltime=walltime,
        executor=slurm_executor,
    )

    status = wait_for_cp2k_single_point(
        submission=hpc_submission.submission,
        executor=slurm_executor,
        monitor_config=monitor_config,
    )

    return collect_cp2k_single_point_archer2(
        hpc_submission=hpc_submission,
        status=status,
        input_file=input_file,
        output_file=output_file,
        result_file=result_file,
    )