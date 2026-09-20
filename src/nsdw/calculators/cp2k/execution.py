"""CP2K execution-request construction for NSDW."""

from __future__ import annotations
from pathlib import Path
from nsdw.execution import ExecutionRequest
from nsdw.execution.profiles.archer2 import (
    build_archer2_job,
)
from nsdw.execution.slurm import SlurmJob

class CP2KExecutionError(RuntimeError):
    """Raised when a CP2K execution request cannot be built."""


def build_cp2k_execution_request(
    *,
    calculation_id: str,
    working_directory: str | Path,
    input_file: str | Path,
    output_file: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
) -> ExecutionRequest:
    """Build a calculator-independent execution request for CP2K."""

    working_directory = Path(
        working_directory
    ).expanduser().resolve()

    input_file = Path(input_file)
    output_file = Path(output_file)

    if not working_directory.is_dir():
        raise CP2KExecutionError(
            "Working directory does not exist: "
            f"{working_directory}"
        )

    input_path = (
        input_file
        if input_file.is_absolute()
        else working_directory / input_file
    )

    if not input_path.is_file():
        raise CP2KExecutionError(
            "CP2K input file does not exist: "
            f"{input_path}"
        )

    if not executable.strip():
        raise CP2KExecutionError(
            "CP2K executable must not be empty"
        )

    return ExecutionRequest(
        calculation_id=calculation_id,
        command=(
            executable,
            "-i",
            str(input_file),
            "-o",
            str(output_file),
        ),
        working_directory=working_directory,
        environment=environment or {},
    )
def build_archer2_cp2k_job(
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
    qos: str = "standard",
    module: str = "cp2k",
    executable: str = "cp2k.psmp",
) -> SlurmJob:
    """Build an ARCHER2 CP2K job."""

    if not module.strip():
        raise CP2KExecutionError(
            "CP2K module cannot be empty"
        )

    if not executable.strip():
        raise CP2KExecutionError(
            "CP2K executable cannot be empty"
        )

    return build_archer2_job(
        name=calculation_id,
        calculation_id=calculation_id,
        working_directory=working_directory,
        command=(
            executable,
            "-i",
            str(input_file),
            "-o",
            str(output_file),
        ),
        account=account,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        walltime=walltime,
        qos=qos,
        modules=(f"load {module}",),
    )