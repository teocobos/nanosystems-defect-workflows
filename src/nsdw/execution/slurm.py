"""Generic SLURM execution models and utilities."""

from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
import shlex
import socket
import subprocess
from enum import StrEnum

from datetime import datetime

class SlurmResources(BaseModel):
    """Calculator-independent resources requested from SLURM."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    nodes: int = Field(default=1, ge=1)
    tasks_per_node: int = Field(default=1, ge=1)
    cpus_per_task: int = Field(default=1, ge=1)

    walltime: str = "01:00:00"

    account: str | None = None
    partition: str | None = None
    qos: str | None = None

    @field_validator("walltime")
    @classmethod
    def validate_walltime(cls, value: str) -> str:
        """Require SLURM-style HH:MM:SS walltime."""

        parts = value.split(":")

        if len(parts) != 3:
            raise ValueError(
                "walltime must use HH:MM:SS format"
            )

        try:
            hours, minutes, seconds = (
                int(part) for part in parts
            )
        except ValueError as exc:
            raise ValueError(
                "walltime must contain integers"
            ) from exc

        if hours < 0:
            raise ValueError(
                "walltime hours must be non-negative"
            )

        if not 0 <= minutes <= 59:
            raise ValueError(
                "walltime minutes must be between 0 and 59"
            )

        if not 0 <= seconds <= 59:
            raise ValueError(
                "walltime seconds must be between 0 and 59"
            )

        if hours == 0 and minutes == 0 and seconds == 0:
            raise ValueError(
                "walltime must be greater than zero"
            )

        return value

class ShellVariable(BaseModel):
    """Reference to an existing shell environment variable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError(
                "shell variable name cannot be empty"
            )

        if not value.replace("_", "").isalnum():
            raise ValueError(
                "shell variable name must contain only "
                "letters, numbers, and underscores"
            )

        if value[0].isdigit():
            raise ValueError(
                "shell variable name cannot start with a digit"
            )

        return value

class SlurmJob(BaseModel):
    """Description of a job to be rendered for SLURM."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    calculation_id: str

    working_directory: Path

    command: tuple[str, ...]

    resources: SlurmResources

    modules: tuple[str, ...] = ()
    environment: dict[str, str | ShellVariable] = {}

    export: str | None = None

    stdout_file: Path | None = None
    stderr_file: Path | None = None

    @field_validator("name", "calculation_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "SLURM job identifiers cannot be empty"
            )

        return value

    @field_validator("command")
    @classmethod
    def validate_command(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError(
                "SLURM command cannot be empty"
            )

        if any(not item.strip() for item in value):
            raise ValueError(
                "SLURM command arguments cannot be empty"
            )

        return value
def render_slurm_script(job: SlurmJob) -> str:
    """Render a generic SLURM batch script."""

    resources = job.resources

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job.name}",
        f"#SBATCH --nodes={resources.nodes}",
        f"#SBATCH --ntasks-per-node={resources.tasks_per_node}",
        f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
        f"#SBATCH --time={resources.walltime}",
    ]

    if resources.account is not None:
        lines.append(
            f"#SBATCH --account={resources.account}"
        )

    if resources.partition is not None:
        lines.append(
            f"#SBATCH --partition={resources.partition}"
        )

    if resources.qos is not None:
        lines.append(
            f"#SBATCH --qos={resources.qos}"
        )

    if job.export is not None:
        lines.append(
            f"#SBATCH --export={job.export}"
        )

    if job.stdout_file is not None:
        lines.append(
            f"#SBATCH --output={job.stdout_file}"
        )

    if job.stderr_file is not None:
        lines.append(
            f"#SBATCH --error={job.stderr_file}"
        )

    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f"cd {shlex.quote(str(job.working_directory))}",
        ]
    )

    if job.modules:
        lines.append("")

        for module in job.modules:
            lines.append(f"module {module}")

    if job.environment:
        lines.append("")

        for key, value in job.environment.items():
            if isinstance(value, ShellVariable):
                rendered_value = f"${value.name}"
            else:
                rendered_value = shlex.quote(value)

            lines.append(
                f"export {key}={rendered_value}"
            )

    lines.extend(
        [
            "",
            shlex.join(job.command),
            "",
        ]
    )

    return "\n".join(lines)
class SlurmExecutionError(RuntimeError):
    """Raised when a SLURM job cannot be submitted."""


def _submission_timestamp() -> str:
    """Return a timezone-aware local timestamp."""

    return datetime.now().astimezone().isoformat()

class SlurmJobState(StrEnum):
    """NSDW representation of SLURM job states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    UNKNOWN = "unknown"


class SlurmStatusResult(BaseModel):
    """Current scheduler status of a submitted SLURM job."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    job_id: str
    state: SlurmJobState
    raw_state: str
    exit_code: str | None = None

def parse_slurm_state(
    raw_state: str,
) -> SlurmJobState:
    """Map a SLURM state string onto an NSDW scheduler state."""

    state = raw_state.strip().upper()

    mapping = {
        "PENDING": SlurmJobState.PENDING,
        "PD": SlurmJobState.PENDING,
        "RUNNING": SlurmJobState.RUNNING,
        "R": SlurmJobState.RUNNING,
        "COMPLETING": SlurmJobState.COMPLETING,
        "CG": SlurmJobState.COMPLETING,
        "COMPLETED": SlurmJobState.COMPLETED,
        "CD": SlurmJobState.COMPLETED,
        "FAILED": SlurmJobState.FAILED,
        "F": SlurmJobState.FAILED,
        "CANCELLED": SlurmJobState.CANCELLED,
        "CA": SlurmJobState.CANCELLED,
        "TIMEOUT": SlurmJobState.TIMEOUT,
        "TO": SlurmJobState.TIMEOUT,
        "OUT_OF_MEMORY": SlurmJobState.OUT_OF_MEMORY,
        "OOM": SlurmJobState.OUT_OF_MEMORY,
    }

    return mapping.get(
        state,
        SlurmJobState.UNKNOWN,
    )

class SlurmSubmissionResult(BaseModel):
    """Result returned when a SLURM job is accepted by the scheduler."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    calculation_id: str
    job_id: str
    host: str
    submitted_at: str
    script_path: Path

    @field_validator(
        "calculation_id",
        "job_id",
        "host",
        "submitted_at",
    )
    @classmethod
    def validate_non_empty(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "SLURM submission fields cannot be empty"
            )

        return value

class SlurmExecutor:
    """Render and submit jobs to a SLURM scheduler."""

    def __init__(
        self,
        *,
        sbatch_executable: str = "sbatch",
        squeue_executable: str = "squeue",
        sacct_executable: str = "sacct",
    ) -> None:
        if not sbatch_executable.strip():
            raise ValueError(
                "sbatch executable cannot be empty"
            )

        if not squeue_executable.strip():
            raise ValueError(
                "squeue executable cannot be empty"
            )

        if not sacct_executable.strip():
            raise ValueError(
                "sacct executable cannot be empty"
            )

        self.sbatch_executable = sbatch_executable
        self.squeue_executable = squeue_executable
        self.sacct_executable = sacct_executable

    def submit(
        self,
        job: SlurmJob,
        *,
        script_file: Path = Path("job.sh"),
    ) -> SlurmSubmissionResult:
        """Render, write, and submit a SLURM job."""

        working_directory = (
            job.working_directory.expanduser().resolve()
        )

        if not working_directory.is_dir():
            raise SlurmExecutionError(
                "SLURM working directory does not exist: "
                f"{working_directory}"
            )

        script_path = (
            script_file
            if script_file.is_absolute()
            else working_directory / script_file
        )

        script_path = script_path.resolve()

        script_path.write_text(
            render_slurm_script(job),
            encoding="utf-8",
        )

        submitted_at = _submission_timestamp()

        try:
            process = subprocess.run(
                [
                    self.sbatch_executable,
                    str(script_path),
                ],
                cwd=working_directory,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SlurmExecutionError(
                "SLURM sbatch executable was not found: "
                f"{self.sbatch_executable}"
            ) from exc

        if process.returncode != 0:
            message = (
                process.stderr.strip()
                or process.stdout.strip()
                or "unknown sbatch error"
            )

            raise SlurmExecutionError(
                f"SLURM submission failed: {message}"
            )

        output = process.stdout.strip()

        prefix = "Submitted batch job "

        if not output.startswith(prefix):
            raise SlurmExecutionError(
                "Could not parse SLURM job ID from sbatch "
                f"output: {output!r}"
            )

        job_id = output[len(prefix):].strip()

        if not job_id:
            raise SlurmExecutionError(
                "SLURM returned an empty job ID"
            )

        return SlurmSubmissionResult(
            calculation_id=job.calculation_id,
            job_id=job_id,
            submitted_at=submitted_at,
            host=socket.gethostname(),
            script_path=script_path,
        )

    def query_active(
        self,
        job_id: str,
    ) -> SlurmStatusResult | None:
        """Query an active SLURM job using squeue."""

        if not job_id.strip():
            raise ValueError(
                "SLURM job ID cannot be empty"
            )

        try:
            process = subprocess.run(
                [
                    self.squeue_executable,
                    "--noheader",
                    "--jobs",
                    job_id,
                    "--format=%T",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SlurmExecutionError(
                "SLURM squeue executable was not found: "
                f"{self.squeue_executable}"
            ) from exc

        if process.returncode != 0:
            message = (
                process.stderr.strip()
                or process.stdout.strip()
                or "unknown squeue error"
            )

            raise SlurmExecutionError(
                f"SLURM status query failed: {message}"
            )

        output = process.stdout.strip()

        if not output:
            return None

        raw_state = output.splitlines()[0].strip()

        return SlurmStatusResult(
            job_id=job_id,
            state=parse_slurm_state(raw_state),
            raw_state=raw_state,
        )

    def query_accounting(
        self,
        job_id: str,
    ) -> SlurmStatusResult | None:
        """Query SLURM accounting information using sacct."""

        if not job_id.strip():
            raise ValueError(
                "SLURM job ID cannot be empty"
            )

        try:
            process = subprocess.run(
                [
                    self.sacct_executable,
                    "--noheader",
                    "--parsable2",
                    "--jobs",
                    job_id,
                    "--format=JobIDRaw,State,ExitCode",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SlurmExecutionError(
                "SLURM sacct executable was not found: "
                f"{self.sacct_executable}"
            ) from exc

        if process.returncode != 0:
            message = (
                process.stderr.strip()
                or process.stdout.strip()
                or "unknown sacct error"
            )

            raise SlurmExecutionError(
                f"SLURM accounting query failed: {message}"
            )

        lines = [
            line.strip()
            for line in process.stdout.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        for line in lines:
            fields = line.split("|")

            if len(fields) < 3:
                continue

            accounting_job_id = fields[0].strip()

            if accounting_job_id != job_id:
                continue

            raw_state = fields[1].strip()
            exit_code = fields[2].strip() or None

            return SlurmStatusResult(
                job_id=job_id,
                state=parse_slurm_state(raw_state),
                raw_state=raw_state,
                exit_code=exit_code,
            )

        return None

    def query_status(
        self,
        job_id: str,
    ) -> SlurmStatusResult | None:
        """Query the current or final status of a SLURM job."""

        active_status = self.query_active(job_id)

        if active_status is not None:
            return active_status

        return self.query_accounting(job_id)