"""ARCHER2 SLURM execution profile."""

from __future__ import annotations

from pathlib import Path

from nsdw.execution.slurm import (
    ShellVariable,
    SlurmJob,
    SlurmResources,
)


ARCHER2_CORES_PER_NODE = 128


class Archer2ProfileError(ValueError):
    """Raised when an invalid ARCHER2 job is requested."""


def build_archer2_job(
    *,
    name: str,
    calculation_id: str,
    working_directory: Path,
    command: tuple[str, ...],
    account: str,
    nodes: int = 1,
    tasks_per_node: int = 128,
    cpus_per_task: int = 1,
    walltime: str = "01:00:00",
    modules: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
    stdout_file: Path | None = None,
    stderr_file: Path | None = None,
) -> SlurmJob:
    """Build a standard parallel ARCHER2 SLURM job."""

    if not account.strip():
        raise Archer2ProfileError(
            "ARCHER2 account cannot be empty"
        )

    cores_requested_per_node = (
        tasks_per_node * cpus_per_task
    )

    if cores_requested_per_node > ARCHER2_CORES_PER_NODE:
        raise Archer2ProfileError(
            "ARCHER2 jobs cannot request more than "
            f"{ARCHER2_CORES_PER_NODE} physical cores per node"
        )

    job_environment = dict(environment or {})

    job_environment.setdefault(
        "OMP_NUM_THREADS",
        str(cpus_per_task),
    )

    job_environment.setdefault(
        "SRUN_CPUS_PER_TASK",
        ShellVariable(
            name="SLURM_CPUS_PER_TASK"
        ),
    )

    if cpus_per_task > 1:
        job_environment.setdefault(
            "OMP_PLACES",
            "cores",
        )

    launch_command = (
        "srun",
        "--hint=nomultithread",
        "--distribution=block:block",
        *command,
    )

    return SlurmJob(
        name=name,
        calculation_id=calculation_id,
        working_directory=working_directory,
        command=launch_command,
        resources=SlurmResources(
            nodes=nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=cpus_per_task,
            walltime=walltime,
            account=account,
            partition="standard",
            qos="standard",
        ),
        modules=modules,
        environment=job_environment,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )
