"""ARCHER2 SLURM execution profile."""

from __future__ import annotations

from enum import StrEnum

from pathlib import Path

from nsdw.execution.slurm import (
    ShellVariable,
    SlurmJob,
    SlurmResources,
)


ARCHER2_CORES_PER_NODE = 128

class Archer2QoS(StrEnum):
    """Supported ARCHER2 CPU QoS profiles."""

    SHORT = "short"
    STANDARD = "standard"
    LONG = "long"


ARCHER2_QOS_LIMITS = {
    Archer2QoS.SHORT: {
        "min_seconds": 1,
        "max_seconds": 20 * 60,
        "max_nodes": 32,
    },
    Archer2QoS.STANDARD: {
        "min_seconds": 1,
        "max_seconds": 24 * 60 * 60,
        "max_nodes": 1024,
    },
    Archer2QoS.LONG: {
        "min_seconds": 24 * 60 * 60,
        "max_seconds": 96 * 60 * 60,
        "max_nodes": 64,
    },
}


class Archer2ProfileError(ValueError):
    """Raised when an invalid ARCHER2 job is requested."""

def _validate_archer2_qos(
    *,
    qos: Archer2QoS | str,
    nodes: int,
    walltime: str,
) -> Archer2QoS:
    """Validate ARCHER2 QoS, node count, and walltime."""

    try:
        selected_qos = Archer2QoS(qos)
    except ValueError as exc:
        raise Archer2ProfileError(
            f"Unsupported ARCHER2 QoS: {qos!r}"
        ) from exc

    limits = ARCHER2_QOS_LIMITS[selected_qos]

    try:
        hours, minutes, seconds = (
            int(part) for part in walltime.split(":")
        )
    except ValueError as exc:
        raise Archer2ProfileError(
            "ARCHER2 walltime must use HH:MM:SS"
        ) from exc

    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise Archer2ProfileError(
            "ARCHER2 walltime must use HH:MM:SS"
        )

    total_seconds = hours * 3600 + minutes * 60 + seconds

    if not (
        limits["min_seconds"]
        <= total_seconds
        <= limits["max_seconds"]
    ):
        raise Archer2ProfileError(
            f"ARCHER2 QoS {selected_qos.value!r} does not "
            f"permit walltime {walltime}"
        )

    if not 1 <= nodes <= limits["max_nodes"]:
        raise Archer2ProfileError(
            f"ARCHER2 QoS {selected_qos.value!r} permits "
            f"between 1 and {limits['max_nodes']} nodes"
        )

    return selected_qos

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
    qos: Archer2QoS | str = Archer2QoS.STANDARD,
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

    if not account.strip():
        raise Archer2ProfileError(
            "ARCHER2 account cannot be empty"
        )

    selected_qos = _validate_archer2_qos(
        qos=qos,
        nodes=nodes,
        walltime=walltime,
    )

    cores_requested_per_node = (
        tasks_per_node * cpus_per_task
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
            qos=selected_qos.value,
        ),
        modules=modules,
        environment=job_environment,
        export="none",
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )
