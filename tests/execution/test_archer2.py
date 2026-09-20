from pathlib import Path
import pytest
from nsdw.execution.profiles.archer2 import (
    ARCHER2_CORES_PER_NODE,
    Archer2ProfileError,
    build_archer2_job,
)
from nsdw.execution import (
    ShellVariable,
    render_slurm_script,
)

def test_archer2_core_count():
    assert ARCHER2_CORES_PER_NODE == 128


def test_build_archer2_mpi_job():
    job = build_archer2_job(
        name="igzo-sp",
        calculation_id="igzo_sp",
        working_directory=Path("/work/e05/e05/user/igzo"),
        command=("cp2k.psmp", "-i", "igzo.inp"),
        account="e05",
        nodes=2,
    )

    assert job.resources.nodes == 2
    assert job.resources.tasks_per_node == 128
    assert job.resources.cpus_per_task == 1

    assert job.resources.account == "e05"
    assert job.resources.partition == "standard"
    assert job.resources.qos == "standard"

    assert job.environment["OMP_NUM_THREADS"] == "1"

    assert job.command == (
        "srun",
        "--hint=nomultithread",
        "--distribution=block:block",
        "cp2k.psmp",
        "-i",
        "igzo.inp",
    )

    assert job.environment[
        "SRUN_CPUS_PER_TASK"
    ] == ShellVariable(
        name="SLURM_CPUS_PER_TASK"
    )

def test_build_archer2_hybrid_job():
    job = build_archer2_job(
        name="hybrid",
        calculation_id="hybrid",
        working_directory=Path("/work/test"),
        command=("program.x",),
        account="e05",
        tasks_per_node=16,
        cpus_per_task=8,
    )

    assert job.environment["OMP_NUM_THREADS"] == "8"
    assert job.environment["OMP_PLACES"] == "cores"


def test_archer2_rejects_too_many_cores():
    with pytest.raises(Archer2ProfileError):
        build_archer2_job(
            name="bad-job",
            calculation_id="bad",
            working_directory=Path("/work/test"),
            command=("program.x",),
            account="e05",
            tasks_per_node=128,
            cpus_per_task=2,
        )


def test_archer2_requires_account():
    with pytest.raises(Archer2ProfileError):
        build_archer2_job(
            name="bad-job",
            calculation_id="bad",
            working_directory=Path("/work/test"),
            command=("program.x",),
            account="",
        )

def test_archer2_renders_scheduler_cpu_propagation():
    job = build_archer2_job(
        name="test",
        calculation_id="test",
        working_directory=Path("/work/test"),
        command=("program.x",),
        account="e05",
    )

    script = render_slurm_script(job)

    assert "#SBATCH --export=none" in script

    assert (
        "export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
        in script
    )

@pytest.mark.parametrize(
    ("qos", "walltime", "nodes"),
    [
        ("short", "00:20:00", 1),
        ("standard", "24:00:00", 2),
        ("long", "24:00:00", 1),
        ("long", "96:00:00", 64),
    ],
)
def test_archer2_accepts_qos_profiles(qos, walltime, nodes):
    job = build_archer2_job(
        name="qos-test",
        calculation_id="qos_test",
        working_directory=Path("/work/test"),
        command=("program.x",),
        account="e05-bulk-shl",
        qos=qos,
        walltime=walltime,
        nodes=nodes,
    )

    assert job.resources.account == "e05-bulk-shl"
    assert job.resources.partition == "standard"
    assert job.resources.qos == qos
    assert job.resources.walltime == walltime
    assert job.resources.nodes == nodes

    script = render_slurm_script(job)

    assert f"#SBATCH --qos={qos}" in script
    assert f"#SBATCH --time={walltime}" in script
    assert "#SBATCH --account=e05-bulk-shl" in script


@pytest.mark.parametrize(
    ("qos", "walltime", "nodes"),
    [
        ("short", "00:20:01", 1),
        ("standard", "24:00:01", 1),
        ("long", "23:59:59", 1),
        ("long", "96:00:01", 1),
        ("short", "00:10:00", 33),
        ("standard", "01:00:00", 1025),
        ("long", "48:00:00", 65),
        ("short", "00:00:00", 1),
        ("short", "00:99:00", 1),
        ("short", "invalid", 1),
        ("short", "00:10:00", 0),
    ],
)
def test_archer2_rejects_invalid_qos_resources(
    qos, walltime, nodes
):
    with pytest.raises(Archer2ProfileError):
        build_archer2_job(
            name="invalid-qos",
            calculation_id="invalid_qos",
            working_directory=Path("/work/test"),
            command=("program.x",),
            account="e05-bulk-shl",
            qos=qos,
            walltime=walltime,
            nodes=nodes,
        )


def test_archer2_rejects_unknown_qos():
    with pytest.raises(
        Archer2ProfileError,
        match="Unsupported ARCHER2 QoS",
    ):
        build_archer2_job(
            name="invalid-qos",
            calculation_id="invalid_qos",
            working_directory=Path("/work/test"),
            command=("program.x",),
            account="e05-bulk-shl",
            qos="ultra",
        )