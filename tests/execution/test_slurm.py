from pathlib import Path

import pytest
from pydantic import ValidationError

from nsdw.execution import (
    ShellVariable,
    SlurmJob,
    SlurmResources,
    render_slurm_script,
)

def test_slurm_resources_defaults():
    resources = SlurmResources()

    assert resources.nodes == 1
    assert resources.tasks_per_node == 1
    assert resources.cpus_per_task == 1
    assert resources.walltime == "01:00:00"


def test_slurm_resources_custom():
    resources = SlurmResources(
        nodes=2,
        tasks_per_node=128,
        cpus_per_task=1,
        walltime="02:30:00",
        account="e05-test",
        partition="standard",
        qos="standard",
    )

    assert resources.nodes == 2
    assert resources.tasks_per_node == 128
    assert resources.account == "e05-test"


@pytest.mark.parametrize(
    "walltime",
    [
        "",
        "01:00",
        "abc",
        "01:60:00",
        "01:00:60",
        "00:00:00",
    ],
)
def test_invalid_walltime_rejected(walltime):
    with pytest.raises(ValidationError):
        SlurmResources(
            walltime=walltime,
        )


def test_invalid_resources_rejected():
    with pytest.raises(ValidationError):
        SlurmResources(nodes=0)

    with pytest.raises(ValidationError):
        SlurmResources(tasks_per_node=0)

    with pytest.raises(ValidationError):
        SlurmResources(cpus_per_task=0)


def test_slurm_job():
    resources = SlurmResources(
        nodes=2,
        tasks_per_node=128,
    )

    job = SlurmJob(
        name="igzo-sp",
        calculation_id="igzo_sp_001",
        working_directory=Path("/work/project/igzo"),
        command=(
            "srun",
            "cp2k.psmp",
            "-i",
            "igzo.inp",
            "-o",
            "igzo.out",
        ),
        resources=resources,
        modules=(
            "load-epcc-module",
            "cp2k",
        ),
        environment={
            "OMP_NUM_THREADS": "1",
        },
    )

    assert job.name == "igzo-sp"
    assert job.calculation_id == "igzo_sp_001"

    assert job.command == (
        "srun",
        "cp2k.psmp",
        "-i",
        "igzo.inp",
        "-o",
        "igzo.out",
    )

    assert job.resources.nodes == 2
    assert job.modules == (
        "load-epcc-module",
        "cp2k",
    )


def test_empty_job_name_rejected():
    with pytest.raises(ValidationError):
        SlurmJob(
            name="",
            calculation_id="test",
            working_directory=Path("."),
            command=("echo", "hello"),
            resources=SlurmResources(),
        )


def test_empty_calculation_id_rejected():
    with pytest.raises(ValidationError):
        SlurmJob(
            name="test",
            calculation_id="",
            working_directory=Path("."),
            command=("echo", "hello"),
            resources=SlurmResources(),
        )


def test_empty_command_rejected():
    with pytest.raises(ValidationError):
        SlurmJob(
            name="test",
            calculation_id="test",
            working_directory=Path("."),
            command=(),
            resources=SlurmResources(),
        )


def test_empty_command_argument_rejected():
    with pytest.raises(ValidationError):
        SlurmJob(
            name="test",
            calculation_id="test",
            working_directory=Path("."),
            command=("echo", ""),
            resources=SlurmResources(),
        )
def test_render_minimal_slurm_script():
    job = SlurmJob(
        name="test-job",
        calculation_id="test_001",
        working_directory=Path("/work/test"),
        command=(
            "srun",
            "cp2k.psmp",
            "-i",
            "test.inp",
            "-o",
            "test.out",
        ),
        resources=SlurmResources(),
    )

    script = render_slurm_script(job)

    assert script.startswith("#!/bin/bash\n")

    assert "#SBATCH --job-name=test-job" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --time=01:00:00" in script

    assert "set -euo pipefail" in script
    assert "cd /work/test" in script

    assert (
        "srun cp2k.psmp -i test.inp -o test.out"
        in script
    )


def test_render_full_slurm_script():
    job = SlurmJob(
        name="igzo-sp",
        calculation_id="igzo_sp_001",
        working_directory=Path(
            "/work/e05/e05/user/igzo calculation"
        ),
        command=(
            "srun",
            "cp2k.psmp",
            "-i",
            "igzo.inp",
            "-o",
            "igzo.out",
        ),
        resources=SlurmResources(
            nodes=2,
            tasks_per_node=128,
            cpus_per_task=1,
            walltime="02:30:00",
            account="e05-test",
            partition="standard",
            qos="standard",
        ),
        modules=(
            "load-epcc-module",
            "load cp2k",
        ),
        environment={
            "OMP_NUM_THREADS": "1",
            "NSDW_TEST_VALUE": "hello world",
        },
        stdout_file=Path("slurm-%j.out"),
        stderr_file=Path("slurm-%j.err"),
    )

    script = render_slurm_script(job)

    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks-per-node=128" in script
    assert "#SBATCH --account=e05-test" in script
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --qos=standard" in script

    assert "#SBATCH --output=slurm-%j.out" in script
    assert "#SBATCH --error=slurm-%j.err" in script

    assert (
        "cd '/work/e05/e05/user/igzo calculation'"
        in script
    )

    assert "module load-epcc-module" in script
    assert "module load cp2k" in script

    assert "export OMP_NUM_THREADS=1" in script

    assert (
        "export NSDW_TEST_VALUE='hello world'"
        in script
    )

    assert (
        "srun cp2k.psmp -i igzo.inp -o igzo.out"
        in script
    )


def test_renderer_shell_quotes_command_arguments():
    job = SlurmJob(
        name="quoted-job",
        calculation_id="quoted_001",
        working_directory=Path("/work/test"),
        command=(
            "python",
            "my script.py",
            "--label",
            "oxygen vacancy",
        ),
        resources=SlurmResources(),
    )

    script = render_slurm_script(job)

    assert (
        "python 'my script.py' --label 'oxygen vacancy'"
        in script
    )
def test_render_shell_variable_environment():
    job = SlurmJob(
        name="shell-variable",
        calculation_id="shell_variable",
        working_directory=Path("/work/test"),
        command=("program.x",),
        resources=SlurmResources(),
        environment={
            "SRUN_CPUS_PER_TASK": ShellVariable(
                name="SLURM_CPUS_PER_TASK"
            ),
        },
    )

    script = render_slurm_script(job)

    assert (
        "export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
        in script
    )


def test_invalid_shell_variable_rejected():
    with pytest.raises(ValidationError):
        ShellVariable(
            name="BAD;rm -rf",
        )