"""Tests for CP2K execution-request construction."""

from pathlib import Path
import pytest
from nsdw.calculators.cp2k.execution import (
    CP2KExecutionError,
    build_cp2k_execution_request,
)
from nsdw.calculators.cp2k import (
    build_archer2_cp2k_job,
    build_cp2k_execution_request,
)
from nsdw.execution import render_slurm_script

def test_build_cp2k_execution_request(
    tmp_path,
):
    input_path = tmp_path / "calculation.inp"

    input_path.write_text(
        "&GLOBAL\n"
        "  PROJECT test\n"
        "  RUN_TYPE ENERGY\n"
        "&END GLOBAL\n"
    )

    request = build_cp2k_execution_request(
        calculation_id="cp2k_test",
        working_directory=tmp_path,
        input_file="calculation.inp",
        output_file="calculation.out",
    )

    assert request.calculation_id == "cp2k_test"

    assert request.command == (
        "cp2k.psmp",
        "-i",
        "calculation.inp",
        "-o",
        "calculation.out",
    )

    assert (
        request.working_directory
        == tmp_path.resolve()
    )


def test_custom_cp2k_executable(
    tmp_path,
):
    input_path = tmp_path / "test.inp"

    input_path.write_text(
        "&GLOBAL\n&END GLOBAL\n"
    )

    request = build_cp2k_execution_request(
        calculation_id="test",
        working_directory=tmp_path,
        input_file="test.inp",
        output_file="test.out",
        executable="cp2k.ssmp",
    )

    assert request.command[0] == "cp2k.ssmp"


def test_cp2k_execution_environment(
    tmp_path,
):
    input_path = tmp_path / "test.inp"

    input_path.write_text(
        "&GLOBAL\n&END GLOBAL\n"
    )

    request = build_cp2k_execution_request(
        calculation_id="test",
        working_directory=tmp_path,
        input_file="test.inp",
        output_file="test.out",
        environment={
            "OMP_NUM_THREADS": "2",
        },
    )

    assert (
        request.environment["OMP_NUM_THREADS"]
        == "2"
    )


def test_missing_cp2k_input_rejected(
    tmp_path,
):
    with pytest.raises(
        CP2KExecutionError,
        match="input file",
    ):
        build_cp2k_execution_request(
            calculation_id="test",
            working_directory=tmp_path,
            input_file="missing.inp",
            output_file="test.out",
        )


def test_missing_working_directory_rejected(
    tmp_path,
):
    with pytest.raises(
        CP2KExecutionError,
        match="Working directory",
    ):
        build_cp2k_execution_request(
            calculation_id="test",
            working_directory=(
                tmp_path / "missing"
            ),
            input_file="test.inp",
            output_file="test.out",
        )


def test_empty_cp2k_executable_rejected(
    tmp_path,
):
    input_path = tmp_path / "test.inp"

    input_path.write_text(
        "&GLOBAL\n&END GLOBAL\n"
    )

    with pytest.raises(
        CP2KExecutionError,
        match="executable",
    ):
        build_cp2k_execution_request(
            calculation_id="test",
            working_directory=tmp_path,
            input_file="test.inp",
            output_file="test.out",
            executable="   ",
        )
def test_build_archer2_cp2k_job(tmp_path):
    job = build_archer2_cp2k_job(
        calculation_id="igzo_sp_001",
        working_directory=tmp_path,
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        nodes=2,
        tasks_per_node=128,
        walltime="02:00:00",
    )

    assert job.name == "igzo_sp_001"
    assert job.calculation_id == "igzo_sp_001"

    assert job.resources.nodes == 2
    assert job.resources.tasks_per_node == 128
    assert job.resources.account == "e05"
    assert job.resources.partition == "standard"
    assert job.resources.qos == "standard"

    assert job.modules == ("load cp2k",)

    assert job.command == (
        "srun",
        "--hint=nomultithread",
        "--distribution=block:block",
        "cp2k.psmp",
        "-i",
        "igzo.inp",
        "-o",
        "igzo.out",
    )


def test_render_archer2_cp2k_script(tmp_path):
    job = build_archer2_cp2k_job(
        calculation_id="igzo_sp_001",
        working_directory=tmp_path,
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        nodes=1,
        tasks_per_node=128,
        walltime="01:00:00",
    )

    script = render_slurm_script(job)

    assert "#SBATCH --job-name=igzo_sp_001" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=128" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --account=e05" in script
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --qos=standard" in script

    assert "module load cp2k" in script
    assert "export OMP_NUM_THREADS=1" in script

    assert (
        "srun --hint=nomultithread "
        "--distribution=block:block "
        "cp2k.psmp -i igzo.inp -o igzo.out"
        in script
    )

def test_render_archer2_cp2k_short_script(tmp_path):
    job = build_archer2_cp2k_job(
        calculation_id="sio2_sp_archer2",
        working_directory=tmp_path,
        input_file=Path("sio2_sp.inp"),
        output_file=Path("sio2_sp.out"),
        account="e05-bulk-shl",
        nodes=1,
        tasks_per_node=128,
        walltime="00:20:00",
        qos="short",
        module="cp2k/cp2k-2025.2",
    )

    script = render_slurm_script(job)

    assert "#SBATCH --account=e05-bulk-shl" in script
    assert "#SBATCH --partition=standard" in script
    assert "#SBATCH --qos=short" in script
    assert "#SBATCH --time=00:20:00" in script
    assert "module load cp2k/cp2k-2025.2" in script
    assert "cp2k.psmp -i sio2_sp.inp -o sio2_sp.out" in script

def test_build_archer2_cp2k_job_versioned_module(
    tmp_path,
):
    job = build_archer2_cp2k_job(
        calculation_id="igzo_sp_001",
        working_directory=tmp_path,
        input_file=Path("igzo.inp"),
        output_file=Path("igzo.out"),
        account="e05",
        module="cp2k/cp2k-2025.2",
        executable="cp2k.psmp",
    )

    assert job.modules == (
        "load cp2k/cp2k-2025.2",
    )

    assert job.command == (
        "srun",
        "--hint=nomultithread",
        "--distribution=block:block",
        "cp2k.psmp",
        "-i",
        "igzo.inp",
        "-o",
        "igzo.out",
    )

    script = render_slurm_script(job)

    assert (
        "module load cp2k/cp2k-2025.2"
        in script
    )

def test_archer2_cp2k_empty_module_rejected(tmp_path):
    with pytest.raises(
        CP2KExecutionError,
        match="CP2K module cannot be empty",
    ):
        build_archer2_cp2k_job(
            calculation_id="igzo_sp",
            working_directory=tmp_path,
            input_file=Path("igzo.inp"),
            output_file=Path("igzo.out"),
            account="e05",
            module="   ",
        )


def test_archer2_cp2k_empty_executable_rejected(tmp_path):
    with pytest.raises(
        CP2KExecutionError,
        match="CP2K executable cannot be empty",
    ):
        build_archer2_cp2k_job(
            calculation_id="igzo_sp",
            working_directory=tmp_path,
            input_file=Path("igzo.inp"),
            output_file=Path("igzo.out"),
            account="e05",
            executable="   ",
        )