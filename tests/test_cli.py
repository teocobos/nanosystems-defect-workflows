"""Tests for the NSDW command-line interface."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pymatgen.core import Lattice, Structure
from typer.testing import CliRunner

from nsdw.cli import app
from nsdw.workflows.convergence.models import (
    ConvergenceParameter,
)
from nsdw.workflows.convergence.reporting import (
    ConvergenceReport,
)
from nsdw.project.models import ProjectConfig
from nsdw.project.scaffold import create_project

runner = CliRunner()


def _write_fake_cp2k(
    path: Path,
) -> None:
    script = """#!/usr/bin/env python3
import sys

args = sys.argv
output = args[args.index("-o") + 1]

with open(output, "w") as handle:
    handle.write(
        "CP2K| version string: CP2K version 2026.2\\n"
        "GLOBAL| Project name cli_test\\n"
        "GLOBAL| Run type ENERGY\\n"
        "DFT| Charge 0\\n"
        "DFT| Multiplicity 1\\n"
        "SCF run converged in 2 steps\\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -10.0000000000\\n"
        "PROGRAM ENDED AT 2026-09-17 12:00:00.000\\n"
    )
"""

    path.write_text(
        script,
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_cp2k_input(
    path: Path,
) -> None:
    path.write_text(
        "&GLOBAL\n"
        "  PROJECT cli_test\n"
        "  RUN_TYPE ENERGY\n"
        "&END GLOBAL\n"
        "&FORCE_EVAL\n"
        "  &DFT\n"
        "    CHARGE 0\n"
        "    MULTIPLICITY 1\n"
        "    &MGRID\n"
        "      CUTOFF 600\n"
        "      REL_CUTOFF 60\n"
        "    &END MGRID\n"
        "    &SCF\n"
        "      EPS_SCF 1.0E-6\n"
        "    &END SCF\n"
        "    &XC\n"
        "      &XC_FUNCTIONAL PBE\n"
        "      &END XC_FUNCTIONAL\n"
        "    &END XC\n"
        "  &END DFT\n"
        "&END FORCE_EVAL\n",
        encoding="utf-8",
    )

def _write_fake_cp2k_convergence(
    path: Path,
) -> None:
    script = """#!/usr/bin/env python3
import sys

args = sys.argv
input_file = args[args.index("-i") + 1]
output_file = args[args.index("-o") + 1]

if "400-Ry" in input_file:
    energy = -10.000000
elif "600-Ry" in input_file:
    energy = -10.005000
elif "800-Ry" in input_file:
    energy = -10.005100
else:
    raise SystemExit(
        f"Unknown convergence candidate: {input_file}"
    )

project_name = input_file.removesuffix(".inp")

with open(output_file, "w") as handle:
    handle.write(
        "CP2K| version string: CP2K version 2026.2\\n"
        f"GLOBAL| Project name {project_name}\\n"
        "GLOBAL| Run type ENERGY\\n"
        "DFT| Charge 0\\n"
        "DFT| Multiplicity 1\\n"
        "SCF run converged in 2 steps\\n"
        f"ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] {energy:.10f}\\n"
        "PROGRAM ENDED AT 2026-10-01 12:00:00.000\\n"
    )
"""

    path.write_text(
        script,
        encoding="utf-8",
    )
    path.chmod(0o755)



def test_cli_cp2k_single_point(
    tmp_path,
):
    executable = tmp_path / "fake_cp2k"
    input_path = tmp_path / "cli_test.inp"
    data_dir = tmp_path / "cp2k-data"

    data_dir.mkdir()

    _write_fake_cp2k(executable)
    _write_cp2k_input(input_path)

    result = runner.invoke(
        app,
        [
            "workflow",
            "single-point",
            "--calculator",
            "cp2k",
            "--workdir",
            str(tmp_path),
            "--input",
            "cli_test.inp",
            "--output",
            "cli_test.out",
            "--result",
            "cli_result.json",
            "--executable",
            str(executable),
            "--cp2k-data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    assert (
        "Single-point workflow completed"
        in result.stdout
    )

    assert "cli_test" in result.stdout
    assert "completed" in result.stdout

    assert (
        tmp_path / "cli_test.out"
    ).is_file()

    assert (
        tmp_path / "cli_result.json"
    ).is_file()


def test_cli_cp2k_single_point_invalid_data_dir(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fake_cp2k"
    input_path = tmp_path / "cli_test.inp"
    missing_data_dir = tmp_path / "missing-cp2k-data"

    _write_fake_cp2k(executable)
    _write_cp2k_input(input_path)

    result = runner.invoke(
        app,
        [
            "workflow",
            "single-point",
            "--calculator",
            "cp2k",
            "--workdir",
            str(tmp_path),
            "--input",
            "cli_test.inp",
            "--executable",
            str(executable),
            "--cp2k-data-dir",
            str(missing_data_dir),
        ],
    )

    assert result.exit_code == 1
    assert "ERROR:" in result.stdout
    assert (
        "CP2K data directory does not exist"
        in result.stdout
    )


@patch("nsdw.cli.run_cp2k_single_point_archer2")
def test_cli_cp2k_single_point_archer2(
    mock_workflow,
    tmp_path,
):
    result = Mock()
    result.calculation.id = "sio2_sp_archer2_auto"
    result.calculation.status.value = "completed"
    result.energy.total.value = -978.2388634470557
    result.energy.total.unit = "eV"
    result.provenance.execution.job_id = "15288186"

    mock_workflow.return_value = result

    cli_result = runner.invoke(
        app,
        [
            "workflow",
            "single-point-archer2",
            "--workdir",
            str(tmp_path),
            "--input",
            "sio2_sp.inp",
            "--output",
            "sio2_sp.out",
            "--result",
            "result.json",
            "--id",
            "sio2_sp_archer2_auto",
            "--account",
            "e05-bulk-shl",
            "--qos",
            "short",
            "--walltime",
            "00:20:00",
            "--module",
            "cp2k/cp2k-2025.2",
            "--poll-interval",
            "10",
            "--timeout",
            "1800",
        ],
    )

    assert cli_result.exit_code == 0, cli_result.output

    assert "ARCHER2 single-point workflow completed" in cli_result.stdout
    assert "sio2_sp_archer2_auto" in cli_result.stdout
    assert "15288186" in cli_result.stdout

    mock_workflow.assert_called_once()

    kwargs = mock_workflow.call_args.kwargs

    assert kwargs["calculation_id"] == "sio2_sp_archer2_auto"
    assert kwargs["working_directory"] == tmp_path.resolve()
    assert kwargs["input_file"] == Path("sio2_sp.inp")
    assert kwargs["output_file"] == Path("sio2_sp.out")
    assert kwargs["result_file"] == Path("result.json")
    assert kwargs["account"] == "e05-bulk-shl"
    assert kwargs["qos"] == "short"
    assert kwargs["walltime"] == "00:20:00"
    assert kwargs["module"] == "cp2k/cp2k-2025.2"

    monitor = kwargs["monitor_config"]
    assert monitor.poll_interval == 10
    assert monitor.timeout == 1800

@patch("nsdw.cli.run_cp2k_single_point_archer2")
def test_cli_cp2k_single_point_archer2_failure(
    mock_workflow,
    tmp_path,
):
    mock_workflow.side_effect = RuntimeError(
        "SLURM submission failed"
    )

    cli_result = runner.invoke(
        app,
        [
            "workflow",
            "single-point-archer2",
            "--workdir",
            str(tmp_path),
            "--input",
            "sio2_sp.inp",
            "--account",
            "e05-bulk-shl",
        ],
    )

    assert cli_result.exit_code == 1
    assert "ERROR:" in cli_result.stdout
    assert "SLURM submission failed" in cli_result.stdout

    mock_workflow.assert_called_once()


def test_workflow_convergence_generate_cutoff(
    tmp_path: Path,
) -> None:
    structure_file = tmp_path / "igzo.xyz"
    output_directory = tmp_path / "cutoff-study"

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["In", "Ga", "Zn", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ],
    )

    with (
        patch(
            "nsdw.cli.load_structure",
            return_value=(structure, []),
        ),
        patch(
            "nsdw.cli.generate_cp2k_convergence_study",
            return_value=(
                output_directory / "400-Ry",
                output_directory / "600-Ry",
                output_directory / "800-Ry",
            ),
        ) as mock_generate,
    ):
        result = runner.invoke(
            app,
            [
                "workflow",
                "convergence-generate",
                str(structure_file),
                "--parameter",
                "cutoff",
                "--values",
                "400",
                "--values",
                "600",
                "--values",
                "800",
                "--preset",
                "igzo-uzh-tzv2p",
                "--output",
                str(output_directory),
            ],
        )

    assert result.exit_code == 0

    assert "Convergence study generated" in result.stdout
    assert "cutoff" in result.stdout
    assert "3" in result.stdout

    mock_generate.assert_called_once()

    kwargs = mock_generate.call_args.kwargs

    assert kwargs["structure"] is structure
    assert kwargs["output_directory"] == output_directory

    base_config = kwargs["base_config"]

    assert base_config.project_name == "igzo"
    assert base_config.cutoff_ry == 600.0
    assert base_config.relative_cutoff_ry == 60.0
    assert base_config.run_type == "ENERGY"

    study = kwargs["study"]

    assert study.parameter == ConvergenceParameter.CUTOFF

    assert [
        candidate.label
        for candidate in study.candidates
    ] == [
        "400-Ry",
        "600-Ry",
        "800-Ry",
    ]

    assert [
        candidate.value.value
        for candidate in study.candidates
    ] == [
        400.0,
        600.0,
        800.0,
    ]

    assert all(
        candidate.value.unit == "Ry"
        for candidate in study.candidates
    )


def test_workflow_convergence_generate_uses_project_directory_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "igzo-project"

    create_project(
        root=project_root,
        config=ProjectConfig(
            name="igzo-project",
            material="IGZO",
            nsdw_version="0.1.0",
            components=["cp2k"],
        ),
    )

    structure_file = (
        project_root
        / "structures"
        / "validated"
        / "igzo.xyz"
    )

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["In", "Ga", "Zn", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ],
    )

    expected_output_directory = (
        project_root
        / "workflows"
        / "convergence"
        / "cutoff"
    )

    monkeypatch.chdir(
        project_root,
    )

    with (
        patch(
            "nsdw.cli.load_structure",
            return_value=(structure, []),
        ),
        patch(
            "nsdw.cli.generate_cp2k_convergence_study",
            return_value=(
                expected_output_directory / "400-Ry",
                expected_output_directory / "600-Ry",
                expected_output_directory / "800-Ry",
            ),
        ) as mock_generate,
    ):
        result = runner.invoke(
            app,
            [
                "workflow",
                "convergence-generate",
                str(structure_file),
                "--parameter",
                "cutoff",
                "--values",
                "400",
                "--values",
                "600",
                "--values",
                "800",
                "--preset",
                "igzo-uzh-tzv2p",
            ],
        )

    assert result.exit_code == 0, result.output

    mock_generate.assert_called_once()

    kwargs = mock_generate.call_args.kwargs

    assert (
        kwargs["output_directory"]
        == expected_output_directory
    )

    assert "Output directory:" in result.stdout


def test_workflow_convergence_generate_uses_standalone_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure_file = tmp_path / "igzo.xyz"

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["In", "Ga", "Zn", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ],
    )

    monkeypatch.chdir(
        tmp_path,
    )

    with (
        patch(
            "nsdw.cli.load_structure",
            return_value=(structure, []),
        ),
        patch(
            "nsdw.cli.generate_cp2k_convergence_study",
            return_value=(
                Path("convergence-study") / "400-Ry",
                Path("convergence-study") / "600-Ry",
                Path("convergence-study") / "800-Ry",
            ),
        ) as mock_generate,
    ):
        result = runner.invoke(
            app,
            [
                "workflow",
                "convergence-generate",
                str(structure_file),
                "--parameter",
                "cutoff",
                "--values",
                "400",
                "--values",
                "600",
                "--values",
                "800",
                "--preset",
                "igzo-uzh-tzv2p",
            ],
        )

    assert result.exit_code == 0, result.output

    mock_generate.assert_called_once()

    kwargs = mock_generate.call_args.kwargs

    assert (
        kwargs["output_directory"]
        == Path("convergence-study")
    )


def test_workflow_convergence_generate_cutoff_end_to_end(
    tmp_path: Path,
) -> None:
    structure_file = tmp_path / "igzo.cif"
    output_directory = tmp_path / "cutoff-study"

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["In", "Ga", "Zn", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ],
    )

    structure.to(
        filename=structure_file,
    )

    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-generate",
            str(structure_file),
            "--parameter",
            "cutoff",
            "--values",
            "400",
            "--values",
            "600",
            "--values",
            "800",
            "--preset",
            "igzo-uzh-tzv2p",
            "--output",
            str(output_directory),
        ],
    )

    assert result.exit_code == 0, result.output

    assert "Convergence study generated" in result.stdout

    manifest = output_directory / "manifest.json"

    assert manifest.is_file()

    for label in (
        "400-Ry",
        "600-Ry",
        "800-Ry",
    ):
        candidate_directory = (
            output_directory / label
        )

        assert candidate_directory.is_dir()

        input_files = list(
            candidate_directory.glob("*.inp")
        )

        coordinate_files = list(
            candidate_directory.glob("*.xyz")
        )

        assert len(input_files) == 1
        assert len(coordinate_files) == 1

    input_400 = next(
        (output_directory / "400-Ry").glob("*.inp")
    ).read_text()

    input_600 = next(
        (output_directory / "600-Ry").glob("*.inp")
    ).read_text()

    input_800 = next(
        (output_directory / "800-Ry").glob("*.inp")
    ).read_text()

    assert "CUTOFF 400" in input_400
    assert "CUTOFF 600" in input_600
    assert "CUTOFF 800" in input_800

    assert "REL_CUTOFF 60" in input_400
    assert "REL_CUTOFF 60" in input_600
    assert "REL_CUTOFF 60" in input_800


def test_workflow_convergence_run(
    tmp_path: Path,
) -> None:
    campaign_directory = tmp_path / "cutoff-study"

    report = ConvergenceReport(
        parameter=ConvergenceParameter.CUTOFF,
        energy_tolerance_ev_per_atom=1.0e-3,
        structure_hash="a" * 64,
        n_atoms=9,
        reference_candidate_label="800-Ry",
        selected_candidate_label="600-Ry",
        energy_tolerance_satisfied=True,
        candidates=(),
    )

    execution_environment = {
        "CP2K_DATA_DIR": "/fake/cp2k/data",
    }

    with (
        patch(
            "nsdw.cli.build_cp2k_environment",
            return_value=execution_environment,
        ) as mock_environment,
        patch(
            "nsdw.cli.run_and_report_cp2k_convergence_campaign",
            return_value=report,
        ) as mock_run,
    ):
        result = runner.invoke(
            app,
            [
                "workflow",
                "convergence-run",
                str(campaign_directory),
                "--executable",
                "cp2k.psmp",
            ],
        )

    assert result.exit_code == 0, result.output

    assert (
        "Convergence campaign completed"
        in result.stdout
    )

    assert "cutoff" in result.stdout
    assert "600-Ry" in result.stdout
    assert "Converged" in result.stdout

    mock_environment.assert_called_once_with(
        explicit_data_dir=None,
        executable="cp2k.psmp",
    )

    mock_run.assert_called_once_with(
        campaign_directory=campaign_directory.resolve(),
        executable="cp2k.psmp",
        environment=execution_environment,
    )



def test_workflow_convergence_run_passes_explicit_cp2k_data_dir(
    tmp_path: Path,
) -> None:
    campaign_directory = tmp_path / "cutoff-study"
    data_dir = tmp_path / "cp2k-data"

    report = ConvergenceReport(
        parameter=ConvergenceParameter.CUTOFF,
        energy_tolerance_ev_per_atom=1.0e-3,
        structure_hash="a" * 64,
        n_atoms=9,
        reference_candidate_label="800-Ry",
        selected_candidate_label="600-Ry",
        energy_tolerance_satisfied=True,
        candidates=(),
    )

    execution_environment = {
        "CP2K_DATA_DIR": str(data_dir.resolve()),
    }

    with (
        patch(
            "nsdw.cli.build_cp2k_environment",
            return_value=execution_environment,
        ) as mock_environment,
        patch(
            "nsdw.cli.run_and_report_cp2k_convergence_campaign",
            return_value=report,
        ) as mock_run,
    ):
        result = runner.invoke(
            app,
            [
                "workflow",
                "convergence-run",
                str(campaign_directory),
                "--executable",
                "custom-cp2k",
                "--cp2k-data-dir",
                str(data_dir),
            ],
        )

    assert result.exit_code == 0, result.output

    mock_environment.assert_called_once_with(
        explicit_data_dir=data_dir,
        executable="custom-cp2k",
    )

    mock_run.assert_called_once_with(
        campaign_directory=campaign_directory.resolve(),
        executable="custom-cp2k",
        environment=execution_environment,
    )


def test_workflow_convergence_run_failure(
    tmp_path: Path,
) -> None:
    campaign_directory = tmp_path / "cutoff-study"

    execution_environment = {
        "CP2K_DATA_DIR": "/fake/cp2k/data",
    }

    with (
        patch(
            "nsdw.cli.build_cp2k_environment",
            return_value=execution_environment,
        ),
        patch(
            "nsdw.cli.run_and_report_cp2k_convergence_campaign",
            side_effect=RuntimeError(
                "CP2K convergence campaign failed"
            ),
        ) as mock_run,
    ):
        result = runner.invoke(
            app,
            [
                "workflow",
                "convergence-run",
                str(campaign_directory),
            ],
        )

    assert result.exit_code == 1
    assert "ERROR:" in result.stdout
    assert (
        "CP2K convergence campaign failed"
        in result.stdout
    )

    mock_run.assert_called_once_with(
        campaign_directory=campaign_directory.resolve(),
        executable="cp2k.psmp",
        environment=execution_environment,
    )

def test_workflow_convergence_run_end_to_end(
    tmp_path: Path,
) -> None:
    structure_file = tmp_path / "igzo.cif"
    campaign_directory = tmp_path / "cutoff-study"
    executable = tmp_path / "fake_cp2k_convergence"
    data_dir = tmp_path / "cp2k-data"

    data_dir.mkdir()

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["In", "Ga", "Zn", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ],
    )

    structure.to(
        filename=structure_file,
    )

    _write_fake_cp2k_convergence(
        executable,
    )

    generate_result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-generate",
            str(structure_file),
            "--parameter",
            "cutoff",
            "--values",
            "400",
            "--values",
            "600",
            "--values",
            "800",
            "--preset",
            "igzo-uzh-tzv2p",
            "--output",
            str(campaign_directory),
        ],
    )

    assert generate_result.exit_code == 0, (
        generate_result.output
    )

    run_result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-run",
            str(campaign_directory),
            "--executable",
            str(executable),
            "--cp2k-data-dir",
            str(data_dir),
        ],
    )

    assert run_result.exit_code == 0, (
        run_result.output
    )

    assert (
        "Convergence campaign completed"
        in run_result.stdout
    )

    assert "Converged" in run_result.stdout
    assert "600-Ry" in run_result.stdout

    assert (
        campaign_directory / "convergence-report.json"
    ).is_file()

    assert (
        campaign_directory / "convergence-report.csv"
    ).is_file()

    for label in (
        "400-Ry",
        "600-Ry",
        "800-Ry",
    ):
        candidate_directory = (
            campaign_directory / label
        )

        assert list(
            candidate_directory.glob("*.out")
        )

        assert (
            candidate_directory / "result.json"
        ).is_file()


def test_cli_structure_validate_xyz_with_lattice(tmp_path: Path):
    """Validate an XYZ structure using six CLI lattice parameters."""

    xyz_file = tmp_path / "sio2.xyz"

    xyz_file.write_text(
        "3\n"
        "Example SiO2\n"
        "Si 0.0 0.0 0.0\n"
        "O 1.6 0.0 0.0\n"
        "O 0.0 1.6 0.0\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "structure",
            "validate",
            str(xyz_file),
            "--a", "10",
            "--b", "11",
            "--c", "12",
            "--alpha", "90",
            "--beta", "90",
            "--gamma", "120",
            "--format", "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"valid": true' in result.stdout.lower()

def test_cli_structure_validate_rejects_incomplete_lattice(
    tmp_path: Path,
):
    xyz_file = tmp_path / "sio2.xyz"

    xyz_file.write_text(
        "3\n"
        "Example SiO2\n"
        "Si 0.0 0.0 0.0\n"
        "O 1.6 0.0 0.0\n"
        "O 0.0 1.6 0.0\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "structure",
            "validate",
            str(xyz_file),
            "--a", "10",
            "--b", "11",
            "--c", "12",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "Supply all six lattice parameters" in result.output

def test_cli_structure_symmetry_xyz_with_lattice(
    tmp_path: Path,
):
    """Analyse symmetry for an XYZ file with an explicit cell."""

    xyz_file = tmp_path / "silicon.xyz"

    xyz_file.write_text(
        "1\n"
        "Single silicon atom in a cubic cell\n"
        "Si 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "structure",
            "symmetry",
            str(xyz_file),
            "--a", "5",
            "--b", "5",
            "--c", "5",
            "--alpha", "90",
            "--beta", "90",
            "--gamma", "90",
            "--format", "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"space_group_number": 221' in result.stdout

def test_cli_structure_supercell_xyz_with_lattice(
    tmp_path: Path,
):
    """Search supercells for an XYZ structure with an explicit cell."""

    xyz_file = tmp_path / "silicon.xyz"

    xyz_file.write_text(
        "1\n"
        "Single silicon atom in a cubic cell\n"
        "Si 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "structure",
            "supercell",
            str(xyz_file),
            "--a", "5",
            "--b", "5",
            "--c", "5",
            "--alpha", "90",
            "--beta", "90",
            "--gamma", "90",
            "--min-atoms", "8",
            "--max-atoms", "8",
            "--min-image-distance", "9",
            "--max-scale", "2",
            "--format", "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"command": "structure.supercell"' in result.stdout
    assert '"acceptable": true' in result.stdout

def test_cli_generate_vacancies_xyz_with_lattice(
    tmp_path: Path,
):
    """Generate an oxygen vacancy from an XYZ structure with a cell."""

    xyz_file = tmp_path / "sio2.xyz"
    output_dir = tmp_path / "vacancies"

    xyz_file.write_text(
        "3\n"
        "Example SiO2 structure\n"
        "Si 0.0 0.0 0.0\n"
        "O 1.6 0.0 0.0\n"
        "O 0.0 1.6 0.0\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "defects",
            "generate-vacancies",
            str(xyz_file),
            "--species", "O",
            "--output", str(output_dir),
            "--a", "10",
            "--b", "10",
            "--c", "10",
            "--alpha", "90",
            "--beta", "90",
            "--gamma", "90",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Vacancy dataset generated" in result.output
    assert output_dir.is_dir()