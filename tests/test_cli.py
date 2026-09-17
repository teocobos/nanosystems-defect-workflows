"""Tests for the NSDW command-line interface."""

from pathlib import Path

from typer.testing import CliRunner

from nsdw.cli import app


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


def test_cli_cp2k_single_point(
    tmp_path,
):
    executable = tmp_path / "fake_cp2k"
    input_path = tmp_path / "cli_test.inp"

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
        ],
    )

    assert result.exit_code == 0

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
