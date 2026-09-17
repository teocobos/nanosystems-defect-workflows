"""Tests for the NSDW single-point workflow."""

from pathlib import Path

import pytest

from nsdw.models.calculation import (
    Backend,
    CalculationStatus,
    CalculationType,
)
from nsdw.workflows import (
    SinglePointWorkflowError,
    run_cp2k_single_point,
)


def _write_fake_cp2k(
    path: Path,
    *,
    return_code: int = 0,
) -> None:
    """Create a minimal executable that mimics CP2K."""

    script = f"""#!/usr/bin/env python3
import sys

args = sys.argv

output = args[args.index("-o") + 1]

with open(output, "w") as handle:
    handle.write(
        "CP2K| version string: CP2K version 2026.2\\n"
        "GLOBAL| Project name test_sp\\n"
        "GLOBAL| Run type ENERGY\\n"
        "DFT| Charge 0\\n"
        "DFT| Multiplicity 1\\n"
        "SCF run converged in 2 steps\\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree] -10.0000000000\\n"
        "PROGRAM ENDED AT 2026-09-17 12:00:00.000\\n"
    )

sys.exit({return_code})
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
        "  PROJECT test_sp\n"
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


def test_cp2k_single_point_workflow(
    tmp_path,
):
    executable = tmp_path / "fake_cp2k"
    input_path = tmp_path / "test.inp"

    _write_fake_cp2k(executable)
    _write_cp2k_input(input_path)

    result = run_cp2k_single_point(
        calculation_id="test_sp",
        working_directory=tmp_path,
        input_file="test.inp",
        output_file="test.out",
        executable=str(executable),
    )

    assert (
        result.calculation.type
        == CalculationType.SINGLE_POINT
    )
    assert (
        result.calculation.status
        == CalculationStatus.COMPLETED
    )
    assert (
        result.calculation.backend
        == Backend.CP2K
    )

    assert result.energy is not None
    assert result.energy.total is not None

    assert result.energy.total.value == pytest.approx(
        -272.11386245988
    )

    assert (
        tmp_path / "result.json"
    ).is_file()

    assert (
        tmp_path / "test.out"
    ).is_file()

    assert result.provenance is not None
    assert len(
        result.provenance.input_files
    ) == 1
    assert len(
        result.provenance.output_files
    ) == 1


def test_cp2k_single_point_failure(
    tmp_path,
):
    executable = tmp_path / "fake_cp2k"
    input_path = tmp_path / "test.inp"

    _write_fake_cp2k(
        executable,
        return_code=7,
    )
    _write_cp2k_input(input_path)

    with pytest.raises(
        SinglePointWorkflowError,
        match="return code 7",
    ):
        run_cp2k_single_point(
            calculation_id="test_sp",
            working_directory=tmp_path,
            input_file="test.inp",
            output_file="test.out",
            executable=str(executable),
        )
