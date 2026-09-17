"""Tests for the CP2K output parser."""

from pathlib import Path

import pytest

from nsdw.calculators.cp2k import (
    CP2KRunType,
    CP2KSCFStatus,
)
from nsdw.calculators.cp2k.parser import (
    CP2KParseError,
    parse_cp2k_output,
    parse_cp2k_text,
)


FIXTURES = (
    Path(__file__).parent
    / "fixtures"
)


def test_parse_completed_energy_output():
    result = parse_cp2k_output(
        FIXTURES / "energy_complete.out"
    )

    assert result.cp2k_version == "2026.2"
    assert result.project_name == "sio2-bulk"

    assert result.run_type == CP2KRunType.ENERGY

    assert result.charge == 0
    assert result.multiplicity == 1

    assert (
        result.energy.total_energy_hartree
        == pytest.approx(
            -101.234567890000
        )
    )

    assert (
        result.scf.status
        == CP2KSCFStatus.CONVERGED
    )

    assert result.scf.iterations == 3
    assert result.normal_termination is True

    assert result.warnings == ()


def test_parser_uses_final_energy():
    text = """
 ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -10.0
 ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -20.0
 PROGRAM ENDED AT 2026-09-17
 """

    result = parse_cp2k_text(text)

    assert (
        result.energy.total_energy_hartree
        == pytest.approx(-20.0)
    )


def test_empty_output_rejected():
    with pytest.raises(
        CP2KParseError,
        match="empty",
    ):
        parse_cp2k_text("   ")


def test_missing_file_rejected(tmp_path):
    missing = (
        tmp_path
        / "missing.out"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        parse_cp2k_output(missing)


def test_incomplete_output_generates_warnings():
    text = """
 CP2K| version string: CP2K version 2026.2
 GLOBAL| Project name test
 GLOBAL| Run type ENERGY
 """

    result = parse_cp2k_text(text)

    assert result.normal_termination is False
    assert (
        result.scf.status
        == CP2KSCFStatus.UNKNOWN
    )

    assert (
        "No total FORCE_EVAL energy found"
        in result.warnings
    )

    assert (
        "Final SCF status could not be determined"
        in result.warnings
    )

    assert (
        "Normal CP2K termination marker not found"
        in result.warnings
    )


def test_unknown_run_type_generates_warning():
    text = """
 GLOBAL| Run type SOME_FUTURE_RUN_TYPE
 """

    result = parse_cp2k_text(text)

    assert result.run_type is None

    assert (
        "Unsupported CP2K run type: "
        "SOME_FUTURE_RUN_TYPE"
        in result.warnings
    )
def test_parse_real_igzo_cp2k_output():
    result = parse_cp2k_output(
        FIXTURES
        / "igzo_ordered_003_sp_real.out"
    )

    assert result.cp2k_version == "2025.2"

    assert (
        result.project_name
        == "igzo_ordered_003_sp"
    )

    assert (
        result.run_type
        == CP2KRunType.ENERGY
    )

    assert result.charge == 0
    assert result.multiplicity == 1

    assert (
        result.energy.total_energy_hartree
        == pytest.approx(
            -767.248205609001843
        )
    )

    assert (
        result.scf.status
        == CP2KSCFStatus.CONVERGED
    )

    assert result.scf.iterations == 2

    assert result.normal_termination is True

    assert result.warnings == ()