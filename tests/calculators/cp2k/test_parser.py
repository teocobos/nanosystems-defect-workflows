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

def test_parse_multigrid_info_from_real_igzo_format():
    text = """
 -------------------------------------------------------------------------------
 ----                             MULTIGRID INFO                            ----
 -------------------------------------------------------------------------------
 count for grid        1:       65027435          cutoff [a.u.]          200.00
 count for grid        2:       30531810          cutoff [a.u.]           66.67
 count for grid        3:       17861146          cutoff [a.u.]           22.22
 count for grid        4:        8080035          cutoff [a.u.]            7.41
 total gridlevel count  :      121500426

 ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -767.248205609001843
 SCF run converged in 2 steps
 PROGRAM ENDED AT 2026-09-27
 """

    result = parse_cp2k_text(text)

    assert result.multigrid is not None

    assert len(result.multigrid.levels) == 4

    assert result.multigrid.levels[0].grid_number == 1
    assert result.multigrid.levels[0].count == 65027435
    assert (
        result.multigrid.levels[0].cutoff_au
        == pytest.approx(200.00)
    )

    assert result.multigrid.levels[1].grid_number == 2
    assert result.multigrid.levels[1].count == 30531810
    assert (
        result.multigrid.levels[1].cutoff_au
        == pytest.approx(66.67)
    )

    assert result.multigrid.levels[2].grid_number == 3
    assert result.multigrid.levels[2].count == 17861146
    assert (
        result.multigrid.levels[2].cutoff_au
        == pytest.approx(22.22)
    )

    assert result.multigrid.levels[3].grid_number == 4
    assert result.multigrid.levels[3].count == 8080035
    assert (
        result.multigrid.levels[3].cutoff_au
        == pytest.approx(7.41)
    )

    assert (
        result.multigrid.total_gridlevel_count
        == 121500426
    )

def test_parser_uses_final_multigrid_block():
    text = """
 ----                             MULTIGRID INFO                            ----
 count for grid        1:            100          cutoff [a.u.]          150.00
 count for grid        2:             50          cutoff [a.u.]           50.00
 total gridlevel count  :            150

 ----                             MULTIGRID INFO                            ----
 count for grid        1:            200          cutoff [a.u.]          200.00
 count for grid        2:            100          cutoff [a.u.]           66.67
 count for grid        3:             50          cutoff [a.u.]           22.22
 total gridlevel count  :            350

 ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -20.0
 SCF run converged in 2 steps
 PROGRAM ENDED AT 2026-09-27
 """

    result = parse_cp2k_text(text)

    assert result.multigrid is not None

    assert len(result.multigrid.levels) == 3

    assert result.multigrid.levels[0].grid_number == 1
    assert result.multigrid.levels[0].count == 200
    assert (
        result.multigrid.levels[0].cutoff_au
        == pytest.approx(200.00)
    )

    assert result.multigrid.levels[-1].grid_number == 3
    assert result.multigrid.levels[-1].count == 50

    assert (
        result.multigrid.total_gridlevel_count
        == 350
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


def test_final_scf_failure_overrides_earlier_success():
    from nsdw.calculators.cp2k.parser import parse_cp2k_text
    from nsdw.calculators.cp2k.models import CP2KSCFStatus

    output = """
 CP2K| version string: CP2K version 2025.1
 SCF run converged in 8 steps
 SCF run NOT converged
 PROGRAM ENDED AT
"""

    parsed = parse_cp2k_text(output)

    assert parsed.scf.status == CP2KSCFStatus.NOT_CONVERGED
    assert parsed.scf.iterations is None


def test_final_scf_success_overrides_earlier_failure():
    from nsdw.calculators.cp2k.parser import parse_cp2k_text
    from nsdw.calculators.cp2k.models import CP2KSCFStatus

    output = """
 CP2K| version string: CP2K version 2025.1
 SCF run NOT converged
 SCF run converged in 12 steps
 PROGRAM ENDED AT
"""

    parsed = parse_cp2k_text(output)

    assert parsed.scf.status == CP2KSCFStatus.CONVERGED
    assert parsed.scf.iterations == 12
