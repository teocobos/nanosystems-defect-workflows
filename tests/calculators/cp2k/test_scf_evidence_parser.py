"""Regression tests for section-aware CP2K SCF evidence parsing."""

import pytest

from nsdw.calculators.cp2k.input_parser import (
    CP2KInputParseError,
    parse_cp2k_input_text,
)


def _input(scf_body: str) -> str:
    return (
        "&GLOBAL\n"
        "  PROJECT scf-evidence-test\n"
        "  RUN_TYPE ENERGY\n"
        "&END GLOBAL\n"
        "&FORCE_EVAL\n"
        "  &DFT\n"
        "    &SCF\n"
        f"{scf_body}\n"
        "    &END SCF\n"
        "  &END DFT\n"
        "&END FORCE_EVAL\n"
    )


def test_extracts_ot_scf_settings():
    text = _input(
        "      SCF_GUESS ATOMIC\n"
        "      EPS_SCF 1.0E-7\n"
        "      MAX_SCF 120\n"
        "      &OT\n"
        "        MINIMIZER CG\n"
        "        PRECONDITIONER FULL_SINGLE_INVERSE\n"
        "        ENERGY_GAP 0.002\n"
        "      &END OT\n"
        "      &OUTER_SCF\n"
        "        MAX_SCF 15\n"
        "      &END OUTER_SCF"
    )

    parsed = parse_cp2k_input_text(text)

    assert parsed.scf_solver == "OT"
    assert parsed.scf_guess == "ATOMIC"
    assert parsed.eps_scf == pytest.approx(1e-7)
    assert parsed.max_scf == 120
    assert parsed.outer_scf_max == 15
    assert parsed.ot_minimizer == "CG"
    assert parsed.ot_preconditioner == "FULL_SINGLE_INVERSE"
    assert parsed.energy_gap == pytest.approx(0.002)


def test_extracts_diagonalization_without_ot_settings():
    text = _input(
        "      SCF_GUESS RESTART\n"
        "      EPS_SCF 1.0E-8\n"
        "      MAX_SCF 80\n"
        "      &DIAGONALIZATION\n"
        "        ALGORITHM STANDARD\n"
        "      &END DIAGONALIZATION"
    )

    parsed = parse_cp2k_input_text(text)

    assert parsed.scf_solver == "DIAGONALIZATION"
    assert parsed.scf_guess == "RESTART"
    assert parsed.eps_scf == pytest.approx(1e-8)
    assert parsed.max_scf == 80

    assert parsed.outer_scf_max is None
    assert parsed.ot_minimizer is None
    assert parsed.ot_preconditioner is None
    assert parsed.energy_gap is None


def test_missing_scf_settings_remain_unknown():
    parsed = parse_cp2k_input_text(
        _input(
            "      &DIAGONALIZATION\n"
            "      &END DIAGONALIZATION"
        )
    )

    assert parsed.scf_guess is None
    assert parsed.eps_scf is None
    assert parsed.max_scf is None
    assert parsed.outer_scf_max is None


def test_rejects_duplicate_max_scf():
    text = _input(
        "      MAX_SCF 100\n"
        "      MAX_SCF 200\n"
        "      &OT\n"
        "      &END OT"
    )

    with pytest.raises(
        CP2KInputParseError,
        match="Repeated SCF setting",
    ):
        parse_cp2k_input_text(text)


def test_rejects_mismatched_section_terminator():
    text = _input(
        "      MAX_SCF 100\n"
        "      &OT\n"
        "      &END OUTER_SCF"
    )

    with pytest.raises(
        CP2KInputParseError,
        match="Mismatched CP2K section terminator",
    ):
        parse_cp2k_input_text(text)


def test_outer_scf_limit_does_not_replace_inner_limit():
    text = _input(
        "      MAX_SCF 90\n"
        "      &OT\n"
        "      &END OT\n"
        "      &OUTER_SCF\n"
        "        MAX_SCF 12\n"
        "      &END OUTER_SCF"
    )

    parsed = parse_cp2k_input_text(text)

    assert parsed.max_scf == 90
    assert parsed.outer_scf_max == 12
