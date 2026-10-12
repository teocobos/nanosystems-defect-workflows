"""Structured CP2K k-point parsing regression tests."""

import pytest

from nsdw.calculators.cp2k.input_parser import (
    CP2KInputParseError,
    parse_cp2k_input_text,
)


def parse_section(content):
    return parse_cp2k_input_text(
        "&FORCE_EVAL\n"
        "  &DFT\n"
        "    &KPOINTS\n"
        f"{content}\n"
        "    &END KPOINTS\n"
        "  &END DFT\n"
        "&END FORCE_EVAL\n"
    )


def test_implicit_gamma():
    parsed = parse_cp2k_input_text(
        "&GLOBAL\n  PROJECT test\n&END GLOBAL\n"
    )
    assert parsed.k_points is None
    assert parsed.kpoint_sampling is None


def test_explicit_gamma():
    parsed = parse_section("      SCHEME GAMMA")
    assert parsed.k_points is None
    assert parsed.kpoint_sampling.scheme == "GAMMA"


def test_monkhorst_pack_preserves_legacy_mesh():
    parsed = parse_section(
        "      SCHEME MONKHORST-PACK 4 4 4"
    )
    assert parsed.k_points == (4, 4, 4)
    assert parsed.kpoint_sampling.mesh == (4, 4, 4)


def test_macdonald_preserves_shift():
    parsed = parse_section(
        "      SCHEME MACDONALD 4 4 4 0.25 0.25 0.25"
    )
    assert parsed.k_points is None
    assert parsed.kpoint_sampling.scheme == "MACDONALD"
    assert parsed.kpoint_sampling.shift == (0.25, 0.25, 0.25)


def test_general_preserves_weighted_points():
    parsed = parse_section(
        "      SCHEME GENERAL\n"
        "      KPOINT 0.0 0.0 0.0 0.25\n"
        "      KPOINT 0.5 0.0 0.0 0.75"
    )
    assert parsed.k_points is None
    assert len(parsed.kpoint_sampling.points) == 2
    assert parsed.kpoint_sampling.points[1].weight == 0.75


@pytest.mark.parametrize(
    "content",
    [
        "SCHEME UNKNOWN",
        "SCHEME MONKHORST-PACK 2 2",
        "SCHEME MONKHORST-PACK 0 2 2",
        "SCHEME GAMMA\nSCHEME GAMMA",
        "SCHEME GENERAL",
        "SCHEME GENERAL\nKPOINT 0 0 0",
        "SCHEME MACDONALD 2 2 2",
        "SCHEME GAMMA\nUNSUPPORTED 1",
    ],
)
def test_malformed_kpoint_sections_rejected(content):
    with pytest.raises(CP2KInputParseError):
        parse_section(content)


def test_comments_ignored():
    parsed = parse_section(
        "      ! SCHEME UNKNOWN\n"
        "      SCHEME MONKHORST-PACK 2 3 4 # valid"
    )
    assert parsed.k_points == (2, 3, 4)


def test_repeated_sections_rejected():
    text = (
        "&KPOINTS\nSCHEME GAMMA\n&END KPOINTS\n"
        "&KPOINTS\nSCHEME GAMMA\n&END KPOINTS\n"
    )
    with pytest.raises(CP2KInputParseError):
        parse_cp2k_input_text(text)
