"""Tests for the CP2K input parser."""

from pathlib import Path

import pytest

from nsdw.calculators.cp2k.models import (
    CP2KRunType,
)
from nsdw.calculators.cp2k.input_parser import (
    CP2KInputParseError,
    parse_cp2k_input,
    parse_cp2k_input_text,
)


FIXTURES = (
    Path(__file__).parent
    / "fixtures"
)


def test_parse_real_igzo_input():
    result = parse_cp2k_input(
        FIXTURES
        / "igzo_ordered_003_sp.inp"
    )

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

    assert result.xc_functional == "PBE"

    assert result.cutoff_ry == pytest.approx(
        700.0
    )

    assert (
        result.relative_cutoff_ry
        == pytest.approx(60.0)
    )

    assert result.eps_scf == pytest.approx(
        1.0e-7
    )

    assert result.k_points == (6, 6, 1)

    assert result.basis_set_file == (
        "BASIS_MOLOPT_UZH"
    )

    assert result.potential_file == (
        "POTENTIAL_UZH"
    )


def test_parse_real_igzo_kinds():
    result = parse_cp2k_input(
        FIXTURES
        / "igzo_ordered_003_sp.inp"
    )

    assert len(result.kinds) == 4

    assert result.kinds[0].kind == "In"
    assert result.kinds[0].element == "In"
    assert (
        result.kinds[0].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q13"
    )
    assert (
        result.kinds[0].potential
        == "GTH-PBE-q13"
    )

    assert result.kinds[1].kind == "Ga"
    assert result.kinds[1].element == "Ga"
    assert (
        result.kinds[1].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q13"
    )
    assert (
        result.kinds[1].potential
        == "GTH-PBE-q13"
    )

    assert result.kinds[2].kind == "Zn"
    assert result.kinds[2].element == "Zn"
    assert (
        result.kinds[2].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q12"
    )
    assert (
        result.kinds[2].potential
        == "GTH-PBE-q12"
    )

    assert result.kinds[3].kind == "O"
    assert result.kinds[3].element == "O"
    assert (
        result.kinds[3].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q6"
    )
    assert (
        result.kinds[3].potential
        == "GTH-PBE-q6"
    )

def test_real_igzo_input_does_not_use_admm():
    result = parse_cp2k_input(
        FIXTURES
        / "igzo_ordered_003_sp.inp"
    )

    assert result.admm is False


def test_empty_cp2k_input_rejected():
    with pytest.raises(
        CP2KInputParseError,
        match="empty",
    ):
        parse_cp2k_input_text("   ")


def test_missing_input_file_rejected(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError
    ):
        parse_cp2k_input(
            tmp_path / "missing.inp"
        )
