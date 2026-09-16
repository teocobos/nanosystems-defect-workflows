from pathlib import Path

import pytest

from nsdw.structures.parser import load_structure
from nsdw.structures.validator import (
    summarise_structure,
    validate_structure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

IGZO_REFERENCE = (
    REPO_ROOT
    / "examples"
    / "igzo"
    / "igzo_crystal_ingazno4_cod1521670.cif"
)


def test_igzo_reference_structure():
    """
    Regression test for the IGZO COD 1521670 reference structure.
    """

    structure, parser_warnings = load_structure(
        IGZO_REFERENCE
    )

    summary = summarise_structure(structure)
    validation = validate_structure(structure)

    assert summary.num_sites == 21

    assert summary.reduced_formula == "ZnInGaO4"

    assert summary.lattice.a == pytest.approx(
        3.299,
        abs=1e-6,
    )

    assert summary.lattice.b == pytest.approx(
        3.299,
        abs=1e-6,
    )

    assert summary.lattice.c == pytest.approx(
        26.101,
        abs=1e-6,
    )

    assert summary.lattice.gamma == pytest.approx(
        120.0,
        abs=1e-6,
    )

    assert validation.minimum_distance == pytest.approx(
        1.93011858,
        abs=1e-6,
    )

    assert validation.valid is True

    # The experimental crystallographic structure contains
    # occupational disorder / partial occupancies.
    assert validation.ordered is False

    assert isinstance(parser_warnings, list)
