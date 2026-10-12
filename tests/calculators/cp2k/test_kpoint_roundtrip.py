"""Round-trip tests for CP2K k-point generation and parsing."""

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KInputConfig,
    CP2KKindConfig,
    CP2KKPointConfig,
    CP2KSCFConfig,
    CP2KWeightedKPoint,
)
from nsdw.calculators.cp2k.generator import render_cp2k_input
from nsdw.calculators.cp2k.input_parser import parse_cp2k_input_text


@pytest.fixture
def structure():
    return Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0, 0, 0]],
    )


def make_config(sampling=None, *, legacy=None):
    return CP2KInputConfig(
        project_name="kpoint-roundtrip",
        coordinate_file="structure.xyz",
        k_points=legacy,
        kpoint_sampling=sampling,
        scf=CP2KSCFConfig(solver="DIAGONALIZATION"),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(
                CP2KKindConfig(
                    element="Si",
                    basis_set="DZVP-MOLOPT-SR-GTH",
                    potential="GTH-PBE",
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "sampling",
    [
        None,
        CP2KKPointConfig(scheme="NONE"),
        CP2KKPointConfig(scheme="GAMMA"),
        CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(3, 3, 3),
        ),
        CP2KKPointConfig(
            scheme="MACDONALD",
            mesh=(4, 4, 4),
            shift=(0.25, 0.25, 0.25),
        ),
        CP2KKPointConfig(
            scheme="GENERAL",
            points=(
                CP2KWeightedKPoint(
                    coordinates=(0.0, 0.0, 0.0),
                    weight=0.25,
                ),
                CP2KWeightedKPoint(
                    coordinates=(0.5, 0.0, 0.0),
                    weight=0.75,
                ),
            ),
        ),
    ],
)
def test_structured_roundtrip(sampling, structure):
    config = make_config(sampling)
    generated = render_cp2k_input(config, structure)
    parsed = parse_cp2k_input_text(generated)

    if sampling is None:
        assert parsed.kpoint_sampling is None
        assert parsed.k_points is None
    elif sampling.scheme == "NONE":
        assert parsed.kpoint_sampling is None
        assert parsed.k_points is None
    else:
        assert parsed.kpoint_sampling == sampling

        if sampling.scheme == "MONKHORST-PACK":
            assert parsed.k_points == sampling.mesh
        else:
            assert parsed.k_points is None


def test_legacy_monkhorst_pack_roundtrip(structure):
    config = make_config(legacy=(2, 2, 2))
    generated = render_cp2k_input(config, structure)
    parsed = parse_cp2k_input_text(generated)

    assert parsed.k_points == (2, 2, 2)
    assert parsed.kpoint_sampling == CP2KKPointConfig(
        scheme="MONKHORST-PACK",
        mesh=(2, 2, 2),
    )


def test_default_ot_gamma_roundtrip(structure):
    config = make_config().model_copy(
        update={"scf": CP2KSCFConfig(solver="OT")}
    )

    generated = render_cp2k_input(config, structure)
    parsed = parse_cp2k_input_text(generated)

    assert "&KPOINTS" not in generated
    assert parsed.k_points is None
    assert parsed.kpoint_sampling is None
    assert parsed.scf_solver == "OT"
