"""Tests for CP2K input generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
)
from nsdw.calculators.cp2k.generator import (
    CP2KGenerationError,
    render_cp2k_input,
)
from nsdw.calculators.cp2k.input_parser import (
    parse_cp2k_input_text,
)
from nsdw.calculators.cp2k.presets import (
    get_basis_potential_preset,
)
from nsdw.structures.parser import load_structure


IGZO_CIF = (
    Path(__file__).parents[2]
    / "data"
    / "igzo"
    / "igzo_crystal_ordered_003.cif"
)


@pytest.fixture
def igzo_structure():
    """Return the ordered 21-atom IGZO test structure."""

    structure, _ = load_structure(IGZO_CIF)

    return structure


def make_config(
    *,
    functional: str = "PBE",
    coordinate_file: str = "igzo.xyz",
    k_points: tuple[int, int, int] | None = None,
) -> CP2KInputConfig:
    """Create a standard IGZO generation configuration."""

    return CP2KInputConfig(
        project_name="igzo_test",
        functional=functional,
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        coordinate_file=coordinate_file,
        k_points=k_points,
        basis_potential=get_basis_potential_preset(
            "igzo-uzh-tzv2p"
        ),
    )


def test_generates_pbe(
    igzo_structure,
):
    """PBE should use the compact CP2K XC form."""

    text = render_cp2k_input(
        make_config(functional="PBE"),
        igzo_structure,
    )

    assert "&XC_FUNCTIONAL PBE" in text

    parsed = parse_cp2k_input_text(text)

    assert parsed.xc_functional == "PBE"


def test_generates_pbesol(
    igzo_structure,
):
    """PBEsol should use explicit exchange and correlation sections."""

    text = render_cp2k_input(
        make_config(functional="PBEsol"),
        igzo_structure,
    )

    assert "&GGA_X_PBE_SOL" in text
    assert "&GGA_C_PBE_SOL" in text

    parsed = parse_cp2k_input_text(text)

    assert parsed.xc_functional == "PBEsol"


def test_generates_external_xyz_topology(
    igzo_structure,
):
    """Generated input should reference the configured external XYZ."""

    text = render_cp2k_input(
        make_config(
            coordinate_file="ordered_igzo.xyz",
        ),
        igzo_structure,
    )

    assert "&TOPOLOGY" in text
    assert (
        "COORD_FILE_NAME ordered_igzo.xyz"
        in text
    )
    assert "COORD_FILE_FORMAT XYZ" in text


def test_external_xyz_requires_coordinate_file(
    igzo_structure,
):
    """External XYZ mode should require a coordinate filename."""

    config = make_config(
        coordinate_file="igzo.xyz",
    ).model_copy(
        update={
            "coordinate_file": None,
        }
    )

    with pytest.raises(
        CP2KGenerationError,
        match="requires coordinate_file",
    ):
        render_cp2k_input(
            config,
            igzo_structure,
        )


def test_generates_cell_from_structure(
    igzo_structure,
):
    """CP2K cell vectors should come from the pymatgen structure."""

    text = render_cp2k_input(
        make_config(),
        igzo_structure,
    )

    assert (
        "A 3.299000000000 "
        "0.000000000000 "
        "0.000000000000"
        in text
    )

    assert (
        "B -1.649500000000 "
        "2.857017807085 "
        "0.000000000000"
        in text
    )

    assert (
        "C 0.000000000000 "
        "0.000000000000 "
        "26.101000000000"
        in text
    )

    assert "PERIODIC XYZ" in text


def test_no_kpoints_by_default(
    igzo_structure,
):
    """Cutoff-convergence inputs should permit no explicit k-points."""

    text = render_cp2k_input(
        make_config(),
        igzo_structure,
    )

    assert "&KPOINTS" not in text

    parsed = parse_cp2k_input_text(text)

    assert parsed.k_points is None


def test_generates_explicit_kpoints(
    igzo_structure,
):
    """Configured Monkhorst-Pack k-points should be rendered."""

    text = render_cp2k_input(
        make_config(
            k_points=(6, 6, 1),
        ),
        igzo_structure,
    )

    assert "&KPOINTS" in text
    assert (
        "SCHEME MONKHORST-PACK 6 6 1"
        in text
    )

    parsed = parse_cp2k_input_text(text)

    assert parsed.k_points == (6, 6, 1)


def test_generates_verified_igzo_kinds(
    igzo_structure,
):
    """The verified IGZO UZH/TZV2P assignments should be preserved."""

    text = render_cp2k_input(
        make_config(),
        igzo_structure,
    )

    parsed = parse_cp2k_input_text(text)

    kinds = {
        kind.element: kind
        for kind in parsed.kinds
    }

    assert set(kinds) == {
        "In",
        "Ga",
        "Zn",
        "O",
    }

    assert (
        kinds["In"].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q13"
    )
    assert kinds["In"].potential == "GTH-PBE-q13"

    assert (
        kinds["Ga"].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q13"
    )
    assert kinds["Ga"].potential == "GTH-PBE-q13"

    assert (
        kinds["Zn"].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q12"
    )
    assert kinds["Zn"].potential == "GTH-PBE-q12"

    assert (
        kinds["O"].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q6"
    )
    assert kinds["O"].potential == "GTH-PBE-q6"


@pytest.mark.parametrize(
    ("functional", "expected"),
    [
        ("PBE", "PBE"),
        ("PBEsol", "PBEsol"),
    ],
)
def test_generation_parser_round_trip(
    igzo_structure,
    functional,
    expected,
):
    """Generated scientific settings should survive parser round trip."""

    config = make_config(
        functional=functional,
    )

    text = render_cp2k_input(
        config,
        igzo_structure,
    )

    parsed = parse_cp2k_input_text(text)

    assert parsed.project_name == "igzo_test"
    assert parsed.run_type.value == "ENERGY_FORCE"

    assert parsed.xc_functional == expected

    assert parsed.charge == 0
    assert parsed.multiplicity == 1

    assert parsed.cutoff_ry == pytest.approx(
        600.0
    )
    assert parsed.relative_cutoff_ry == pytest.approx(
        60.0
    )

    assert parsed.eps_scf == pytest.approx(
        1.0e-6
    )

    assert parsed.k_points is None

    assert (
        parsed.basis_set_file
        == "BASIS_MOLOPT_UZH"
    )
    assert (
        parsed.potential_file
        == "POTENTIAL_UZH"
    )

def test_rejects_missing_kind_definition(
    igzo_structure,
):
    """Generation should fail when a structure species has no KIND."""

    config = make_config()

    basis_potential = config.basis_potential

    assert basis_potential is not None

    incomplete_basis = basis_potential.model_copy(
        update={
            "kinds": tuple(
                kind
                for kind in basis_potential.kinds
                if kind.element != "O"
            ),
        }
    )

    invalid_config = config.model_copy(
        update={
            "basis_potential": incomplete_basis,
        }
    )

    with pytest.raises(
        CP2KGenerationError,
        match=(
            "Missing CP2K KIND definitions "
            "for structure elements: O"
        ),
    ):
        render_cp2k_input(
            invalid_config,
            igzo_structure,
        )


def test_rejects_extra_kind_definition(
    igzo_structure,
):
    """Generation should fail when the preset contains an unused KIND."""

    config = make_config()

    basis_potential = config.basis_potential

    assert basis_potential is not None

    extra_kind = (
        basis_potential.kinds[0].model_copy(
            update={
                "element": "Si",
            }
        )
    )

    invalid_basis = basis_potential.model_copy(
        update={
            "kinds": (
                *basis_potential.kinds,
                extra_kind,
            ),
        }
    )

    invalid_config = config.model_copy(
        update={
            "basis_potential": invalid_basis,
        }
    )

    with pytest.raises(
        CP2KGenerationError,
        match=(
            "CP2K KIND definitions contain elements "
            "not present in the structure: Si"
        ),
    ):
        render_cp2k_input(
            invalid_config,
            igzo_structure,
        )