"""Tests for verified CP2K basis-set and pseudopotential presets."""

from __future__ import annotations

from nsdw.calculators.cp2k.presets import (
    get_basis_potential_preset,
)


def test_sio2_pbe_dzvp_preset():
    """SiO2 smoke-test preset should use verified CP2K PBE data."""

    preset = get_basis_potential_preset(
        "sio2-pbe-dzvp"
    )

    assert preset.basis_set_file == "BASIS_MOLOPT"
    assert preset.potential_file == "GTH_POTENTIALS"

    kinds = {
        kind.element: kind
        for kind in preset.kinds
    }

    assert set(kinds) == {"Si", "O"}

    assert (
        kinds["Si"].basis_set
        == "DZVP-MOLOPT-GTH-q4"
    )
    assert kinds["Si"].potential == "GTH-PBE-q4"

    assert (
        kinds["O"].basis_set
        == "DZVP-MOLOPT-GTH-q6"
    )
    assert kinds["O"].potential == "GTH-PBE-q6"
