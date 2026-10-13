from nsdw.calculators.cp2k.presets import (
    get_basis_potential_preset,
)


def test_sio2_tzv2px_basis_preset():
    config = get_basis_potential_preset("sio2-pbe-tzv2px")

    assert config.basis_set_file == "BASIS_MOLOPT"
    assert config.potential_file == "GTH_POTENTIALS"

    kinds = {kind.element: kind for kind in config.kinds}

    assert set(kinds) == {"Si", "O"}

    assert kinds["Si"].basis_set == "TZV2PX-MOLOPT-GTH-q4"
    assert kinds["Si"].potential == "GTH-PBE-q4"

    assert kinds["O"].basis_set == "TZV2PX-MOLOPT-GTH-q6"
    assert kinds["O"].potential == "GTH-PBE-q6"
