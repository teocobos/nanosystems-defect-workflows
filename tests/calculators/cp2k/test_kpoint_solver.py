import pytest
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KInputConfig,
    CP2KKindConfig,
    CP2KSCFConfig,
)
from nsdw.calculators.cp2k.generator import (
    CP2KGenerationError,
    render_cp2k_input,
)


def make_input(*, k_points=None, solver="OT"):
    structure = Structure(
        Lattice.cubic(5.0),
        ["Si"],
        [[0, 0, 0]],
    )
    config = CP2KInputConfig(
        project_name="solver-test",
        coordinate_file="structure.xyz",
        k_points=k_points,
        scf=CP2KSCFConfig(solver=solver),
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
    return config, structure


def test_default_solver_preserves_ot():
    config, structure = make_input()
    text = render_cp2k_input(config, structure)
    assert "&OT" in text
    assert "&DIAGONALIZATION" not in text


def test_explicit_kpoints_with_diagonalization():
    config, structure = make_input(
        k_points=(2, 2, 2),
        solver="DIAGONALIZATION",
    )
    text = render_cp2k_input(config, structure)
    assert "SCHEME MONKHORST-PACK 2 2 2" in text
    assert "&DIAGONALIZATION" in text
    assert "&OT" not in text


def test_explicit_kpoints_reject_ot():
    config, structure = make_input(
        k_points=(2, 2, 2),
    )
    with pytest.raises(
        CP2KGenerationError,
        match="requires.*DIAGONALIZATION",
    ):
        render_cp2k_input(config, structure)


def test_gamma_mesh_explicitly_rejects_ot():
    config, structure = make_input(
        k_points=(1, 1, 1),
    )
    with pytest.raises(CP2KGenerationError):
        render_cp2k_input(config, structure)


def test_invalid_solver_rejected():
    with pytest.raises(ValidationError):
        CP2KSCFConfig(solver="UNKNOWN")
