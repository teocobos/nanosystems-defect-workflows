"""Tests for CP2K AiiDA input adaptation."""

from __future__ import annotations

import pytest

from pymatgen.core import (
    Lattice,
    Structure,
)

from nsdw.calculators.cp2k.aiida import (
    CP2KAiiDAAdapterError,
    build_aiida_cp2k_parameters,
)
from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KInputConfig,
    CP2KKindConfig,
    CP2KSCFConfig,
)


def _structure() -> Structure:
    return Structure(
        lattice=Lattice.cubic(5.0),
        species=["Si", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
    )


def _config(
    *,
    functional: str = "PBE",
    k_points: tuple[int, int, int] | None = None,
) -> CP2KInputConfig:
    return CP2KInputConfig(
        project_name="sio2-production",
        run_type="ENERGY_FORCE",
        charge=1,
        multiplicity=2,
        functional=functional,
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        k_points=k_points,
        scf=CP2KSCFConfig(
            scf_guess="ATOMIC",
            eps_scf=1.0e-7,
            max_scf=150,
            outer_scf_max=12,
            ot_minimizer="CG",
            ot_preconditioner="FULL_SINGLE_INVERSE",
            energy_gap=0.002,
        ),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(
                CP2KKindConfig(
                    element="Si",
                    basis_set="DZVP-MOLOPT-GTH-q4",
                    potential="GTH-PBE-q4",
                ),
                CP2KKindConfig(
                    element="O",
                    basis_set="DZVP-MOLOPT-GTH-q6",
                    potential="GTH-PBE-q6",
                ),
            ),
        ),
    )


def test_build_aiida_cp2k_parameters_preserves_science() -> None:
    parameters = build_aiida_cp2k_parameters(
        config=_config(),
        structure=_structure(),
    )

    assert parameters["GLOBAL"] == {
        "RUN_TYPE": "ENERGY_FORCE",
        "PRINT_LEVEL": "MEDIUM",
    }

    dft = parameters["FORCE_EVAL"]["DFT"]

    assert dft["BASIS_SET_FILE_NAME"] == "BASIS_MOLOPT"
    assert dft["POTENTIAL_FILE_NAME"] == "GTH_POTENTIALS"
    assert dft["CHARGE"] == 1
    assert dft["MULTIPLICITY"] == 2
    assert dft["QS"] == {
        "METHOD": "GPW",
    }

    assert dft["MGRID"] == {
        "CUTOFF": 560.0,
        "REL_CUTOFF": 40.0,
    }

    assert dft["SCF"] == {
        "SCF_GUESS": "ATOMIC",
        "EPS_SCF": 1.0e-7,
        "MAX_SCF": 150,
        "OT": {
            "MINIMIZER": "CG",
            "PRECONDITIONER": "FULL_SINGLE_INVERSE",
            "ENERGY_GAP": 0.002,
        },
        "OUTER_SCF": {
            "MAX_SCF": 12,
        },
    }

    assert dft["XC"] == {
        "XC_FUNCTIONAL": {
            "_": "PBE",
        },
    }


def test_aiida_parameters_omit_project_and_coordinates() -> None:
    parameters = build_aiida_cp2k_parameters(
        config=_config(),
        structure=_structure(),
    )

    assert "PROJECT" not in parameters["GLOBAL"]
    assert "PROJECT_NAME" not in parameters["GLOBAL"]

    subsys = parameters["FORCE_EVAL"]["SUBSYS"]

    assert "COORD" not in subsys
    assert "TOPOLOGY" not in subsys


def test_aiida_parameters_include_cell_and_kinds() -> None:
    parameters = build_aiida_cp2k_parameters(
        config=_config(),
        structure=_structure(),
    )

    subsys = parameters["FORCE_EVAL"]["SUBSYS"]

    assert subsys["CELL"] == {
        "A": "5.000000000000 0.000000000000 0.000000000000",
        "B": "0.000000000000 5.000000000000 0.000000000000",
        "C": "0.000000000000 0.000000000000 5.000000000000",
        "PERIODIC": "XYZ",
    }

    assert subsys["KIND"] == [
        {
            "_": "Si",
            "ELEMENT": "Si",
            "BASIS_SET": "DZVP-MOLOPT-GTH-q4",
            "POTENTIAL": "GTH-PBE-q4",
        },
        {
            "_": "O",
            "ELEMENT": "O",
            "BASIS_SET": "DZVP-MOLOPT-GTH-q6",
            "POTENTIAL": "GTH-PBE-q6",
        },
    ]


def test_aiida_parameters_include_kpoints() -> None:
    parameters = build_aiida_cp2k_parameters(
        config=_config(
            k_points=(6, 6, 1),
        ),
        structure=_structure(),
    )

    assert (
        parameters["FORCE_EVAL"]["DFT"]["KPOINTS"]
        == {
            "SCHEME": "MONKHORST-PACK 6 6 1",
        }
    )


def test_aiida_parameters_support_pbesol() -> None:
    parameters = build_aiida_cp2k_parameters(
        config=_config(
            functional="PBEsol",
        ),
        structure=_structure(),
    )

    assert (
        parameters["FORCE_EVAL"]["DFT"]["XC"]
        == {
            "XC_FUNCTIONAL": {
                "GGA_X_PBE_SOL": {},
                "GGA_C_PBE_SOL": {},
            },
        }
    )


def test_aiida_parameters_require_basis_configuration() -> None:
    config = _config().model_copy(
        update={
            "basis_potential": None,
        }
    )

    with pytest.raises(
        CP2KAiiDAAdapterError,
        match="basis/potential configuration",
    ):
        build_aiida_cp2k_parameters(
            config=config,
            structure=_structure(),
        )


def test_aiida_parameters_require_kind_coverage() -> None:
    config = _config().model_copy(
        update={
            "basis_potential": CP2KBasisPotentialConfig(
                basis_set_file="BASIS_MOLOPT",
                potential_file="GTH_POTENTIALS",
                kinds=(
                    CP2KKindConfig(
                        element="O",
                        basis_set="DZVP-MOLOPT-GTH-q6",
                        potential="GTH-PBE-q6",
                    ),
                ),
            ),
        }
    )

    with pytest.raises(
        CP2KAiiDAAdapterError,
        match="Missing CP2K KIND definitions",
    ):
        build_aiida_cp2k_parameters(
            config=config,
            structure=_structure(),
        )
