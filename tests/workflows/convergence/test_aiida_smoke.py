"""Tests for the LUMI-C SiO2 AiiDA smoke-test configuration."""

from pathlib import Path

from nsdw.workflows.convergence.aiida_smoke import (
    CP2K_DATA,
    build_smoke_test_inputs,
)


STRUCTURE = Path("tests/data/sio2/alpha_quartz_cod1526860.cif")


def test_smoke_test_scientific_configuration():
    structure, parameters = build_smoke_test_inputs(STRUCTURE)

    assert len(structure) == 9
    assert structure.composition.reduced_formula == "SiO2"

    dft = parameters["FORCE_EVAL"]["DFT"]

    assert dft["MGRID"]["CUTOFF"] == 400.0
    assert dft["MGRID"]["REL_CUTOFF"] == 30.0
    assert dft["SCF"]["EPS_SCF"] == 1.0e-6
    assert "OT" in dft["SCF"]
    assert "KPOINTS" not in dft

    assert dft["BASIS_SET_FILE_NAME"] == f"{CP2K_DATA}/BASIS_MOLOPT"
    assert dft["POTENTIAL_FILE_NAME"] == f"{CP2K_DATA}/GTH_POTENTIALS"

    kinds = parameters["FORCE_EVAL"]["SUBSYS"]["KIND"]
    assert {kind["ELEMENT"] for kind in kinds} == {"Si", "O"}
    assert all("TZV2P-MOLOPT-GTH" in kind["BASIS_SET"] for kind in kinds)
