"""Regression tests for automatic k-point campaign provenance."""

import json
from pathlib import Path

import pytest

from nsdw.calculators.cp2k.generation_models import CP2KInputConfig
from nsdw.calculators.cp2k.presets import get_basis_potential_preset
from nsdw.structures.parser import load_structure
from nsdw.workflows.convergence.cp2k_generation import (
    CP2KConvergenceGenerationError,
    generate_cp2k_convergence_study,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)


@pytest.fixture
def campaign_inputs():
    structure, _ = load_structure(
        Path("tests/data/sio2/alpha_quartz_cod1526860.cif")
    )

    config = CP2KInputConfig(
        project_name="provenance-test",
        run_type="ENERGY",
        basis_potential=get_basis_potential_preset("sio2-pbe-dzvp"),
        scf={"solver": "DIAGONALIZATION"},
    )

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.KPOINTS,
        candidates=[
            ConvergenceCandidate(
                label="1x1x1",
                order=0,
                value=(1, 1, 1),
            ),
            ConvergenceCandidate(
                label="2x2x2",
                order=1,
                value=(2, 2, 2),
            ),
        ],
        criterion={"energy_tolerance_ev_per_atom": 0.001},
    )

    return structure, config, study


def generate(tmp_path, campaign_inputs, provenance):
    structure, config, study = campaign_inputs
    destination = tmp_path / "campaign"

    generate_cp2k_convergence_study(
        structure=structure,
        base_config=config,
        study=study,
        output_directory=destination,
        kpoint_planner_provenance=provenance,
    )

    return destination


def valid_provenance():
    return {
        "schema_version": 1,
        "mode": "automatic",
        "meshes": [[1, 1, 1], [2, 2, 2]],
        "energy_tolerance_ev_per_atom": 0.001,
    }


def test_provenance_is_published_with_manifest(
    tmp_path, campaign_inputs
):
    destination = generate(
        tmp_path,
        campaign_inputs,
        valid_provenance(),
    )

    record = json.loads(
        (destination / "kpoint-planner.json").read_text()
    )

    assert record["meshes"] == [[1, 1, 1], [2, 2, 2]]
    assert (destination / "manifest.json").is_file()


@pytest.mark.parametrize(
    "provenance, message",
    [
        (
            {"meshes": [[1, 1, 1], [3, 3, 3]]},
            "does not match candidates",
        ),
        (
            {"meshes": [[1, 1, 1], [2, 2, 2]], "bad": float("nan")},
            "not valid finite JSON",
        ),
        (
            {"meshes": [[1, 1, 1], [2, 2, 2]], "bad": object()},
            "not valid finite JSON",
        ),
        (
            "invalid",
            "must be a dictionary",
        ),
    ],
)
def test_invalid_provenance_writes_nothing(
    tmp_path, campaign_inputs, provenance, message
):
    destination = tmp_path / "campaign"

    with pytest.raises(
        CP2KConvergenceGenerationError,
        match=message,
    ):
        generate(
            tmp_path,
            campaign_inputs,
            provenance,
        )

    assert not destination.exists()
