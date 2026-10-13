"""Validation of basis-selection provenance."""

import pytest

from nsdw.calculators.cp2k.generation_models import CP2KInputConfig
from nsdw.calculators.cp2k.presets import get_basis_potential_preset
from nsdw.workflows.convergence.cp2k_generation import (
    CP2KConvergenceGenerationError,
    generate_cp2k_convergence_study,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceCriterion,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)


@pytest.fixture
def basis_study():
    return ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.BASIS,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=0.001
        ),
        candidates=[
            ConvergenceCandidate(
                label="DZVP", order=0, value="DZVP"
            ),
            ConvergenceCandidate(
                label="TZVP", order=1, value="TZVP"
            ),
        ],
    )


@pytest.mark.parametrize(
    "provenance",
    [
        {"schema_version": 2, "strategy": "standard",
         "candidates": [
             {"label": "DZVP", "preset": "sio2-pbe-dzvp"},
             {"label": "TZVP", "preset": "sio2-pbe-tzvp"},
         ]},
        {"schema_version": 1, "strategy": "unknown",
         "candidates": [
             {"label": "DZVP", "preset": "sio2-pbe-dzvp"},
             {"label": "TZVP", "preset": "sio2-pbe-tzvp"},
         ]},
        {"schema_version": 1, "strategy": "standard",
         "candidates": [
             {"label": "TZVP", "preset": "sio2-pbe-tzvp"},
             {"label": "DZVP", "preset": "sio2-pbe-dzvp"},
         ]},
        {"schema_version": 1, "strategy": "custom",
         "candidates": [
             {"label": "DZVP", "preset": "sio2-pbe-dzvp"},
         ]},
        {"schema_version": 1, "strategy": "custom",
         "candidates": [
             {"label": "DZVP", "preset": ""},
             {"label": "TZVP", "preset": "sio2-pbe-tzvp"},
         ]},
        {"schema_version": 1, "strategy": "custom",
         "candidates": [
             {"label": "DZVP", "preset": "sio2-pbe-tzvp"},
             {"label": "TZVP", "preset": "sio2-pbe-dzvp"},
         ]},
    ],
)
def test_invalid_basis_provenance_fails_before_creating_files(
    tmp_path, basis_study, provenance
):
    from pymatgen.core import Structure

    structure = Structure(
        lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        species=["Si", "O", "O"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75]],
    )

    output = tmp_path / "campaign"

    with pytest.raises(CP2KConvergenceGenerationError):
        generate_cp2k_convergence_study(
            structure=structure,
            base_config=CP2KInputConfig(
                project_name="sio2-basis-provenance-test",
                basis_potential=get_basis_potential_preset(
                    "sio2-pbe-dzvp"
                )
            ),
            study=basis_study,
            output_directory=output,
            basis_candidates={
                "DZVP": get_basis_potential_preset(
                    "sio2-pbe-dzvp"
                ),
                "TZVP": get_basis_potential_preset(
                    "sio2-pbe-tzvp"
                ),
            },
            basis_selection_provenance=provenance,
        )

    assert not output.exists()
