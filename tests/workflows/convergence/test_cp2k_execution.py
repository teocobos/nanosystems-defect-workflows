from pathlib import Path
from unittest.mock import patch

import pytest
from pymatgen.core import Lattice, Structure
import shutil

from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.manifest import (
    ConvergenceManifestCandidate,
    ConvergenceStudyManifest,
    write_convergence_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCriterion,
    ConvergenceParameter,
)


def _manifest() -> ConvergenceStudyManifest:
    return ConvergenceStudyManifest(
        calculator="cp2k",
        parameter=ConvergenceParameter.CUTOFF,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=1.0e-3,
        ),
        candidates=(
            ConvergenceManifestCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(
                    value=400.0,
                    unit="Ry",
                ),
                directory="400-Ry",
                input_file="test-400-Ry.inp",
                coordinate_file="test-400-Ry.xyz",
            ),
            ConvergenceManifestCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(
                    value=600.0,
                    unit="Ry",
                ),
                directory="600-Ry",
                input_file="test-600-Ry.inp",
                coordinate_file="test-600-Ry.xyz",
            ),
        ),
    )


def test_run_cp2k_convergence_campaign_executes_candidates_in_order(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_execution import (
        run_cp2k_convergence_campaign,
    )

    campaign_directory = tmp_path / "study"

    (campaign_directory / "400-Ry").mkdir(
        parents=True
    )
    (campaign_directory / "600-Ry").mkdir()

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["In", "Ga", "Zn", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ],
    )

    structure.to(
        filename=campaign_directory / "structure.json",
        fmt="json",
    )

    write_convergence_manifest(
        manifest=_manifest(),
        path=campaign_directory / "manifest.json",
    )

    with patch(
        "nsdw.workflows.convergence.cp2k_execution."
        "run_cp2k_single_point"
    ) as run_single_point:
        run_cp2k_convergence_campaign(
            campaign_directory=campaign_directory,
        )

    assert run_single_point.call_count == 2

    first_call = run_single_point.call_args_list[0]
    second_call = run_single_point.call_args_list[1]

    assert first_call.kwargs["calculation_id"] == "400-Ry"

    assert (
        first_call.kwargs["working_directory"]
        == campaign_directory / "400-Ry"
    )

    assert (
        first_call.kwargs["input_file"]
        == Path("test-400-Ry.inp")
    )

    assert (
        first_call.kwargs["output_file"]
        == Path("test-400-Ry.out")
    )

    first_structure = first_call.kwargs["structure"]

    assert isinstance(
        first_structure,
        Structure,
    )

    assert len(first_structure) == 4

    assert (
        first_structure.composition.reduced_formula
        == structure.composition.reduced_formula
    )

    assert first_structure == structure

    assert second_call.kwargs["calculation_id"] == "600-Ry"

    assert (
        second_call.kwargs["working_directory"]
        == campaign_directory / "600-Ry"
    )

    assert (
        second_call.kwargs["input_file"]
        == Path("test-600-Ry.inp")
    )

    assert (
        second_call.kwargs["output_file"]
        == Path("test-600-Ry.out")
    )

    second_structure = second_call.kwargs["structure"]

    assert isinstance(
        second_structure,
        Structure,
    )

    assert len(second_structure) == 4

    assert (
        second_structure.composition.reduced_formula
        == structure.composition.reduced_formula
    )

    assert second_structure == structure


def test_run_cp2k_convergence_campaign_rejects_non_cp2k_manifest(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_execution import (
        CP2KConvergenceExecutionError,
        run_cp2k_convergence_campaign,
    )

    campaign_directory = tmp_path / "study"
    campaign_directory.mkdir()

    manifest = _manifest().model_copy(
        update={"calculator": "vasp"}
    )

    write_convergence_manifest(
        manifest=manifest,
        path=campaign_directory / "manifest.json",
    )

    with pytest.raises(
        CP2KConvergenceExecutionError,
        match="requires a CP2K manifest",
    ):
        run_cp2k_convergence_campaign(
            campaign_directory=campaign_directory,
        )


def test_collect_existing_cp2k_convergence_results(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_execution import (
        collect_existing_cp2k_convergence_results,
    )

    campaign_directory = tmp_path / "study"

    candidate_directory = (
        campaign_directory / "400-Ry"
    )
    candidate_directory.mkdir(parents=True)

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["In", "Ga", "Zn", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
        ],
    )

    structure.to(
        filename=campaign_directory / "structure.json",
        fmt="json",
    )

    manifest = _manifest().model_copy(
        update={
            "candidates": (
                _manifest().candidates[0],
            )
        }
    )

    write_convergence_manifest(
        manifest=manifest,
        path=campaign_directory / "manifest.json",
    )

    fixture_directory = (
        Path("tests/calculators/cp2k/fixtures")
        .resolve()
    )

    shutil.copy(
        fixture_directory / "igzo_ordered_003_sp.inp",
        candidate_directory / "test-400-Ry.inp",
    )

    shutil.copy(
        fixture_directory / "igzo_ordered_003_sp_real.out",
        candidate_directory / "calculation.out",
    )

    results = collect_existing_cp2k_convergence_results(
        campaign_directory=campaign_directory,
    )

    assert len(results) == 1

    result = results[0]

    assert result.calculation.id == "400-Ry"
    assert result.calculation.status.value == "completed"

    assert result.structure is not None
    assert result.structure.n_atoms == 4

    assert result.energy is not None
    assert result.energy.total is not None
    assert result.energy.total.unit == "eV"