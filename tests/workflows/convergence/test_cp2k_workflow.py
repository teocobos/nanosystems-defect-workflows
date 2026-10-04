from pathlib import Path
from unittest.mock import patch

from nsdw.models.calculation import (
    Backend,
    CalculationMetadata,
    CalculationStatus,
    CalculationType,
)
from nsdw.models.quantity import Quantity
from nsdw.models.result import (
    EnergyResult,
    NSDWResult,
)
from nsdw.models.structure import StructureResult
from nsdw.workflows.convergence.manifest import (
    ConvergenceManifestCandidate,
    ConvergenceStudyManifest,
    write_convergence_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCriterion,
    ConvergenceParameter,
)
from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KInputConfig,
    CP2KSCFConfig,
)
from nsdw.workflows.convergence.reporting import (
    ConvergenceReport,
)
import pytest

from nsdw.workflows.convergence.recipes import (
    ConvergenceRecipeError,
)
from nsdw.project.models import ProjectConfig
from nsdw.project.scaffold import create_project
from nsdw.project.workspace import load_project_workspace


def _result(
    *,
    calculation_id: str,
    energy_ev: float,
) -> NSDWResult:
    return NSDWResult(
        calculation=CalculationMetadata(
            id=calculation_id,
            type=CalculationType.SINGLE_POINT,
            status=CalculationStatus.COMPLETED,
            backend=Backend.CP2K,
        ),
        structure=StructureResult(
            formula="SiO2",
            n_atoms=9,
            periodic=True,
            structure_hash="a" * 64,
        ),
        energy=EnergyResult(
            total=Quantity(
                value=energy_ev,
                unit="eV",
            ),
        ),
    )


def test_run_and_analyse_cp2k_convergence_campaign(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        run_and_analyse_cp2k_convergence_campaign,
    )

    campaign_directory = tmp_path / "study"

    for label in (
        "400-Ry",
        "600-Ry",
        "800-Ry",
    ):
        (campaign_directory / label).mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = ConvergenceStudyManifest(
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
            ConvergenceManifestCandidate(
                label="800-Ry",
                order=2,
                value=Quantity(
                    value=800.0,
                    unit="Ry",
                ),
                directory="800-Ry",
                input_file="test-800-Ry.inp",
                coordinate_file="test-800-Ry.xyz",
            ),
        ),
    )

    write_convergence_manifest(
        manifest=manifest,
        path=campaign_directory / "manifest.json",
    )

    results = (
        _result(
            calculation_id="400-Ry",
            energy_ev=-90.000,
        ),
        _result(
            calculation_id="600-Ry",
            energy_ev=-90.045,
        ),
        _result(
            calculation_id="800-Ry",
            energy_ev=-90.0495,
        ),
    )

    with patch(
        "nsdw.workflows.convergence.cp2k_workflow."
        "run_cp2k_convergence_campaign",
        return_value=results,
    ):
        analysis = (
            run_and_analyse_cp2k_convergence_campaign(
                campaign_directory=campaign_directory,
            )
        )

    assert analysis.energy_tolerance_satisfied
    assert analysis.selected_candidate_label == "600-Ry"

def test_run_and_report_cp2k_convergence_campaign(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        run_and_report_cp2k_convergence_campaign,
    )

    campaign_directory = tmp_path / "study"

    for label in (
        "400-Ry",
        "600-Ry",
        "800-Ry",
    ):
        (campaign_directory / label).mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = ConvergenceStudyManifest(
        calculator="cp2k",
        parameter=ConvergenceParameter.CUTOFF,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=1.0e-3,
        ),
        candidates=(
            ConvergenceManifestCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
                directory="400-Ry",
                input_file="test-400-Ry.inp",
                coordinate_file="test-400-Ry.xyz",
            ),
            ConvergenceManifestCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
                directory="600-Ry",
                input_file="test-600-Ry.inp",
                coordinate_file="test-600-Ry.xyz",
            ),
            ConvergenceManifestCandidate(
                label="800-Ry",
                order=2,
                value=Quantity(value=800.0, unit="Ry"),
                directory="800-Ry",
                input_file="test-800-Ry.inp",
                coordinate_file="test-800-Ry.xyz",
            ),
        ),
    )

    write_convergence_manifest(
        manifest=manifest,
        path=campaign_directory / "manifest.json",
    )

    results = (
        _result(
            calculation_id="400-Ry",
            energy_ev=-90.000,
        ),
        _result(
            calculation_id="600-Ry",
            energy_ev=-90.045,
        ),
        _result(
            calculation_id="800-Ry",
            energy_ev=-90.0495,
        ),
    )

    with patch(
        "nsdw.workflows.convergence.cp2k_workflow."
        "run_cp2k_convergence_campaign",
        return_value=results,
    ):
        report = run_and_report_cp2k_convergence_campaign(
            campaign_directory=campaign_directory,
        )

    assert report.selected_candidate_label == "600-Ry"
    assert report.energy_tolerance_satisfied

    assert (
        campaign_directory / "convergence-report.json"
    ).is_file()

    assert (
        campaign_directory / "convergence-report.csv"
    ).is_file()


def test_analyse_and_report_existing_cp2k_convergence_campaign(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        analyse_and_report_existing_cp2k_convergence_campaign,
    )

    campaign_directory = tmp_path / "study"
    campaign_directory.mkdir()

    manifest = ConvergenceStudyManifest(
        calculator="cp2k",
        parameter=ConvergenceParameter.CUTOFF,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=1.0e-3,
        ),
        candidates=(
            ConvergenceManifestCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
                directory="400-Ry",
                input_file="test-400-Ry.inp",
                coordinate_file="test-400-Ry.xyz",
            ),
            ConvergenceManifestCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
                directory="600-Ry",
                input_file="test-600-Ry.inp",
                coordinate_file="test-600-Ry.xyz",
            ),
            ConvergenceManifestCandidate(
                label="800-Ry",
                order=2,
                value=Quantity(value=800.0, unit="Ry"),
                directory="800-Ry",
                input_file="test-800-Ry.inp",
                coordinate_file="test-800-Ry.xyz",
            ),
        ),
    )

    write_convergence_manifest(
        manifest=manifest,
        path=campaign_directory / "manifest.json",
    )

    results = (
        _result(
            calculation_id="400-Ry",
            energy_ev=-90.000,
        ),
        _result(
            calculation_id="600-Ry",
            energy_ev=-90.045,
        ),
        _result(
            calculation_id="800-Ry",
            energy_ev=-90.0495,
        ),
    )

    with patch(
        "nsdw.workflows.convergence.cp2k_workflow."
        "collect_existing_cp2k_convergence_results",
        return_value=results,
    ) as collect_existing:
        report = (
            analyse_and_report_existing_cp2k_convergence_campaign(
                campaign_directory=campaign_directory,
            )
        )

    collect_existing.assert_called_once_with(
        campaign_directory=campaign_directory.resolve(),
    )

    assert report.selected_candidate_label == "600-Ry"
    assert report.energy_tolerance_satisfied

    assert (
        campaign_directory / "convergence-report.json"
    ).is_file()

    assert (
        campaign_directory / "convergence-report.csv"
    ).is_file()


def _report(
    *,
    parameter: ConvergenceParameter,
    selected_candidate_label: str,
) -> ConvergenceReport:
    return ConvergenceReport(
        parameter=parameter,
        energy_tolerance_ev_per_atom=1.0e-3,
        structure_hash="a" * 64,
        n_atoms=9,
        reference_candidate_label="reference",
        selected_candidate_label=selected_candidate_label,
        energy_tolerance_satisfied=True,
        candidates=(),
    )


def test_run_standard_cp2k_convergence_recipe_chains_stages(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        run_standard_cp2k_convergence_recipe,
    )

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["O"],
        coords=[[0.0, 0.0, 0.0]],
    )

    base_config = CP2KInputConfig(
        project_name="test",
        run_type="ENERGY",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
    )

    cutoff_report = _report(
        parameter=ConvergenceParameter.CUTOFF,
        selected_candidate_label="560-Ry",
    )

    relative_cutoff_report = _report(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        selected_candidate_label="40-Ry",
    )

    execution_environment = {
    "CP2K_DATA_DIR": "/fake/cp2k/data",
    }

    workflow_directory = tmp_path / "convergence"

    with (
        patch(
            "nsdw.workflows.convergence.cp2k_workflow."
            "generate_cp2k_convergence_study",
        ) as generate,
        patch(
            "nsdw.workflows.convergence.cp2k_workflow."
            "run_and_report_cp2k_convergence_campaign",
            side_effect=(
                cutoff_report,
                relative_cutoff_report,
            ),
        ) as run_and_report,
        patch(
            "nsdw.workflows.convergence.cp2k_workflow."
            "write_cp2k_methodology_candidate",
        ) as write_candidate,
    ):
        result = run_standard_cp2k_convergence_recipe(
            structure=structure,
            base_config=base_config,
            workflow_directory=workflow_directory,
            executable="custom-cp2k",
            environment=execution_environment,
        )

    assert generate.call_count == 2
    assert run_and_report.call_count == 2

    first_run = run_and_report.call_args_list[0].kwargs

    assert first_run == {
        "campaign_directory": (
            workflow_directory / "cutoff"
        ),
        "executable": "custom-cp2k",
        "environment": execution_environment,
    }

    second_run = run_and_report.call_args_list[1].kwargs

    assert second_run == {
        "campaign_directory": (
            workflow_directory / "relative_cutoff"
        ),
        "executable": "custom-cp2k",
        "environment": execution_environment,
    }

    cutoff_generation = generate.call_args_list[0].kwargs

    assert (
        cutoff_generation["study"].parameter
        == ConvergenceParameter.CUTOFF
    )
    assert len(cutoff_generation["study"].candidates) == 10
    assert (
        cutoff_generation["output_directory"]
        == workflow_directory / "cutoff"
    )

    relative_generation = generate.call_args_list[1].kwargs

    assert (
        relative_generation["study"].parameter
        == ConvergenceParameter.RELATIVE_CUTOFF
    )
    assert len(relative_generation["study"].candidates) == 10
    assert relative_generation["base_config"].cutoff_ry == 560.0
    assert (
        relative_generation["output_directory"]
        == workflow_directory / "relative_cutoff"
    )

    assert result.cutoff_ry == 560.0
    assert result.relative_cutoff_ry == 40.0
    assert result.converged_config.cutoff_ry == 560.0
    assert result.converged_config.relative_cutoff_ry == 40.0

    assert result.cutoff_report is cutoff_report
    assert result.relative_cutoff_report is relative_cutoff_report

    write_candidate.assert_called_once_with(
        result=result,
        workflow_directory=workflow_directory,
    )

def test_standard_cp2k_convergence_recipe_stops_if_cutoff_not_converged(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        run_standard_cp2k_convergence_recipe,
    )

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["O"],
        coords=[[0.0, 0.0, 0.0]],
    )

    base_config = CP2KInputConfig(
        project_name="test",
        run_type="ENERGY",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
    )

    cutoff_report = ConvergenceReport(
        parameter=ConvergenceParameter.CUTOFF,
        energy_tolerance_ev_per_atom=1.0e-3,
        structure_hash="a" * 64,
        n_atoms=1,
        reference_candidate_label="850-Ry",
        selected_candidate_label=None,
        energy_tolerance_satisfied=False,
        candidates=(),
    )

    workflow_directory = tmp_path / "convergence"

    with (
        patch(
            "nsdw.workflows.convergence.cp2k_workflow."
            "generate_cp2k_convergence_study",
        ) as generate,
        patch(
            "nsdw.workflows.convergence.cp2k_workflow."
            "run_and_report_cp2k_convergence_campaign",
            return_value=cutoff_report,
        ) as run_and_report,
    ):
        with pytest.raises(
            ConvergenceRecipeError,
            match="did not select a candidate",
        ):
            run_standard_cp2k_convergence_recipe(
                structure=structure,
                base_config=base_config,
                workflow_directory=workflow_directory,
            )

    # Only the cutoff stage may have been generated/executed.
    assert generate.call_count == 1
    assert run_and_report.call_count == 1


def test_standard_cp2k_convergence_recipe_stops_if_relative_cutoff_not_converged(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        run_standard_cp2k_convergence_recipe,
    )

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["O"],
        coords=[[0.0, 0.0, 0.0]],
    )

    base_config = CP2KInputConfig(
        project_name="test",
        run_type="ENERGY",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
    )

    cutoff_report = _report(
        parameter=ConvergenceParameter.CUTOFF,
        selected_candidate_label="560-Ry",
    )

    relative_cutoff_report = ConvergenceReport(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        energy_tolerance_ev_per_atom=1.0e-3,
        structure_hash="a" * 64,
        n_atoms=1,
        reference_candidate_label="100-Ry",
        selected_candidate_label=None,
        energy_tolerance_satisfied=False,
        candidates=(),
    )

    workflow_directory = tmp_path / "convergence"

    with (
        patch(
            "nsdw.workflows.convergence.cp2k_workflow."
            "generate_cp2k_convergence_study",
        ) as generate,
        patch(
            "nsdw.workflows.convergence.cp2k_workflow."
            "run_and_report_cp2k_convergence_campaign",
            side_effect=(
                cutoff_report,
                relative_cutoff_report,
            ),
        ) as run_and_report,
    ):
        with pytest.raises(
            ConvergenceRecipeError,
            match="did not select a candidate",
        ):
            run_standard_cp2k_convergence_recipe(
                structure=structure,
                base_config=base_config,
                workflow_directory=workflow_directory,
            )

    # Both stages were generated and executed, but no final converged
    # configuration may be returned without relative-cutoff convergence.
    assert generate.call_count == 2
    assert run_and_report.call_count == 2


def test_build_cp2k_production_methodology_from_convergence_result(
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        CP2KStandardConvergenceResult,
        build_cp2k_production_methodology,
    )

    cutoff_report = _report(
        parameter=ConvergenceParameter.CUTOFF,
        selected_candidate_label="560-Ry",
    )

    relative_cutoff_report = _report(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        selected_candidate_label="40-Ry",
    )

    scf = CP2KSCFConfig(
        eps_scf=1.0e-7,
        max_scf=150,
    )

    basis_potential = CP2KBasisPotentialConfig(
        basis_set_file="BASIS_MOLOPT",
        potential_file="GTH_POTENTIALS",
        kinds=(),
    )

    converged_config = CP2KInputConfig(
        project_name="igzo",
        run_type="ENERGY",
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        scf=scf,
        basis_potential=basis_potential,
    )

    result = CP2KStandardConvergenceResult(
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        cutoff_report=cutoff_report,
        relative_cutoff_report=relative_cutoff_report,
        converged_config=converged_config,
    )

    methodology = build_cp2k_production_methodology(
        result=result,
        cutoff_report_path=(
            "workflows/convergence/cutoff/"
            "convergence-report.json"
        ),
        relative_cutoff_report_path=(
            "workflows/convergence/relative_cutoff/"
            "convergence-report.json"
        ),
    )

    assert methodology.status == "validated"
    assert methodology.functional.value == "PBE"

    assert methodology.cutoff_ry == 560.0
    assert methodology.relative_cutoff_ry == 40.0

    assert methodology.scf.eps_scf == 1.0e-7
    assert methodology.scf.max_scf == 150

    assert methodology.basis_potential is not None
    assert (
        methodology.basis_potential.basis_set_file
        == "BASIS_MOLOPT"
    )
    assert (
        methodology.basis_potential.potential_file
        == "GTH_POTENTIALS"
    )

    assert methodology.provenance.source == "convergence"
    assert (
        methodology.provenance.workflow
        == "standard_cp2k_convergence"
    )

    assert (
        methodology.provenance.cutoff_report
        == (
            "workflows/convergence/cutoff/"
            "convergence-report.json"
        )
    )

    assert (
        methodology.provenance.relative_cutoff_report
        == (
            "workflows/convergence/relative_cutoff/"
            "convergence-report.json"
        )
    )


def test_persist_cp2k_convergence_methodology(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        CP2KStandardConvergenceResult,
        persist_cp2k_convergence_methodology,
    )

    project_root = tmp_path / "igzo-project"

    create_project(
        root=project_root,
        config=ProjectConfig(
            name="IGZO defect study",
            material="InGaZnO4",
            nsdw_version="0.1.0",
            components=["cp2k"],
        ),
    )

    workflow_directory = (
        project_root
        / "workflows"
        / "convergence"
    )

    cutoff_directory = (
        workflow_directory
        / "cutoff"
    )

    relative_cutoff_directory = (
        workflow_directory
        / "relative_cutoff"
    )

    cutoff_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_cutoff_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        cutoff_directory
        / "convergence-report.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    (
        relative_cutoff_directory
        / "convergence-report.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    cutoff_report = _report(
        parameter=ConvergenceParameter.CUTOFF,
        selected_candidate_label="560-Ry",
    )

    relative_cutoff_report = _report(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        selected_candidate_label="40-Ry",
    )

    converged_config = CP2KInputConfig(
        project_name="igzo",
        run_type="ENERGY",
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        scf=CP2KSCFConfig(),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(),
        ),
    )

    convergence_result = CP2KStandardConvergenceResult(
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        cutoff_report=cutoff_report,
        relative_cutoff_report=relative_cutoff_report,
        converged_config=converged_config,
    )

    updated_workspace = (
        persist_cp2k_convergence_methodology(
            project_root=project_root,
            result=convergence_result,
            workflow_directory=workflow_directory,
        )
    )

    assert updated_workspace.name == "IGZO defect study"
    assert updated_workspace.material == "InGaZnO4"
    assert updated_workspace.components == ("cp2k",)

    reloaded = load_project_workspace(
        project_root
    )

    methodology = reloaded.config.methodology.cp2k

    assert methodology is not None
    assert methodology.status == "validated"

    assert methodology.cutoff_ry == 560.0
    assert methodology.relative_cutoff_ry == 40.0

    assert methodology.basis_potential is not None
    assert (
        methodology.basis_potential.basis_set_file
        == "BASIS_MOLOPT"
    )
    assert (
        methodology.basis_potential.potential_file
        == "GTH_POTENTIALS"
    )

    assert (
        methodology.provenance.cutoff_report
        == (
            "workflows/convergence/cutoff/"
            "convergence-report.json"
        )
    )

    assert (
        methodology.provenance.relative_cutoff_report
        == (
            "workflows/convergence/relative_cutoff/"
            "convergence-report.json"
        )
    )


def test_write_and_load_cp2k_methodology_candidate(
    tmp_path: Path,
) -> None:
    from nsdw.workflows.convergence.cp2k_workflow import (
        CP2KStandardConvergenceResult,
        load_cp2k_methodology_candidate,
        write_cp2k_methodology_candidate,
    )

    workflow_directory = (
        tmp_path
        / "workflows"
        / "convergence"
    )

    cutoff_report = _report(
        parameter=ConvergenceParameter.CUTOFF,
        selected_candidate_label="560-Ry",
    )

    relative_cutoff_report = _report(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        selected_candidate_label="40-Ry",
    )

    converged_config = CP2KInputConfig(
        project_name="igzo",
        run_type="ENERGY",
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        scf=CP2KSCFConfig(
            eps_scf=1.0e-7,
            max_scf=150,
        ),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(),
        ),
    )

    result = CP2KStandardConvergenceResult(
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        cutoff_report=cutoff_report,
        relative_cutoff_report=relative_cutoff_report,
        converged_config=converged_config,
    )

    candidate_path = (
        write_cp2k_methodology_candidate(
            result=result,
            workflow_directory=workflow_directory,
        )
    )

    assert candidate_path == (
        workflow_directory
        / "methodology-candidate.yaml"
    ).resolve()

    assert candidate_path.is_file()

    methodology = load_cp2k_methodology_candidate(
        candidate_path
    )

    assert methodology.status == "candidate"
    assert methodology.functional.value == "PBE"

    assert methodology.cutoff_ry == 560.0
    assert methodology.relative_cutoff_ry == 40.0

    assert methodology.scf.eps_scf == 1.0e-7
    assert methodology.scf.max_scf == 150

    assert methodology.basis_potential is not None
    assert (
        methodology.basis_potential.basis_set_file
        == "BASIS_MOLOPT"
    )
    assert (
        methodology.basis_potential.potential_file
        == "GTH_POTENTIALS"
    )

    assert (
        methodology.provenance.cutoff_report
        == "cutoff/convergence-report.json"
    )

    assert (
        methodology.provenance.relative_cutoff_report
        == (
            "relative_cutoff/"
            "convergence-report.json"
        )
    )


def test_load_cp2k_methodology_candidate_rejects_validated_status(
    tmp_path: Path,
) -> None:
    candidate_path = (
        tmp_path
        / "methodology-candidate.yaml"
    )

    candidate_path.write_text(
        "\n".join(
            [
                "status: validated",
                "functional: PBE",
                "cutoff_ry: 560.0",
                "relative_cutoff_ry: 40.0",
                "k_points: null",
                "scf:",
                "  scf_guess: ATOMIC",
                "  eps_scf: 1.0e-6",
                "  max_scf: 100",
                "  outer_scf_max: 10",
                "  ot_minimizer: CG",
                "  ot_preconditioner: FULL_SINGLE_INVERSE",
                "  energy_gap: 0.001",
                "basis_potential: null",
                "provenance:",
                "  source: convergence",
                "  workflow: standard_cp2k_convergence",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from nsdw.workflows.convergence.cp2k_workflow import (
        load_cp2k_methodology_candidate,
    )

    with pytest.raises(
        ValueError,
        match="status 'candidate'",
    ):
        load_cp2k_methodology_candidate(
            candidate_path
        )


def test_promote_cp2k_methodology_candidate(
    tmp_path: Path,
) -> None:
    from nsdw.project.models import ProjectConfig
    from nsdw.project.scaffold import create_project
    from nsdw.project.workspace import (
        load_project_workspace,
    )
    from nsdw.workflows.convergence.cp2k_workflow import (
        CP2KStandardConvergenceResult,
        promote_cp2k_methodology_candidate,
        write_cp2k_methodology_candidate,
    )

    project_root = tmp_path / "igzo-project"

    create_project(
        root=project_root,
        config=ProjectConfig(
            name="IGZO defect study",
            material="InGaZnO4",
            nsdw_version="0.1.0",
            components=["cp2k"],
        ),
    )

    workflow_directory = (
        project_root
        / "workflows"
        / "convergence"
    )

    cutoff_directory = (
        workflow_directory / "cutoff"
    )

    relative_cutoff_directory = (
        workflow_directory / "relative_cutoff"
    )

    cutoff_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_cutoff_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        cutoff_directory
        / "convergence-report.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    (
        relative_cutoff_directory
        / "convergence-report.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    result = CP2KStandardConvergenceResult(
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        cutoff_report=_report(
            parameter=ConvergenceParameter.CUTOFF,
            selected_candidate_label="560-Ry",
        ),
        relative_cutoff_report=_report(
            parameter=(
                ConvergenceParameter.RELATIVE_CUTOFF
            ),
            selected_candidate_label="40-Ry",
        ),
        converged_config=CP2KInputConfig(
            project_name="igzo",
            run_type="ENERGY",
            cutoff_ry=560.0,
            relative_cutoff_ry=40.0,
            scf=CP2KSCFConfig(),
            basis_potential=CP2KBasisPotentialConfig(
                basis_set_file="BASIS_MOLOPT",
                potential_file="GTH_POTENTIALS",
                kinds=(),
            ),
        ),
    )

    candidate_path = (
        write_cp2k_methodology_candidate(
            result=result,
            workflow_directory=workflow_directory,
        )
    )

    workspace = promote_cp2k_methodology_candidate(
        project_root=project_root,
        candidate_path=candidate_path,
    )

    methodology = (
        workspace.config.methodology.cp2k
    )

    assert methodology is not None
    assert methodology.status == "validated"

    assert methodology.cutoff_ry == 560.0
    assert methodology.relative_cutoff_ry == 40.0

    assert (
        methodology.provenance.cutoff_report
        == (
            "workflows/convergence/cutoff/"
            "convergence-report.json"
        )
    )

    assert (
        methodology.provenance.relative_cutoff_report
        == (
            "workflows/convergence/relative_cutoff/"
            "convergence-report.json"
        )
    )

    reloaded = load_project_workspace(
        project_root
    )

    assert (
        reloaded.config.methodology.cp2k
        == methodology
    )