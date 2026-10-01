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