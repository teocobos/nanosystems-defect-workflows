"""End-to-end CP2K convergence workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from pymatgen.core import Structure

from nsdw.calculators.cp2k.generation_models import CP2KInputConfig

from nsdw.workflows.convergence.cp2k_execution import (
    collect_existing_cp2k_convergence_results,
    run_cp2k_convergence_campaign,
)
from nsdw.workflows.convergence.cp2k_generation import (
    generate_cp2k_convergence_study,
)
from nsdw.workflows.convergence.manifest import (
    convergence_study_from_manifest,
    load_convergence_manifest,
)
from nsdw.workflows.convergence.observation import (
    collect_convergence_observations,
)
from nsdw.workflows.convergence.recipes import (
    apply_selected_cutoff,
    build_standard_cp2k_cutoff_study,
    build_standard_cp2k_relative_cutoff_study,
    selected_ry_value,
)
from nsdw.workflows.convergence.reporting import (
    ConvergenceReport,
    build_convergence_report,
)
from nsdw.workflows.convergence.analyser import (
    ConvergenceTailAnalysis,
    analyse_convergence_against_reference,
    analyse_convergence_energy,
    analyse_convergence_tail_stability,
)
from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
)

from nsdw.project.workspace import (
    ProjectWorkspace,
    update_cp2k_methodology,
)
from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
    MethodologyStatus,
)


def _run_cp2k_convergence_pipeline(
    *,
    campaign_directory: Path,
    executable: str,
    environment: dict[str, str] | None,
):
    """Execute a CP2K convergence campaign and collect observations."""

    manifest = load_convergence_manifest(
        campaign_directory / "manifest.json"
    )

    study = convergence_study_from_manifest(
        manifest
    )

    results = run_cp2k_convergence_campaign(
        campaign_directory=campaign_directory,
        executable=executable,
        environment=environment,
    )

    observations = collect_convergence_observations(
        candidates=tuple(study.candidates),
        results=results,
    )

    return study, observations


def _build_and_write_convergence_report(
    *,
    campaign_directory: Path,
    study,
    observations,
) -> ConvergenceReport:
    """Analyse observations and write convergence reports."""

    observation_list = list(observations)

    adjacent_analysis = analyse_convergence_energy(
        study,
        observation_list,
    )

    reference_analysis = analyse_convergence_against_reference(
        study,
        observation_list,
    )

    tail_analysis = analyse_convergence_tail_stability(
        study,
        observation_list,
    )

    report = build_convergence_report(
        study=study,
        observations=observation_list,
        adjacent_analysis=adjacent_analysis,
        reference_analysis=reference_analysis,
        tail_analysis=tail_analysis,
    )

    report.write_json(
        campaign_directory / "convergence-report.json"
    )

    report.write_csv(
        campaign_directory / "convergence-report.csv"
    )

    return report


def analyse_and_report_existing_cp2k_convergence_campaign(
    *,
    campaign_directory: str | Path,
) -> ConvergenceReport:
    """Analyse and report an already completed CP2K campaign."""

    campaign_directory = (
        Path(campaign_directory)
        .expanduser()
        .resolve()
    )

    manifest = load_convergence_manifest(
        campaign_directory / "manifest.json"
    )

    study = convergence_study_from_manifest(
        manifest
    )

    results = collect_existing_cp2k_convergence_results(
        campaign_directory=campaign_directory,
    )

    observations = collect_convergence_observations(
        candidates=tuple(study.candidates),
        results=results,
    )

    return _build_and_write_convergence_report(
        campaign_directory=campaign_directory,
        study=study,
        observations=observations,
    )


def run_and_analyse_cp2k_convergence_campaign(
    *,
    campaign_directory: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
) -> ConvergenceTailAnalysis:
    """Execute and analyse a generated CP2K convergence campaign."""

    campaign_directory = (
        Path(campaign_directory)
        .expanduser()
        .resolve()
    )

    study, observations = _run_cp2k_convergence_pipeline(
        campaign_directory=campaign_directory,
        executable=executable,
        environment=environment,
    )

    return analyse_convergence_tail_stability(
        study,
        list(observations),
    )


def run_and_report_cp2k_convergence_campaign(
    *,
    campaign_directory: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
) -> ConvergenceReport:
    """Execute, analyse, and report a CP2K convergence campaign."""

    campaign_directory = (
        Path(campaign_directory)
        .expanduser()
        .resolve()
    )

    study, observations = _run_cp2k_convergence_pipeline(
        campaign_directory=campaign_directory,
        executable=executable,
        environment=environment,
    )

    return _build_and_write_convergence_report(
        campaign_directory=campaign_directory,
        study=study,
        observations=observations,
    )


@dataclass(frozen=True)
class CP2KStandardConvergenceResult:
    """Result of the standard chained CP2K convergence recipe."""

    cutoff_ry: float
    relative_cutoff_ry: float
    cutoff_report: ConvergenceReport
    relative_cutoff_report: ConvergenceReport
    converged_config: CP2KInputConfig


def build_cp2k_production_methodology(
    *,
    result: CP2KStandardConvergenceResult,
    cutoff_report_path: str,
    relative_cutoff_report_path: str,
    status: MethodologyStatus = "validated",
) -> CP2KProductionMethodology:
    """Build a persistent production methodology from convergence output."""

    config = result.converged_config

    return CP2KProductionMethodology(
        status=status,
        functional=config.functional,
        cutoff_ry=result.cutoff_ry,
        relative_cutoff_ry=result.relative_cutoff_ry,
        k_points=config.k_points,
        scf=config.scf,
        basis_potential=config.basis_potential,
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
            cutoff_report=cutoff_report_path,
            relative_cutoff_report=relative_cutoff_report_path,
        ),
    )


def write_cp2k_methodology_candidate(
    *,
    result: CP2KStandardConvergenceResult,
    workflow_directory: str | Path,
) -> Path:
    """Write a reviewable CP2K methodology candidate artifact."""

    workflow_directory = (
        Path(workflow_directory)
        .expanduser()
        .resolve()
    )

    workflow_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    methodology = build_cp2k_production_methodology(
        result=result,
        cutoff_report_path=(
            "cutoff/convergence-report.json"
        ),
        relative_cutoff_report_path=(
            "relative_cutoff/convergence-report.json"
        ),
        status="candidate",
    )

    candidate_path = (
        workflow_directory
        / "methodology-candidate.yaml"
    )

    candidate_path.write_text(
        yaml.safe_dump(
            methodology.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return candidate_path


def load_cp2k_methodology_candidate(
    candidate_path: str | Path,
) -> CP2KProductionMethodology:
    """Load and validate a CP2K methodology candidate artifact."""

    candidate_path = (
        Path(candidate_path)
        .expanduser()
        .resolve()
    )

    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Methodology candidate not found: "
            f"{candidate_path}"
        )

    data = yaml.safe_load(
        candidate_path.read_text(
            encoding="utf-8",
        )
    )

    methodology = CP2KProductionMethodology.model_validate(
        data
    )

    if methodology.status != "candidate":
        raise ValueError(
            "Methodology candidate must have "
            "status 'candidate'."
        )

    return methodology


def promote_cp2k_methodology_candidate(
    *,
    project_root: str | Path,
    candidate_path: str | Path,
) -> ProjectWorkspace:
    """Promote a reviewed CP2K methodology candidate into project metadata."""

    project_root = (
        Path(project_root)
        .expanduser()
        .resolve()
    )

    candidate_path = (
        Path(candidate_path)
        .expanduser()
        .resolve()
    )

    methodology = load_cp2k_methodology_candidate(
        candidate_path
    )

    candidate_directory = candidate_path.parent

    def project_relative_provenance_path(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        resolved = (
            candidate_directory
            / value
        ).resolve()

        try:
            relative = resolved.relative_to(
                project_root
            )
        except ValueError as exc:
            raise ValueError(
                "Methodology provenance must remain "
                "inside the NSDW project."
            ) from exc

        return relative.as_posix()

    promoted_provenance = (
        methodology.provenance.model_copy(
            update={
                "cutoff_report": (
                    project_relative_provenance_path(
                        methodology.provenance.cutoff_report
                    )
                ),
                "relative_cutoff_report": (
                    project_relative_provenance_path(
                        methodology.provenance.relative_cutoff_report
                    )
                ),
            }
        )
    )

    promoted_methodology = methodology.model_copy(
        update={
            "status": "validated",
            "provenance": promoted_provenance,
        }
    )

    return update_cp2k_methodology(
        project_root,
        promoted_methodology,
    )


def persist_cp2k_convergence_methodology(
    *,
    project_root: str | Path,
    result: CP2KStandardConvergenceResult,
    workflow_directory: str | Path,
) -> ProjectWorkspace:
    """Persist converged CP2K settings into the NSDW project."""

    project_root = Path(project_root).expanduser().resolve()
    workflow_directory = Path(
        workflow_directory
    ).expanduser().resolve()

    cutoff_report = (
        workflow_directory
        / "cutoff"
        / "convergence-report.json"
    )

    relative_cutoff_report = (
        workflow_directory
        / "relative_cutoff"
        / "convergence-report.json"
    )

    try:
        cutoff_report_path = str(
            cutoff_report.relative_to(project_root)
        )
    except ValueError:
        cutoff_report_path = str(cutoff_report)

    try:
        relative_cutoff_report_path = str(
            relative_cutoff_report.relative_to(project_root)
        )
    except ValueError:
        relative_cutoff_report_path = str(
            relative_cutoff_report
        )

    methodology = build_cp2k_production_methodology(
        result=result,
        cutoff_report_path=cutoff_report_path,
        relative_cutoff_report_path=relative_cutoff_report_path,
    )

    return update_cp2k_methodology(
        project_root,
        methodology,
    )


def run_standard_cp2k_convergence_recipe(
    *,
    structure: Structure,
    base_config: CP2KInputConfig,
    workflow_directory: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
) -> CP2KStandardConvergenceResult:
    """Run the standard chained CP2K convergence recipe."""

    workflow_directory = Path(workflow_directory)

    cutoff_study = build_standard_cp2k_cutoff_study()
    cutoff_directory = workflow_directory / "cutoff"

    generate_cp2k_convergence_study(
        structure=structure,
        base_config=base_config,
        study=cutoff_study,
        output_directory=cutoff_directory,
    )

    cutoff_report = run_and_report_cp2k_convergence_campaign(
        campaign_directory=cutoff_directory,
        executable=executable,
        environment=environment,
    )

    cutoff_ry = selected_ry_value(
        study=cutoff_study,
        analysis=cutoff_report,
    )

    relative_base_config = apply_selected_cutoff(
        base_config=base_config,
        study=cutoff_study,
        analysis=cutoff_report,
    )

    relative_cutoff_study = (
        build_standard_cp2k_relative_cutoff_study()
    )
    relative_cutoff_directory = (
        workflow_directory / "relative_cutoff"
    )

    generate_cp2k_convergence_study(
        structure=structure,
        base_config=relative_base_config,
        study=relative_cutoff_study,
        output_directory=relative_cutoff_directory,
    )

    relative_cutoff_report = (
        run_and_report_cp2k_convergence_campaign(
            campaign_directory=relative_cutoff_directory,
            executable=executable,
            environment=environment,
        )
    )

    relative_cutoff_ry = selected_ry_value(
        study=relative_cutoff_study,
        analysis=relative_cutoff_report,
    )

    converged_config = relative_base_config.model_copy(
        update={
            "relative_cutoff_ry": relative_cutoff_ry,
        }
    )

    result = CP2KStandardConvergenceResult(
        cutoff_ry=cutoff_ry,
        relative_cutoff_ry=relative_cutoff_ry,
        cutoff_report=cutoff_report,
        relative_cutoff_report=relative_cutoff_report,
        converged_config=converged_config,
    )

    write_cp2k_methodology_candidate(
        result=result,
        workflow_directory=workflow_directory,
    )

    return result