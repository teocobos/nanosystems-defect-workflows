"""End-to-end CP2K convergence workflow orchestration."""

from __future__ import annotations

from pathlib import Path

from nsdw.workflows.convergence.analyser import (
    ConvergenceTailAnalysis,
    analyse_convergence_against_reference,
    analyse_convergence_energy,
    analyse_convergence_tail_stability,
)
from nsdw.workflows.convergence.cp2k_execution import (
    run_cp2k_convergence_campaign,
)
from nsdw.workflows.convergence.manifest import (
    convergence_study_from_manifest,
    load_convergence_manifest,
)
from nsdw.workflows.convergence.observation import (
    collect_convergence_observations,
)
from nsdw.workflows.convergence.reporting import (
    ConvergenceReport,
    build_convergence_report,
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