"""End-to-end CP2K convergence workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
    apply_selected_kpoints,
    build_cp2k_kpoint_study,
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
    _commit_verified_cp2k_methodology_locked,
    load_project_workspace,
    update_cp2k_methodology,
)
from nsdw.project.locking import project_write_lock
from nsdw.workflows.convergence.cp2k_methodology_evidence import (
    validate_cp2k_methodology_evidence,
)
from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
    MethodologyStatus,
)
from nsdw.models.quantity import Quantity


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
    kpoint_mesh: tuple[int, int, int] | None = None
    kpoint_report: ConvergenceReport | None = None
    final_cutoff_verification_report: ConvergenceReport | None = None
    final_relative_cutoff_verification_report: ConvergenceReport | None = None


def build_cp2k_production_methodology(
    *,
    result: CP2KStandardConvergenceResult,
    cutoff_report_path: str,
    relative_cutoff_report_path: str,
    kpoint_report_path: str | None = None,
    final_cutoff_verification_report_path: str | None = None,
    final_relative_cutoff_verification_report_path: str | None = None,
    status: MethodologyStatus = "candidate",
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
            kpoint_report=kpoint_report_path,
            final_cutoff_verification_report=(
                final_cutoff_verification_report_path
            ),
            final_relative_cutoff_verification_report=(
                final_relative_cutoff_verification_report_path
            ),
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
        kpoint_report_path=(
            "kpoints/convergence-report.json"
            if result.kpoint_report is not None
            else None
        ),
        final_cutoff_verification_report_path=(
            "final_cutoff_verification/convergence-report.json"
            if result.final_cutoff_verification_report is not None
            else None
        ),
        final_relative_cutoff_verification_report_path=(
            "final_relative_cutoff_verification/convergence-report.json"
            if result.final_relative_cutoff_verification_report is not None
            else None
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



class CP2KMethodologyValidationError(ValueError):
    """Raised when methodology evidence is insufficient."""


def _validate_kpoint_convergence_report(
    *,
    candidate_directory: Path,
    methodology: CP2KProductionMethodology,
) -> None:
    """Validate recorded k-point convergence evidence."""

    provenance_path = methodology.provenance.kpoint_report

    if not provenance_path:
        raise CP2KMethodologyValidationError(
            "K-point methodology requires a k-point convergence report."
        )

    report_path = (candidate_directory / provenance_path).resolve()

    if not report_path.is_file():
        raise CP2KMethodologyValidationError(
            f"K-point convergence report not found: {report_path}"
        )

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CP2KMethodologyValidationError(
            "K-point convergence report is unreadable or invalid JSON."
        ) from exc

    if not isinstance(report, dict):
        raise CP2KMethodologyValidationError(
            "K-point convergence report must contain a JSON object."
        )

    if report.get("parameter") != "kpoints":
        raise CP2KMethodologyValidationError(
            "Expected a k-point convergence report."
        )

    if report.get("energy_tolerance_satisfied") is not True:
        raise CP2KMethodologyValidationError(
            "K-point energy convergence has not been demonstrated."
        )

    selected_label = report.get("selected_candidate_label")

    if not isinstance(selected_label, str) or not selected_label:
        raise CP2KMethodologyValidationError(
            "K-point convergence report has no selected candidate."
        )

    candidates = report.get("candidates")

    if not isinstance(candidates, list):
        raise CP2KMethodologyValidationError(
            "K-point convergence report has no candidate data."
        )

    selected = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("label") == selected_label
        and candidate.get("selected") is True
    ]

    if len(selected) != 1:
        raise CP2KMethodologyValidationError(
            "K-point report selection is missing or inconsistent."
        )

    value = selected[0].get("value")

    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            type(component) is not int or component <= 0
            for component in value
        )
        or tuple(value) != methodology.k_points
    ):
        raise CP2KMethodologyValidationError(
            "Selected k-point mesh does not match the methodology."
        )

    if methodology.scf.solver != "DIAGONALIZATION":
        raise CP2KMethodologyValidationError(
            "Explicit k-point sampling requires diagonalisation."
        )

def promote_cp2k_methodology_candidate(
    *,
    project_root: str | Path,
    candidate_path: str | Path,
) -> ProjectWorkspace:
    """Promote only after independent scientific evidence verification.

    Candidate loading, evidence verification and atomic persistence
    are performed while holding the exclusive project lock.
    """
    root = Path(project_root).expanduser().resolve()

    with project_write_lock(root):
        workspace = load_project_workspace(root)

        if "cp2k" not in workspace.config.components:
            raise CP2KMethodologyValidationError(
                "Cannot promote methodology: CP2K component is disabled."
            )

        existing = workspace.config.methodology.cp2k

        if existing is not None and existing.status == "validated":
            raise CP2KMethodologyValidationError(
                "A validated CP2K methodology already exists."
            )

        candidate = load_cp2k_methodology_candidate(
            candidate_path
        )

        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=candidate,
        )

        validated = CP2KProductionMethodology.model_validate(
            {
                **candidate.model_dump(
                    mode="json",
                    warnings="error",
                ),
                "status": "validated",
            }
        )

        return _commit_verified_cp2k_methodology_locked(
            root,
            validated,
            candidate_path=candidate_path,
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

    kpoint_report_path = None
    if result.kpoint_report is not None:
        kpoint_report = (
            workflow_directory / "kpoints" / "convergence-report.json"
        )
        try:
            kpoint_report_path = str(
                kpoint_report.relative_to(project_root)
            )
        except ValueError:
            kpoint_report_path = str(kpoint_report)

    methodology = build_cp2k_production_methodology(
        result=result,
        cutoff_report_path=cutoff_report_path,
        relative_cutoff_report_path=relative_cutoff_report_path,
        kpoint_report_path=kpoint_report_path,
        final_cutoff_verification_report_path=(
            (
                workflow_directory
                / "final_cutoff_verification"
                / "convergence-report.json"
            ).relative_to(project_root).as_posix()
            if result.final_cutoff_verification_report is not None
            else None
        ),
        final_relative_cutoff_verification_report_path=(
            (
                workflow_directory
                / "final_relative_cutoff_verification"
                / "convergence-report.json"
            ).relative_to(project_root).as_posix()
            if result.final_relative_cutoff_verification_report is not None
            else None
        ),
        status="candidate",
    )

    return update_cp2k_methodology(
        project_root,
        methodology,
    )



@dataclass(frozen=True)
class CP2KFinalMeshVerificationResult:
    """Verification results using final production k-point settings."""

    cutoff_report: ConvergenceReport
    relative_cutoff_report: ConvergenceReport


def _require_selected_parameter_stable(
    *,
    report: ConvergenceReport,
    study,
    selected_value_ry: float,
) -> None:
    """Require the original parameter choice to remain tail-stable."""

    expected_label = next(
        (
            candidate.label
            for candidate in study.candidates
            if isinstance(candidate.value, Quantity)
            and candidate.value.unit == "Ry"
            and candidate.value.value == selected_value_ry
        ),
        None,
    )

    if expected_label is None:
        raise CP2KMethodologyValidationError(
            f"Original parameter {selected_value_ry:g} Ry "
            "is absent from the verification grid."
        )

    if report.parameter != study.parameter:
        raise CP2KMethodologyValidationError(
            "Verification report has the wrong parameter."
        )

    if report.energy_tolerance_satisfied is not True:
        raise CP2KMethodologyValidationError(
            "Final-mesh verification did not converge."
        )

    matches = [
        candidate
        for candidate in report.candidates
        if candidate.label == expected_label
    ]

    if len(matches) != 1:
        raise CP2KMethodologyValidationError(
            "Original parameter is missing from verification results."
        )

    candidate = matches[0]
    difference = (
        candidate.maximum_higher_cost_energy_difference_ev_per_atom
    )

    if (
        candidate.tail_stable is not True
        or difference is None
        or not math.isfinite(difference)
        or not math.isfinite(report.energy_tolerance_ev_per_atom)
        or report.energy_tolerance_ev_per_atom <= 0
        or difference > report.energy_tolerance_ev_per_atom
    ):
        raise CP2KMethodologyValidationError(
            f"Original parameter {selected_value_ry:g} Ry "
            "is not tail-stable at the final k-point mesh."
        )


def verify_cp2k_final_mesh_convergence(
    *,
    structure: Structure,
    converged_config: CP2KInputConfig,
    workflow_directory: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
) -> CP2KFinalMeshVerificationResult:
    """Verify numerical convergence using final k-point settings."""

    if converged_config.k_points is None:
        raise CP2KMethodologyValidationError(
            "Final-mesh verification requires explicit k-points."
        )

    if converged_config.scf.solver != "DIAGONALIZATION":
        raise CP2KMethodologyValidationError(
            "Final-mesh verification requires diagonalisation."
        )

    workflow_directory = Path(workflow_directory)

    cutoff_study = build_standard_cp2k_cutoff_study()
    relative_study = build_standard_cp2k_relative_cutoff_study()

    cutoff_directory = (
        workflow_directory / "final_cutoff_verification"
    )
    relative_directory = (
        workflow_directory / "final_relative_cutoff_verification"
    )

    generate_cp2k_convergence_study(
        structure=structure,
        base_config=converged_config,
        study=cutoff_study,
        output_directory=cutoff_directory,
    )

    cutoff_report = run_and_report_cp2k_convergence_campaign(
        campaign_directory=cutoff_directory,
        executable=executable,
        environment=environment,
    )

    _require_selected_parameter_stable(
        report=cutoff_report,
        study=cutoff_study,
        selected_value_ry=converged_config.cutoff_ry,
    )

    generate_cp2k_convergence_study(
        structure=structure,
        base_config=converged_config,
        study=relative_study,
        output_directory=relative_directory,
    )

    relative_report = run_and_report_cp2k_convergence_campaign(
        campaign_directory=relative_directory,
        executable=executable,
        environment=environment,
    )

    if cutoff_report.structure_hash != relative_report.structure_hash:
        raise CP2KMethodologyValidationError(
            "Verification reports describe different structures."
        )

    if cutoff_report.n_atoms != relative_report.n_atoms:
        raise CP2KMethodologyValidationError(
            "Verification reports have different atom counts."
        )

    _require_selected_parameter_stable(
        report=relative_report,
        study=relative_study,
        selected_value_ry=converged_config.relative_cutoff_ry,
    )

    return CP2KFinalMeshVerificationResult(
        cutoff_report=cutoff_report,
        relative_cutoff_report=relative_report,
    )


def run_standard_cp2k_convergence_recipe(
    *,
    structure: Structure,
    base_config: CP2KInputConfig,
    workflow_directory: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
    kpoint_meshes: tuple[tuple[int, int, int], ...] | None = None,
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

    kpoint_mesh = None
    kpoint_report = None
    final_verification = None

    if kpoint_meshes is not None:
        kpoint_study = build_cp2k_kpoint_study(
            meshes=kpoint_meshes,
        )

        # Explicit k-point sampling is incompatible with OT.
        # Every candidate must use diagonalisation.
        kpoint_base_config = converged_config.model_copy(
            update={
                "scf": converged_config.scf.model_copy(
                    update={"solver": "DIAGONALIZATION"}
                ),
            }
        )

        kpoint_directory = workflow_directory / "kpoints"

        generate_cp2k_convergence_study(
            structure=structure,
            base_config=kpoint_base_config,
            study=kpoint_study,
            output_directory=kpoint_directory,
        )

        kpoint_report = run_and_report_cp2k_convergence_campaign(
            campaign_directory=kpoint_directory,
            executable=executable,
            environment=environment,
        )

        converged_config = apply_selected_kpoints(
            base_config=kpoint_base_config,
            study=kpoint_study,
            analysis=kpoint_report,
        )
        kpoint_mesh = converged_config.k_points

        final_verification = verify_cp2k_final_mesh_convergence(
            structure=structure,
            converged_config=converged_config,
            workflow_directory=workflow_directory,
            executable=executable,
            environment=environment,
        )

    result = CP2KStandardConvergenceResult(
        cutoff_ry=cutoff_ry,
        relative_cutoff_ry=relative_cutoff_ry,
        cutoff_report=cutoff_report,
        relative_cutoff_report=relative_cutoff_report,
        converged_config=converged_config,
        kpoint_mesh=kpoint_mesh,
        kpoint_report=kpoint_report,
        final_cutoff_verification_report=(
            final_verification.cutoff_report
            if final_verification is not None
            else None
        ),
        final_relative_cutoff_verification_report=(
            final_verification.relative_cutoff_report
            if final_verification is not None
            else None
        ),
    )

    write_cp2k_methodology_candidate(
        result=result,
        workflow_directory=workflow_directory,
    )

    return result