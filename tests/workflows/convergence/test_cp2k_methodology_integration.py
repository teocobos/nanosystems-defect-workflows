
"""Five-campaign CP2K methodology evidence integration tests."""

from pathlib import Path
import json
import shutil

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
    CP2KSCFConfig,
)
from nsdw.calculators.cp2k.input_parser import parse_cp2k_input
from nsdw.calculators.cp2k.parser import parse_cp2k_output
from nsdw.calculators.cp2k.presets import (
    get_basis_potential_preset,
)
from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
)
from nsdw.workflows.convergence.analyser import (
    analyse_convergence_against_reference,
    analyse_convergence_energy,
    analyse_convergence_tail_stability,
)
from nsdw.workflows.convergence.cp2k_generation import (
    generate_cp2k_convergence_study,
)
from nsdw.workflows.convergence.cp2k_methodology_evidence import (
    validate_cp2k_methodology_evidence,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceObservation,
)
from nsdw.workflows.convergence.recipes import (
    build_cp2k_kpoint_study,
    build_standard_cp2k_cutoff_study,
    build_standard_cp2k_relative_cutoff_study,
)
from nsdw.workflows.convergence.reporting import (
    build_convergence_report,
)
from nsdw.structures.result import calculate_structure_hash


def _make_campaign(
    *,
    root,
    name,
    structure,
    config,
    study,
    fixture_output,
):
    directory = root / name

    generate_cp2k_convergence_study(
        structure=structure,
        base_config=config,
        study=study,
        output_directory=directory,
    )

    manifest = load_convergence_manifest(
        directory / "manifest.json"
    )

    parsed_output = parse_cp2k_output(fixture_output)
    energy_hartree = (
        parsed_output.energy.total_energy_hartree
    )
    assert energy_hartree is not None

    observations = []

    for candidate in manifest.candidates:
        candidate_directory = directory / candidate.directory
        input_path = candidate_directory / candidate.input_file
        parsed_input = parse_cp2k_input(input_path)

        output_path = input_path.with_suffix(".out")

        output_text = fixture_output.read_text(
            encoding="utf-8"
        )

        # Preserve the synthetic fixture's SCF/energy/termination
        # evidence while matching generated input project identity.
        fixture_project = parsed_output.project_name
        assert fixture_project is not None
        assert parsed_input.project_name is not None

        output_text = output_text.replace(
            fixture_project,
            parsed_input.project_name,
        )

        output_path.write_text(
            output_text,
            encoding="utf-8",
        )

        observations.append(
            ConvergenceObservation(
                candidate=study.candidates[candidate.order],
                total_energy_ev=(
                    energy_hartree * HARTREE_TO_EV
                ),
                n_atoms=len(structure),
                structure_hash=calculate_structure_hash(
                    structure
                ),
            )
        )

    adjacent = analyse_convergence_energy(
        study,
        observations,
    )

    reference = analyse_convergence_against_reference(
        study,
        observations,
    )

    tail = analyse_convergence_tail_stability(
        study,
        observations,
    )

    report = build_convergence_report(
        study=study,
        observations=observations,
        adjacent_analysis=adjacent,
        reference_analysis=reference,
        tail_analysis=tail,
    )

    report.write_json(
        directory / "convergence-report.json"
    )

    return report


@pytest.fixture
def complete_methodology(tmp_path):
    root = tmp_path
    workflow = root / "workflow"
    workflow.mkdir()

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    fixture_output = (
        Path("tests/calculators/cp2k/fixtures")
        / "energy_complete.out"
    )

    assert fixture_output.is_file()

    base = CP2KInputConfig(
        project_name="methodology-test",
        functional="PBE",
        run_type="ENERGY",
        cutoff_ry=400.0,
        relative_cutoff_ry=40.0,
        basis_potential=get_basis_potential_preset(
            "sio2-pbe-dzvp"
        ),
    )

    cutoff_study = build_standard_cp2k_cutoff_study()
    relative_study = (
        build_standard_cp2k_relative_cutoff_study()
    )
    kpoint_study = build_cp2k_kpoint_study(
        meshes=((1, 1, 1), (2, 2, 2), (3, 3, 3)),
    )

    # Deterministic synthetic energies give stable convergence.
    cutoff = _make_campaign(
        root=workflow,
        name="cutoff",
        structure=structure,
        config=base,
        study=cutoff_study,
        fixture_output=fixture_output,
    )

    selected_cutoff = cutoff.candidates[0].value.value

    relative_base = base.model_copy(
        update={"cutoff_ry": selected_cutoff}
    )

    relative = _make_campaign(
        root=workflow,
        name="relative_cutoff",
        structure=structure,
        config=relative_base,
        study=relative_study,
        fixture_output=fixture_output,
    )

    selected_relative = (
        relative.candidates[0].value.value
    )

    converged = relative_base.model_copy(
        update={
            "relative_cutoff_ry": selected_relative,
            "scf": CP2KSCFConfig(
                solver="DIAGONALIZATION"
            ),
        }
    )

    kpoints = _make_campaign(
        root=workflow,
        name="kpoints",
        structure=structure,
        config=converged,
        study=kpoint_study,
        fixture_output=fixture_output,
    )

    selected_mesh = tuple(
        kpoints.candidates[0].value
    )

    final_config = converged.model_copy(
        update={"k_points": selected_mesh}
    )

    _make_campaign(
        root=workflow,
        name="final_cutoff_verification",
        structure=structure,
        config=final_config,
        study=cutoff_study,
        fixture_output=fixture_output,
    )

    _make_campaign(
        root=workflow,
        name="final_relative_cutoff_verification",
        structure=structure,
        config=final_config,
        study=relative_study,
        fixture_output=fixture_output,
    )

    candidate_path = workflow / "methodology-candidate.yaml"

    methodology = CP2KProductionMethodology(
        status="candidate",
        functional="PBE",
        cutoff_ry=selected_cutoff,
        relative_cutoff_ry=selected_relative,
        k_points=selected_mesh,
        scf=CP2KSCFConfig(
            solver="DIAGONALIZATION"
        ),
        basis_potential=get_basis_potential_preset(
            "sio2-pbe-dzvp"
        ),
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
            cutoff_report="cutoff/convergence-report.json",
            relative_cutoff_report=(
                "relative_cutoff/convergence-report.json"
            ),
            kpoint_report="kpoints/convergence-report.json",
            final_cutoff_verification_report=(
                "final_cutoff_verification/convergence-report.json"
            ),
            final_relative_cutoff_verification_report=(
                "final_relative_cutoff_verification/"
                "convergence-report.json"
            ),
        ),
    )

    import yaml

    candidate_path.write_text(
        yaml.safe_dump(
            methodology.model_dump(
                mode="json",
                warnings="error",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return root, candidate_path, methodology


def test_complete_five_campaign_methodology(complete_methodology):
    root, candidate_path, methodology = complete_methodology

    validated = validate_cp2k_methodology_evidence(
        project_root=root,
        candidate_path=candidate_path,
        methodology=methodology,
    )

    assert set(validated) == {
        "cutoff_report",
        "relative_cutoff_report",
        "kpoint_report",
        "final_cutoff_verification_report",
        "final_relative_cutoff_verification_report",
    }


def _first_candidate_input(root, campaign):
    """Return the first generated CP2K input in a campaign."""
    directory = root / "workflow" / campaign
    manifest = load_convergence_manifest(
        directory / "manifest.json"
    )
    candidate = manifest.candidates[0]
    return (
        directory
        / candidate.directory
        / candidate.input_file
    )


def _replace_input_setting(path, old, new):
    """Change one CP2K input setting without changing its manifest."""
    original = path.read_text(encoding="utf-8")

    if original.count(old) != 1:
        raise AssertionError(
            f"Expected one occurrence of {old!r} "
            f"in {path}; found {original.count(old)}"
        )

    path.write_text(
        original.replace(old, new, 1),
        encoding="utf-8",
    )


def test_rejects_changed_relative_campaign_cutoff(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(
        root, "relative_cutoff"
    )

    parsed = parse_cp2k_input(path)
    assert parsed.cutoff_ry == methodology.cutoff_ry

    _replace_input_setting(
        path,
        f"CUTOFF {parsed.cutoff_ry:g}",
        f"CUTOFF {parsed.cutoff_ry + 100:g}",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=methodology,
        )


def test_rejects_changed_final_campaign_charge(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(
        root, "final_cutoff_verification"
    )

    _replace_input_setting(
        path,
        "CHARGE 0",
        "CHARGE 1",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=methodology,
        )


def test_rejects_wrong_final_mesh(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(
        root, "final_relative_cutoff_verification"
    )

    parsed = parse_cp2k_input(path)
    assert parsed.k_points == methodology.k_points

    original_mesh = " ".join(
        str(value) for value in methodology.k_points
    )

    wrong_mesh = " ".join(
        str(value + 1) for value in methodology.k_points
    )

    _replace_input_setting(
        path,
        f"SCHEME MONKHORST-PACK {original_mesh}",
        f"SCHEME MONKHORST-PACK {wrong_mesh}",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=methodology,
        )


def test_rejects_wrong_proposed_cutoff(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    altered = methodology.model_copy(
        update={
            "cutoff_ry": methodology.cutoff_ry + 100.0
        }
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=altered,
        )



def test_rejects_methodology_xc_mismatch(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    assert methodology.functional.value == "PBE"

    altered = methodology.model_copy(
        update={"functional": "PBEsol"}
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Production XC functional differs",
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=altered,
        )


def test_rejects_methodology_scf_tolerance_mismatch(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    altered_scf = methodology.scf.model_copy(
        update={"eps_scf": 1.0e-7}
    )

    altered = methodology.model_copy(
        update={"scf": altered_scf}
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Production SCF tolerance differs",
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=altered,
        )


def test_rejects_methodology_pseudopotential_mismatch(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    basis = methodology.basis_potential
    assert basis is not None

    original_kinds = basis.kinds
    assert any(kind.element == "Si" for kind in original_kinds)

    altered_kinds = tuple(
        kind.model_copy(
            update={"potential": "GTH-PBE-q99"}
        )
        if kind.element == "Si"
        else kind
        for kind in original_kinds
    )

    altered_basis = basis.model_copy(
        update={"kinds": altered_kinds}
    )

    altered = methodology.model_copy(
        update={"basis_potential": altered_basis}
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Production atomic basis/potential assignments differ",
    ):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=altered,
        )

def _assert_methodology_rejected(
    root,
    candidate_path,
    methodology,
):
    """Run the complete evidence validator and require rejection."""
    with pytest.raises(ConvergenceEvidenceValidationError):
        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=candidate_path,
            methodology=methodology,
        )


def test_rejects_tampered_final_scf_max_iterations(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(
        root, "final_cutoff_verification"
    )

    parsed = parse_cp2k_input(path)
    assert parsed.max_scf == methodology.scf.max_scf

    _replace_input_setting(
        path,
        f"MAX_SCF {parsed.max_scf}",
        f"MAX_SCF {parsed.max_scf + 25}",
    )

    _assert_methodology_rejected(
        root, candidate_path, methodology
    )


def test_rejects_tampered_final_scf_guess(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(
        root, "final_relative_cutoff_verification"
    )

    parsed = parse_cp2k_input(path)
    assert parsed.scf_guess == "ATOMIC"

    _replace_input_setting(
        path,
        "SCF_GUESS ATOMIC",
        "SCF_GUESS RESTART",
    )

    _assert_methodology_rejected(
        root, candidate_path, methodology
    )


def test_rejects_missing_final_max_scf(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(
        root, "final_cutoff_verification"
    )

    parsed = parse_cp2k_input(path)
    assert parsed.max_scf is not None

    _replace_input_setting(
        path,
        f"MAX_SCF {parsed.max_scf}",
        "! MAX_SCF removed for provenance test",
    )

    assert parse_cp2k_input(path).max_scf is None

    _assert_methodology_rejected(
        root, candidate_path, methodology
    )


def test_rejects_tampered_initial_ot_minimizer(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(
        root, "relative_cutoff"
    )

    parsed = parse_cp2k_input(path)
    assert parsed.scf_solver == "OT"
    assert parsed.ot_minimizer == "CG"

    _replace_input_setting(
        path,
        "MINIMIZER CG",
        "MINIMIZER DIIS",
    )

    _assert_methodology_rejected(
        root, candidate_path, methodology
    )


def test_rejects_tampered_initial_outer_scf_limit(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    path = _first_candidate_input(root, "cutoff")

    parsed = parse_cp2k_input(path)
    assert parsed.scf_solver == "OT"
    assert parsed.outer_scf_max is not None

    # This replacement targets the OUTER_SCF block, not
    # the inner SCF MAX_SCF parameter.
    original = path.read_text(encoding="utf-8")

    old = (
        "      &OUTER_SCF\n"
        f"        MAX_SCF {parsed.outer_scf_max}\n"
    )

    new = (
        "      &OUTER_SCF\n"
        f"        MAX_SCF {parsed.outer_scf_max + 5}\n"
    )

    if original.count(old) != 1:
        raise AssertionError(
            "Expected one generated OUTER_SCF block."
        )

    path.write_text(
        original.replace(old, new, 1),
        encoding="utf-8",
    )

    assert (
        parse_cp2k_input(path).outer_scf_max
        == parsed.outer_scf_max + 5
    )

    _assert_methodology_rejected(
        root, candidate_path, methodology
    )


def test_accepts_legitimate_ot_to_diagonalization_transition(
    complete_methodology,
):
    root, candidate_path, methodology = complete_methodology

    initial = parse_cp2k_input(
        _first_candidate_input(root, "cutoff")
    )

    kpoint = parse_cp2k_input(
        _first_candidate_input(root, "kpoints")
    )

    final = parse_cp2k_input(
        _first_candidate_input(
            root, "final_cutoff_verification"
        )
    )

    assert initial.scf_solver == "OT"
    assert kpoint.scf_solver == "DIAGONALIZATION"
    assert final.scf_solver == "DIAGONALIZATION"

    validated = validate_cp2k_methodology_evidence(
        project_root=root,
        candidate_path=candidate_path,
        methodology=methodology,
    )

    assert len(validated) == 5



def _initialise_promotion_project(root):
    """Add project metadata without disturbing generated campaigns."""
    from nsdw.project.models import ProjectConfig
    from nsdw.project.workspace import write_project_config

    write_project_config(
        root,
        ProjectConfig(
            name="Methodology Promotion Test",
            material="SiO2",
            nsdw_version="0.1.0",
            components=["cp2k"],
        ),
    )


def test_promotion_with_complete_evidence(complete_methodology):
    """Five verified campaigns permit a validated methodology."""
    from nsdw.project.workspace import load_project_workspace
    from nsdw.workflows.convergence.cp2k_workflow import (
        promote_cp2k_methodology_candidate,
    )

    root, candidate_path, candidate = complete_methodology
    _initialise_promotion_project(root)

    promoted = promote_cp2k_methodology_candidate(
        project_root=root,
        candidate_path=candidate_path,
    )

    saved = load_project_workspace(root)
    methodology = saved.config.methodology.cp2k

    assert methodology is not None
    assert methodology.status == "validated"
    assert methodology.cutoff_ry == candidate.cutoff_ry
    assert methodology.relative_cutoff_ry == candidate.relative_cutoff_ry
    assert methodology.k_points == candidate.k_points
    assert promoted.config.methodology.cp2k == methodology


def test_tampered_evidence_blocks_promotion(complete_methodology):
    """Changed CP2K input must leave project.yaml untouched."""
    from nsdw.workflows.convergence.cp2k_workflow import (
        promote_cp2k_methodology_candidate,
    )

    root, candidate_path, methodology = complete_methodology
    _initialise_promotion_project(root)

    metadata = root / "project.yaml"
    before = metadata.read_bytes()

    input_path = _first_candidate_input(
        root, "relative_cutoff"
    )

    parsed = parse_cp2k_input(input_path)

    _replace_input_setting(
        input_path,
        f"CUTOFF {parsed.cutoff_ry:g}",
        f"CUTOFF {parsed.cutoff_ry + 100:g}",
    )

    with pytest.raises(ConvergenceEvidenceValidationError):
        promote_cp2k_methodology_candidate(
            project_root=root,
            candidate_path=candidate_path,
        )

    assert metadata.read_bytes() == before



def test_promotion_atomic_replace_failure_preserves_project(
    complete_methodology,
    monkeypatch,
):
    """A failed atomic replacement leaves project metadata intact."""
    import os

    from nsdw.workflows.convergence.cp2k_workflow import (
        promote_cp2k_methodology_candidate,
    )

    root, candidate_path, _ = complete_methodology
    _initialise_promotion_project(root)

    metadata = root / "project.yaml"
    before = metadata.read_bytes()

    def reject_replace(source, destination):
        if Path(destination).resolve() == metadata.resolve():
            raise OSError("Simulated atomic promotion failure")
        return original_replace(source, destination)

    original_replace = os.replace

    monkeypatch.setattr(
        os,
        "replace",
        reject_replace,
    )

    with pytest.raises(
        OSError,
        match="Simulated atomic promotion failure",
    ):
        promote_cp2k_methodology_candidate(
            project_root=root,
            candidate_path=candidate_path,
        )

    assert metadata.read_bytes() == before
    assert list(root.glob(".project-*.yaml.tmp")) == []



def _promotion_process_worker(root, candidate_path, start, ready, results):
    """Attempt promotion in a separate Linux process."""
    from nsdw.workflows.convergence.cp2k_workflow import (
        promote_cp2k_methodology_candidate,
    )

    ready.put("ready")

    if not start.wait(timeout=30):
        results.put(("error", "Timed out waiting for start"))
        return

    try:
        promote_cp2k_methodology_candidate(
            project_root=root,
            candidate_path=candidate_path,
        )
    except Exception as exc:
        results.put((type(exc).__name__, str(exc)))
    else:
        results.put(("success", ""))


def test_concurrent_promotion_allows_only_one_commit(
    complete_methodology,
):
    """Concurrent promotions cannot overwrite validated methodology."""
    import multiprocessing
    import queue
    import sys

    from nsdw.project.locking import project_write_lock
    from nsdw.project.workspace import load_project_workspace

    if sys.platform != "linux":
        pytest.skip("Requires Linux fcntl project locking")

    root, candidate_path, candidate = complete_methodology
    _initialise_promotion_project(root)

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    ready = context.Queue()
    results = context.Queue()

    workers = [
        context.Process(
            target=_promotion_process_worker,
            args=(root, candidate_path, start, ready, results),
        )
        for _ in range(2)
    ]

    try:
        # Neither child can commit before the parent releases this lock.
        with project_write_lock(root):
            for worker in workers:
                worker.start()

            for _ in workers:
                assert ready.get(timeout=15) == "ready"

            start.set()

        outcomes = [
            results.get(timeout=60)
            for _ in workers
        ]

        for worker in workers:
            worker.join(timeout=15)

        assert all(worker.exitcode == 0 for worker in workers)

        statuses = sorted(status for status, _ in outcomes)

        assert statuses == [
            "CP2KMethodologyValidationError",
            "success",
        ], outcomes

        saved = load_project_workspace(root)
        methodology = saved.config.methodology.cp2k

        assert methodology is not None
        assert methodology.status == "validated"
        assert methodology.cutoff_ry == candidate.cutoff_ry
        assert methodology.relative_cutoff_ry == (
            candidate.relative_cutoff_ry
        )

    finally:
        start.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        ready.close()
        results.close()
