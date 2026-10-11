"""Regression tests for resumable CP2K convergence stages."""

from pathlib import Path

import pytest

from nsdw.calculators.cp2k.generation_models import CP2KInputConfig
from nsdw.calculators.cp2k.presets import get_basis_potential_preset
from nsdw.workflows.convergence.cp2k_staged import (
    CP2KConvergenceStageError,
    advance_cp2k_convergence_campaign,
)


@pytest.fixture
def base_config():
    return CP2KInputConfig(
        project_name="alpha_quartz_cod1526860",
        run_type="ENERGY",
        basis_potential=get_basis_potential_preset("sio2-pbe-dzvp"),
    )


def test_advance_rejects_missing_source(tmp_path, base_config):
    destination = tmp_path / "next"

    with pytest.raises(
        CP2KConvergenceStageError,
        match="Source campaign does not exist",
    ):
        advance_cp2k_convergence_campaign(
            completed_campaign_directory=tmp_path / "missing",
            next_campaign_directory=destination,
            base_config=base_config,
        )

    assert not destination.exists()


def test_advance_rejects_existing_destination(tmp_path, base_config):
    source = tmp_path / "source"
    source.mkdir()

    destination = tmp_path / "next"
    destination.mkdir()

    sentinel = destination / "important.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(
        CP2KConvergenceStageError,
        match="Destination already exists",
    ):
        advance_cp2k_convergence_campaign(
            completed_campaign_directory=source,
            next_campaign_directory=destination,
            base_config=base_config,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_advance_rejects_destination_inside_source(tmp_path, base_config):
    source = tmp_path / "source"
    source.mkdir()

    destination = source / "next"

    with pytest.raises(
        CP2KConvergenceStageError,
        match="Destination must not be inside",
    ):
        advance_cp2k_convergence_campaign(
            completed_campaign_directory=source,
            next_campaign_directory=destination,
            base_config=base_config,
        )

    assert not destination.exists()


def test_advance_real_sio2_campaign(tmp_path, base_config):
    """Integration test against previously validated ARCHER2 evidence."""
    import json

    from nsdw.calculators.cp2k.input_parser import parse_cp2k_input
    from nsdw.workflows.convergence.manifest import (
        load_convergence_manifest,
    )

    source = Path(
        "outputs/archer2-validation/sio2-pbe-cutoff-extended"
    )

    if not source.is_dir():
        pytest.skip("Historical ARCHER2 evidence is not available.")

    destination = tmp_path / "relative-cutoff"

    original_report = (
        source / "convergence-report.json"
    ).read_bytes()

    packages = advance_cp2k_convergence_campaign(
        completed_campaign_directory=source,
        next_campaign_directory=destination,
        base_config=base_config,
    )

    assert len(packages) == 10

    manifest = load_convergence_manifest(
        destination / "manifest.json"
    )

    assert manifest.parameter.value == "relative_cutoff"
    assert len(manifest.candidates) == 10

    for index, candidate in enumerate(manifest.candidates, start=1):
        parsed = parse_cp2k_input(
            destination
            / candidate.directory
            / candidate.input_file
        )

        assert parsed.cutoff_ry == 400.0
        assert parsed.relative_cutoff_ry == index * 10.0

    provenance = json.loads(
        (destination / "stage-provenance.json").read_text()
    )

    assert provenance["selected_cutoff_ry"] == 400.0

    assert (
        source / "convergence-report.json"
    ).read_bytes() == original_report


def test_atomic_publication_rejects_existing_destination(tmp_path):
    from nsdw.workflows.convergence.cp2k_staged import (
        _publish_directory_no_replace,
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "generated.txt").write_text("new", encoding="utf-8")

    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "important.txt").write_text(
        "preserve", encoding="utf-8"
    )

    with pytest.raises(
        CP2KConvergenceStageError,
        match="Destination already exists",
    ):
        _publish_directory_no_replace(staging, destination)

    assert (staging / "generated.txt").read_text() == "new"
    assert (destination / "important.txt").read_text() == "preserve"


def test_advance_cleans_staging_after_generation_failure(
    tmp_path,
    base_config,
    monkeypatch,
):
    """Generation failure must not publish an incomplete campaign."""
    from nsdw.workflows.convergence import cp2k_generation

    source = Path(
        "outputs/archer2-validation/sio2-pbe-cutoff-extended"
    )

    if not source.is_dir():
        pytest.skip("Historical ARCHER2 evidence is not available.")

    destination = tmp_path / "relative-cutoff"

    def fail_generation(*, output_directory, **kwargs):
        output_directory = Path(output_directory)
        (output_directory / "partial.txt").write_text(
            "incomplete", encoding="utf-8"
        )
        raise RuntimeError("Simulated generation failure")

    monkeypatch.setattr(
        cp2k_generation,
        "generate_cp2k_convergence_study",
        fail_generation,
    )

    with pytest.raises(RuntimeError, match="Simulated generation failure"):
        advance_cp2k_convergence_campaign(
            completed_campaign_directory=source,
            next_campaign_directory=destination,
            base_config=base_config,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".relative-cutoff.staging-*"))


@pytest.fixture
def synthetic_cutoff_campaign(tmp_path, base_config):
    """Build a self-contained completed CP2K cutoff campaign."""
    from pymatgen.core import Lattice, Structure

    from nsdw.models.quantity import Quantity
    from nsdw.workflows.convergence.cp2k_generation import (
        generate_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.cp2k_workflow import (
        analyse_and_report_existing_cp2k_convergence_campaign,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )
    from nsdw.workflows.convergence.manifest import (
        load_convergence_manifest,
    )

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    source = tmp_path / "cutoff"

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label=f"{value}-Ry",
                order=index,
                value=Quantity(value=float(value), unit="Ry"),
            )
            for index, value in enumerate((400, 560, 700))
        ],
    )

    generate_cp2k_convergence_study(
        structure=structure,
        base_config=base_config,
        study=study,
        output_directory=source,
    )

    manifest = load_convergence_manifest(
        source / "manifest.json"
    )

    fixture = Path(
        "tests/calculators/cp2k/fixtures/energy_complete.out"
    )

    output_template = fixture.read_text(encoding="utf-8")

    assert output_template.count("sio2-bulk") == 1

    for candidate in manifest.candidates:
        candidate_dir = source / candidate.directory

        project_name = (
            f"{base_config.project_name}-{candidate.label}"
        )

        output_text = output_template.replace(
            "sio2-bulk",
            project_name,
        )

        output_path = (
            candidate_dir
            / Path(candidate.input_file).with_suffix(".out").name
        )

        output_path.write_text(
            output_text,
            encoding="utf-8",
        )

    report = analyse_and_report_existing_cp2k_convergence_campaign(
        campaign_directory=source,
    )

    assert report.energy_tolerance_satisfied
    assert report.selected_candidate_label == "400-Ry"

    return source


def test_advance_self_contained_campaign(
    tmp_path,
    synthetic_cutoff_campaign,
):
    """Exercise the full stage transition without ARCHER2 evidence."""
    import json

    from nsdw.calculators.cp2k.input_parser import parse_cp2k_input
    from nsdw.workflows.convergence.manifest import (
        load_convergence_manifest,
    )

    source = synthetic_cutoff_campaign
    destination = tmp_path / "relative-cutoff"

    report_before = (
        source / "convergence-report.json"
    ).read_bytes()

    packages = advance_cp2k_convergence_campaign(
        completed_campaign_directory=source,
        next_campaign_directory=destination,
    )

    assert len(packages) == 10

    manifest = load_convergence_manifest(
        destination / "manifest.json"
    )

    assert manifest.parameter.value == "relative_cutoff"

    for index, candidate in enumerate(manifest.candidates, start=1):
        parsed = parse_cp2k_input(
            destination
            / candidate.directory
            / candidate.input_file
        )

        assert parsed.cutoff_ry == 400.0
        assert parsed.relative_cutoff_ry == index * 10.0

    provenance = json.loads(
        (destination / "stage-provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert provenance["selected_cutoff_ry"] == 400.0

    assert (
        source / "convergence-report.json"
    ).read_bytes() == report_before


def test_self_contained_generation_failure_cleanup(
    tmp_path,
    synthetic_cutoff_campaign,
    monkeypatch,
):
    from nsdw.workflows.convergence import cp2k_generation

    destination = tmp_path / "failed-relative-cutoff"

    def fail_generation(*, output_directory, **kwargs):
        output_directory = Path(output_directory)
        (output_directory / "partial.txt").write_text(
            "incomplete",
            encoding="utf-8",
        )
        raise RuntimeError("Simulated generation failure")

    monkeypatch.setattr(
        cp2k_generation,
        "generate_cp2k_convergence_study",
        fail_generation,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated generation failure",
    ):
        advance_cp2k_convergence_campaign(
            completed_campaign_directory=synthetic_cutoff_campaign,
            next_campaign_directory=destination,
        )

    assert not destination.exists()
    assert not list(
        tmp_path.glob(".failed-relative-cutoff.staging-*")
    )
