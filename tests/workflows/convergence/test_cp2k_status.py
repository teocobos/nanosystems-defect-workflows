"""Tests for read-only CP2K campaign status inspection."""

from pathlib import Path

import pytest

from nsdw.workflows.convergence.cp2k_status import (
    CP2KCandidateState,
    inspect_cp2k_convergence_campaign,
)


def test_generated_relative_cutoff_campaign_is_pending():
    campaign = Path(
        "outputs/archer2-validation/sio2-pbe-relative-cutoff"
    )

    if not campaign.is_dir():
        pytest.skip("Local ARCHER2 campaign not available.")

    result = inspect_cp2k_convergence_campaign(campaign)

    assert len(result.candidates) == 10
    assert all(
        candidate.state == CP2KCandidateState.PENDING
        for candidate in result.candidates
    )
    assert not result.all_completed


def test_missing_campaign_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_cp2k_convergence_campaign(tmp_path)


def test_rejects_output_symlink_escape(tmp_path):
    """An output symlink outside the campaign must not appear pending."""
    import json

    campaign = tmp_path / "campaign"
    candidate_dir = campaign / "400-Ry"
    candidate_dir.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "calculator": "cp2k",
        "parameter": "cutoff",
        "criterion": {
            "energy_tolerance_ev_per_atom": 0.001
        },
        "candidates": [
            {
                "label": "400-Ry",
                "order": 0,
                "value": {"value": 400.0, "unit": "Ry"},
                "directory": "400-Ry",
                "input_file": "calculation.inp",
                "coordinate_file": "structure.xyz",
            }
        ],
    }

    (campaign / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    (candidate_dir / "calculation.inp").write_text(
        (
            "&GLOBAL\n"
            "  PROJECT sio2-400-Ry\n"
            "  RUN_TYPE ENERGY\n"
            "&END GLOBAL\n"
            "&FORCE_EVAL\n"
            "  &DFT\n"
            "    &MGRID\n"
            "      CUTOFF 400\n"
            "      REL_CUTOFF 60\n"
            "    &END MGRID\n"
            "  &END DFT\n"
            "&END FORCE_EVAL\n"
        ),
        encoding="utf-8",
    )

    external = tmp_path / "external.out"
    external.write_text("external data", encoding="utf-8")

    (candidate_dir / "calculation.out").symlink_to(external)

    result = inspect_cp2k_convergence_campaign(campaign)

    assert len(result.candidates) == 1
    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "escapes" in result.candidates[0].reason


def _copy_relative_cutoff_candidate(tmp_path):
    """Create an isolated copy of one real generated candidate."""
    import json
    import shutil

    source = Path(
        "outputs/archer2-validation/sio2-pbe-relative-cutoff"
    )

    if not source.is_dir():
        pytest.skip("Generated relative-cutoff campaign unavailable.")

    destination = tmp_path / "campaign"
    destination.mkdir()

    shutil.copy2(
        source / "structure.json",
        destination / "structure.json",
    )

    shutil.copy2(
        source / "base-config.json",
        destination / "base-config.json",
    )

    manifest = json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )

    candidate = manifest["candidates"][0]
    manifest["candidates"] = [candidate]

    (destination / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    candidate_directory = Path(candidate["directory"])

    shutil.copytree(
        source / candidate_directory,
        destination / candidate_directory,
    )

    return destination, candidate


def test_rejects_incorrect_relative_cutoff(tmp_path):
    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    input_path = (
        campaign
        / candidate["directory"]
        / candidate["input_file"]
    )

    source = input_path.read_text(encoding="utf-8")

    import re

    modified, count = re.subn(
        r"(?im)^(\s*REL_CUTOFF\s+)10(?:\.0+)?(\s*(?:!.*)?)$",
        r"\g<1>999\2",
        source,
        count=1,
    )

    assert count == 1, "Expected REL_CUTOFF 10 in test input."

    input_path.write_text(modified, encoding="utf-8")

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "relative_cutoff_ry" in result.candidates[0].reason


def test_rejects_incorrect_project_name(tmp_path):
    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    input_path = (
        campaign
        / candidate["directory"]
        / candidate["input_file"]
    )

    source = input_path.read_text(encoding="utf-8")

    import re

    modified, count = re.subn(
        r"(?im)^(\s*PROJECT\s+)\S+",
        r"\g<1>incorrect-project",
        source,
        count=1,
    )

    assert count == 1, "Expected PROJECT declaration."

    input_path.write_text(modified, encoding="utf-8")

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "project name" in result.candidates[0].reason.lower()


def test_rejects_missing_candidate_input(tmp_path):
    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    input_path = (
        campaign
        / candidate["directory"]
        / candidate["input_file"]
    )

    input_path.unlink()

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "missing" in result.candidates[0].reason.lower()


def test_rejects_modified_atomic_coordinates(tmp_path):
    """Candidate coordinates must agree with the canonical structure."""
    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    xyz_path = (
        campaign
        / candidate["directory"]
        / candidate["coordinate_file"]
    )

    lines = xyz_path.read_text(encoding="utf-8").splitlines()

    # XYZ format: atom count, comment, then atomic coordinates.
    assert len(lines) >= 3

    tokens = lines[2].split()
    assert len(tokens) >= 4

    tokens[1] = f"{float(tokens[1]) + 0.5:.10f}"
    lines[2] = " ".join(tokens)

    xyz_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert not result.all_completed
    assert "structure" in result.candidates[0].reason.lower() or (
        "coordinate" in result.candidates[0].reason.lower()
        or "xyz" in result.candidates[0].reason.lower()
    )


def _modify_candidate_input(tmp_path, pattern, replacement):
    """Modify an isolated generated candidate without touching real data."""
    import re

    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    input_path = (
        campaign / candidate["directory"] / candidate["input_file"]
    )

    source = input_path.read_text(encoding="utf-8")

    modified, count = re.subn(
        pattern,
        replacement,
        source,
        count=1,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    assert count == 1, f"Expected exactly one match: {pattern}"

    input_path.write_text(modified, encoding="utf-8")

    return campaign


def test_rejects_changed_fixed_cutoff(tmp_path):
    campaign = _modify_candidate_input(
        tmp_path,
        r"^(\s*CUTOFF\s+)400(?:\.0+)?(\s*(?:!.*)?)$",
        r"\g<1>500\2",
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "cutoff" in result.candidates[0].reason.lower()


def test_rejects_changed_scf_tolerance(tmp_path):
    campaign = _modify_candidate_input(
        tmp_path,
        r"^(\s*EPS_SCF\s+)1(?:\.0+)?[Ee]-0?6(\s*(?:!.*)?)$",
        r"\g<1>1.0E-4\2",
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "eps_scf" in result.candidates[0].reason.lower()


def test_rejects_changed_xc_functional(tmp_path):
    campaign = _modify_candidate_input(
        tmp_path,
        r"^(\s*&XC_FUNCTIONAL\s+)PBE(\s*)$",
        r"\g<1>BLYP\2",
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "functional" in result.candidates[0].reason.lower()




def test_rejects_changed_basis_set(tmp_path):
    campaign = _modify_candidate_input(
        tmp_path,
        r"^(\s*BASIS_SET\s+)DZVP-MOLOPT-GTH-q4(\s*)$",
        r"\g<1>TZVP-MOLOPT-GTH-q4\2",
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "basis" in result.candidates[0].reason.lower()


def test_rejects_changed_pseudopotential(tmp_path):
    campaign = _modify_candidate_input(
        tmp_path,
        r"^(\s*POTENTIAL\s+)GTH-PBE-q4(\s*)$",
        r"\g<1>GTH-PBE-q6\2",
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "potential" in result.candidates[0].reason.lower()


def test_missing_base_config_is_invalid(tmp_path):
    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    (campaign / "base-config.json").unlink()

    result = inspect_cp2k_convergence_campaign(campaign)

    assert len(result.candidates) == 1
    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "base-config.json" in result.candidates[0].reason
    assert not result.all_completed


def test_rejects_completed_output_from_different_project(tmp_path):
    """A converged output from another project is not valid evidence."""
    import shutil

    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    directory = campaign / candidate["directory"]
    input_path = directory / candidate["input_file"]
    output_path = input_path.with_suffix(".out")

    fixture = Path(
        "tests/calculators/cp2k/fixtures/energy_complete.out"
    )

    shutil.copy2(fixture, output_path)

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INVALID
    assert "project_name" in result.candidates[0].reason
    assert not result.all_completed


def test_classifies_truncated_output_as_incomplete(tmp_path):
    """An unfinished CP2K output must not be considered completed."""
    import shutil

    campaign, candidate = _copy_relative_cutoff_candidate(tmp_path)

    directory = campaign / candidate["directory"]
    input_path = directory / candidate["input_file"]
    output_path = input_path.with_suffix(".out")

    fixture = Path(
        "tests/calculators/cp2k/fixtures/energy_complete.out"
    )

    shutil.copy2(fixture, output_path)

    text = output_path.read_text(encoding="utf-8")

    # Remove the final termination record to simulate an interrupted run.
    import re

    modified, count = re.subn(
        r"(?m)^.*PROGRAM ENDED AT.*(?:\n|$)",
        "",
        text,
    )

    assert count >= 1, "Expected CP2K termination record."

    output_path.write_text(modified, encoding="utf-8")

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[0].state == CP2KCandidateState.INCOMPLETE
    assert not result.all_completed


def test_completed_archer2_cutoff_campaign(tmp_path):
    """Inspect genuine completed CP2K outputs in an isolated campaign."""
    import json
    import shutil

    from nsdw.calculators.cp2k.generation_models import CP2KInputConfig
    from nsdw.workflows.convergence.cp2k_staged import (
        validate_cutoff_campaign_configuration,
    )

    source = Path(
        "outputs/archer2-validation/sio2-pbe-cutoff-extended"
    )

    reference = Path(
        "outputs/archer2-validation/sio2-pbe-relative-cutoff"
    )

    if not source.is_dir() or not reference.is_dir():
        pytest.skip("Local ARCHER2 validation campaigns unavailable.")

    campaign = tmp_path / "completed-cutoff"
    shutil.copytree(source, campaign)

    config = CP2KInputConfig.model_validate_json(
        (reference / "base-config.json").read_text(encoding="utf-8")
    )

    # CUTOFF is varied by the manifest, so its reference value is
    # not used as a fixed scientific constraint.
    (campaign / "base-config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )

    validate_cutoff_campaign_configuration(
        campaign_directory=campaign,
        base_config=config,
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert len(result.candidates) == 4, [
        (candidate.label, candidate.state, candidate.reason)
        for candidate in result.candidates
    ]

    assert all(
        candidate.state == CP2KCandidateState.COMPLETED
        for candidate in result.candidates
    ), [
        (candidate.label, candidate.state, candidate.reason)
        for candidate in result.candidates
    ]

    assert result.all_completed


def _generated_basis_campaign(tmp_path):
    from pymatgen.core import Lattice, Structure

    from nsdw.calculators.cp2k.generation_models import (
        CP2KBasisPotentialConfig,
        CP2KInputConfig,
        CP2KKindConfig,
    )
    from nsdw.workflows.convergence.cp2k_generation import (
        generate_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    reference = CP2KBasisPotentialConfig(
        basis_set_file="BASIS_MOLOPT",
        potential_file="GTH_POTENTIALS",
        kinds=(
            CP2KKindConfig(
                element="O",
                basis_set="DZVP-MOLOPT-GTH-q6",
                potential="GTH-PBE-q6",
            ),
        ),
    )

    alternative = reference.model_copy(
        update={
            "kinds": (
                CP2KKindConfig(
                    element="O",
                    basis_set="TZVP-MOLOPT-GTH-q6",
                    potential="GTH-PBE-q6",
                ),
            ),
        }
    )

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.BASIS,
        candidates=[
            ConvergenceCandidate(
                label="dzvp",
                order=0,
                value="dzvp",
            ),
            ConvergenceCandidate(
                label="tzvp",
                order=1,
                value="tzvp",
            ),
        ],
    )

    campaign = tmp_path / "basis-campaign"

    generate_cp2k_convergence_study(
        structure=Structure(
            Lattice.cubic(5.0),
            ["O"],
            [[0, 0, 0]],
        ),
        base_config=CP2KInputConfig(
            project_name="basis-test",
            run_type="ENERGY",
            basis_potential=reference,
        ),
        study=study,
        output_directory=campaign,
        basis_candidates={
            "dzvp": reference,
            "tzvp": alternative,
        },
    )

    return campaign


def test_generated_basis_campaign_candidates_are_pending(tmp_path):
    campaign = _generated_basis_campaign(tmp_path)

    result = inspect_cp2k_convergence_campaign(campaign)

    assert len(result.candidates) == 2
    assert all(
        candidate.state == CP2KCandidateState.PENDING
        for candidate in result.candidates
    )


def test_basis_campaign_rejects_missing_candidate_provenance(tmp_path):
    import json

    campaign = _generated_basis_campaign(tmp_path)
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    manifest["candidates"][1].pop("basis_potential")
    manifest_path.write_text(json.dumps(manifest))

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[1].state == CP2KCandidateState.INVALID
    assert "provenance" in result.candidates[1].reason.lower()


def test_basis_campaign_rejects_unexpected_basis_assignment(tmp_path):
    campaign = _generated_basis_campaign(tmp_path)

    input_path = next((campaign / "tzvp").glob("*.inp"))
    source = input_path.read_text()

    assert "TZVP-MOLOPT-GTH-q6" in source

    input_path.write_text(
        source.replace(
            "TZVP-MOLOPT-GTH-q6",
            "DZVP-MOLOPT-GTH-q6",
        )
    )

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[1].state == CP2KCandidateState.INVALID
    assert "basis" in result.candidates[1].reason.lower()


def test_basis_campaign_rejects_changed_pseudopotential_provenance(tmp_path):
    import json

    campaign = _generated_basis_campaign(tmp_path)
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    manifest["candidates"][1]["basis_potential"]["kinds"][0][
        "potential"
    ] = "GTH-BLYP-q6"

    manifest_path.write_text(json.dumps(manifest))

    result = inspect_cp2k_convergence_campaign(campaign)

    assert result.candidates[1].state == CP2KCandidateState.INVALID
    assert "pseudopotential" in result.candidates[1].reason.lower()
