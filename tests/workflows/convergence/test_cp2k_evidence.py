"""Adversarial tests for independent CP2K convergence evidence."""

import json
import shutil
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
from nsdw.calculators.cp2k.parser import parse_cp2k_output
from nsdw.models.quantity import Quantity
from nsdw.structures.result import calculate_structure_hash
from nsdw.workflows.convergence.cp2k_evidence import (
    validate_cp2k_calculation_evidence,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)
from nsdw.workflows.convergence.manifest import (
    ConvergenceManifestCandidate,
    ConvergenceStudyManifest,
    write_convergence_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCriterion,
    ConvergenceParameter,
)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path
    directory = root / "campaign"
    directory.mkdir()

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    structure.to(
        filename=directory / "structure.json",
        fmt="json",
    )

    fixture = (
        Path("tests/calculators/cp2k/fixtures")
        / "energy_complete.out"
    )
    assert fixture.is_file()

    parsed = parse_cp2k_output(fixture)
    assert parsed.normal_termination
    assert parsed.energy.total_energy_hartree is not None
    assert parsed.project_name is not None

    energy = parsed.energy.total_energy_hartree * HARTREE_TO_EV

    labels = ("400-Ry", "560-Ry", "700-Ry")
    values = (400.0, 560.0, 700.0)

    manifest_candidates = []
    report_candidates = []

    for index, (label, value) in enumerate(
        zip(labels, values, strict=True)
    ):
        candidate_dir = directory / label
        candidate_dir.mkdir()

        input_text = f"""&GLOBAL
  PROJECT {parsed.project_name}
  RUN_TYPE ENERGY
&END GLOBAL
&FORCE_EVAL
  &DFT
    CHARGE 0
    MULTIPLICITY 1
    &MGRID
      CUTOFF {value}
      REL_CUTOFF 60
    &END MGRID
    &SCF
      EPS_SCF 1.0E-6
      &OT
        MINIMIZER DIIS
      &END OT
    &END SCF
  &END DFT
  &SUBSYS
    &CELL
      A 5.0 0.0 0.0
      B 0.0 5.0 0.0
      C 0.0 0.0 5.0
      PERIODIC XYZ
    &END CELL
    &TOPOLOGY
      COORD_FILE_NAME structure.xyz
      COORD_FILE_FORMAT XYZ
    &END TOPOLOGY
  &END SUBSYS
&END FORCE_EVAL
"""

        (candidate_dir / "calculation.inp").write_text(
            input_text,
            encoding="utf-8",
        )

        (candidate_dir / "structure.xyz").write_text(
            "2\nTest\n"
            "Si 0.0 0.0 0.0\n"
            "O 1.25 1.25 1.25\n",
            encoding="utf-8",
        )

        shutil.copy2(
            fixture,
            candidate_dir / "calculation.out",
        )

        manifest_candidates.append(
            ConvergenceManifestCandidate(
                label=label,
                order=index,
                value=Quantity(value=value, unit="Ry"),
                directory=label,
                input_file="calculation.inp",
                coordinate_file="structure.xyz",
            )
        )

        report_candidates.append({
            "label": label,
            "order": index,
            "value": {"value": value, "unit": "Ry"},
            "total_energy_ev": energy,
            "energy_ev_per_atom": energy / len(structure),
            "adjacent_energy_difference_ev_per_atom": (
                0.0 if index < 2 else None
            ),
            "reference_energy_difference_ev_per_atom": 0.0,
            "maximum_higher_cost_energy_difference_ev_per_atom": (
                0.0 if index < 2 else None
            ),
            "tail_stable": True if index < 2 else None,
            "selected": index == 0,
        })

    manifest = ConvergenceStudyManifest(
        calculator="cp2k",
        parameter=ConvergenceParameter.CUTOFF,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=0.001
        ),
        candidates=tuple(manifest_candidates),
    )

    write_convergence_manifest(
        manifest=manifest,
        path=directory / "manifest.json",
    )

    report = {
        "parameter": "cutoff",
        "energy_tolerance_ev_per_atom": 0.001,
        "structure_hash": calculate_structure_hash(structure),
        "n_atoms": len(structure),
        "reference_candidate_label": labels[-1],
        "selected_candidate_label": labels[0],
        "energy_tolerance_satisfied": True,
        "candidates": report_candidates,
    }

    report_path = directory / "convergence-report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    return root, directory, report_path


def validate(campaign):
    root, _, report_path = campaign

    return validate_cp2k_calculation_evidence(
        project_root=root,
        report_path=report_path,
        expected_parameter="cutoff",
        expected_value=400.0,
    )


def test_accepts_consistent_calculation_evidence(campaign):
    report = validate(campaign)
    assert report["selected_candidate_label"] == "400-Ry"


def test_rejects_modified_report_energy(campaign):
    _, _, report_path = campaign
    report = json.loads(report_path.read_text())
    report["candidates"][0]["total_energy_ev"] += 1.0
    report_path.write_text(json.dumps(report))

    with pytest.raises(ConvergenceEvidenceValidationError):
        validate(campaign)


def test_rejects_input_cutoff_mismatch(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.inp"
    source = path.read_text()
    assert "CUTOFF 400.0" in source
    path.write_text(
        source.replace("CUTOFF 400.0", "CUTOFF 450.0")
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Incorrect cutoff",
    ):
        validate(campaign)


def test_rejects_missing_output(campaign):
    _, directory, _ = campaign
    (directory / "400-Ry/calculation.out").unlink()

    with pytest.raises(ConvergenceEvidenceValidationError):
        validate(campaign)


def test_rejects_ambiguous_output(campaign):
    _, directory, _ = campaign
    candidate = directory / "400-Ry"
    shutil.copy2(
        candidate / "calculation.out",
        candidate / "alternative.out",
    )

    # An unrelated .out file is not currently an ambiguity:
    # discovery considers only the derived and legacy names.
    assert validate(campaign)

    # Use a different input basename so both recognised output
    # conventions can coexist.
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidates"][0]["input_file"] = "alternative.inp"
    manifest_path.write_text(json.dumps(manifest))

    shutil.copy2(
        candidate / "calculation.inp",
        candidate / "alternative.inp",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="unambiguous",
    ):
        validate(campaign)


def test_rejects_unterminated_output(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.out"
    text = path.read_text()
    assert "PROGRAM ENDED AT" in text
    path.write_text(
        text.replace("PROGRAM ENDED AT", "PROGRAM INTERRUPTED AT")
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="did not complete",
    ):
        validate(campaign)


def test_rejects_nonconverged_scf(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.out"
    text = path.read_text()

    import re

    updated, count = re.subn(
        r"SCF run converged in\s+\d+\s+steps",
        "SCF run NOT converged",
        text,
    )
    assert count > 0
    path.write_text(updated)

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="did not complete",
    ):
        validate(campaign)


def test_rejects_nonvaried_setting_mismatch(campaign):
    _, directory, _ = campaign
    path = directory / "560-Ry/calculation.inp"
    source = path.read_text()
    assert "REL_CUTOFF 60" in source
    path.write_text(
        source.replace("REL_CUTOFF 60", "REL_CUTOFF 80")
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Non-varied",
    ):
        validate(campaign)


def test_rejects_candidate_input_symlink_escape(campaign, tmp_path):
    _, directory, _ = campaign
    original = directory / "400-Ry/calculation.inp"
    external = tmp_path.parent / (
        tmp_path.name + "-external-cp2k.inp"
    )
    shutil.copy2(original, external)
    original.unlink()
    original.symlink_to(external)

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="escapes",
    ):
        validate(campaign)


# Step 8C.3D.3: Combined calculation + structure validation.


def test_combined_rejects_modified_xyz_coordinates(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/structure.xyz"
    original = path.read_text(encoding="utf-8")

    assert "O 1.25 1.25 1.25" in original

    path.write_text(
        original.replace(
            "O 1.25 1.25 1.25",
            "O 1.35 1.25 1.25",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="XYZ coordinate mismatch",
    ):
        validate(campaign)


def test_combined_rejects_modified_xyz_species(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/structure.xyz"
    original = path.read_text(encoding="utf-8")

    assert "O 1.25 1.25 1.25" in original

    path.write_text(
        original.replace(
            "O 1.25 1.25 1.25",
            "Si 1.25 1.25 1.25",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="XYZ species mismatch",
    ):
        validate(campaign)


def test_combined_rejects_modified_cell(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.inp"
    original = path.read_text(encoding="utf-8")

    assert "A 5.0 0.0 0.0" in original

    path.write_text(
        original.replace(
            "A 5.0 0.0 0.0",
            "A 6.0 0.0 0.0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="cell vector mismatch",
    ):
        validate(campaign)


def test_combined_rejects_wrong_xyz_reference(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.inp"
    original = path.read_text(encoding="utf-8")

    assert "COORD_FILE_NAME structure.xyz" in original

    path.write_text(
        original.replace(
            "COORD_FILE_NAME structure.xyz",
            "COORD_FILE_NAME other.xyz",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="coordinate reference",
    ):
        validate(campaign)


@pytest.mark.parametrize(
    "directive",
    [
        "@INCLUDE extra.inp",
        "@SET CUTOFF 100",
        "@IF ${ENABLE_EXTRA}",
    ],
)
def test_combined_rejects_preprocessor_directives(
    campaign,
    directive,
):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.inp"

    path.write_text(
        directive + "\n" + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="preprocessing",
    ):
        validate(campaign)


def test_rejects_later_scf_cycle_without_energy(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.out"
    text = path.read_text(encoding="utf-8")

    marker = "PROGRAM ENDED AT"
    assert marker in text

    text = text.replace(
        marker,
        "*** SCF run converged in 2 steps ***\n"
        + marker,
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Ambiguous CP2K single-point event sequence",
    ):
        validate(campaign)


def test_rejects_duplicate_final_energy(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.out"
    text = path.read_text(encoding="utf-8")

    energy_line = next(
        line
        for line in text.splitlines()
        if "ENERGY| Total FORCE_EVAL" in line
    )

    text = text.replace(
        energy_line,
        energy_line + "\n" + energy_line,
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Ambiguous CP2K single-point event sequence",
    ):
        validate(campaign)


def test_rejects_energy_before_scf_convergence(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.out"
    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    scf_index = next(
        i for i, line in enumerate(lines)
        if "SCF run converged in" in line
    )
    energy_index = next(
        i for i, line in enumerate(lines)
        if "ENERGY| Total FORCE_EVAL" in line
    )

    lines[scf_index], lines[energy_index] = (
        lines[energy_index],
        lines[scf_index],
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Ambiguous CP2K single-point event sequence",
    ):
        validate(campaign)


def test_rejects_duplicate_termination_marker(campaign):
    _, directory, _ = campaign
    path = directory / "400-Ry/calculation.out"
    text = path.read_text(encoding="utf-8")

    assert "PROGRAM ENDED AT" in text
    text = text.replace(
        "PROGRAM ENDED AT",
        "PROGRAM ENDED AT\nPROGRAM ENDED AT",
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Ambiguous CP2K single-point event sequence",
    ):
        validate(campaign)


@pytest.mark.parametrize(
    ("input_old", "input_new", "output_old", "output_new"),
    [
        ("CHARGE 0", "CHARGE 1", None, None),
        ("MULTIPLICITY 1", "MULTIPLICITY 2", None, None),
        ("RUN_TYPE ENERGY", "RUN_TYPE GEO_OPT", None, None),
        ("CHARGE 0", "", None, None),
        ("MULTIPLICITY 1", "", None, None),
        (None, None, "DFT| Charge", "DFT| MissingCharge"),
    ],
)
def test_rejects_input_output_identity_mismatch(
    campaign,
    input_old,
    input_new,
    output_old,
    output_new,
):
    _, directory, _ = campaign
    candidate = directory / "400-Ry"

    if input_old is not None:
        path = candidate / "calculation.inp"
        text = path.read_text(encoding="utf-8")

        assert input_old in text

        path.write_text(
            text.replace(input_old, input_new, 1),
            encoding="utf-8",
        )

    if output_old is not None:
        path = candidate / "calculation.out"
        text = path.read_text(encoding="utf-8")

        assert output_old in text

        path.write_text(
            text.replace(output_old, output_new, 1),
            encoding="utf-8",
        )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
    ):
        validate(campaign)
