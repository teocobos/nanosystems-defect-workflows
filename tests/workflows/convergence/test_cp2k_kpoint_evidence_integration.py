"""Integration tests for CP2K mesh-convergence evidence validation.

The shared CP2K output fixture is used to test validation mechanics.
It does not establish physical convergence for the synthetic meshes.
"""

import json
import shutil
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
from nsdw.calculators.cp2k.parser import parse_cp2k_output
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
def kpoint_campaign(tmp_path):
    root = tmp_path
    campaign = root / "campaign"
    campaign.mkdir()

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    structure.to(filename=campaign / "structure.json", fmt="json")

    output_fixture = (
        Path("tests/calculators/cp2k/fixtures")
        / "energy_complete.out"
    )
    assert output_fixture.is_file()

    parsed_output = parse_cp2k_output(output_fixture)
    assert parsed_output.normal_termination
    assert parsed_output.project_name is not None
    assert parsed_output.energy.total_energy_hartree is not None

    energy_ev = (
        parsed_output.energy.total_energy_hartree * HARTREE_TO_EV
    )
    energy_per_atom = energy_ev / len(structure)

    meshes = ((1, 1, 1), (2, 2, 2), (3, 3, 3))
    manifest_candidates = []
    report_candidates = []

    for order, mesh in enumerate(meshes):
        label = "x".join(map(str, mesh))
        candidate_dir = campaign / label
        candidate_dir.mkdir()

        input_text = f"""&GLOBAL
  PROJECT {parsed_output.project_name}
  RUN_TYPE ENERGY
&END GLOBAL
&FORCE_EVAL
  &DFT
    CHARGE 0
    MULTIPLICITY 1
    &MGRID
      CUTOFF 400
      REL_CUTOFF 30
    &END MGRID
    &SCF
      EPS_SCF 1.0E-6
      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION
    &END SCF
    &KPOINTS
      SCHEME MONKHORST-PACK {mesh[0]} {mesh[1]} {mesh[2]}
    &END KPOINTS
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
            input_text, encoding="utf-8"
        )
        (candidate_dir / "structure.xyz").write_text(
            "2\nTest\n"
            "Si 0.0 0.0 0.0\n"
            "O 1.25 1.25 1.25\n",
            encoding="utf-8",
        )
        shutil.copy2(
            output_fixture,
            candidate_dir / "calculation.out",
        )

        manifest_candidates.append(
            ConvergenceManifestCandidate(
                label=label,
                order=order,
                value=mesh,
                directory=label,
                input_file="calculation.inp",
                coordinate_file="structure.xyz",
            )
        )

        report_candidates.append({
            "label": label,
            "order": order,
            "value": list(mesh),
            "total_energy_ev": energy_ev,
            "energy_ev_per_atom": energy_per_atom,
            "adjacent_energy_difference_ev_per_atom": (
                0.0 if order < len(meshes) - 1 else None
            ),
            "reference_energy_difference_ev_per_atom": 0.0,
            "maximum_higher_cost_energy_difference_ev_per_atom": (
                0.0 if order < len(meshes) - 1 else None
            ),
            "tail_stable": (
                True if order < len(meshes) - 1 else None
            ),
            "selected": order == 0,
        })

    write_convergence_manifest(
        manifest=ConvergenceStudyManifest(
            calculator="cp2k",
            parameter=ConvergenceParameter.KPOINTS,
            criterion=ConvergenceCriterion(
                energy_tolerance_ev_per_atom=0.001
            ),
            candidates=tuple(manifest_candidates),
        ),
        path=campaign / "manifest.json",
    )

    report = {
        "parameter": "kpoints",
        "energy_tolerance_ev_per_atom": 0.001,
        "structure_hash": calculate_structure_hash(structure),
        "n_atoms": len(structure),
        "reference_candidate_label": "3x3x3",
        "selected_candidate_label": "1x1x1",
        "energy_tolerance_satisfied": True,
        "candidates": report_candidates,
    }

    report_path = campaign / "convergence-report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    return root, campaign, report_path


def validate(campaign):
    root, _, report_path = campaign
    return validate_cp2k_calculation_evidence(
        project_root=root,
        report_path=report_path,
        expected_parameter="kpoints",
        expected_value=(1, 1, 1),
    )


def replace_input(campaign, label, old, new):
    _, directory, _ = campaign
    path = directory / label / "calculation.inp"
    original = path.read_text(encoding="utf-8")
    assert original.count(old) == 1
    path.write_text(
        original.replace(old, new, 1),
        encoding="utf-8",
    )


def test_accepts_structured_monkhorst_pack_campaign(kpoint_campaign):
    report = validate(kpoint_campaign)
    assert report["selected_candidate_label"] == "1x1x1"


def test_rejects_wrong_candidate_mesh(kpoint_campaign):
    replace_input(
        kpoint_campaign,
        "2x2x2",
        "SCHEME MONKHORST-PACK 2 2 2",
        "SCHEME MONKHORST-PACK 2 2 3",
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Incorrect k-point mesh",
    ):
        validate(kpoint_campaign)


def test_rejects_gamma_in_mesh_campaign(kpoint_campaign):
    replace_input(
        kpoint_campaign,
        "2x2x2",
        "SCHEME MONKHORST-PACK 2 2 2",
        "SCHEME GAMMA",
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Unsupported k-point sampling",
    ):
        validate(kpoint_campaign)


def test_rejects_macdonald_in_mesh_campaign(kpoint_campaign):
    replace_input(
        kpoint_campaign,
        "2x2x2",
        "SCHEME MONKHORST-PACK 2 2 2",
        "SCHEME MACDONALD 2 2 2 0.25 0.25 0.25",
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Unsupported k-point sampling",
    ):
        validate(kpoint_campaign)


def test_rejects_ot_solver_for_explicit_mesh(kpoint_campaign):
    replace_input(
        kpoint_campaign,
        "2x2x2",
        """&DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION""",
        """&OT
        MINIMIZER DIIS
      &END OT""",
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Explicit k-point evidence requires DIAGONALIZATION|Non-varied",
    ):
        validate(kpoint_campaign)


def test_rejects_nonvaried_cutoff_change(kpoint_campaign):
    replace_input(
        kpoint_campaign,
        "2x2x2",
        "CUTOFF 400",
        "CUTOFF 500",
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Non-varied CP2K settings",
    ):
        validate(kpoint_campaign)
