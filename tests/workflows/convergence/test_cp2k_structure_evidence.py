"""Tests for CP2K structure provenance validation."""

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.workflows.convergence.cp2k_structure_evidence import (
    validate_cp2k_structure_evidence,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


@pytest.fixture
def geometry(tmp_path):
    campaign = tmp_path / "campaign"
    candidate = campaign / "400-Ry"
    candidate.mkdir(parents=True)

    structure = Structure(
        Lattice.cubic(5.0),
        ["Si", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    structure.to(
        filename=campaign / "structure.json",
        fmt="json",
    )

    (candidate / "structure.xyz").write_text(
        "2\nTest\n"
        "Si 0.0 0.0 0.0\n"
        "O 1.25 1.25 1.25\n",
        encoding="utf-8",
    )

    (candidate / "calculation.inp").write_text(
        """&GLOBAL
 PROJECT test
 RUN_TYPE ENERGY
&END GLOBAL
&FORCE_EVAL
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
""",
        encoding="utf-8",
    )

    return tmp_path, campaign, candidate


def check(geometry):
    root, campaign, candidate = geometry

    validate_cp2k_structure_evidence(
        project_root=root,
        campaign_directory=campaign,
        candidate_directory=candidate,
        input_filename="calculation.inp",
        coordinate_filename="structure.xyz",
    )


def test_accepts_matching_geometry(geometry):
    check(geometry)


@pytest.mark.parametrize(
    "old,new",
    [
        ("O 1.25 1.25 1.25", "O 1.35 1.25 1.25"),
        ("O 1.25 1.25 1.25", "Si 1.25 1.25 1.25"),
        ("2\nTest", "3\nTest"),
    ],
)
def test_rejects_modified_xyz(geometry, old, new):
    _, _, candidate = geometry
    path = candidate / "structure.xyz"
    path.write_text(path.read_text().replace(old, new))

    with pytest.raises(ConvergenceEvidenceValidationError):
        check(geometry)


def test_rejects_modified_cell(geometry):
    _, _, candidate = geometry
    path = candidate / "calculation.inp"
    path.write_text(
        path.read_text().replace(
            "A 5.0 0.0 0.0",
            "A 6.0 0.0 0.0",
        )
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="cell vector mismatch",
    ):
        check(geometry)


def test_rejects_wrong_coordinate_reference(geometry):
    _, _, candidate = geometry
    path = candidate / "calculation.inp"
    path.write_text(
        path.read_text().replace(
            "COORD_FILE_NAME structure.xyz",
            "COORD_FILE_NAME other.xyz",
        )
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="coordinate reference",
    ):
        check(geometry)


@pytest.mark.parametrize(
    "directive",
    [
        "@INCLUDE extra.inp",
        "@SET CUTOFF 100",
        "@IF ${ENABLE_EXTRA}",
    ],
)
def test_rejects_preprocessor_directives(geometry, directive):
    _, _, candidate = geometry
    path = candidate / "calculation.inp"
    path.write_text(directive + "\n" + path.read_text())

    with pytest.raises(ConvergenceEvidenceValidationError):
        check(geometry)


def test_rejects_xyz_symlink_escape(geometry, tmp_path):
    _, _, candidate = geometry
    xyz = candidate / "structure.xyz"
    external = tmp_path.parent / (
        tmp_path.name + "-external-geometry.xyz"
    )
    external.write_bytes(xyz.read_bytes())
    xyz.unlink()
    xyz.symlink_to(external)

    try:
        with pytest.raises(
            ConvergenceEvidenceValidationError,
            match="escapes",
        ):
            check(geometry)
    finally:
        external.unlink(missing_ok=True)
