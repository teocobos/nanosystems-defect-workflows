"""Structure evidence tests using actual NSDW CP2K packages."""

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
)
from nsdw.calculators.cp2k.package import write_cp2k_package
from nsdw.workflows.convergence.cp2k_structure_evidence import (
    validate_cp2k_structure_evidence,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


@pytest.fixture
def generated_package(tmp_path):
    campaign = tmp_path / "campaign"
    candidate = campaign / "600-Ry"
    campaign.mkdir()

    # Hexagonal lattice with a nonorthogonal basal plane.
    structure = Structure(
        Lattice.hexagonal(3.299, 26.101),
        ["Zn", "In", "Ga", "O"],
        [
            [0.0, 0.0, 0.0],
            [1 / 3, 2 / 3, 0.25],
            [2 / 3, 1 / 3, 0.50],
            [0.25, 0.25, 0.75],
        ],
    )

    structure.to(
        filename=campaign / "structure.json",
        fmt="json",
    )

    from nsdw.calculators.cp2k.presets import (
        get_basis_potential_preset,
    )

    config = CP2KInputConfig(
        project_name="igzo-evidence",
        functional="PBEsol",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        basis_potential=get_basis_potential_preset(
            "igzo-uzh-tzv2p"
        ),
    )

    write_cp2k_package(
        structure=structure,
        config=config,
        output_directory=candidate,
    )

    return tmp_path, campaign, candidate


def validate_package(generated_package):
    root, campaign, candidate = generated_package

    validate_cp2k_structure_evidence(
        project_root=root,
        campaign_directory=campaign,
        candidate_directory=candidate,
        input_filename="igzo-evidence.inp",
        coordinate_filename="igzo-evidence.xyz",
    )


def test_accepts_generated_nonorthogonal_package(generated_package):
    validate_package(generated_package)


def test_rejects_modified_generated_xyz(generated_package):
    _, _, candidate = generated_package
    path = candidate / "igzo-evidence.xyz"

    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[2].split()
    fields[1] = str(float(fields[1]) + 0.01)
    lines[2] = " ".join(fields)

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="XYZ coordinate mismatch",
    ):
        validate_package(generated_package)


def test_rejects_modified_generated_cell(generated_package):
    _, _, candidate = generated_package
    path = candidate / "igzo-evidence.inp"

    lines = path.read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        if line.strip().startswith("B "):
            fields = line.split()
            fields[1] = str(float(fields[1]) + 0.01)
            lines[index] = " ".join(fields)
            break
    else:
        raise AssertionError("Generated B cell vector not found.")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="cell vector mismatch",
    ):
        validate_package(generated_package)


def test_rejects_modified_generated_xyz_reference(
    generated_package,
):
    _, _, candidate = generated_package
    path = candidate / "igzo-evidence.inp"

    original = path.read_text(encoding="utf-8")
    assert "COORD_FILE_NAME igzo-evidence.xyz" in original

    path.write_text(
        original.replace(
            "COORD_FILE_NAME igzo-evidence.xyz",
            "COORD_FILE_NAME incorrect.xyz",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="coordinate reference",
    ):
        validate_package(generated_package)
