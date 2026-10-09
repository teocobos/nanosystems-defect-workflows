"""Structure provenance using NSDW's ordered 21-atom IGZO CIF."""

import pytest
from pymatgen.core import Structure

from nsdw.calculators.cp2k.generation_models import CP2KInputConfig
from nsdw.calculators.cp2k.package import write_cp2k_package
from nsdw.calculators.cp2k.presets import get_basis_potential_preset
from nsdw.workflows.convergence.cp2k_structure_evidence import (
    validate_cp2k_structure_evidence,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


@pytest.fixture
def igzo21_package(tmp_path):
    campaign = tmp_path / "campaign"
    candidate = campaign / "600-Ry"
    campaign.mkdir()

    cif = (
        "tests/data/igzo/"
        "igzo_crystal_ordered_003.cif"
    )

    structure = Structure.from_file(cif)

    assert len(structure) == 21
    assert structure.is_ordered

    structure.to(
        filename=campaign / "structure.json",
        fmt="json",
    )

    config = CP2KInputConfig(
        project_name="igzo21-evidence",
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

    return tmp_path, campaign, candidate, structure


def check(igzo21_package):
    root, campaign, candidate, _ = igzo21_package

    validate_cp2k_structure_evidence(
        project_root=root,
        campaign_directory=campaign,
        candidate_directory=candidate,
        input_filename="igzo21-evidence.inp",
        coordinate_filename="igzo21-evidence.xyz",
    )


def test_accepts_full_igzo21_package(igzo21_package):
    check(igzo21_package)


def test_generated_xyz_matches_all_21_atoms(igzo21_package):
    _, _, candidate, structure = igzo21_package
    path = candidate / "igzo21-evidence.xyz"

    lines = path.read_text(encoding="utf-8").splitlines()

    assert int(lines[0]) == 21
    assert len(lines[2:]) == 21

    for line, site in zip(lines[2:], structure, strict=True):
        fields = line.split()

        assert fields[0] == site.specie.symbol

        for actual, expected in zip(
            map(float, fields[1:]),
            site.coords,
            strict=True,
        ):
            assert abs(actual - expected) < 1e-10

    check(igzo21_package)


def test_rejects_igzo21_coordinate_change(igzo21_package):
    _, _, candidate, _ = igzo21_package
    path = candidate / "igzo21-evidence.xyz"

    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[3].split()
    fields[1] = str(float(fields[1]) + 0.001)
    lines[3] = " ".join(fields)

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="XYZ coordinate mismatch",
    ):
        check(igzo21_package)


def test_rejects_igzo21_atom_reordering(igzo21_package):
    _, _, candidate, _ = igzo21_package
    path = candidate / "igzo21-evidence.xyz"

    lines = path.read_text(encoding="utf-8").splitlines()

    # Swap records with different species if possible;
    # otherwise swap different atomic positions.
    first = 2
    second = next(
        (
            i for i in range(3, len(lines))
            if lines[i].split()[0] != lines[first].split()[0]
        ),
        3,
    )

    lines[first], lines[second] = lines[second], lines[first]

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="XYZ species mismatch|XYZ coordinate mismatch",
    ):
        check(igzo21_package)


def test_rejects_igzo21_cell_change(igzo21_package):
    _, _, candidate, _ = igzo21_package
    path = candidate / "igzo21-evidence.inp"

    lines = path.read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        if line.strip().startswith("C "):
            fields = line.split()
            fields[3] = str(float(fields[3]) + 0.01)
            lines[index] = " ".join(fields)
            break
    else:
        raise AssertionError("Generated C cell vector not found.")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="cell vector mismatch",
    ):
        check(igzo21_package)
