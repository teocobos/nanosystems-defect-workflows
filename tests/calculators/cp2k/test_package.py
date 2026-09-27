"""Tests for portable CP2K calculation packages."""

from __future__ import annotations

from pathlib import Path

import pytest

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
)
from nsdw.calculators.cp2k.input_parser import (
    parse_cp2k_input,
)
from nsdw.calculators.cp2k.package import (
    CP2KPackageError,
    write_cp2k_package,
)
from nsdw.calculators.cp2k.presets import (
    get_basis_potential_preset,
)
from nsdw.structures.parser import load_structure


IGZO_CIF = (
    Path(__file__).parents[2]
    / "data"
    / "igzo"
    / "igzo_crystal_ordered_003.cif"
)


@pytest.fixture
def igzo_structure():
    """Return the ordered 21-atom IGZO test structure."""

    structure, _ = load_structure(IGZO_CIF)

    return structure


def make_config(
    *,
    functional: str = "PBEsol",
) -> CP2KInputConfig:
    """Create a standard IGZO CP2K package configuration."""

    return CP2KInputConfig(
        project_name="igzo_003_test",
        functional=functional,
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        basis_potential=get_basis_potential_preset(
            "igzo-uzh-tzv2p"
        ),
    )


def test_writes_input_and_xyz(
    tmp_path,
    igzo_structure,
):
    """A package should contain matching input and XYZ files."""

    package_directory = tmp_path / "cp2k"

    result = write_cp2k_package(
        structure=igzo_structure,
        config=make_config(),
        output_directory=package_directory,
    )

    assert result == package_directory.resolve()

    input_path = (
        package_directory
        / "igzo_003_test.inp"
    )
    xyz_path = (
        package_directory
        / "igzo_003_test.xyz"
    )

    assert input_path.is_file()
    assert xyz_path.is_file()


def test_input_references_packaged_xyz(
    tmp_path,
    igzo_structure,
):
    """The generated input should reference the packaged XYZ by name."""

    package_directory = tmp_path / "cp2k"

    write_cp2k_package(
        structure=igzo_structure,
        config=make_config(),
        output_directory=package_directory,
    )

    input_path = (
        package_directory
        / "igzo_003_test.inp"
    )

    text = input_path.read_text(
        encoding="utf-8"
    )

    assert (
        "COORD_FILE_NAME igzo_003_test.xyz"
        in text
    )
    assert "COORD_FILE_FORMAT XYZ" in text


def test_packaged_xyz_contains_all_atoms(
    tmp_path,
    igzo_structure,
):
    """The packaged XYZ should contain the full IGZO structure."""

    package_directory = tmp_path / "cp2k"

    write_cp2k_package(
        structure=igzo_structure,
        config=make_config(),
        output_directory=package_directory,
    )

    xyz_path = (
        package_directory
        / "igzo_003_test.xyz"
    )

    lines = xyz_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines[0] == "21"

    coordinate_lines = lines[2:]

    assert len(coordinate_lines) == 21


@pytest.mark.parametrize(
    ("functional", "expected"),
    [
        ("PBE", "PBE"),
        ("PBEsol", "PBEsol"),
    ],
)
def test_packaged_input_round_trip(
    tmp_path,
    igzo_structure,
    functional,
    expected,
):
    """Packaged PBE and PBEsol inputs should parse correctly."""

    package_directory = tmp_path / functional.lower()

    write_cp2k_package(
        structure=igzo_structure,
        config=make_config(
            functional=functional,
        ),
        output_directory=package_directory,
    )

    parsed = parse_cp2k_input(
        package_directory
        / "igzo_003_test.inp"
    )

    assert parsed.project_name == "igzo_003_test"
    assert parsed.run_type.value == "ENERGY_FORCE"
    assert parsed.xc_functional == expected

    assert parsed.cutoff_ry == pytest.approx(
        600.0
    )
    assert parsed.relative_cutoff_ry == pytest.approx(
        60.0
    )

    assert parsed.eps_scf == pytest.approx(
        1.0e-6
    )
    assert parsed.k_points is None

    assert (
        parsed.basis_set_file
        == "BASIS_MOLOPT_UZH"
    )
    assert (
        parsed.potential_file
        == "POTENTIAL_UZH"
    )


def test_custom_filenames(
    tmp_path,
    igzo_structure,
):
    """Users should be able to select package filenames."""

    package_directory = tmp_path / "cp2k"

    write_cp2k_package(
        structure=igzo_structure,
        config=make_config(),
        output_directory=package_directory,
        input_filename="cutoff_600.inp",
        coordinate_filename="ordered_igzo.xyz",
    )

    input_path = (
        package_directory
        / "cutoff_600.inp"
    )
    xyz_path = (
        package_directory
        / "ordered_igzo.xyz"
    )

    assert input_path.is_file()
    assert xyz_path.is_file()

    text = input_path.read_text(
        encoding="utf-8"
    )

    assert (
        "COORD_FILE_NAME ordered_igzo.xyz"
        in text
    )


def test_refuses_existing_input(
    tmp_path,
    igzo_structure,
):
    """Package creation should not silently overwrite an input."""

    package_directory = tmp_path / "cp2k"

    write_cp2k_package(
        structure=igzo_structure,
        config=make_config(),
        output_directory=package_directory,
    )

    with pytest.raises(
        CP2KPackageError,
        match="input already exists",
    ):
        write_cp2k_package(
            structure=igzo_structure,
            config=make_config(),
            output_directory=package_directory,
        )


def test_rejects_input_path_as_filename(
    tmp_path,
    igzo_structure,
):
    """Input filenames must not escape the package directory."""

    with pytest.raises(
        CP2KPackageError,
        match="must be a filename",
    ):
        write_cp2k_package(
            structure=igzo_structure,
            config=make_config(),
            output_directory=tmp_path / "cp2k",
            input_filename="nested/test.inp",
        )


def test_rejects_coordinate_path_as_filename(
    tmp_path,
    igzo_structure,
):
    """Coordinate filenames must not escape the package directory."""

    with pytest.raises(
        CP2KPackageError,
        match="must be a filename",
    ):
        write_cp2k_package(
            structure=igzo_structure,
            config=make_config(),
            output_directory=tmp_path / "cp2k",
            coordinate_filename="nested/test.xyz",
        )


def test_requires_inp_extension(
    tmp_path,
    igzo_structure,
):
    """CP2K input packages should require an .inp filename."""

    with pytest.raises(
        CP2KPackageError,
        match=r"must end in '\.inp'",
    ):
        write_cp2k_package(
            structure=igzo_structure,
            config=make_config(),
            output_directory=tmp_path / "cp2k",
            input_filename="test.txt",
        )


def test_requires_xyz_extension(
    tmp_path,
    igzo_structure,
):
    """External coordinate packages should require an .xyz file."""

    with pytest.raises(
        CP2KPackageError,
        match=r"must end in '\.xyz'",
    ):
        write_cp2k_package(
            structure=igzo_structure,
            config=make_config(),
            output_directory=tmp_path / "cp2k",
            coordinate_filename="test.cif",
        )

def test_generation_failure_leaves_no_partial_package(
    tmp_path,
    igzo_structure,
):
    """Scientific validation failure should leave no partial package."""

    config = make_config()

    basis_potential = config.basis_potential

    assert basis_potential is not None

    # Deliberately remove oxygen so CP2K generation fails
    # the structure-to-KIND consistency check.
    incomplete_basis = basis_potential.model_copy(
        update={
            "kinds": tuple(
                kind
                for kind in basis_potential.kinds
                if kind.element != "O"
            ),
        }
    )

    invalid_config = config.model_copy(
        update={
            "basis_potential": incomplete_basis,
        }
    )

    package_directory = (
        tmp_path
        / "failed_cp2k_package"
    )

    with pytest.raises(
        Exception,
        match=(
            "Missing CP2K KIND definitions "
            "for structure elements: O"
        ),
    ):
        write_cp2k_package(
            structure=igzo_structure,
            config=invalid_config,
            output_directory=package_directory,
        )

    assert not package_directory.exists()

    assert not (
        package_directory
        / "igzo_003_test.inp"
    ).exists()

    assert not (
        package_directory
        / "igzo_003_test.xyz"
    ).exists()