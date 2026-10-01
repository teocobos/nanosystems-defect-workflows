from pathlib import Path
import json

from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KInputConfig,
    CP2KKindConfig,
)
from nsdw.models.quantity import Quantity

from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)
import pytest

from nsdw.workflows.convergence.cp2k_generation import (
    CP2KConvergenceGenerationError,
    generate_cp2k_convergence_study,
)

from unittest.mock import patch

def _structure() -> Structure:
    return Structure(
        lattice=Lattice.cubic(5.0),
        species=["O"],
        coords=[[0.0, 0.0, 0.0]],
    )


def _base_config() -> CP2KInputConfig:
    return CP2KInputConfig(
        project_name="test",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_SET",
            potential_file="POTENTIAL",
            kinds=(
                CP2KKindConfig(
                    element="O",
                    basis_set="DZVP-MOLOPT-SR-GTH",
                    potential="GTH-PBE",
                ),
            ),
        ),
    )


def test_generate_cutoff_study_creates_candidate_packages(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="800-Ry",
                order=2,
                value=Quantity(value=800.0, unit="Ry"),
            ),
        ],
    )

    output_directory = tmp_path / "cutoff-study"

    packages = generate_cp2k_convergence_study(
        structure=_structure(),
        base_config=_base_config(),
        study=study,
        output_directory=output_directory,
    )

    assert len(packages) == 3

    for label in ("400-Ry", "600-Ry", "800-Ry"):
        candidate_directory = output_directory / label

        assert candidate_directory.is_dir()

        input_files = list(candidate_directory.glob("*.inp"))
        coordinate_files = list(candidate_directory.glob("*.xyz"))

        assert len(input_files) == 1
        assert len(coordinate_files) == 1

    input_400 = next(
        (output_directory / "400-Ry").glob("*.inp")
    ).read_text()

    input_600 = next(
        (output_directory / "600-Ry").glob("*.inp")
    ).read_text()

    input_800 = next(
        (output_directory / "800-Ry").glob("*.inp")
    ).read_text()

    assert "CUTOFF 400" in input_400
    assert "CUTOFF 600" in input_600
    assert "CUTOFF 800" in input_800

    assert "REL_CUTOFF 60" in input_400
    assert "REL_CUTOFF 60" in input_600
    assert "REL_CUTOFF 60" in input_800

def test_generate_relative_cutoff_study(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="40-Ry",
                order=0,
                value=Quantity(value=40.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="60-Ry",
                order=1,
                value=Quantity(value=60.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="80-Ry",
                order=2,
                value=Quantity(value=80.0, unit="Ry"),
            ),
        ],
    )

    output_directory = tmp_path / "relative-cutoff-study"

    packages = generate_cp2k_convergence_study(
        structure=_structure(),
        base_config=_base_config(),
        study=study,
        output_directory=output_directory,
    )

    assert len(packages) == 3

    for label, value in (
        ("40-Ry", "40"),
        ("60-Ry", "60"),
        ("80-Ry", "80"),
    ):
        input_text = next(
            (output_directory / label).glob("*.inp")
        ).read_text()

        assert f"REL_CUTOFF {value}" in input_text

        # CUTOFF is not the parameter under study and must remain fixed.
        assert "CUTOFF 600" in input_text


def test_generate_kpoint_study(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.KPOINTS,
        candidates=[
            ConvergenceCandidate(
                label="1x1x1",
                order=0,
                value=(1, 1, 1),
            ),
            ConvergenceCandidate(
                label="2x2x2",
                order=1,
                value=(2, 2, 2),
            ),
            ConvergenceCandidate(
                label="3x3x3",
                order=2,
                value=(3, 3, 3),
            ),
        ],
    )

    output_directory = tmp_path / "kpoint-study"

    packages = generate_cp2k_convergence_study(
        structure=_structure(),
        base_config=_base_config(),
        study=study,
        output_directory=output_directory,
    )

    assert len(packages) == 3

    for label, mesh in (
        ("1x1x1", "1 1 1"),
        ("2x2x2", "2 2 2"),
        ("3x3x3", "3 3 3"),
    ):
        input_text = next(
            (output_directory / label).glob("*.inp")
        ).read_text()

        assert (
            f"SCHEME MONKHORST-PACK {mesh}"
            in input_text
        )

        # Grid settings are not under study and must remain fixed.
        assert "CUTOFF 600" in input_text
        assert "REL_CUTOFF 60" in input_text

def test_generation_failure_does_not_leave_partial_study(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.BASIS,
        candidates=[
            ConvergenceCandidate(
                label="dzvp",
                order=0,
                value="DZVP-MOLOPT-SR-GTH",
            ),
            ConvergenceCandidate(
                label="tzvp",
                order=1,
                value="TZVP-MOLOPT-GTH",
            ),
        ],
    )

    output_directory = tmp_path / "basis-study"

    with pytest.raises(
        CP2KConvergenceGenerationError,
        match="basis-set convergence generation is not yet supported",
    ):
        generate_cp2k_convergence_study(
            structure=_structure(),
            base_config=_base_config(),
            study=study,
            output_directory=output_directory,
        )

    assert not output_directory.exists()

def test_generation_rolls_back_after_partial_failure(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="800-Ry",
                order=2,
                value=Quantity(value=800.0, unit="Ry"),
            ),
        ],
    )

    output_directory = tmp_path / "cutoff-study"

    from nsdw.calculators.cp2k.package import (
        write_cp2k_package as real_write_cp2k_package,
    )

    call_count = 0

    def failing_writer(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 2:
            raise RuntimeError("simulated package failure")

        return real_write_cp2k_package(**kwargs)

    with patch(
        "nsdw.workflows.convergence.cp2k_generation.write_cp2k_package",
        side_effect=failing_writer,
    ):
        with pytest.raises(
            RuntimeError,
            match="simulated package failure",
        ):
            generate_cp2k_convergence_study(
                structure=_structure(),
                base_config=_base_config(),
                study=study,
                output_directory=output_directory,
            )

    assert call_count == 2
    assert not output_directory.exists()

def test_generation_failure_preserves_preexisting_output_directory(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
            ),
        ],
    )

    output_directory = tmp_path / "cutoff-study"
    output_directory.mkdir()

    sentinel = output_directory / "user-data.txt"
    sentinel.write_text(
        "do not delete\n",
        encoding="utf-8",
    )

    with patch(
        "nsdw.workflows.convergence.cp2k_generation.write_cp2k_package",
        side_effect=RuntimeError("simulated package failure"),
    ):
        with pytest.raises(
            RuntimeError,
            match="simulated package failure",
        ):
            generate_cp2k_convergence_study(
                structure=_structure(),
                base_config=_base_config(),
                study=study,
                output_directory=output_directory,
            )

    assert output_directory.is_dir()
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "do not delete\n"

def test_generate_cutoff_study_writes_manifest(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
            ),
        ],
    )

    output_directory = tmp_path / "cutoff-study"

    generate_cp2k_convergence_study(
        structure=_structure(),
        base_config=_base_config(),
        study=study,
        output_directory=output_directory,
    )

    manifest_path = output_directory / "manifest.json"

    assert manifest_path.is_file()

def test_generated_manifest_records_study_metadata_and_paths(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
            ),
        ],
    )

    output_directory = tmp_path / "cutoff-study"

    generate_cp2k_convergence_study(
        structure=_structure(),
        base_config=_base_config(),
        study=study,
        output_directory=output_directory,
    )

    manifest = json.loads(
        (output_directory / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == 1
    assert manifest["calculator"] == "cp2k"
    assert manifest["parameter"] == "cutoff"

    assert manifest["criterion"] == {
        "energy_tolerance_ev_per_atom": 1.0e-3,
    }

    assert len(manifest["candidates"]) == 2

    first = manifest["candidates"][0]
    second = manifest["candidates"][1]

    assert first == {
        "label": "400-Ry",
        "order": 0,
        "value": {
            "value": 400.0,
            "unit": "Ry",
        },
        "directory": "400-Ry",
        "input_file": "test-400-Ry.inp",
        "coordinate_file": "test-400-Ry.xyz",
    }

    assert second == {
        "label": "600-Ry",
        "order": 1,
        "value": {
            "value": 600.0,
            "unit": "Ry",
        },
        "directory": "600-Ry",
        "input_file": "test-600-Ry.inp",
        "coordinate_file": "test-600-Ry.xyz",
    }

    # Manifest paths must remain portable.
    assert not Path(first["directory"]).is_absolute()
    assert not Path(first["input_file"]).is_absolute()
    assert not Path(first["coordinate_file"]).is_absolute()

def test_manifest_failure_rolls_back_generated_study(
    tmp_path: Path,
) -> None:
    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=[
            ConvergenceCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(value=400.0, unit="Ry"),
            ),
            ConvergenceCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(value=600.0, unit="Ry"),
            ),
        ],
    )

    output_directory = tmp_path / "cutoff-study"

    with patch(
        "nsdw.workflows.convergence.cp2k_generation."
        "write_convergence_manifest",
        side_effect=RuntimeError("simulated manifest failure"),
    ):
        with pytest.raises(
            RuntimeError,
            match="simulated manifest failure",
        ):
            generate_cp2k_convergence_study(
                structure=_structure(),
                base_config=_base_config(),
                study=study,
                output_directory=output_directory,
            )

    assert not output_directory.exists()