import hashlib
import json
from pathlib import Path

from pymatgen.core import Structure
from typer.testing import CliRunner

from nsdw.cli import app


runner = CliRunner()

IGZO_CIF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "igzo"
    / "igzo_crystal_ordered_003.cif"
)

MANUAL_MATRIX = "[[3,0,0],[0,3,0],[0,0,1]]"


def manual_command(output_dir):
    return [
        "structure",
        "select-supercell",
        str(IGZO_CIF),
        "--engine",
        "manual",
        "--matrix",
        MANUAL_MATRIX,
        "--output-dir",
        str(output_dir),
    ]


def test_manual_cli_exports_valid_cif_and_json(tmp_path):
    output_dir = tmp_path / "manual"

    result = runner.invoke(
        app,
        manual_command(output_dir),
    )

    assert result.exit_code == 0, result.output

    cif_files = list(output_dir.glob("*.cif"))
    json_files = list(output_dir.glob("*.json"))

    assert len(cif_files) == 1
    assert len(json_files) == 1

    structure = Structure.from_file(str(cif_files[0]))
    metadata = json.loads(json_files[0].read_text())

    assert len(structure) == 189
    assert metadata["engine"] == "manual"
    assert metadata["num_atoms"] == 189
    assert metadata["determinant"] == 9
    assert metadata["transformation_matrix"] == [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 1],
    ]

    assert metadata["cif_filename"] == cif_files[0].name
    assert metadata["cif_sha256"] == hashlib.sha256(
        cif_files[0].read_bytes()
    ).hexdigest()

    assert metadata["source_sha256"] == hashlib.sha256(
        IGZO_CIF.read_bytes()
    ).hexdigest()

    assert abs(
        metadata["minimum_image_distance_angstrom"]
        - 9.897
    ) < 0.01


def test_cli_rejects_invalid_matrix_json(tmp_path):
    output_dir = tmp_path / "invalid"

    args = manual_command(output_dir)
    matrix_index = args.index("--matrix") + 1
    args[matrix_index] = "not-valid-json"

    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "ERROR" in result.output
    assert not list(output_dir.glob("*.cif"))
    assert not list(output_dir.glob("*.json"))


def test_cli_does_not_overwrite_existing_exports(tmp_path):
    output_dir = tmp_path / "protected"
    args = manual_command(output_dir)

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output

    files = sorted(output_dir.iterdir())
    assert len(files) == 2

    original_contents = {
        path.name: path.read_bytes()
        for path in files
    }

    second = runner.invoke(app, args)

    assert second.exit_code != 0
    assert "already exists" in second.output

    assert {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
    } == original_contents



def test_cli_preserves_existing_json_on_export_failure(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "partial"
    output_dir.mkdir()

    existing_json = (
        output_dir
        / "igzo_crystal_ordered_003_manual_189atoms.json"
    )
    existing_json.write_text("existing provenance\n")

    result = runner.invoke(
        app,
        manual_command(output_dir),
    )

    assert result.exit_code != 0
    assert existing_json.read_text() == "existing provenance\n"
    assert not list(output_dir.glob("*.cif"))



def test_cli_preserves_json_created_during_export(
    tmp_path,
    monkeypatch,
):
    from pathlib import Path

    output_dir = tmp_path / "collision"
    output_dir.mkdir()

    json_path = (
        output_dir
        / "igzo_crystal_ordered_003_manual_189atoms.json"
    )

    original_open = Path.open

    def competing_open(self, mode="r", *args, **kwargs):
        if self == json_path and mode == "x":
            with original_open(
                json_path, "w", encoding="utf-8"
            ) as handle:
                handle.write("created by another process\n")

        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", competing_open)

    result = runner.invoke(
        app,
        manual_command(output_dir),
    )

    assert result.exit_code != 0
    assert json_path.read_text() == (
        "created by another process\n"
    )
    assert not list(output_dir.glob("*.cif"))
