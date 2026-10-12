"""CLI regression tests for CP2K k-point convergence generation."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nsdw.cli import app


runner = CliRunner()

STRUCTURE = Path(
    "tests/data/sio2/alpha_quartz_cod1526860.cif"
)


def invoke_generate(
    tmp_path: Path,
    *options: str,
):
    output = tmp_path / "campaign"
    args = [
        "workflow",
        "convergence-generate",
        str(STRUCTURE.resolve()),
        "--preset",
        "sio2-pbe-dzvp",
        "--output",
        str(output),
        *options,
    ]
    return runner.invoke(app, args), output


def test_kpoint_campaign_generation(tmp_path):
    import json

    result, output = invoke_generate(
        tmp_path,
        "--parameter", "kpoints",
        "--mesh", "1x1x1",
        "--mesh", "2x2x2",
        "--mesh", "3x3x3",
        "--cutoff", "400",
        "--relative-cutoff", "30",
    )

    assert result.exit_code == 0, result.output

    manifest = json.loads(
        (output / "manifest.json").read_text()
    )

    assert manifest["parameter"] == "kpoints"
    assert [
        tuple(candidate["value"])
        for candidate in manifest["candidates"]
    ] == [
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
    ]

    for candidate in manifest["candidates"]:
        input_path = (
            output
            / candidate["directory"]
            / candidate["input_file"]
        )
        text = input_path.read_text()

        mesh = tuple(candidate["value"])

        assert (
            "SCHEME MONKHORST-PACK "
            + " ".join(map(str, mesh))
        ) in text
        assert "CUTOFF 400" in text
        assert "REL_CUTOFF 30" in text
        assert "&DIAGONALIZATION" in text
        assert "&OT" not in text


@pytest.mark.parametrize(
    "mesh",
    [
        "2x2",
        "2x2x0",
        "2x-1x2",
        "2x2xabc",
        "2.5x2x2",
    ],
)
def test_rejects_invalid_mesh(tmp_path, mesh):
    result, output = invoke_generate(
        tmp_path,
        "--parameter", "kpoints",
        "--mesh", "1x1x1",
        "--mesh", mesh,
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_rejects_duplicate_meshes(tmp_path):
    result, output = invoke_generate(
        tmp_path,
        "--parameter", "kpoints",
        "--mesh", "2x2x2",
        "--mesh", "2x2x2",
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_rejects_decreasing_mesh_cost(tmp_path):
    result, output = invoke_generate(
        tmp_path,
        "--parameter", "kpoints",
        "--mesh", "3x3x3",
        "--mesh", "2x2x2",
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_rejects_values_with_kpoints(tmp_path):
    result, output = invoke_generate(
        tmp_path,
        "--parameter", "kpoints",
        "--mesh", "1x1x1",
        "--mesh", "2x2x2",
        "--values", "400",
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_rejects_mesh_with_cutoff(tmp_path):
    result, output = invoke_generate(
        tmp_path,
        "--parameter", "cutoff",
        "--values", "400",
        "--values", "600",
        "--mesh", "2x2x2",
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_legacy_cutoff_cli_remains_supported(tmp_path):
    result, output = invoke_generate(
        tmp_path,
        "--parameter", "cutoff",
        "--values", "400",
        "--values", "600",
    )

    assert result.exit_code == 0, result.output
    assert (output / "manifest.json").is_file()


def test_automatic_kpoint_cli_generation(tmp_path):
    import json
    from typer.testing import CliRunner
    from nsdw.cli import app

    structure_file = (
        Path(__file__).resolve().parent
        / "data/sio2/alpha_quartz_cod1526860.cif"
    )
    output = tmp_path / "automatic-kpoints"

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "convergence-generate",
            str(structure_file),
            "--parameter", "kpoints",
            "--preset", "sio2-pbe-dzvp",
            "--auto",
            "--cutoff", "400",
            "--relative-cutoff", "30",
            "--output", str(output),
        ],
    )

    assert result.exit_code == 0, result.output

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["parameter"] == "kpoints"
    assert [
        tuple(candidate["value"])
        for candidate in manifest["candidates"]
    ] == [
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 2),
        (3, 3, 3),
    ]

    for candidate in manifest["candidates"]:
        input_path = (
            output
            / candidate["directory"]
            / candidate["input_file"]
        )

        text = input_path.read_text(encoding="utf-8")
        mesh = candidate["value"]

        assert (
            "SCHEME MONKHORST-PACK "
            + " ".join(map(str, mesh))
        ) in text
        assert "CUTOFF 400" in text
        assert "REL_CUTOFF 30" in text
        assert "&DIAGONALIZATION" in text
        assert "&OT" not in text


def test_automatic_kpoint_cli_rejects_manual_mesh(tmp_path):
    from typer.testing import CliRunner
    from nsdw.cli import app

    structure_file = (
        Path(__file__).resolve().parent
        / "data/sio2/alpha_quartz_cod1526860.cif"
    )

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "convergence-generate",
            str(structure_file),
            "--parameter", "kpoints",
            "--preset", "sio2-pbe-dzvp",
            "--auto",
            "--mesh", "1x1x1",
            "--mesh", "2x2x2",
            "--output", str(tmp_path / "invalid"),
        ],
    )

    assert result.exit_code != 0
    assert "either --auto or --mesh" in result.output


def test_automatic_kpoint_cli_rejects_cutoff_parameter(tmp_path):
    from typer.testing import CliRunner
    from nsdw.cli import app

    structure_file = (
        Path(__file__).resolve().parent
        / "data/sio2/alpha_quartz_cod1526860.cif"
    )

    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "convergence-generate",
            str(structure_file),
            "--parameter", "cutoff",
            "--preset", "sio2-pbe-dzvp",
            "--auto",
            "--values", "400",
            "--values", "500",
            "--output", str(tmp_path / "invalid"),
        ],
    )

    assert result.exit_code != 0
    assert "only supported for k-point convergence" in result.output
