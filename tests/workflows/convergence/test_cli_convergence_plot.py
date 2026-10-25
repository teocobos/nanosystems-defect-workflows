
import json

from typer.testing import CliRunner

from nsdw.cli import app


runner = CliRunner()


def make_dataset(path):
    path.write_text(
        json.dumps([
            {
                "cutoff_ry": 300,
                "grid_numbers": [1, 2],
                "counts": [80, 20],
                "grid_cutoffs_au": [150.0, 50.0],
                "reported_total": 100,
            },
            {
                "cutoff_ry": 400,
                "grid_numbers": [1, 2],
                "counts": [60, 40],
                "grid_cutoffs_au": [200.0, 66.67],
                "reported_total": 100,
            },
        ]),
        encoding="utf-8",
    )


def test_explicit_dataset(tmp_path):
    dataset = tmp_path / "multigrid_data.json"
    make_dataset(dataset)

    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-plot",
            str(dataset),
            "--format",
            "png",
            "--dpi",
            "150",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list(
        (tmp_path / "figures").glob("*.png")
    )) == 3


def test_directory_autodiscovery(tmp_path):
    dataset = tmp_path / "multigrid_data.json"
    make_dataset(dataset)

    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-plot",
            str(tmp_path),
            "--format",
            "svg",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list(
        (tmp_path / "figures").glob("*.svg")
    )) == 3


def test_default_input(tmp_path, monkeypatch):
    make_dataset(tmp_path / "multigrid_data.json")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-plot",
            "--format",
            "png",
            "--dpi",
            "150",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        tmp_path / "figures" / "gaussian_grid_counts.png"
    ).exists()


def test_existing_figures_are_preserved(tmp_path):
    dataset = tmp_path / "multigrid_data.json"
    make_dataset(dataset)

    destination = tmp_path / "figures"
    destination.mkdir()
    existing = destination / "gaussian_grid_counts.png"
    existing.write_bytes(b"original")

    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-plot",
            str(dataset),
            "--format",
            "png",
        ],
    )

    assert result.exit_code != 0
    assert "overwrite" in result.output.lower()
    assert existing.read_bytes() == b"original"


def test_overwrite_explicitly(tmp_path):
    dataset = tmp_path / "multigrid_data.json"
    make_dataset(dataset)

    destination = tmp_path / "figures"
    destination.mkdir()
    existing = destination / "gaussian_grid_counts.png"
    existing.write_bytes(b"original")

    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-plot",
            str(dataset),
            "--format",
            "png",
            "--dpi",
            "150",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert existing.read_bytes() != b"original"


def test_missing_dataset(tmp_path):
    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-plot",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "ERROR" in result.output


def test_ambiguous_dataset(tmp_path):
    make_dataset(tmp_path / "multigrid_data.json")
    figures = tmp_path / "figures"
    figures.mkdir()
    make_dataset(figures / "multigrid_data.json")

    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-plot",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Multiple" in result.output
