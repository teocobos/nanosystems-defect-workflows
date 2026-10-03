"""CLI tests for NSDW project commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nsdw.cli import app

from nsdw.project.models import ProjectConfig
from nsdw.project.scaffold import create_project

runner = CliRunner()


def test_project_init_creates_project(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-project"

    result = runner.invoke(
        app,
        [
            "project",
            "init",
            "igzo-project",
            "--material",
            "IGZO",
            "--with",
            "cp2k",
            "--with",
            "vasp",
            "--directory",
            str(root),
        ],
    )

    assert result.exit_code == 0
    assert "Created NSDW project:" in result.stdout
    assert "Material: IGZO" in result.stdout
    assert "Components: cp2k, vasp" in result.stdout

    assert (root / "project.yaml").is_file()
    assert (root / "README.md").is_file()
    assert (root / ".gitignore").is_file()

    assert (root / "calculations" / "cp2k").is_dir()
    assert (root / "calculations" / "vasp").is_dir()

    assert (
        root
        / "working"
        / "calculations"
        / "cp2k"
    ).is_dir()


def test_project_init_with_git_creates_repository(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-project"

    result = runner.invoke(
        app,
        [
            "project",
            "init",
            "igzo-project",
            "--material",
            "IGZO",
            "--with",
            "cp2k",
            "--directory",
            str(root),
            "--git",
        ],
    )

    assert result.exit_code == 0
    assert "Git: initialised on branch main" in result.stdout

    assert (root / ".git").is_dir()


def test_project_init_rejects_invalid_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "invalid-project"

    result = runner.invoke(
        app,
        [
            "project",
            "init",
            "invalid-project",
            "--material",
            "SiO2",
            "--with",
            "quantum-espresso",
            "--directory",
            str(root),
        ],
    )

    assert result.exit_code == 2

    assert "Invalid project component." in result.stdout
    assert (
        "Allowed components: cp2k, vasp, lammps, mace"
        in result.stdout
    )

    assert not root.exists()


def test_project_init_rejects_non_empty_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "existing-project"
    root.mkdir()

    marker = root / "important-data.txt"
    marker.write_text(
        "do not overwrite",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "project",
            "init",
            "existing-project",
            "--material",
            "IGZO",
            "--directory",
            str(root),
        ],
    )

    assert result.exit_code == 1
    assert "Project directory is not empty" in result.stdout

    assert marker.read_text(
        encoding="utf-8",
    ) == "do not overwrite"

def test_project_info_discovers_project_from_nested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "igzo-project"

    create_project(
        root=root,
        config=ProjectConfig(
            name="igzo-project",
            material="IGZO",
            nsdw_version="0.1.0",
            components=["cp2k"],
        ),
    )

    nested_directory = (
        root
        / "workflows"
        / "convergence"
        / "cutoff"
    )
    nested_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    monkeypatch.chdir(
        nested_directory,
    )

    result = runner.invoke(
        app,
        [
            "project",
            "info",
        ],
    )

    assert result.exit_code == 0
    assert "igzo-project" in result.stdout
    assert "IGZO" in result.stdout
    assert "cp2k" in result.stdout
    assert str(root.resolve()) in result.stdout


def test_project_info_rejects_directory_outside_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "not-a-project"
    directory.mkdir()

    monkeypatch.chdir(
        directory,
    )

    result = runner.invoke(
        app,
        [
            "project",
            "info",
        ],
    )

    assert result.exit_code == 1
    assert "No NSDW project found" in result.stdout