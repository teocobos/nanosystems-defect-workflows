"""Tests for NSDW project scaffolding."""

from pathlib import Path

import pytest
import yaml

from nsdw.project.models import ProjectConfig
from nsdw.project.scaffold import (
    CORE_DIRECTORIES,
    ProjectScaffoldError,
    create_project,
)


def make_config(
    components: list[str] | None = None,
) -> ProjectConfig:
    """Return a representative NSDW project configuration."""

    return ProjectConfig(
        name="igzo-defect-modelling",
        material="IGZO",
        nsdw_version="0.1.0",
        components=components or [],
    )


def test_create_project_creates_core_structure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"

    created = create_project(
        root=root,
        config=make_config(),
    )

    assert created == root.resolve()
    assert created.is_dir()

    for relative_directory in CORE_DIRECTORIES:
        assert (created / relative_directory).is_dir()

    assert (created / "project.yaml").is_file()


def test_create_project_creates_component_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"

    create_project(
        root=root,
        config=make_config(
            ["mace", "vasp", "cp2k"],
        ),
    )

    for component in ("cp2k", "vasp", "mace"):
        assert (
            root
            / "calculations"
            / component
        ).is_dir()

        assert (
            root
            / "working"
            / "calculations"
            / component
        ).is_dir()

    assert not (
        root
        / "calculations"
        / "lammps"
    ).exists()


def test_project_yaml_contains_resolved_configuration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"

    create_project(
        root=root,
        config=make_config(
            ["mace", "cp2k", "cp2k"],
        ),
    )

    data = yaml.safe_load(
        (root / "project.yaml").read_text(
            encoding="utf-8",
        )
    )

    assert data == {
        "schema_version": 1,
        "name": "igzo-defect-modelling",
        "material": "IGZO",
        "nsdw_version": "0.1.0",
        "components": [
            "cp2k",
            "mace",
        ],
        "methodology": {
            "cp2k": None,
    },
}


def test_create_project_allows_existing_empty_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"
    root.mkdir()

    created = create_project(
        root=root,
        config=make_config(),
    )

    assert created == root.resolve()
    assert (created / "project.yaml").is_file()


def test_create_project_rejects_non_empty_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"
    root.mkdir()

    existing = root / "existing.txt"
    existing.write_text(
        "do not overwrite\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ProjectScaffoldError,
        match="Project directory is not empty",
    ):
        create_project(
            root=root,
            config=make_config(),
        )

    assert existing.read_text(
        encoding="utf-8",
    ) == "do not overwrite\n"


def test_create_project_rejects_existing_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"
    root.write_text(
        "not a directory\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ProjectScaffoldError,
        match="exists and is not a directory",
    ):
        create_project(
            root=root,
            config=make_config(),
        )

def test_create_project_writes_collaboration_readme(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"

    create_project(
        root=root,
        config=make_config(
            ["cp2k", "mace"],
        ),
    )

    readme = (
        root / "README.md"
    ).read_text(
        encoding="utf-8",
    )

    assert "# igzo-defect-modelling" in readme
    assert "**IGZO**" in readme
    assert "- cp2k" in readme
    assert "- mace" in readme

    assert "working/calculations/" in readme
    assert "calculations/" in readme

    assert (
        "This project was initialised with NSDW 0.1.0."
        in readme
    )


def test_create_project_writes_research_gitignore(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"

    create_project(
        root=root,
        config=make_config(
            ["cp2k", "vasp"],
        ),
    )

    gitignore = (
        root / ".gitignore"
    ).read_text(
        encoding="utf-8",
    )

    # Active calculations should stay outside Git.
    assert "working/" in gitignore

    # Common large calculator artefacts should be ignored.
    assert "*.wfn" in gitignore
    assert "*-RESTART*" in gitignore
    assert "WAVECAR" in gitignore
    assert "CHGCAR" in gitignore
    assert "slurm-*.out" in gitignore

    # The curated calculation record must remain shareable.
    ignored_lines = {
        line.strip()
        for line in gitignore.splitlines()
        if line.strip()
        and not line.startswith("#")
    }

    assert "calculations/" not in ignored_lines

def test_create_project_preserves_shareable_directories_for_git(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-defect-modelling"

    create_project(
        root=root,
        config=make_config(
            ["cp2k", "vasp"],
        ),
    )

    tracked_directories = (
        "docs",
        "structures/raw",
        "structures/validated",
        "structures/supercells",
        "structures/defects",
        "calculations",
        "calculations/cp2k",
        "calculations/vasp",
        "workflows",
        "reports",
    )

    for relative_directory in tracked_directories:
        assert (
            root
            / relative_directory
            / ".gitkeep"
        ).is_file()

    # working/ is deliberately local scratch space.
    assert not (
        root
        / "working"
        / ".gitkeep"
    ).exists()

    assert not (
        root
        / "working"
        / "calculations"
        / "cp2k"
        / ".gitkeep"
    ).exists()