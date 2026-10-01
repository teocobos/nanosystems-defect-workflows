"""Tests for NSDW project workspace loading and discovery."""

from pathlib import Path

import pytest
import yaml

from nsdw.project.models import ProjectConfig
from nsdw.project.scaffold import create_project
from nsdw.project.workspace import (
    ProjectWorkspaceError,
    find_project_workspace,
    load_project_workspace,
)


def make_config(
    *,
    name: str = "IGZO",
) -> ProjectConfig:
    """Return a representative NSDW project configuration."""

    return ProjectConfig(
        name=name,
        material="InGaZnO4",
        nsdw_version="0.1.0",
        components=["cp2k"],
    )


def test_load_project_workspace(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"

    create_project(
        root=project_directory,
        config=make_config(),
    )

    workspace = load_project_workspace(
        project_directory,
    )

    assert workspace.root == project_directory.resolve()
    assert workspace.name == "IGZO"
    assert workspace.schema_version == 1


def test_load_project_workspace_rejects_unsupported_schema_version(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"

    create_project(
        root=project_directory,
        config=make_config(),
    )

    metadata_path = project_directory / "project.yaml"

    metadata = yaml.safe_load(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata["schema_version"] = 999

    metadata_path.write_text(
        yaml.safe_dump(
            metadata,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="Unsupported NSDW project schema version",
    ):
        load_project_workspace(
            project_directory,
        )


def test_load_project_workspace_rejects_missing_metadata_fields(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"
    project_directory.mkdir()

    metadata_path = project_directory / "project.yaml"

    metadata_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="Could not load NSDW project metadata",
    ):
        load_project_workspace(
            project_directory,
        )


def test_load_project_workspace_rejects_invalid_yaml(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"
    project_directory.mkdir()

    metadata_path = project_directory / "project.yaml"

    metadata_path.write_text(
        "schema_version: [\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="Could not load NSDW project metadata",
    ):
        load_project_workspace(
            project_directory,
        )


def test_find_project_workspace_from_nested_directory(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"

    create_project(
        root=project_directory,
        config=make_config(),
    )

    nested_directory = (
        project_directory
        / "workflows"
        / "convergence"
        / "cutoff"
        / "600-Ry"
    )

    nested_directory.mkdir(
        parents=True,
    )

    workspace = find_project_workspace(
        nested_directory,
    )

    assert workspace.root == project_directory.resolve()
    assert workspace.name == "IGZO"
    assert workspace.schema_version == 1


def test_find_project_workspace_from_project_root(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"

    create_project(
        root=project_directory,
        config=make_config(),
    )

    workspace = find_project_workspace(
        project_directory,
    )

    assert workspace.root == project_directory.resolve()
    assert workspace.name == "IGZO"


def test_find_project_workspace_raises_when_project_not_found(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "not-a-project"
    directory.mkdir()

    with pytest.raises(
        ProjectWorkspaceError,
        match="No NSDW project found",
    ):
        find_project_workspace(
            directory,
        )


def test_load_project_workspace_from_scaffolded_project(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"

    config = ProjectConfig(
        name="IGZO",
        material="InGaZnO4",
        nsdw_version="0.1.0",
        components=["cp2k"],
    )

    create_project(
        root=project_directory,
        config=config,
    )

    workspace = load_project_workspace(
        project_directory,
    )

    assert workspace.root == project_directory.resolve()
    assert workspace.name == "IGZO"
    assert workspace.schema_version == 1