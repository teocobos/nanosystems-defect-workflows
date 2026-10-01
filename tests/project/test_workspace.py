from pathlib import Path

import pytest
import yaml

from nsdw.project.workspace import (
    ProjectWorkspaceError,
    create_project_workspace,
    find_project_workspace,
    load_project_workspace,
)


def test_create_project_workspace(tmp_path: Path) -> None:
    project_directory = tmp_path / "IGZO"

    workspace = create_project_workspace(
        project_directory,
        name="IGZO",
    )

    assert workspace.root == project_directory.resolve()

    assert (project_directory / "nsdw-project.yaml").is_file()
    assert (project_directory / "structure").is_dir()
    assert (project_directory / "convergence").is_dir()
    assert (project_directory / "production").is_dir()
    assert (project_directory / "reports").is_dir()

    metadata = yaml.safe_load(
        (project_directory / "nsdw-project.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert metadata == {
        "schema_version": 1,
        "name": "IGZO",
    }


def test_create_project_workspace_refuses_existing_project(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"

    create_project_workspace(
        project_directory,
        name="IGZO",
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="already exists",
    ):
        create_project_workspace(
            project_directory,
            name="IGZO",
        )


def test_create_project_workspace_initialises_existing_directory(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "IGZO"
    project_directory.mkdir()

    existing_file = project_directory / "notes.txt"
    existing_file.write_text(
        "Existing research notes\n",
        encoding="utf-8",
    )

    workspace = create_project_workspace(
        project_directory,
        name="IGZO",
    )

    assert workspace.root == project_directory.resolve()
    assert existing_file.read_text(
        encoding="utf-8"
    ) == "Existing research notes\n"

    assert (project_directory / "nsdw-project.yaml").is_file()
    assert (project_directory / "structure").is_dir()
    assert (project_directory / "convergence").is_dir()
    assert (project_directory / "production").is_dir()
    assert (project_directory / "reports").is_dir()


def test_load_project_workspace(tmp_path: Path) -> None:
    project_directory = tmp_path / "IGZO"

    create_project_workspace(
        project_directory,
        name="IGZO",
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

    create_project_workspace(
        project_directory,
        name="IGZO",
    )

    metadata_path = project_directory / "nsdw-project.yaml"

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

    metadata_path = project_directory / "nsdw-project.yaml"

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

    metadata_path = project_directory / "nsdw-project.yaml"

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

    create_project_workspace(
        project_directory,
        name="IGZO",
    )

    nested_directory = (
        project_directory
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

    create_project_workspace(
        project_directory,
        name="IGZO",
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