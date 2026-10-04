"""NSDW project workspace creation and representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pydantic import ValidationError

from nsdw.project.models import (
    CP2KProductionMethodology,
    ProjectConfig,
)

PROJECT_SCHEMA_VERSION = 1


class ProjectWorkspaceError(RuntimeError):
    """Raised when an NSDW project workspace cannot be created or loaded."""


@dataclass(frozen=True)
class ProjectWorkspace:
    """Filesystem representation of an NSDW project."""

    root: Path
    config: ProjectConfig

    @property
    def name(self) -> str:
        """Return the project name."""

        return self.config.name

    @property
    def schema_version(self) -> int:
        """Return the project schema version."""

        return self.config.schema_version

    @property
    def material(self) -> str:
        """Return the project material."""

        return self.config.material

    @property
    def components(self) -> tuple[str, ...]:
        """Return the enabled modelling components."""

        return tuple(self.config.components)


def write_project_config(
    root: str | Path,
    config: ProjectConfig,
) -> Path:
    """Persist validated NSDW project metadata."""

    root = Path(root).expanduser().resolve()

    metadata_path = root / "project.yaml"

    metadata_path.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return metadata_path


def update_cp2k_methodology(
    root: str | Path,
    methodology: CP2KProductionMethodology,
) -> ProjectWorkspace:
    """Persist the project's selected CP2K production methodology."""

    workspace = load_project_workspace(root)

    if "cp2k" not in workspace.config.components:
        raise ProjectWorkspaceError(
            "Cannot set CP2K methodology because the "
            "CP2K component is not enabled for this project."
        )

    updated_methodology = (
        workspace.config.methodology.model_copy(
            update={
                "cp2k": methodology,
            }
        )
    )

    updated_config = workspace.config.model_copy(
        update={
            "methodology": updated_methodology,
        }
    )

    write_project_config(
        workspace.root,
        updated_config,
    )

    return ProjectWorkspace(
        root=workspace.root,
        config=updated_config,
    )


def load_project_workspace(
    root: str | Path,
) -> ProjectWorkspace:
    """Load an existing NSDW project workspace."""

    root = Path(root).expanduser().resolve()

    metadata_path = root / "project.yaml"

    if not metadata_path.is_file():
        raise ProjectWorkspaceError(
            f"NSDW project metadata not found: {metadata_path}"
        )

    try:
        raw_metadata = yaml.safe_load(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        config = ProjectConfig.model_validate(
            raw_metadata
        )

    except (
        OSError,
        TypeError,
        yaml.YAMLError,
        ValidationError,
    ) as exc:
        raise ProjectWorkspaceError(
            f"Could not load NSDW project metadata: {metadata_path}"
        ) from exc

    if config.schema_version != PROJECT_SCHEMA_VERSION:
        raise ProjectWorkspaceError(
            "Unsupported NSDW project schema version: "
            f"{config.schema_version}. "
            f"Supported version: {PROJECT_SCHEMA_VERSION}."
        )

    return ProjectWorkspace(
    root=root,
    config=config,
    )


def find_project_workspace(
    start: str | Path,
) -> ProjectWorkspace:
    """Find the nearest NSDW project workspace from a path."""

    start = Path(start).expanduser().resolve()

    current = (
        start
        if start.is_dir()
        else start.parent
    )

    for candidate in (
        current,
        *current.parents,
    ):
        metadata_path = candidate / "project.yaml"

        if metadata_path.is_file():
            return load_project_workspace(candidate)

    raise ProjectWorkspaceError(
        f"No NSDW project found from: {start}"
    )