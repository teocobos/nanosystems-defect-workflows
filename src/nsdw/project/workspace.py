"""NSDW project workspace creation and representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pydantic import ValidationError

from nsdw.project.models import ProjectConfig

PROJECT_SCHEMA_VERSION = 1


class ProjectWorkspaceError(RuntimeError):
    """Raised when an NSDW project workspace cannot be created or loaded."""


@dataclass(frozen=True)
class ProjectMetadata:
    """Persistent metadata describing an NSDW project."""

    schema_version: int
    name: str


@dataclass(frozen=True)
class ProjectWorkspace:
    """Filesystem representation of an NSDW project."""

    root: Path
    metadata: ProjectMetadata

    @property
    def name(self) -> str:
        """Return the project name."""

        return self.metadata.name

    @property
    def schema_version(self) -> int:
        """Return the project schema version."""

        return self.metadata.schema_version


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

    metadata = ProjectMetadata(
        schema_version=config.schema_version,
        name=config.name,
    )

    return ProjectWorkspace(
        root=root,
        metadata=metadata,
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