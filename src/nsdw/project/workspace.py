"""NSDW project workspace creation and representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_SCHEMA_VERSION = 1

PROJECT_DIRECTORIES = (
    "structure",
    "convergence",
    "production",
    "reports",
)


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


def create_project_workspace(
    root: str | Path,
    *,
    name: str,
) -> ProjectWorkspace:
    """Create a new NSDW project workspace."""

    root = Path(root).expanduser().resolve()

    metadata_path = root / "nsdw-project.yaml"

    if metadata_path.exists():
        raise ProjectWorkspaceError(
            f"NSDW project already exists: {root}"
        )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for directory in PROJECT_DIRECTORIES:
        (root / directory).mkdir(exist_ok=True)

    metadata = ProjectMetadata(
        schema_version=PROJECT_SCHEMA_VERSION,
        name=name,
    )

    metadata_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": metadata.schema_version,
                "name": metadata.name,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return ProjectWorkspace(
        root=root,
        metadata=metadata,
    )


def load_project_workspace(
    root: str | Path,
) -> ProjectWorkspace:
    """Load an existing NSDW project workspace."""

    root = Path(root).expanduser().resolve()

    metadata_path = root / "nsdw-project.yaml"

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

        metadata = ProjectMetadata(
            schema_version=raw_metadata["schema_version"],
            name=raw_metadata["name"],
        )

    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise ProjectWorkspaceError(
            f"Could not load NSDW project metadata: {metadata_path}"
        ) from exc

    if metadata.schema_version != PROJECT_SCHEMA_VERSION:
        raise ProjectWorkspaceError(
            "Unsupported NSDW project schema version: "
            f"{metadata.schema_version}. "
            f"Supported version: {PROJECT_SCHEMA_VERSION}."
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
        metadata_path = candidate / "nsdw-project.yaml"

        if metadata_path.is_file():
            return load_project_workspace(candidate)

    raise ProjectWorkspaceError(
        f"No NSDW project found from: {start}"
    )