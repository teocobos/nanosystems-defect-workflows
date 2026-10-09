"""NSDW project workspace creation and representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pydantic import ValidationError

from nsdw.project.locking import project_write_lock
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


def _write_project_config_locked(
    root: str | Path,
    config: ProjectConfig,
    *,
    _verified_candidate_path: str | Path | None = None,
) -> Path:
    """Atomically write project metadata under the project lock.

    Validated methodology requires independent evidence verification
    at the persistence boundary.
    """
    import os
    import tempfile

    root = Path(root).expanduser().resolve()
    metadata_path = root / "project.yaml"

    # model_copy(update=...) does not validate nested updates.
    config = ProjectConfig.model_validate(
        config.model_dump(
            mode="json",
            warnings="error",
        )
    )

    if config.schema_version != PROJECT_SCHEMA_VERSION:
        raise ProjectWorkspaceError(
            "Cannot persist unsupported project schema version."
        )

    methodology = config.methodology.cp2k

    if _verified_candidate_path is not None:
        if methodology is None or methodology.status != "validated":
            raise ProjectWorkspaceError(
                "Verified promotion requires validated methodology."
            )

        # Local imports avoid a module-import cycle.
        from nsdw.workflows.convergence.cp2k_workflow import (
            load_cp2k_methodology_candidate,
        )
        from nsdw.workflows.convergence.cp2k_methodology_evidence import (
            validate_cp2k_methodology_evidence,
        )

        candidate = load_cp2k_methodology_candidate(
            _verified_candidate_path
        )

        proposed = methodology.model_dump(
            mode="json",
            warnings="error",
        )

        expected = candidate.model_dump(
            mode="json",
            warnings="error",
        )
        expected["status"] = "validated"

        if proposed != expected:
            raise ProjectWorkspaceError(
                "Validated methodology differs from its "
                "verified candidate."
            )

        validate_cp2k_methodology_evidence(
            project_root=root,
            candidate_path=_verified_candidate_path,
            methodology=candidate,
        )

    if (
        methodology is not None
        and methodology.status == "validated"
        and _verified_candidate_path is None
    ):
        raise ProjectWorkspaceError(
            "Cannot persist validated CP2K methodology "
            "through the ordinary project configuration writer."
        )

    if metadata_path.exists():
        existing = load_project_workspace(root)
        existing_methodology = existing.config.methodology.cp2k

        if (
            existing_methodology is not None
            and existing_methodology.status == "validated"

        ):
            raise ProjectWorkspaceError(
                "Cannot overwrite an existing validated CP2K "
                "methodology through ordinary project persistence."
            )

    content = yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=False,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".project-",
            suffix=".yaml.tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, metadata_path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return metadata_path


def _commit_verified_cp2k_methodology_locked(
    root: str | Path,
    methodology: CP2KProductionMethodology,
    *,
    candidate_path: str | Path,
) -> ProjectWorkspace:
    """Commit an independently verified methodology under the project lock.

    The caller must hold the project lock and must have completed
    scientific evidence verification in the same transaction.
    """
    workspace = load_project_workspace(root)

    if "cp2k" not in workspace.config.components:
        raise ProjectWorkspaceError(
            "Cannot promote CP2K methodology: "
            "CP2K is not enabled for this project."
        )

    existing = workspace.config.methodology.cp2k

    if existing is not None and existing.status == "validated":
        raise ProjectWorkspaceError(
            "Cannot overwrite an existing validated CP2K methodology."
        )

    if methodology.status != "validated":
        raise ProjectWorkspaceError(
            "Promotion commit requires validated methodology status."
        )

    updated_methodology = workspace.config.methodology.model_copy(
        update={"cp2k": methodology}
    )

    updated_config = workspace.config.model_copy(
        update={"methodology": updated_methodology}
    )

    _write_project_config_locked(
        workspace.root,
        updated_config,
        _verified_candidate_path=candidate_path,
    )

    return load_project_workspace(workspace.root)


def write_project_config(
    root: str | Path,
    config: ProjectConfig,
) -> Path:
    """Persist ordinary project metadata under the project lock."""
    with project_write_lock(root):
        return _write_project_config_locked(
            root,
            config,
        )




def update_cp2k_methodology(
    root: str | Path,
    methodology: CP2KProductionMethodology,
) -> ProjectWorkspace:
    """Persist the project's selected CP2K production methodology."""

    with project_write_lock(root):
        workspace = load_project_workspace(root)

        if methodology.status == "validated":
            raise ProjectWorkspaceError(
                "Cannot persist a validated CP2K methodology "
                "through the ordinary project update API. "
                "Use evidence-verified methodology promotion."
            )

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

        _write_project_config_locked(
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