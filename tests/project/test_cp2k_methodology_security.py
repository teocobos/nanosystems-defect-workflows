"""Security regression tests for CP2K methodology persistence."""

from pathlib import Path

import pytest

from nsdw.calculators.cp2k.generation_models import (
    CP2KSCFConfig,
)
from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
    ProjectConfig,
)
from nsdw.project.scaffold import create_project
from nsdw.project.workspace import (
    ProjectWorkspaceError,
    load_project_workspace,
    update_cp2k_methodology,
)
from nsdw.workflows.convergence.cp2k_workflow import (
    CP2KMethodologyValidationError,
    promote_cp2k_methodology_candidate,
)


def _methodology(status="candidate"):
    return CP2KProductionMethodology(
        status=status,
        functional="PBE",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        scf=CP2KSCFConfig(),
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
            cutoff_report="cutoff/convergence-report.json",
            relative_cutoff_report=(
                "relative_cutoff/convergence-report.json"
            ),
        ),
    )


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"

    create_project(
        root=root,
        config=ProjectConfig(
            name="Security Test",
            material="SiO2",
            nsdw_version="0.1.0",
            components=["cp2k"],
        ),
    )

    return root


def test_new_methodology_defaults_to_candidate():
    methodology = _methodology()
    assert methodology.status == "candidate"


def test_rejects_direct_validated_methodology(project):
    with pytest.raises(
        ProjectWorkspaceError,
        match="evidence-verified methodology promotion",
    ):
        update_cp2k_methodology(
            project,
            _methodology(status="validated"),
        )


def test_rejected_validated_write_preserves_project_file(project):
    config_path = project / "project.yaml"
    before = config_path.read_bytes()

    with pytest.raises(ProjectWorkspaceError):
        update_cp2k_methodology(
            project,
            _methodology(status="validated"),
        )

    assert config_path.read_bytes() == before


def test_candidate_methodology_persists(project):
    update_cp2k_methodology(project, _methodology())

    persisted = load_project_workspace(project)
    assert persisted.config.methodology.cp2k is not None
    assert persisted.config.methodology.cp2k.status == "candidate"


def test_invalid_candidate_preserves_project_file(project):
    """Malformed candidates cannot modify project metadata."""
    from pydantic import ValidationError

    config_path = project / "project.yaml"
    before = config_path.read_bytes()

    candidate_path = project / "methodology-candidate.yaml"
    candidate_path.write_text(
        "status: candidate\\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        promote_cp2k_methodology_candidate(
            project_root=project,
            candidate_path=candidate_path,
        )

    assert config_path.read_bytes() == before



def _config_with_methodology(project, methodology):
    workspace = load_project_workspace(project)

    updated_methodology = (
        workspace.config.methodology.model_copy(
            update={"cp2k": methodology}
        )
    )

    return workspace.config.model_copy(
        update={"methodology": updated_methodology}
    )


def test_writer_rejects_validated_methodology(project):
    from nsdw.project.workspace import write_project_config

    config = _config_with_methodology(
        project,
        _methodology(status="validated"),
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="Cannot persist validated CP2K methodology",
    ):
        write_project_config(project, config)


def test_writer_rejection_preserves_project_yaml(project):
    from nsdw.project.workspace import write_project_config

    metadata = project / "project.yaml"
    before = metadata.read_bytes()

    config = _config_with_methodology(
        project,
        _methodology(status="validated"),
    )

    with pytest.raises(ProjectWorkspaceError):
        write_project_config(project, config)

    assert metadata.read_bytes() == before
    assert list(project.glob(".project-*.yaml.tmp")) == []


def test_atomic_replace_failure_preserves_original(
    project,
    monkeypatch,
):
    import nsdw.project.workspace as workspace_module

    metadata = project / "project.yaml"
    before = metadata.read_bytes()

    config = _config_with_methodology(
        project,
        _methodology(),
    )

    def fail_replace(source, destination):
        raise OSError("Simulated atomic replacement failure")

    import os

    monkeypatch.setattr(
        os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="Simulated atomic replacement failure",
    ):
        workspace_module.write_project_config(
            project,
            config,
        )

    assert metadata.read_bytes() == before
    assert list(project.glob(".project-*.yaml.tmp")) == []


def test_successful_atomic_write_cleans_temporary_files(project):
    from nsdw.project.workspace import write_project_config

    config = _config_with_methodology(
        project,
        _methodology(),
    )

    write_project_config(project, config)

    persisted = load_project_workspace(project)

    assert persisted.config.methodology.cp2k is not None
    assert persisted.config.methodology.cp2k.status == "candidate"
    assert list(project.glob(".project-*.yaml.tmp")) == []


def test_existing_validated_methodology_cannot_be_overwritten(
    project,
):
    import yaml

    from nsdw.project.workspace import write_project_config

    metadata = project / "project.yaml"
    config = _config_with_methodology(
        project,
        _methodology(status="validated"),
    )

    # Simulate a previously completed evidence-verified promotion.
    metadata.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    before = metadata.read_bytes()

    candidate = config.model_copy(
        update={
            "methodology": config.methodology.model_copy(
                update={"cp2k": _methodology()}
            )
        }
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="Cannot overwrite an existing validated",
    ):
        write_project_config(project, candidate)

    assert metadata.read_bytes() == before


def test_writer_revalidates_nested_methodology(project):
    from pydantic import ValidationError
    from pydantic_core import PydanticSerializationError

    from nsdw.project.workspace import write_project_config

    metadata = project / "project.yaml"
    before = metadata.read_bytes()

    config = _config_with_methodology(
        project,
        _methodology(),
    )

    malformed = config.methodology.model_copy(
        update={"cp2k": {"status": "candidate"}}
    )

    invalid_config = config.model_copy(
        update={"methodology": malformed}
    )

    with pytest.raises(
        (
            ValidationError,
            PydanticSerializationError,
            AttributeError,
        )
    ):
        write_project_config(project, invalid_config)

    assert metadata.read_bytes() == before


def test_writer_rejects_unsupported_schema_version(project):
    from nsdw.project.workspace import write_project_config

    metadata = project / "project.yaml"
    before = metadata.read_bytes()

    workspace = load_project_workspace(project)

    invalid_config = workspace.config.model_copy(
        update={"schema_version": 999}
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="unsupported project schema version",
    ):
        write_project_config(project, invalid_config)

    assert metadata.read_bytes() == before


def test_public_writer_acquires_project_lock(
    project,
    monkeypatch,
):
    import nsdw.project.workspace as workspace_module

    original_lock = workspace_module.project_write_lock
    observed = []

    from contextlib import contextmanager

    @contextmanager
    def observed_lock(root):
        observed.append("entered")
        with original_lock(root):
            yield
        observed.append("released")

    monkeypatch.setattr(
        workspace_module,
        "project_write_lock",
        observed_lock,
    )

    config = load_project_workspace(project).config

    workspace_module.write_project_config(
        project,
        config,
    )

    assert observed == ["entered", "released"]


def test_candidate_update_works_with_locked_writer(project):
    update_cp2k_methodology(
        project,
        _methodology(),
    )

    persisted = load_project_workspace(project)

    assert persisted.config.methodology.cp2k is not None
    assert persisted.config.methodology.cp2k.status == "candidate"
    assert (project / ".project.lock").is_file()



def test_direct_validated_writer_bypass_is_rejected(project):
    """Ordinary persistence cannot bypass evidence verification."""
    from nsdw.project.workspace import (
        _write_project_config_locked,
    )

    metadata = project / "project.yaml"
    before = metadata.read_bytes()

    config = _config_with_methodology(
        project,
        _methodology(status="validated"),
    )

    with pytest.raises(ProjectWorkspaceError):
        _write_project_config_locked(
            project,
            config,
        )

    assert metadata.read_bytes() == before
