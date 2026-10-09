"""Tests for NSDW project workspace loading and discovery."""

from pathlib import Path

import pytest
import yaml

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KSCFConfig,
    CP2KXCFunctional,
)

from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
    ProjectConfig,
)

from nsdw.project.scaffold import create_project

from nsdw.project.workspace import (
    ProjectWorkspaceError,
    find_project_workspace,
    load_project_workspace,
    update_cp2k_methodology,
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


def test_load_project_workspace_defaults_missing_methodology(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "legacy-project"
    project_directory.mkdir()

    metadata_path = project_directory / "project.yaml"

    metadata_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: Legacy IGZO",
                "material: IGZO",
                "nsdw_version: 0.1.0",
                "components:",
                "  - cp2k",
                "",
            ]
        ),
        encoding="utf-8",
    )

    workspace = load_project_workspace(
        project_directory
    )

    assert workspace.config.methodology.cp2k is None


def test_load_project_workspace_with_cp2k_methodology(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "igzo-project"
    project_directory.mkdir()

    metadata_path = project_directory / "project.yaml"

    metadata_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: IGZO",
                "material: InGaZnO4",
                "nsdw_version: 0.1.0",
                "components:",
                "  - cp2k",
                "methodology:",
                "  cp2k:",
                "    status: validated",
                "    functional: PBE",
                "    cutoff_ry: 600.0",
                "    relative_cutoff_ry: 60.0",
                "    k_points: null",
                "    scf:",
                "      scf_guess: ATOMIC",
                "      eps_scf: 1.0e-6",
                "      max_scf: 100",
                "      outer_scf_max: 10",
                "      ot_minimizer: CG",
                "      ot_preconditioner: FULL_SINGLE_INVERSE",
                "      energy_gap: 0.001",
                "    basis_potential:",
                "      basis_set_file: BASIS_MOLOPT",
                "      potential_file: GTH_POTENTIALS",
                "      kinds: []",
                "    provenance:",
                "      source: convergence",
                "      workflow: standard_cp2k_convergence",
                "      cutoff_report: workflows/convergence/cutoff/convergence-report.json",
                "      relative_cutoff_report: workflows/convergence/relative_cutoff/convergence-report.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    workspace = load_project_workspace(
        project_directory
    )

    methodology = workspace.config.methodology.cp2k

    assert methodology is not None
    assert methodology.status == "validated"
    assert methodology.functional.value == "PBE"
    assert methodology.cutoff_ry == 600.0
    assert methodology.relative_cutoff_ry == 60.0
    assert methodology.k_points is None

    assert methodology.scf.scf_guess == "ATOMIC"
    assert methodology.scf.eps_scf == 1.0e-6
    assert methodology.scf.max_scf == 100

    assert methodology.basis_potential is not None
    assert (
        methodology.basis_potential.basis_set_file
        == "BASIS_MOLOPT"
    )
    assert (
        methodology.basis_potential.potential_file
        == "GTH_POTENTIALS"
    )

    assert methodology.provenance.source == "convergence"
    assert (
        methodology.provenance.workflow
        == "standard_cp2k_convergence"
    )


def test_load_project_workspace_with_cp2k_methodology(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "igzo-project"
    project_directory.mkdir()

    metadata_path = project_directory / "project.yaml"

    metadata_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: IGZO",
                "material: InGaZnO4",
                "nsdw_version: 0.1.0",
                "components:",
                "  - cp2k",
                "methodology:",
                "  cp2k:",
                "    status: validated",
                "    functional: PBE",
                "    cutoff_ry: 600.0",
                "    relative_cutoff_ry: 60.0",
                "    k_points: null",
                "    scf:",
                "      scf_guess: ATOMIC",
                "      eps_scf: 1.0e-6",
                "      max_scf: 100",
                "      outer_scf_max: 10",
                "      ot_minimizer: CG",
                "      ot_preconditioner: FULL_SINGLE_INVERSE",
                "      energy_gap: 0.001",
                "    basis_potential:",
                "      basis_set_file: BASIS_MOLOPT",
                "      potential_file: GTH_POTENTIALS",
                "      kinds: []",
                "    provenance:",
                "      source: convergence",
                "      workflow: standard_cp2k_convergence",
                "      cutoff_report: workflows/convergence/cutoff/convergence-report.json",
                "      relative_cutoff_report: workflows/convergence/relative_cutoff/convergence-report.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    workspace = load_project_workspace(
        project_directory
    )

    methodology = workspace.config.methodology.cp2k

    assert methodology is not None
    assert methodology.status == "validated"
    assert methodology.functional.value == "PBE"
    assert methodology.cutoff_ry == 600.0
    assert methodology.relative_cutoff_ry == 60.0
    assert methodology.k_points is None

    assert methodology.scf.scf_guess == "ATOMIC"
    assert methodology.scf.eps_scf == 1.0e-6
    assert methodology.scf.max_scf == 100

    assert methodology.basis_potential is not None
    assert (
        methodology.basis_potential.basis_set_file
        == "BASIS_MOLOPT"
    )
    assert (
        methodology.basis_potential.potential_file
        == "GTH_POTENTIALS"
    )

    assert methodology.provenance.source == "convergence"
    assert (
        methodology.provenance.workflow
        == "standard_cp2k_convergence"
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


def test_update_cp2k_methodology_persists_configuration(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "igzo-project"

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

    methodology = CP2KProductionMethodology(
        status="candidate",
        functional=CP2KXCFunctional.PBE,
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        k_points=None,
        scf=CP2KSCFConfig(),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(),
        ),
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
            cutoff_report=(
                "workflows/convergence/cutoff/"
                "convergence-report.json"
            ),
            relative_cutoff_report=(
                "workflows/convergence/relative_cutoff/"
                "convergence-report.json"
            ),
        ),
    )

    updated = update_cp2k_methodology(
        project_directory,
        methodology,
    )

    assert updated.name == "IGZO"
    assert updated.material == "InGaZnO4"
    assert updated.components == ("cp2k",)

    persisted = load_project_workspace(
        project_directory
    )

    persisted_methodology = (
        persisted.config.methodology.cp2k
    )

    assert persisted_methodology is not None
    assert persisted_methodology.status == "candidate"
    assert persisted_methodology.functional.value == "PBE"
    assert persisted_methodology.cutoff_ry == 600.0
    assert persisted_methodology.relative_cutoff_ry == 60.0

    assert persisted_methodology.basis_potential is not None
    assert (
        persisted_methodology.basis_potential.basis_set_file
        == "BASIS_MOLOPT"
    )

    assert (
        persisted_methodology.provenance.workflow
        == "standard_cp2k_convergence"
    )


def test_update_cp2k_methodology_requires_cp2k_component(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / "mace-project"

    config = ProjectConfig(
        name="MACE project",
        material="IGZO",
        nsdw_version="0.1.0",
        components=["mace"],
    )

    create_project(
        root=project_directory,
        config=config,
    )

    methodology = CP2KProductionMethodology(
        status="candidate",
        functional=CP2KXCFunctional.PBE,
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        scf=CP2KSCFConfig(),
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
        ),
    )

    with pytest.raises(
        ProjectWorkspaceError,
        match="CP2K component is not enabled",
    ):
        update_cp2k_methodology(
            project_directory,
            methodology,
        )