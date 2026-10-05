"""CP2K production configuration builders."""

from pathlib import Path

from pymatgen.core import Structure

from nsdw.calculators.cp2k import (
    build_aiida_cp2k_parameters,
    write_cp2k_package,
)
from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
)
from nsdw.project.models import (
    CP2KProductionMethodology,
)
from nsdw.workflows.production.models import (
    CP2KProductionRecipe,
)
from nsdw.project.workspace import (
    ProjectWorkspace,
)
from nsdw.execution import (
    AiiDACp2kResources,
    AiiDASubmission,
    submit_cp2k_aiida,
)


def build_cp2k_production_config(
    *,
    methodology: CP2KProductionMethodology,
    recipe: CP2KProductionRecipe,
) -> CP2KInputConfig:
    """Combine validated methodology with calculation-specific settings."""

    if methodology.status != "validated":
        raise ValueError(
            "CP2K production calculations require a "
            "validated project methodology."
        )

    return CP2KInputConfig(
        project_name=recipe.project_name,
        run_type=recipe.run_type,
        charge=recipe.charge,
        multiplicity=recipe.multiplicity,
        functional=methodology.functional,
        cutoff_ry=methodology.cutoff_ry,
        relative_cutoff_ry=methodology.relative_cutoff_ry,
        k_points=methodology.k_points,
        scf=methodology.scf,
        basis_potential=methodology.basis_potential,
    )


def build_cp2k_production_config_from_workspace(
    *,
    workspace: ProjectWorkspace,
    recipe: CP2KProductionRecipe,
) -> CP2KInputConfig:
    """Build a CP2K production config from an NSDW project workspace."""

    methodology = workspace.config.methodology.cp2k

    if methodology is None:
        raise ValueError(
            "No validated CP2K methodology is stored "
            "for this project."
        )

    return build_cp2k_production_config(
        methodology=methodology,
        recipe=recipe,
    )


def generate_cp2k_production_package(
    *,
    workspace: ProjectWorkspace,
    structure: Structure,
    recipe: CP2KProductionRecipe,
    output_directory: str | Path | None = None,
) -> Path:
    """Generate a portable CP2K production calculation package."""

    config = build_cp2k_production_config_from_workspace(
        workspace=workspace,
        recipe=recipe,
    )

    if output_directory is None:
        output_directory = (
            workspace.root
            / "working"
            / "calculations"
            / "cp2k"
            / recipe.project_name
        )

    return write_cp2k_package(
        structure=structure,
        config=config,
        output_directory=output_directory,
    )

def submit_cp2k_production_aiida(
    *,
    workspace: ProjectWorkspace,
    structure: Structure,
    recipe: CP2KProductionRecipe,
    code_label: str,
    resources: AiiDACp2kResources,
    profile: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> AiiDASubmission:
    """Submit a validated CP2K production calculation through AiiDA.

    NSDW owns the validated scientific configuration. The configuration
    is adapted into the structured parameters expected by aiida-cp2k,
    while AiiDA owns execution, transport, scheduling, parsing, and
    provenance.
    """

    config = build_cp2k_production_config_from_workspace(
        workspace=workspace,
        recipe=recipe,
    )

    parameters = build_aiida_cp2k_parameters(
        config=config,
        structure=structure,
    )

    return submit_cp2k_aiida(
        calculation_id=recipe.project_name,
        code_label=code_label,
        parameters=parameters,
        structure=structure.to_ase_atoms(),
        resources=resources,
        profile=profile,
        label=label,
        description=description,
    )
