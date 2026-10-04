import pytest

from pymatgen.core import (
    Lattice,
    Structure,
)

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KKindConfig,
    CP2KSCFConfig,
    CP2KXCFunctional,
)
from nsdw.workflows.production.models import (
    CP2KProductionRecipe,
)
from nsdw.project.models import (
    CP2KMethodologyProvenance,
    CP2KProductionMethodology,
    ProjectConfig,
)

from nsdw.project.scaffold import (
    create_project,
)

from nsdw.project.workspace import (
    load_project_workspace,
)

from nsdw.workflows.production.cp2k import (
    build_cp2k_production_config,
    build_cp2k_production_config_from_workspace,
    generate_cp2k_production_package,
)

def _methodology(
    *,
    status: str = "validated",
) -> CP2KProductionMethodology:
    return CP2KProductionMethodology(
        status=status,
        functional=CP2KXCFunctional.PBE,
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        k_points=(2, 2, 2),
        scf=CP2KSCFConfig(
            eps_scf=1.0e-7,
            max_scf=150,
        ),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(),
        ),
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
        ),
    )


def test_build_cp2k_production_config_combines_methodology_and_recipe() -> None:
    recipe = CP2KProductionRecipe(
        project_name="oxygen-vacancy-plus-two",
        run_type="ENERGY_FORCE",
        charge=2,
        multiplicity=1,
    )

    config = build_cp2k_production_config(
        methodology=_methodology(),
        recipe=recipe,
    )

    assert config.project_name == "oxygen-vacancy-plus-two"
    assert config.run_type == "ENERGY_FORCE"
    assert config.charge == 2
    assert config.multiplicity == 1

    assert config.functional == CP2KXCFunctional.PBE
    assert config.cutoff_ry == 560.0
    assert config.relative_cutoff_ry == 40.0
    assert config.k_points == (2, 2, 2)

    assert config.scf.eps_scf == 1.0e-7
    assert config.scf.max_scf == 150

    assert config.basis_potential is not None
    assert (
        config.basis_potential.basis_set_file
        == "BASIS_MOLOPT"
    )
    assert (
        config.basis_potential.potential_file
        == "GTH_POTENTIALS"
    )


def test_build_cp2k_production_config_rejects_candidate_methodology() -> None:
    recipe = CP2KProductionRecipe(
        project_name="igzo",
    )

    with pytest.raises(
        ValueError,
        match="validated project methodology",
    ):
        build_cp2k_production_config(
            methodology=_methodology(
                status="candidate",
            ),
            recipe=recipe,
        )


def test_build_cp2k_production_config_from_workspace(
    tmp_path,
) -> None:
    project_root = tmp_path / "igzo-project"

    config = ProjectConfig(
        name="igzo-project",
        material="IGZO",
        nsdw_version="0.1.0",
        components=["cp2k"],
        methodology={
            "cp2k": _methodology(),
        },
    )

    create_project(
        root=project_root,
        config=config,
    )

    workspace = load_project_workspace(
        project_root
    )

    recipe = CP2KProductionRecipe(
        project_name="igzo-neutral",
        charge=0,
        multiplicity=1,
    )

    production_config = (
        build_cp2k_production_config_from_workspace(
            workspace=workspace,
            recipe=recipe,
        )
    )

    assert production_config.project_name == "igzo-neutral"
    assert production_config.charge == 0
    assert production_config.multiplicity == 1

    assert production_config.cutoff_ry == 560.0
    assert production_config.relative_cutoff_ry == 40.0
    assert production_config.functional == CP2KXCFunctional.PBE


def test_build_cp2k_production_config_from_workspace_requires_methodology(
    tmp_path,
) -> None:
    project_root = tmp_path / "igzo-project"

    create_project(
        root=project_root,
        config=ProjectConfig(
            name="igzo-project",
            material="IGZO",
            nsdw_version="0.1.0",
            components=["cp2k"],
        ),
    )

    workspace = load_project_workspace(
        project_root
    )

    recipe = CP2KProductionRecipe(
        project_name="igzo-neutral",
    )

    with pytest.raises(
        ValueError,
        match="No validated CP2K methodology",
    ):
        build_cp2k_production_config_from_workspace(
            workspace=workspace,
            recipe=recipe,
        )


def test_generate_cp2k_production_package_from_workspace(
    tmp_path,
) -> None:
    project_root = tmp_path / "oxide-project"

    methodology = CP2KProductionMethodology(
        status="validated",
        functional=CP2KXCFunctional.PBE,
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
        k_points=None,
        scf=CP2KSCFConfig(
            eps_scf=1.0e-7,
            max_scf=150,
        ),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_MOLOPT",
            potential_file="GTH_POTENTIALS",
            kinds=(
                CP2KKindConfig(
                    element="O",
                    basis_set="DZVP-MOLOPT-SR-GTH",
                    potential="GTH-PBE-q6",
                ),
            ),
        ),
        provenance=CP2KMethodologyProvenance(
            source="convergence",
            workflow="standard_cp2k_convergence",
        ),
    )

    create_project(
        root=project_root,
        config=ProjectConfig(
            name="oxide-project",
            material="O",
            nsdw_version="0.1.0",
            components=["cp2k"],
            methodology={
                "cp2k": methodology,
            },
        ),
    )

    workspace = load_project_workspace(
        project_root
    )

    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["O"],
        coords=[
            [0.0, 0.0, 0.0],
        ],
    )

    recipe = CP2KProductionRecipe(
        project_name="oxygen-neutral",
        run_type="ENERGY_FORCE",
        charge=0,
        multiplicity=1,
    )

    package_directory = (
        generate_cp2k_production_package(
            workspace=workspace,
            structure=structure,
            recipe=recipe,
        )
    )

    expected_directory = (
        project_root
        / "working"
        / "calculations"
        / "cp2k"
        / "oxygen-neutral"
    ).resolve()

    assert package_directory == expected_directory

    input_path = (
        expected_directory
        / "oxygen-neutral.inp"
    )

    coordinate_path = (
        expected_directory
        / "oxygen-neutral.xyz"
    )

    assert input_path.is_file()
    assert coordinate_path.is_file()

    input_text = input_path.read_text(
        encoding="utf-8",
    )

    assert "PROJECT oxygen-neutral" in input_text
    assert "RUN_TYPE ENERGY_FORCE" in input_text
    assert "CHARGE 0" in input_text
    assert "MULTIPLICITY 1" in input_text

    assert "CUTOFF 560" in input_text
    assert "REL_CUTOFF 40" in input_text

    assert (
        "BASIS_SET_FILE_NAME BASIS_MOLOPT"
        in input_text
    )

    assert (
        "POTENTIAL_FILE_NAME GTH_POTENTIALS"
        in input_text
    )

    assert (
        "BASIS_SET DZVP-MOLOPT-SR-GTH"
        in input_text
    )

    assert (
        "POTENTIAL GTH-PBE-q6"
        in input_text
    )

    assert (
        "COORD_FILE_NAME oxygen-neutral.xyz"
        in input_text
    )