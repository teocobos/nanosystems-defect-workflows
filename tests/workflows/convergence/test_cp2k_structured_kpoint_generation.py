"""Structured k-point convergence candidate generation tests."""

import pytest

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
    CP2KKPointConfig,
    CP2KSCFConfig,
)
from nsdw.workflows.convergence.cp2k_generation import (
    CP2KConvergenceGenerationError,
    _candidate_config,
)
from nsdw.workflows.convergence.models import ConvergenceParameter


def candidate(base, value=(2, 2, 2)):
    return _candidate_config(
        base_config=base,
        parameter=ConvergenceParameter.KPOINTS,
        value=value,
        project_name="test-kpoints",
    )


def base_config(*, sampling=None, solver="DIAGONALIZATION"):
    return CP2KInputConfig(
        project_name="test",
        kpoint_sampling=sampling,
        scf=CP2KSCFConfig(solver=solver),
    )


def test_legacy_mesh_generation():
    result = candidate(base_config())
    assert result.k_points == (2, 2, 2)
    assert result.kpoint_sampling is None


def test_structured_monkhorst_pack_generation():
    base = base_config(
        sampling=CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(1, 1, 1),
        )
    )
    result = candidate(base, (3, 3, 3))
    assert result.k_points is None
    assert result.kpoint_sampling.mesh == (3, 3, 3)
    assert result.kpoint_sampling.scheme == "MONKHORST-PACK"


def test_structured_none_can_become_monkhorst_pack():
    base = base_config(
        sampling=CP2KKPointConfig(scheme="NONE")
    )
    result = candidate(base)
    assert result.k_points is None
    assert result.kpoint_sampling.scheme == "MONKHORST-PACK"


@pytest.mark.parametrize("scheme", ["GAMMA", "MACDONALD"])
def test_incompatible_base_sampling_rejected(scheme):
    if scheme == "MACDONALD":
        sampling = CP2KKPointConfig(
            scheme=scheme,
            mesh=(2, 2, 2),
            shift=(0.25, 0.25, 0.25),
        )
    else:
        sampling = CP2KKPointConfig(scheme=scheme)

    with pytest.raises(
        CP2KConvergenceGenerationError,
        match="Mesh convergence supports only",
    ):
        candidate(base_config(sampling=sampling))


def test_ot_solver_rejected():
    with pytest.raises(
        CP2KConvergenceGenerationError,
        match="DIAGONALIZATION",
    ):
        candidate(base_config(solver="OT"))


@pytest.mark.parametrize(
    "mesh",
    [
        (0, 2, 2),
        (-1, 2, 2),
        (2, 2),
        (2.0, 2, 2),
    ],
)
def test_invalid_mesh_rejected(mesh):
    with pytest.raises(
        CP2KConvergenceGenerationError,
        match="positive three-integer meshes",
    ):
        candidate(base_config(), mesh)


def test_model_copy_conflict_cannot_bypass_validation():
    base = base_config(
        sampling=CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(1, 1, 1),
        )
    ).model_copy(update={"k_points": (4, 4, 4)})

    with pytest.raises(
        CP2KConvergenceGenerationError,
        match="Invalid CP2K convergence candidate configuration",
    ):
        candidate(base)
