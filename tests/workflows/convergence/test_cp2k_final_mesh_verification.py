"""Regression tests for final-mesh CP2K convergence verification."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nsdw.workflows.convergence import cp2k_workflow as workflow


def _config(*, mesh=(2, 2, 2), solver="DIAGONALIZATION"):
    return SimpleNamespace(
        k_points=mesh,
        scf=SimpleNamespace(solver=solver),
        cutoff_ry=560.0,
        relative_cutoff_ry=40.0,
    )


def _report(
    study,
    selected_value,
    *,
    stable=True,
    difference=0.0002,
    tolerance=0.001,
    converged=True,
    structure_hash="a" * 64,
    n_atoms=9,
):
    selected = next(
        candidate
        for candidate in study.candidates
        if isinstance(candidate.value, workflow.Quantity)
        and candidate.value.value == selected_value
    )

    result = SimpleNamespace(
        label=selected.label,
        tail_stable=stable,
        maximum_higher_cost_energy_difference_ev_per_atom=difference,
    )

    return SimpleNamespace(
        parameter=study.parameter,
        energy_tolerance_satisfied=converged,
        energy_tolerance_ev_per_atom=tolerance,
        candidates=(result,),
        structure_hash=structure_hash,
        n_atoms=n_atoms,
    )


def _verification_reports(**overrides):
    cutoff_study = workflow.build_standard_cp2k_cutoff_study()
    relative_study = workflow.build_standard_cp2k_relative_cutoff_study()

    cutoff_options = overrides.get("cutoff_options", {})
    relative_options = overrides.get("relative_options", {})

    return (
        _report(cutoff_study, 560.0, **cutoff_options),
        _report(relative_study, 40.0, **relative_options),
    )


def test_final_mesh_verification_succeeds(tmp_path):
    cutoff, relative = _verification_reports()

    with (
        patch.object(
            workflow,
            "generate_cp2k_convergence_study",
        ) as generate,
        patch.object(
            workflow,
            "run_and_report_cp2k_convergence_campaign",
            side_effect=(cutoff, relative),
        ) as execute,
    ):
        result = workflow.verify_cp2k_final_mesh_convergence(
            structure=object(),
            converged_config=_config(),
            workflow_directory=tmp_path,
        )

    assert result.cutoff_report is cutoff
    assert result.relative_cutoff_report is relative
    assert generate.call_count == 2
    assert execute.call_count == 2

    for call in generate.call_args_list:
        assert call.kwargs["base_config"].k_points == (2, 2, 2)
        assert (
            call.kwargs["base_config"].scf.solver
            == "DIAGONALIZATION"
        )


@pytest.mark.parametrize(
    "options",
    [
        {"stable": False},
        {"difference": 0.002},
        {"difference": float("nan")},
        {"converged": False},
        {"tolerance": 0.0},
    ],
)
def test_unstable_cutoff_stops_before_relative_campaign(
    tmp_path, options
):
    cutoff, relative = _verification_reports(
        cutoff_options=options
    )

    with (
        patch.object(
            workflow,
            "generate_cp2k_convergence_study",
        ) as generate,
        patch.object(
            workflow,
            "run_and_report_cp2k_convergence_campaign",
            side_effect=(cutoff, relative),
        ) as execute,
    ):
        with pytest.raises(
            workflow.CP2KMethodologyValidationError
        ):
            workflow.verify_cp2k_final_mesh_convergence(
                structure=object(),
                converged_config=_config(),
                workflow_directory=tmp_path,
            )

    assert generate.call_count == 1
    assert execute.call_count == 1


@pytest.mark.parametrize(
    "options",
    [
        {"stable": False},
        {"difference": 0.002},
        {"converged": False},
        {"structure_hash": "b" * 64},
        {"n_atoms": 10},
    ],
)
def test_invalid_relative_verification_fails(
    tmp_path, options
):
    cutoff, relative = _verification_reports(
        relative_options=options
    )

    with (
        patch.object(
            workflow,
            "generate_cp2k_convergence_study",
        ),
        patch.object(
            workflow,
            "run_and_report_cp2k_convergence_campaign",
            side_effect=(cutoff, relative),
        ),
    ):
        with pytest.raises(
            workflow.CP2KMethodologyValidationError
        ):
            workflow.verify_cp2k_final_mesh_convergence(
                structure=object(),
                converged_config=_config(),
                workflow_directory=tmp_path,
            )


@pytest.mark.parametrize(
    "config",
    [
        _config(mesh=None),
        _config(solver="OT"),
    ],
)
def test_invalid_production_configuration_stops_before_execution(
    tmp_path, config
):
    with (
        patch.object(
            workflow,
            "generate_cp2k_convergence_study",
        ) as generate,
        patch.object(
            workflow,
            "run_and_report_cp2k_convergence_campaign",
        ) as execute,
    ):
        with pytest.raises(
            workflow.CP2KMethodologyValidationError
        ):
            workflow.verify_cp2k_final_mesh_convergence(
                structure=object(),
                converged_config=config,
                workflow_directory=tmp_path,
            )

    generate.assert_not_called()
    execute.assert_not_called()
