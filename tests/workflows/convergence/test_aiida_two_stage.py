"""Contract tests for the two-stage AiiDA controller."""

from aiida.orm import Dict, Int

from nsdw.workflows.convergence.aiida_two_stage import (
    CP2KTwoStageConvergenceWorkChain,
)


def test_controller_inputs():
    inputs = CP2KTwoStageConvergenceWorkChain.spec().inputs

    assert "code" in inputs
    assert "structure" in inputs
    assert "cutoff_candidates" in inputs
    assert "resources" in inputs
    assert "relative_candidate_count" in inputs
    accepted_types = inputs["relative_candidate_count"].valid_type
    assert Int in accepted_types
    assert type(None) in accepted_types


def test_controller_outputs():
    outputs = CP2KTwoStageConvergenceWorkChain.spec().outputs

    assert set((
        "cutoff_summary",
        "cutoff_analysis",
        "relative_candidates",
        "relative_summary",
        "relative_analysis",
    )).issubset(outputs)


def test_controller_exit_codes():
    codes = CP2KTwoStageConvergenceWorkChain.spec().exit_codes

    assert codes.ERROR_INVALID_INPUT.status == 410
    assert codes.ERROR_CUTOFF_FAILED.status == 411
    assert codes.ERROR_STAGE_PREPARATION_FAILED.status == 412
    assert codes.ERROR_RELATIVE_FAILED.status == 413


def test_candidate_output_type():
    outputs = CP2KTwoStageConvergenceWorkChain.spec().outputs
    assert outputs["relative_candidates"].valid_type is Dict



from types import SimpleNamespace
from unittest.mock import Mock, patch

from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Str

from nsdw.workflows.convergence.aiida_two_stage import (
    prepare_relative_candidates,
)


class FakeDict:
    """Minimal stand-in for an AiiDA Dict in unit tests."""

    def __init__(self, data):
        self.data = data

    def get_dict(self):
        from copy import deepcopy
        return deepcopy(self.data)


def make_fake_controller():
    controller = Mock()
    controller.inputs = AttributeDict({
        "code": Mock(name="mock_cp2k_code"),
        "structure": Mock(name="mock_structure"),
        "cutoff_candidates": FakeDict({
            "400-Ry": {
                "FORCE_EVAL": {
                    "DFT": {
                        "MGRID": {
                            "CUTOFF": 400.0,
                            "REL_CUTOFF": 30.0,
                        }
                    }
                }
            }
        }),
        "resources": FakeDict({
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 600,
        }),
    })

    controller.ctx = AttributeDict()
    controller.exit_codes = (
        CP2KTwoStageConvergenceWorkChain.spec().exit_codes
    )

    return controller


def test_failed_cutoff_stops_workflow():
    controller = make_fake_controller()

    controller.ctx.cutoff_process = SimpleNamespace(
        pk=123,
        exit_status=400,
        is_finished_ok=False,
        outputs=AttributeDict(),
    )

    result = CP2KTwoStageConvergenceWorkChain.inspect_cutoff(
        controller
    )

    assert result.status == 411
    controller.out.assert_not_called()


def test_successful_cutoff_exposes_analysis():
    controller = make_fake_controller()

    summary = FakeDict({"candidate": "completed"})
    analysis = FakeDict({"report": {"parameter": "cutoff"}})

    controller.ctx.cutoff_process = SimpleNamespace(
        pk=123,
        exit_status=0,
        is_finished_ok=True,
        outputs=AttributeDict({
            "summary": summary,
            "analysis": analysis,
        }),
    )

    result = CP2KTwoStageConvergenceWorkChain.inspect_cutoff(
        controller
    )

    assert result is None
    controller.out.assert_any_call("cutoff_summary", summary)
    controller.out.assert_any_call("cutoff_analysis", analysis)


def test_failed_relative_cutoff_returns_exit_code():
    controller = make_fake_controller()

    controller.ctx.relative_process = SimpleNamespace(
        pk=124,
        exit_status=400,
        is_finished_ok=False,
        outputs=AttributeDict(),
    )

    result = CP2KTwoStageConvergenceWorkChain.inspect_relative_cutoff(
        controller
    )

    assert result.status == 413


def test_successful_relative_cutoff_exposes_analysis():
    controller = make_fake_controller()

    summary = FakeDict({"candidate": "completed"})
    analysis = FakeDict({
        "report": {"parameter": "relative_cutoff"},
        "provisional": True,
    })

    controller.ctx.relative_process = SimpleNamespace(
        pk=124,
        exit_status=0,
        is_finished_ok=True,
        outputs=AttributeDict({
            "summary": summary,
            "analysis": analysis,
        }),
    )

    result = CP2KTwoStageConvergenceWorkChain.inspect_relative_cutoff(
        controller
    )

    assert result is None
    controller.out.assert_any_call("relative_summary", summary)
    controller.out.assert_any_call("relative_analysis", analysis)



def make_cutoff_analysis(*, selected="400-Ry", satisfied=True):
    return FakeDict({
        "report": {
            "parameter": "cutoff",
            "energy_tolerance_satisfied": satisfied,
            "selected_candidate_label": selected,
        },
        "validation_scope": "standard_prefix",
        "production_certified": False,
        "provisional": True,
    })


def test_relative_stage_submits_correct_candidates():
    controller = make_fake_controller()

    analysis = make_cutoff_analysis()
    controller.ctx.cutoff_process = SimpleNamespace(
        outputs=AttributeDict({"analysis": analysis}),
    )

    generated = {}

    def fake_prepare(*, base_parameters, cutoff_analysis, candidate_count):
        assert base_parameters.get_dict()["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] == 400
        assert cutoff_analysis is analysis
        assert candidate_count.value == 5

        from nsdw.workflows.convergence.aiida_stage_candidates import (
            build_relative_cutoff_candidates,
        )

        result = build_relative_cutoff_candidates(
            base_parameters=base_parameters.get_dict(),
            cutoff_analysis=cutoff_analysis.get_dict(),
            candidate_count=candidate_count.value,
        )

        generated.update(result)
        return FakeDict(result)

    with patch(
        "nsdw.workflows.convergence.aiida_two_stage.Dict",
        side_effect=lambda dict: FakeDict(dict),
    ), patch(
        "nsdw.workflows.convergence.aiida_two_stage.Int",
        side_effect=lambda value: SimpleNamespace(value=value),
    ), patch(
        "nsdw.workflows.convergence.aiida_two_stage.Str",
        side_effect=lambda value: SimpleNamespace(value=value),
    ), patch(
        "nsdw.workflows.convergence.aiida_two_stage.prepare_relative_candidates",
        side_effect=fake_prepare,
    ):
        result = CP2KTwoStageConvergenceWorkChain.submit_relative_cutoff(
            controller
        )

    assert len(generated) == 5
    assert list(generated) == [
        "10-Ry", "20-Ry", "30-Ry", "40-Ry", "50-Ry"
    ]

    for index, parameters in enumerate(generated.values(), start=1):
        mgrid = parameters["FORCE_EVAL"]["DFT"]["MGRID"]
        assert mgrid["CUTOFF"] == 400
        assert mgrid["REL_CUTOFF"] == index * 10

    controller.submit.assert_called_once()

    kwargs = controller.submit.call_args.kwargs
    assert kwargs["parameter"].value == "relative_cutoff"
    assert kwargs["candidates"].get_dict() == generated

    controller.out.assert_called_once_with(
        "relative_candidates",
        kwargs["candidates"],
    )

    assert "relative_process" in result


def test_invalid_cutoff_analysis_prevents_relative_submission():
    controller = make_fake_controller()

    controller.ctx.cutoff_process = SimpleNamespace(
        outputs=AttributeDict({
            "analysis": make_cutoff_analysis(satisfied=False),
        }),
    )

    with patch(
        "nsdw.workflows.convergence.aiida_two_stage.Dict",
        side_effect=lambda dict: FakeDict(dict),
    ), patch(
        "nsdw.workflows.convergence.aiida_two_stage.Int",
        side_effect=lambda value: SimpleNamespace(value=value),
    ), patch(
        "nsdw.workflows.convergence.aiida_two_stage.prepare_relative_candidates",
        side_effect=lambda **kwargs: (
            __import__(
                "nsdw.workflows.convergence.aiida_stage_candidates",
                fromlist=["build_relative_cutoff_candidates"],
            ).build_relative_cutoff_candidates(
                base_parameters=kwargs["base_parameters"].get_dict(),
                cutoff_analysis=kwargs["cutoff_analysis"].get_dict(),
                candidate_count=kwargs["candidate_count"].value,
            )
        ),
    ):
        result = CP2KTwoStageConvergenceWorkChain.submit_relative_cutoff(
            controller
        )

    assert result.status == 412
    controller.submit.assert_not_called()
    controller.out.assert_not_called()



def test_production_stage_parameter_contract():
    """Both convergence stages must accept the controller's parameter input."""
    from aiida.orm import Str
    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
    )

    child_spec = CP2KConvergenceWorkChain.spec()
    parent_spec = CP2KTwoStageConvergenceWorkChain.spec()

    assert "parameter" in child_spec.inputs
    assert "code" in child_spec.inputs
    assert "structure" in child_spec.inputs
    assert "candidates" in child_spec.inputs
    assert "resources" in child_spec.inputs

    assert "cutoff_candidates" in parent_spec.inputs
    assert "resources" in parent_spec.inputs

    parameter_port = child_spec.inputs["parameter"]

    assert parameter_port.valid_type is not None
    assert Str in parameter_port.valid_type

    assert parameter_port.required is False
