"""Contract tests for AiiDA CP2K convergence orchestration."""

import pytest

from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)


def test_workchain_has_required_inputs():
    spec = CP2KConvergenceWorkChain.spec()

    assert "code" in spec.inputs
    assert "structure" in spec.inputs
    assert "candidates" in spec.inputs
    assert "resources" in spec.inputs


def test_workchain_has_summary_and_analysis_outputs():
    spec = CP2KConvergenceWorkChain.spec()
    assert "summary" in spec.outputs
    assert "analysis" in spec.outputs
    assert "ERROR_ANALYSIS_FAILED" in spec.exit_codes


def test_workchain_has_failure_exit_code():
    spec = CP2KConvergenceWorkChain.spec()
    assert "ERROR_CHILD_FAILED" in spec.exit_codes


def test_validate_candidate_payload():
    from nsdw.workflows.convergence.aiida_workchain import (
        validate_candidate_payload,
    )

    assert validate_candidate_payload(
        {"cutoff-400Ry": {"FORCE_EVAL": {"METHOD": "QS"}}},
        {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 600,
        },
    )


def test_validate_candidate_payload_rejects_empty_candidates():
    import pytest
    from nsdw.workflows.convergence.aiida_workchain import (
        validate_candidate_payload,
    )

    with pytest.raises(ValueError, match="At least one"):
        validate_candidate_payload(
            {},
            {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "max_wallclock_seconds": 600,
            },
        )


def test_validate_candidate_payload_rejects_bad_resources():
    import pytest
    from nsdw.workflows.convergence.aiida_workchain import (
        validate_candidate_payload,
    )

    with pytest.raises(ValueError, match="num_machines"):
        validate_candidate_payload(
            {"cutoff-400Ry": {"FORCE_EVAL": {}}},
            {
                "num_machines": 0,
                "num_mpiprocs_per_machine": 1,
                "max_wallclock_seconds": 600,
            },
        )


def test_workchain_requires_structure_data():
    from aiida.orm import StructureData
    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
    )

    assert (
        CP2KConvergenceWorkChain.spec()
        .inputs["structure"].valid_type
        is StructureData
    )


@pytest.mark.aiida_integration
def test_inspect_candidates_with_completed_cp2k():
    """Inspect five completed CP2K children without submitting jobs."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aiida.common.extendeddicts import AttributeDict
    from aiida.manage.configuration import load_profile
    from aiida.orm import Bool, Str, load_node

    load_profile("nsdw-dev")

    parent = load_node(112)
    assert parent.is_finished_ok

    candidates = parent.inputs.candidates.get_dict()
    entries = parent.outputs.summary.get_dict()

    context = {
        "candidate_names": {},
    }

    for index, name in enumerate(candidates):
        internal_id = f"candidate_{index:04d}"
        context["candidate_names"][internal_id] = name
        context[internal_id] = load_node(entries[name]["pk"])

    fake = SimpleNamespace(
        ctx=AttributeDict(context),
        inputs=AttributeDict({
            "structure": parent.inputs.structure,
            "candidates": parent.inputs.candidates,
            "parameter": Str("cutoff"),
            "retry_only": Bool(False),
        }),
        out=Mock(),
        report=Mock(),
        exit_codes=SimpleNamespace(
            ERROR_CHILD_FAILED=400,
            ERROR_ANALYSIS_FAILED=402,
        ),
    )

    result = CP2KConvergenceWorkChain.inspect_candidates(fake)

    assert result is None

    emitted = {
        call.args[0]: call.args[1]
        for call in fake.out.call_args_list
    }

    assert set(emitted) == {"summary", "analysis"}
    assert emitted["summary"].is_stored
    assert emitted["analysis"].is_stored

    analysis = emitted["analysis"].get_dict()

    assert analysis["validation_scope"] == "standard_prefix"
    assert analysis["production_certified"] is False
    assert analysis["report"]["selected_candidate_label"] == "400-Ry"



@pytest.mark.aiida_integration
def test_retry_only_input_contract():
    """Retry mode is optional and disabled by default."""
    from aiida import load_profile
    from aiida.orm import Bool

    load_profile("nsdw-dev")
    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
    )

    spec = CP2KConvergenceWorkChain.spec()

    assert "retry_only" in spec.inputs

    port = spec.inputs["retry_only"]

    assert Bool in port.valid_type
    assert port.required is False
    assert port.has_default()

    default = port.default
    if callable(default):
        default = default()

    assert isinstance(default, Bool)
    assert default.value is False



@pytest.mark.aiida_integration
def test_retry_only_skips_analysis():
    """Successful retry-only execution emits a summary, not analysis."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aiida import load_profile
    from aiida.common.extendeddicts import AttributeDict
    from aiida.orm import Bool, Str, load_node

    load_profile("nsdw-dev")

    parent = load_node(112)
    assert parent.is_finished_ok

    candidates = parent.inputs.candidates.get_dict()
    entries = parent.outputs.summary.get_dict()

    context = {"candidate_names": {}}

    for index, name in enumerate(candidates):
        internal_id = f"candidate_{index:04d}"
        context["candidate_names"][internal_id] = name
        context[internal_id] = load_node(entries[name]["pk"])

    fake = SimpleNamespace(
        ctx=AttributeDict(context),
        inputs=AttributeDict({
            "structure": parent.inputs.structure,
            "candidates": parent.inputs.candidates,
            "parameter": Str("cutoff"),
            "retry_only": Bool(True),
        }),
        out=Mock(),
        report=Mock(),
        exit_codes=SimpleNamespace(
            ERROR_CHILD_FAILED=400,
            ERROR_ANALYSIS_FAILED=402,
        ),
    )

    result = CP2KConvergenceWorkChain.inspect_candidates(fake)

    assert result is None

    emitted = {
        call.args[0]: call.args[1]
        for call in fake.out.call_args_list
    }

    assert set(emitted) == {"summary"}
    assert emitted["summary"].is_stored
    assert set(emitted["summary"].get_dict()) == set(candidates)
    fake.report.assert_called_once()


@pytest.mark.aiida_integration
def test_retry_only_preserves_child_failure():
    """Failed CP2K children remain fatal in retry-only mode."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aiida import load_profile
    from aiida.common.extendeddicts import AttributeDict
    from aiida.orm import Bool, load_node

    load_profile("nsdw-dev")

    campaign = load_node(268)
    candidates = campaign.inputs.candidates.get_dict()
    entries = campaign.outputs.summary.get_dict()

    context = {"candidate_names": {}}

    for index, name in enumerate(candidates):
        internal_id = f"candidate_{index:04d}"
        context["candidate_names"][internal_id] = name
        context[internal_id] = load_node(entries[name]["pk"])

    fake = SimpleNamespace(
        ctx=AttributeDict(context),
        inputs=AttributeDict({
            "structure": campaign.inputs.structure,
            "candidates": campaign.inputs.candidates,
            "retry_only": Bool(True),
        }),
        out=Mock(),
        report=Mock(),
        exit_codes=SimpleNamespace(
            ERROR_CHILD_FAILED=400,
            ERROR_ANALYSIS_FAILED=402,
        ),
    )

    result = CP2KConvergenceWorkChain.inspect_candidates(fake)

    assert result == 400

    emitted = {
        call.args[0]: call.args[1]
        for call in fake.out.call_args_list
    }

    assert set(emitted) == {"summary"}
    assert emitted["summary"].is_stored

    summary = emitted["summary"].get_dict()

    assert summary["40-Ry"]["finished_ok"] is False
    assert summary["50-Ry"]["finished_ok"] is False
    assert "analysis" not in emitted
