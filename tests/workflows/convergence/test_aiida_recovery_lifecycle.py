

import pytest
from aiida import load_profile
from aiida.engine import run_get_node
from aiida.orm import Int, QueryBuilder, load_node
from nsdw.workflows.convergence.aiida_recovery_workchain import (
    CP2KConvergenceRecoveryWorkChain,
)




@pytest.mark.aiida_integration
def test_targeted_retry_builder():
    """Only provenance-validated retries may enter the builder."""
    import pytest

    from aiida import load_profile
    from aiida.orm import Dict, load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        build_retry_workchain_builder,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    plan = recovery.outputs.recovery_plan
    candidates = recovery.outputs.retry_candidates
    resources = recovery.outputs.retry_resources

    builder = build_retry_workchain_builder(
        campaign,
        plan,
        candidates,
        resources,
    )

    assert builder.code.uuid == campaign.inputs.code.uuid
    assert builder.structure.uuid == campaign.inputs.structure.uuid
    assert list(builder.candidates.get_dict()) == ["40-Ry", "50-Ry"]
    assert builder.resources.get_dict()["max_wallclock_seconds"] == 1800
    assert builder.resources.get_dict()["queue_name"] == "small"
    assert builder.resources.get_dict()["account"] == "project_465003407"
    assert builder.retry_only.value is True

    # A successfully completed candidate must never be retried.
    original_candidates = campaign.inputs.candidates.get_dict()

    invalid_candidates = Dict(dict={
        "30-Ry": original_candidates["30-Ry"],
        "40-Ry": original_candidates["40-Ry"],
    })

    with pytest.raises(ValueError, match="Selected retry candidates"):
        build_retry_workchain_builder(
            campaign,
            plan,
            invalid_candidates,
            resources,
        )

    # Reject altered HPC resources.
    invalid_resources = resources.get_dict().copy()
    invalid_resources["queue_name"] = "different-queue"

    with pytest.raises(ValueError, match="Retry resources differ"):
        build_retry_workchain_builder(
            campaign,
            plan,
            candidates,
            Dict(dict=invalid_resources),
        )

    # Reject a recovery plan referring to another campaign.
    invalid_plan = plan.get_dict().copy()
    invalid_plan["original_campaign_pk"] = 999999

    with pytest.raises(ValueError, match="different campaign"):
        build_retry_workchain_builder(
            campaign,
            Dict(dict=invalid_plan),
            candidates,
            resources,
        )

    print("PASS: provenance-validated retry builder; no submission.")


def test_recovery_submission_gate():
    """Exercise all controller gate combinations without AiiDA jobs."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from aiida.orm import Bool

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        CP2KConvergenceRecoveryWorkChain,
    )

    submit = Mock(
        side_effect=AssertionError(
            "Unexpected AiiDA submission during testing."
        )
    )

    report = Mock()

    fake = SimpleNamespace(
        inputs=SimpleNamespace(
            submit_retries=Bool(False),
            execution_enabled=Bool(False),
        ),
        submit=submit,
        report=report,
        exit_codes=SimpleNamespace(
            ERROR_SUBMISSION_NOT_IMPLEMENTED=422,
        ),
    )

    cases = [
        (False, False, None),
        (True, False, 422),
        (False, True, 422),
        (True, True, 422),
    ]

    for requested, enabled, expected in cases:
        fake.inputs.submit_retries = Bool(requested)
        fake.inputs.execution_enabled = Bool(enabled)

        result = (
            CP2KConvergenceRecoveryWorkChain.submit_targeted_retries(
                fake
            )
        )

        assert result == expected
        submit.assert_not_called()

    assert report.call_count == 4

    print(
        "PASS: controller rejects execution under all "
        "gate combinations; no HPC submissions."
    )


@pytest.mark.aiida_integration
def test_recovery_validation_lifecycle():
    load_profile("nsdw-dev")

    # Count CP2K calculation nodes, not all AiiDA provenance nodes.
    from aiida.plugins import CalculationFactory

    Cp2kCalculation = CalculationFactory("cp2k")

    def count_cp2k_calculations():
        query = QueryBuilder()
        query.append(Cp2kCalculation)
        return query.count()

    cp2k_count_before = count_cp2k_calculations()

    outputs, process = run_get_node(
        CP2KConvergenceRecoveryWorkChain,
        original_campaign_pk=Int(268),
        retry_walltime=Int(1800),
    )

    assert process.is_finished_ok, (
        f"Recovery validation failed: "
        f"PK={process.pk}, exit={process.exit_status}"
    )

    assert "recovery_plan" in outputs

    plan = outputs["recovery_plan"].get_dict()

    assert plan["original_campaign_pk"] == 268
    assert plan["reusable"] == [
        "10-Ry",
        "20-Ry",
        "30-Ry",
    ]
    assert plan["retry"] == [
        "40-Ry",
        "50-Ry",
    ]
    assert plan["retry_walltime"] == 1800
    assert plan["submitted"] is False

    # Verify the exact retry payload against the original campaign.
    assert "retry_candidates" in outputs

    original_campaign = load_node(268)
    original_candidates = original_campaign.inputs.candidates.get_dict()
    retry_candidates = outputs["retry_candidates"].get_dict()

    assert list(retry_candidates) == ["40-Ry", "50-Ry"]

    for label in ("40-Ry", "50-Ry"):
        assert retry_candidates[label] == original_candidates[label]

    assert len(original_candidates) == 5

    # Verify provenance-tracked retry resources.
    assert "retry_resources" in outputs

    retry_resources = outputs["retry_resources"].get_dict()
    original_resources = original_campaign.inputs.resources.get_dict()

    assert retry_resources == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 1800,
        "queue_name": "small",
        "account": "project_465003407",
    }

    assert original_resources == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 600,
        "queue_name": "small",
        "account": "project_465003407",
    }

    assert (
        retry_resources["max_wallclock_seconds"]
        > original_resources["max_wallclock_seconds"]
    )

    # The recovery controller must not create any CP2K calculations.
    cp2k_count_after = count_cp2k_calculations()

    assert cp2k_count_after == cp2k_count_before, (
        "Recovery validation unexpectedly created CP2K calculations: "
        f"before={cp2k_count_before}, after={cp2k_count_after}"
    )

    print(f"Recovery validation WorkChain PK: {process.pk}")
    print(f"Recovery plan PK: {outputs['recovery_plan'].pk}")
    print("PASS: Recovery lifecycle completed without CP2K submission.")


@pytest.mark.aiida_integration
def test_targeted_submission_preparation():
    """Prepare only failed candidates without launching calculations."""
    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        prepare_targeted_retry_submission,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    builder = prepare_targeted_retry_submission(
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    assert list(builder.candidates.get_dict()) == [
        "40-Ry",
        "50-Ry",
    ]
    assert builder.retry_only.value is True
    assert (
        builder.resources.get_dict()["max_wallclock_seconds"]
        == 1800
    )

    print("PASS: targeted submission prepared without HPC execution.")


@pytest.mark.aiida_integration
def test_guarded_retry_submission():
    """Reject tampering before mocked AiiDA submission."""
    from unittest.mock import Mock
    import pytest

    from aiida import load_profile
    from aiida.orm import Bool, Dict, load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        prepare_targeted_retry_submission,
        submit_validated_retry,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    kwargs = dict(
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    def fresh_builder():
        return prepare_targeted_retry_submission(**kwargs)

    submit = Mock(return_value="mock-child-workchain")

    with pytest.raises(PermissionError):
        submit_validated_retry(
            fresh_builder(),
            submit,
            **kwargs,
            authorised=False,
        )

    submit.assert_not_called()

    result = submit_validated_retry(
        fresh_builder(),
        submit,
        **kwargs,
        authorised=True,
    )

    assert result == "mock-child-workchain"
    assert submit.call_count == 1
    submit.reset_mock()

    # Reject altered retry mode.
    builder = fresh_builder()
    builder.retry_only = Bool(False)

    with pytest.raises(ValueError):
        submit_validated_retry(
            builder, submit, **kwargs, authorised=True
        )

    submit.assert_not_called()

    # Reject altered candidate selection.
    builder = fresh_builder()
    builder.candidates = Dict(dict={
        "40-Ry": builder.candidates.get_dict()["40-Ry"],
    })

    with pytest.raises(ValueError):
        submit_validated_retry(
            builder, submit, **kwargs, authorised=True
        )

    submit.assert_not_called()

    # Reject altered SLURM resources.
    builder = fresh_builder()
    resources = builder.resources.get_dict()
    resources["queue_name"] = "tampered-queue"
    builder.resources = Dict(dict=resources)

    with pytest.raises(ValueError):
        submit_validated_retry(
            builder, submit, **kwargs, authorised=True
        )

    submit.assert_not_called()

    # Reject altered recovery provenance.
    builder = fresh_builder()
    plan = kwargs["recovery_plan"].get_dict()
    plan["original_campaign_pk"] = -1

    with pytest.raises(ValueError):
        submit_validated_retry(
            builder,
            submit,
            **{**kwargs, "recovery_plan": Dict(dict=plan)},
            authorised=True,
        )

    submit.assert_not_called()

    print(
        "PASS: provenance, candidate and resource checks "
        "prevent unauthorised or modified submissions."
    )


def test_retry_execution_authorisation():
    """Verify every dual-gate combination without AiiDA execution."""
    import pytest

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        retry_execution_authorised,
    )

    cases = [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ]

    for submit_retries, execution_enabled, expected in cases:
        actual = retry_execution_authorised(
            submit_retries,
            execution_enabled,
        )
        assert actual is expected

    with pytest.raises(TypeError):
        retry_execution_authorised(1, True)

    with pytest.raises(TypeError):
        retry_execution_authorised(True, "yes")

    print(
        "PASS: all four gate combinations and invalid inputs "
        "behave as expected."
    )


@pytest.mark.aiida_integration
def test_prepare_recovery_child():
    """Prepare the child builder without submitting any jobs."""
    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        prepare_recovery_child,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    context_key, builder = prepare_recovery_child(
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    assert context_key == "retry_workchain"
    assert list(builder.candidates.get_dict()) == [
        "40-Ry",
        "50-Ry",
    ]
    assert builder.retry_only.value is True

    resources = builder.resources.get_dict()

    assert resources["max_wallclock_seconds"] == 1800
    assert resources["queue_name"] == "small"
    assert resources["account"] == "project_465003407"

    print(
        "PASS: recovery child prepared with validated "
        "inputs; no calculations submitted."
    )


@pytest.mark.aiida_integration
def test_mocked_recovery_child_context():
    """Verify AiiDA child tracking without submitting jobs."""
    from unittest.mock import Mock
    import pytest

    from aiida import load_profile
    from aiida.engine import ToContext
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        create_recovery_child_context,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    kwargs = dict(
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    fake_child = Mock(name="mock-retry-workchain")
    submit = Mock(return_value=fake_child)

    with pytest.raises(PermissionError):
        create_recovery_child_context(
            **kwargs,
            submit_callback=submit,
            authorised=False,
        )

    submit.assert_not_called()

    context = create_recovery_child_context(
        **kwargs,
        submit_callback=submit,
        authorised=True,
    )

    assert isinstance(context, ToContext)
    assert list(context) == ["retry_workchain"]
    assert context["retry_workchain"] is fake_child
    submit.assert_called_once()

    builder = submit.call_args.args[0]

    assert builder.retry_only.value is True
    assert list(builder.candidates.get_dict()) == [
        "40-Ry",
        "50-Ry",
    ]
    assert builder.resources.get_dict()["max_wallclock_seconds"] == 1800

    print(
        "PASS: mocked child submitted through validated "
        "boundary and registered in ToContext."
    )


@pytest.mark.aiida_integration
def test_recovery_child_inspection_rejects_invalid():
    """Reject invalid or unrelated retry children."""
    import pytest

    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        inspect_recovery_child,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    kwargs = dict(
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
        recovery_parent=recovery,
    )

    with pytest.raises(TypeError):
        inspect_recovery_child(
            child=object(),
            **kwargs,
        )

    # The original campaign is not a successful retry child.
    with pytest.raises(ValueError):
        inspect_recovery_child(
            child=campaign,
            **kwargs,
        )

    print(
        "PASS: invalid and unrelated recovery children rejected."
    )


@pytest.mark.aiida_integration
def test_recovery_child_requires_valid_parent():
    """Reject absent, invalid and original-campaign parents."""
    import pytest

    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        inspect_recovery_child,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    kwargs = dict(
        child=recovery,
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    with pytest.raises(TypeError, match="Recovery parent must"):
        inspect_recovery_child(
            **kwargs,
            recovery_parent=None,
        )

    with pytest.raises(TypeError, match="Recovery parent must"):
        inspect_recovery_child(
            **kwargs,
            recovery_parent=object(),
        )

    with pytest.raises(ValueError, match="cannot be the original"):
        inspect_recovery_child(
            **kwargs,
            recovery_parent=campaign,
        )

    print("PASS: recovery parent validation rejects invalid inputs.")


@pytest.mark.aiida_integration
def test_retry_calculation_provenance_validation():
    """Validate actual CP2K parser provenance and scheduler options."""
    import pytest

    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        validate_retry_calculation_provenance,
    )

    load_profile("nsdw-dev")

    calculation = load_node(270)
    parser_output = calculation.outputs.output_parameters

    original_resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 600,
        "queue_name": "small",
        "account": "project_465003407",
    }

    assert validate_retry_calculation_provenance(
        calculation=calculation,
        parser_output=parser_output,
        expected_resources=original_resources,
    )

    incorrect_resources = {
        **original_resources,
        "max_wallclock_seconds": 1800,
    }

    with pytest.raises(ValueError, match="scheduler mismatch"):
        validate_retry_calculation_provenance(
            calculation=calculation,
            parser_output=parser_output,
            expected_resources=incorrect_resources,
        )

    print(
        "PASS: actual CP2K parser provenance verified; "
        "incorrect scheduler resources rejected."
    )


@pytest.mark.aiida_integration
def test_convergence_summary_provenance():
    """Validate actual summary calcfunction and parser input links."""
    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        validate_convergence_summary_provenance,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)

    assert validate_convergence_summary_provenance(
        campaign=campaign
    )


@pytest.mark.aiida_integration
def test_recovery_parent_identity():
    """Verify actual recovery-parent identity and output nodes."""
    import pytest

    from aiida import load_profile
    from aiida.orm import Dict, load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        inspect_recovery_child,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    kwargs = dict(
        child=campaign,
        original_campaign=campaign,
        recovery_parent=recovery,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    # Correct recovery-parent identity and outputs allow inspection
    # to proceed to the child-success check.
    with pytest.raises(
        ValueError,
        match="Recovery child did not finish successfully",
    ):
        inspect_recovery_child(**kwargs)

    # A different recovery-plan node must be rejected.
    with pytest.raises(
        ValueError,
        match="Recovery parent output mismatch: recovery_plan",
    ):
        inspect_recovery_child(
            **{
                **kwargs,
                "recovery_plan": Dict(
                    dict=recovery.outputs.recovery_plan.get_dict()
                ),
            }
        )


@pytest.mark.aiida_integration
def test_original_campaign_recovery_provenance():
    """Validate historical CP2K outcomes without retry submission."""
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery import (
        validate_recovery_provenance,
    )
    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        build_retry_workchain_builder,
    )

    from aiida import load_profile

    load_profile("nsdw-dev")

    original = load_node(268)
    recovery = load_node(605)

    plan = validate_recovery_provenance(original)

    assert list(plan.reusable) == [
        "10-Ry",
        "20-Ry",
        "30-Ry",
    ]
    assert list(plan.retry) == [
        "40-Ry",
        "50-Ry",
    ]

    assert recovery.is_finished_ok

    recovery_plan = recovery.outputs.recovery_plan
    retry_candidates = recovery.outputs.retry_candidates
    retry_resources = recovery.outputs.retry_resources

    assert recovery_plan.get_dict()["submitted"] is False
    assert recovery_plan.get_dict()["original_campaign_pk"] == 268

    builder = build_retry_workchain_builder(
        original_campaign=original,
        recovery_plan=recovery_plan,
        retry_candidates=retry_candidates,
        retry_resources=retry_resources,
    )

    assert list(builder.candidates.get_dict()) == [
        "40-Ry",
        "50-Ry",
    ]
    assert builder.retry_only.value is True

    assert (
        builder.resources.get_dict()["max_wallclock_seconds"]
        == 1800
    )
    assert (
        builder.resources.get_dict()["account"]
        == "project_465003407"
    )


@pytest.mark.aiida_integration
def test_historical_cp2k_parser_reproducibility():
    """Replay a real CP2K output without storing provenance."""
    from math import isclose

    from aiida import load_profile
    from aiida.orm import load_node
    from aiida.plugins import ParserFactory

    load_profile("nsdw-dev")

    calculation = load_node(270)

    assert calculation.is_finished_ok

    parser_name = calculation.get_option("parser_name")
    assert parser_name == "cp2k_base_parser"

    parser_class = ParserFactory(parser_name)

    results, _ = parser_class.parse_from_node(
        calculation,
        store_provenance=False,
    )

    assert results is not None
    assert "output_parameters" in results

    original = calculation.outputs.output_parameters.get_dict()
    replayed = results["output_parameters"].get_dict()

    assert isclose(
        original["energy"],
        replayed["energy"],
        rel_tol=0.0,
        abs_tol=1e-10,
    )

    for key in (
        "energy_units",
        "exceeded_walltime",
        "nwarnings",
    ):
        assert original[key] == replayed[key]


@pytest.mark.aiida_integration
def test_retry_duplicate_submission_guard():
    """Duplicate checks must block callbacks before submission."""
    from unittest.mock import Mock

    import pytest

    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        prepare_targeted_retry_submission,
        submit_validated_retry,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    kwargs = dict(
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    def submit_with_guard(guard):
        builder = prepare_targeted_retry_submission(**kwargs)
        callback = Mock(return_value="mock-child")
        return builder, callback

    builder, callback = submit_with_guard(None)

    duplicate = Mock(return_value=True)

    with pytest.raises(RuntimeError, match="Duplicate"):
        submit_validated_retry(
            builder,
            callback,
            **kwargs,
            authorised=True,
            duplicate_check=duplicate,
        )

    duplicate.assert_called_once()
    callback.assert_not_called()

    builder, callback = submit_with_guard(None)
    clear = Mock(return_value=False)

    result = submit_validated_retry(
        builder,
        callback,
        **kwargs,
        authorised=True,
        duplicate_check=clear,
    )

    assert result == "mock-child"
    clear.assert_called_once()
    callback.assert_called_once()

    builder, callback = submit_with_guard(None)
    broken = Mock(side_effect=RuntimeError("Guard unavailable"))

    with pytest.raises(RuntimeError, match="Guard unavailable"):
        submit_validated_retry(
            builder,
            callback,
            **kwargs,
            authorised=True,
            duplicate_check=broken,
        )

    callback.assert_not_called()

    builder, callback = submit_with_guard(None)

    with pytest.raises(TypeError, match="Python bool"):
        submit_validated_retry(
            builder,
            callback,
            **kwargs,
            authorised=True,
            duplicate_check=lambda **_: None,
        )

    callback.assert_not_called()


@pytest.mark.aiida_integration
def test_retry_duplicate_guard_rejects_invalid_callback():
    """Reject a non-callable duplicate guard."""
    from unittest.mock import Mock

    import pytest

    from aiida import load_profile
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        prepare_targeted_retry_submission,
        submit_validated_retry,
    )

    load_profile("nsdw-dev")

    campaign = load_node(268)
    recovery = load_node(605)

    kwargs = dict(
        original_campaign=campaign,
        recovery_plan=recovery.outputs.recovery_plan,
        retry_candidates=recovery.outputs.retry_candidates,
        retry_resources=recovery.outputs.retry_resources,
    )

    builder = prepare_targeted_retry_submission(**kwargs)
    callback = Mock()

    with pytest.raises(TypeError, match="callable"):
        submit_validated_retry(
            builder,
            callback,
            **kwargs,
            authorised=True,
            duplicate_check="not-a-callback",
        )

    callback.assert_not_called()
