"""Provenance-aware recovery of incomplete CP2K convergence campaigns."""

from __future__ import annotations

from aiida.engine import WorkChain, calcfunction
from aiida.orm import Bool, Dict, Int, WorkChainNode, load_node

from nsdw.workflows.convergence.aiida_recovery import (
    build_retry_candidates,
    build_retry_resources,
    validate_recovery_provenance,
)



@calcfunction
def create_recovery_plan(
    original_summary: Dict,
    original_candidates: Dict,
    campaign_pk: Int,
    retry_walltime: Int,
    reusable: Dict,
    retry: Dict,
) -> Dict:
    """Create a provenance-tracked recovery plan."""
    return Dict(dict={
        "original_campaign_pk": campaign_pk.value,
        "reusable": reusable.get_dict()["labels"],
        "retry": retry.get_dict()["labels"],
        "retry_walltime": retry_walltime.value,
        "submitted": False,
    })




@calcfunction
def create_retry_candidates(
    original_candidates: Dict,
    recovery_plan: Dict,
    selected: Dict,
) -> Dict:
    """Create a provenance-tracked subset of retry candidates."""
    original = original_candidates.get_dict()
    subset = selected.get_dict()
    plan = recovery_plan.get_dict()

    expected = plan["retry"]

    if list(subset) != expected:
        raise ValueError("Retry candidate ordering mismatch.")

    for name, parameters in subset.items():
        if name not in original:
            raise ValueError(f"Unknown retry candidate: {name}")

        if parameters != original[name]:
            raise ValueError(
                f"Modified CP2K parameters for retry candidate: {name}"
            )

    return Dict(dict=subset)



@calcfunction
def create_retry_resources(
    original_resources: Dict,
    recovery_plan: Dict,
    selected: Dict,
) -> Dict:
    """Create provenance-tracked retry resources."""
    original = original_resources.get_dict()
    plan = recovery_plan.get_dict()
    actual = selected.get_dict()

    expected = build_retry_resources(
        original,
        plan["retry_walltime"],
    )

    if actual != expected:
        raise ValueError(
            "Retry resources differ from the validated configuration."
        )

    return Dict(dict=actual)



def build_retry_workchain_builder(
    original_campaign: WorkChainNode,
    recovery_plan: Dict,
    retry_candidates: Dict,
    retry_resources: Dict,
):
    """Build only the validated failed candidates; submit nothing."""
    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
    )

    if not isinstance(original_campaign, WorkChainNode):
        raise TypeError("Original campaign must be a WorkChainNode.")

    validated = validate_recovery_provenance(original_campaign)
    plan = recovery_plan.get_dict()

    if plan.get("original_campaign_pk") != original_campaign.pk:
        raise ValueError("Recovery plan references a different campaign.")

    if plan.get("submitted") is not False:
        raise ValueError("Recovery plan is not in preparation state.")

    if plan.get("reusable") != list(validated.reusable):
        raise ValueError("Reusable candidates differ from provenance.")

    if plan.get("retry") != list(validated.retry):
        raise ValueError("Retry candidates differ from provenance.")

    selected = retry_candidates.get_dict()
    expected_labels = list(validated.retry)

    if not expected_labels:
        raise ValueError("No candidates require retry.")

    if list(selected) != expected_labels:
        raise ValueError(
            "Selected retry candidates differ from recovery plan."
        )

    original_candidates = original_campaign.inputs.candidates.get_dict()

    for name, parameters in selected.items():
        if parameters != original_candidates[name]:
            raise ValueError(
                f"Retry parameters differ from original: {name}"
            )

    expected_resources = build_retry_resources(
        original_campaign.inputs.resources.get_dict(),
        plan["retry_walltime"],
    )

    if retry_resources.get_dict() != expected_resources:
        raise ValueError(
            "Retry resources differ from validated configuration."
        )

    builder = CP2KConvergenceWorkChain.get_builder()
    builder.code = original_campaign.inputs.code
    builder.structure = original_campaign.inputs.structure
    builder.candidates = retry_candidates
    builder.resources = retry_resources
    builder.retry_only = Bool(True)

    if "parameter" in original_campaign.inputs:
        builder.parameter = original_campaign.inputs.parameter

    return builder


def prepare_targeted_retry_submission(
    original_campaign: WorkChainNode,
    recovery_plan: Dict,
    retry_candidates: Dict,
    retry_resources: Dict,
):
    """Reconstruct and validate a targeted retry; submit nothing."""
    builder = build_retry_workchain_builder(
        original_campaign=original_campaign,
        recovery_plan=recovery_plan,
        retry_candidates=retry_candidates,
        retry_resources=retry_resources,
    )

    selected = list(builder.candidates.get_dict())
    expected = recovery_plan.get_dict()["retry"]

    if selected != expected:
        raise ValueError(
            "Submission candidates differ from validated recovery plan."
        )

    if not builder.retry_only.value:
        raise ValueError(
            "Targeted retry must run in retry-only mode."
        )

    return builder


def submit_validated_retry(
    builder,
    submit_callback,
    *,
    original_campaign: WorkChainNode,
    recovery_plan: Dict,
    retry_candidates: Dict,
    retry_resources: Dict,
    authorised: bool = False,
    duplicate_check=None,
):
    """Submit only a provenance-validated targeted retry.

    duplicate_check is an injectable pre-submission guard.
    It does not provide atomic distributed locking.
    """
    if not authorised:
        raise PermissionError(
            "Targeted retry submission requires explicit authorisation."
        )

    expected = prepare_targeted_retry_submission(
        original_campaign=original_campaign,
        recovery_plan=recovery_plan,
        retry_candidates=retry_candidates,
        retry_resources=retry_resources,
    )

    if not builder.retry_only.value:
        raise ValueError("Refusing to submit a non-retry WorkChain.")

    if builder.code.uuid != expected.code.uuid:
        raise ValueError("Retry CP2K code differs from original.")

    if builder.structure.uuid != expected.structure.uuid:
        raise ValueError("Retry structure differs from original.")

    if (
        builder.candidates.get_dict()
        != expected.candidates.get_dict()
    ):
        raise ValueError("Retry candidates differ from provenance.")

    if (
        list(builder.candidates.get_dict())
        != list(expected.candidates.get_dict())
    ):
        raise ValueError("Retry candidate ordering differs.")

    if (
        builder.resources.get_dict()
        != expected.resources.get_dict()
    ):
        raise ValueError("Retry resources differ from provenance.")

    expected_parameter = expected.get("parameter", None)
    actual_parameter = builder.get("parameter", None)

    if (expected_parameter is None) != (actual_parameter is None):
        raise ValueError("Retry convergence parameter differs.")

    if expected_parameter is not None:
        if actual_parameter.value != expected_parameter.value:
            raise ValueError("Retry convergence parameter differs.")

    if duplicate_check is not None:
        if not callable(duplicate_check):
            raise TypeError("duplicate_check must be callable.")

        duplicate_found = duplicate_check(
            original_campaign=original_campaign,
            recovery_plan=recovery_plan,
            retry_candidates=retry_candidates,
            retry_resources=retry_resources,
        )

        if type(duplicate_found) is not bool:
            raise TypeError(
                "duplicate_check must return a Python bool."
            )

        if duplicate_found:
            raise RuntimeError(
                "Duplicate targeted recovery submission detected."
            )

    return submit_callback(builder)


def retry_execution_authorised(
    submit_retries: bool,
    execution_enabled: bool,
) -> bool:
    """Require both explicit execution controls."""
    if type(submit_retries) is not bool:
        raise TypeError("submit_retries must be a Python bool.")

    if type(execution_enabled) is not bool:
        raise TypeError("execution_enabled must be a Python bool.")

    return submit_retries and execution_enabled


def prepare_recovery_child(
    original_campaign: WorkChainNode,
    recovery_plan: Dict,
    retry_candidates: Dict,
    retry_resources: Dict,
):
    """Prepare a validated child WorkChain without submitting it."""
    builder = prepare_targeted_retry_submission(
        original_campaign=original_campaign,
        recovery_plan=recovery_plan,
        retry_candidates=retry_candidates,
        retry_resources=retry_resources,
    )

    context_key = "retry_workchain"

    if not builder.retry_only.value:
        raise ValueError(
            "Recovery child must use retry-only execution."
        )

    return context_key, builder


def create_recovery_child_context(
    *,
    original_campaign: WorkChainNode,
    recovery_plan: Dict,
    retry_candidates: Dict,
    retry_resources: Dict,
    submit_callback,
    authorised: bool = False,
):
    """Submit a validated retry and return its AiiDA context mapping."""
    from aiida.engine import ToContext

    context_key, builder = prepare_recovery_child(
        original_campaign=original_campaign,
        recovery_plan=recovery_plan,
        retry_candidates=retry_candidates,
        retry_resources=retry_resources,
    )

    child = submit_validated_retry(
        builder,
        submit_callback,
        original_campaign=original_campaign,
        recovery_plan=recovery_plan,
        retry_candidates=retry_candidates,
        retry_resources=retry_resources,
        authorised=authorised,
    )

    return ToContext(**{context_key: child})


def validate_retry_calculation_provenance(
    *,
    calculation,
    parser_output,
    expected_resources: dict,
):
    """Verify CP2K parser provenance and scheduler resources."""
    from aiida.common.links import LinkType

    if calculation.base.attributes.get(
        calculation.IMMIGRATED_KEY,
        False,
    ) is True:
        raise ValueError(
            "Imported CP2K calculations cannot validate "
            "production recovery results."
        )

    creators = parser_output.base.links.get_incoming(
        link_type=LinkType.CREATE
    ).all()

    if (
        len(creators) != 1
        or creators[0].node.uuid != calculation.uuid
        or creators[0].link_label != "output_parameters"
    ):
        raise ValueError(
            "CP2K parser output has incorrect CREATE provenance."
        )

    expected_scheduler = {
        "resources": {
            "num_machines": expected_resources["num_machines"],
            "num_mpiprocs_per_machine": (
                expected_resources["num_mpiprocs_per_machine"]
            ),
        },
        "max_wallclock_seconds": (
            expected_resources["max_wallclock_seconds"]
        ),
        "queue_name": expected_resources["queue_name"],
        "account": expected_resources["account"],
    }

    for option, expected in expected_scheduler.items():
        actual = calculation.get_option(option)

        if actual != expected:
            raise ValueError(
                f"Retry calculation scheduler mismatch: {option}"
            )

    return True


def validate_convergence_summary_provenance(*, campaign):
    """Validate the provenance of a convergence summary and its parser inputs."""
    from aiida.common.links import LinkType
    from aiida.orm import WorkChainNode, load_node

    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
    )

    if not isinstance(campaign, WorkChainNode):
        raise TypeError("Campaign must be a WorkChainNode.")

    if campaign.process_type != CP2KConvergenceWorkChain.build_process_type():
        raise ValueError("Unexpected convergence campaign process type.")

    if "summary" not in campaign.outputs:
        raise ValueError("Convergence campaign has no summary.")

    summary = campaign.outputs.summary
    entries = summary.get_dict()
    candidates = campaign.inputs.candidates.get_dict()

    if list(entries) != list(candidates):
        raise ValueError("Summary candidate order mismatch.")

    creators = summary.base.links.get_incoming(
        link_type=LinkType.CREATE
    ).all()

    if len(creators) != 1:
        raise ValueError("Summary must have exactly one creator.")

    creator_link = creators[0]
    creator = creator_link.node

    expected_process = (
        "nsdw.workflows.convergence.aiida_workchain."
        "build_convergence_summary"
    )

    if (
        creator_link.link_label != "result"
        or creator.process_type != expected_process
    ):
        raise ValueError("Unexpected summary creator.")

    callers = creator.base.links.get_incoming(
        link_type=LinkType.CALL_CALC
    ).all()

    if len(callers) != 1 or callers[0].node.uuid != campaign.uuid:
        raise ValueError("Summary creator caller mismatch.")

    input_links = creator.base.links.get_incoming(
        link_type=LinkType.INPUT_CALC
    ).all()

    inputs = {link.link_label: link.node for link in input_links}

    if len(inputs) != len(input_links):
        raise ValueError("Duplicate summary input labels.")

    expected_labels = {
        "entries",
        *(
            f"candidate_{index:04d}"
            for index in range(len(candidates))
        ),
    }

    if set(inputs) != expected_labels:
        raise ValueError("Summary creator input labels mismatch.")

    if inputs["entries"].get_dict() != entries:
        raise ValueError("Summary entries input mismatch.")

    for index, name in enumerate(candidates):
        entry = entries[name]

        if type(entry.get("pk")) is not int:
            raise ValueError(f"Missing calculation PK: {name}")

        calculation = load_node(entry["pk"])

        if "output_parameters" not in calculation.outputs:
            raise ValueError(
                f"Missing parser output for summary candidate: {name}"
            )

        actual_parser = calculation.outputs.output_parameters
        linked_parser = inputs[f"candidate_{index:04d}"]

        if linked_parser.uuid != actual_parser.uuid:
            raise ValueError(
                f"Summary parser provenance mismatch: {name}"
            )

    return True


def validate_recovery_child_import_boundary(child):
    """Reject imported CP2K outputs in production recovery children."""
    if "import_folders" in child.inputs:
        raise ValueError(
            "Imported CP2K results are not permitted "
            "in production recovery children."
        )
    return True


def inspect_recovery_child(
    *,
    child,
    original_campaign: WorkChainNode,
    recovery_plan: Dict,
    retry_candidates: Dict,
    retry_resources: Dict,
    recovery_parent: WorkChainNode,
):
    """Validate retry-child results against original provenance."""
    from aiida.common.links import LinkType
    from aiida.orm import load_node

    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
        valid_cp2k_energy,
    )

    if not isinstance(recovery_parent, WorkChainNode):
        raise TypeError("Recovery parent must be a WorkChainNode.")

    if recovery_parent.uuid == original_campaign.uuid:
        raise ValueError(
            "Recovery parent cannot be the original campaign."
        )

    if (
        recovery_parent.process_type
        != CP2KConvergenceRecoveryWorkChain.build_process_type()
    ):
        raise ValueError("Unexpected recovery parent process type.")

    for name, expected_node in (
        ("recovery_plan", recovery_plan),
        ("retry_candidates", retry_candidates),
        ("retry_resources", retry_resources),
    ):
        if name not in recovery_parent.outputs:
            raise ValueError(f"Missing recovery parent output: {name}")

        if recovery_parent.outputs[name].uuid != expected_node.uuid:
            raise ValueError(f"Recovery parent output mismatch: {name}")

    # Revalidate original campaign and retry inputs.
    expected = prepare_targeted_retry_submission(
        original_campaign=original_campaign,
        recovery_plan=recovery_plan,
        retry_candidates=retry_candidates,
        retry_resources=retry_resources,
    )

    if not isinstance(child, WorkChainNode):
        raise TypeError("Recovery child must be a WorkChainNode.")

    if child.process_type != CP2KConvergenceWorkChain.build_process_type():
        raise ValueError("Unexpected recovery child process type.")

    validate_recovery_child_import_boundary(child)

    if not child.is_finished_ok or child.exit_status != 0:
        raise ValueError("Recovery child did not finish successfully.")

    if child.inputs.code.uuid != expected.code.uuid:
        raise ValueError("Recovery child CP2K code mismatch.")

    if child.inputs.structure.uuid != expected.structure.uuid:
        raise ValueError("Recovery child structure mismatch.")

    if (
        child.inputs.candidates.get_dict()
        != expected.candidates.get_dict()
    ):
        raise ValueError("Recovery child candidate mismatch.")

    if (
        list(child.inputs.candidates.get_dict())
        != list(expected.candidates.get_dict())
    ):
        raise ValueError("Recovery child candidate order mismatch.")

    if (
        child.inputs.resources.get_dict()
        != expected.resources.get_dict()
    ):
        raise ValueError("Recovery child resource mismatch.")

    if (
        "retry_only" not in child.inputs
        or child.inputs.retry_only.value is not True
    ):
        raise ValueError("Recovery child is not retry-only.")

    expected_parameter = expected.get("parameter", None)
    actual_parameter = child.inputs.get("parameter", None)

    if (expected_parameter is None) != (actual_parameter is None):
        raise ValueError("Recovery child parameter mismatch.")

    if expected_parameter is not None:
        if actual_parameter.value != expected_parameter.value:
            raise ValueError("Recovery child parameter mismatch.")

    parents = [
        link.node
        for link in child.base.links.get_incoming(
            link_type=LinkType.CALL_WORK
        ).all()
    ]

    if len(parents) != 1 or parents[0].uuid != recovery_parent.uuid:
        raise ValueError("Recovery child parent provenance mismatch.")

    if "summary" not in child.outputs:
        raise ValueError("Recovery child has no summary.")

    validate_convergence_summary_provenance(campaign=child)

    summary = child.outputs.summary.get_dict()
    candidates = expected.candidates.get_dict()

    if list(summary) != list(candidates):
        raise ValueError("Recovery child summary labels mismatch.")

    for name in candidates:
        entry = summary[name]

        if not isinstance(entry, dict):
            raise ValueError(f"Invalid retry summary for {name}.")

        if type(entry.get("pk")) is not int:
            raise ValueError(f"Missing calculation PK for {name}.")

        calculation = load_node(entry["pk"])

        callers = [
            link.node
            for link in calculation.base.links.get_incoming(
                link_type=LinkType.CALL_CALC
            ).all()
        ]

        if len(callers) != 1 or callers[0].uuid != child.uuid:
            raise ValueError(f"Incorrect calculation parent: {name}")

        if calculation.label != name:
            raise ValueError(f"Calculation label mismatch: {name}")

        if str(calculation.uuid) != str(entry.get("uuid")):
            raise ValueError(f"Calculation UUID mismatch: {name}")

        if (
            not calculation.is_finished_ok
            or calculation.exit_status != 0
            or entry.get("finished_ok") is not True
            or entry.get("exit_status") != 0
        ):
            raise ValueError(f"Retry calculation failed: {name}")

        if (
            calculation.inputs.parameters.get_dict()
            != candidates[name]
        ):
            raise ValueError(f"Calculation parameters mismatch: {name}")

        if calculation.inputs.code.uuid != expected.code.uuid:
            raise ValueError(f"Calculation code mismatch: {name}")

        if calculation.inputs.structure.uuid != expected.structure.uuid:
            raise ValueError(f"Calculation structure mismatch: {name}")

        if "output_parameters" not in calculation.outputs:
            raise ValueError(f"Missing parser output: {name}")

        parser_output = calculation.outputs.output_parameters

        validate_retry_calculation_provenance(
            calculation=calculation,
            parser_output=parser_output,
            expected_resources=expected.resources.get_dict(),
        )

        parser = parser_output.get_dict()

        if not valid_cp2k_energy(
            parser.get("energy"),
            parser.get("energy_units"),
        ):
            raise ValueError(f"Invalid parsed energy: {name}")

        if parser != entry.get("output_parameters"):
            raise ValueError(f"Parser summary mismatch: {name}")

        if parser["energy"] != entry.get("energy_hartree"):
            raise ValueError(f"Energy mismatch: {name}")

    return summary


class CP2KConvergenceRecoveryWorkChain(WorkChain):
    """Validate a failed campaign before targeted recovery submission."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("original_campaign_pk", valid_type=Int)
        spec.input("retry_walltime", valid_type=Int)
        spec.input(
            "submit_retries",
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
        )

        spec.input(
            "execution_enabled",
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
        )

        spec.outline(
            cls.validate_original_campaign,
            cls.prepare_retry_builder,
            cls.submit_targeted_retries,
        )

        spec.output("recovery_plan", valid_type=Dict)
        spec.output("retry_candidates", valid_type=Dict)
        spec.output("retry_resources", valid_type=Dict)

        spec.exit_code(
            420,
            "ERROR_INVALID_CAMPAIGN",
            message="Original convergence campaign is invalid.",
        )
        spec.exit_code(
            421,
            "ERROR_INVALID_WALLTIME",
            message="Recovery walltime must be a positive integer.",
        )
        spec.exit_code(
            422,
            "ERROR_SUBMISSION_NOT_IMPLEMENTED",
            message="Retry submission is not yet implemented.",
        )

    def prepare_retry_builder(self):
        """Validate the targeted retry builder without submitting jobs."""
        campaign = load_node(self.inputs.original_campaign_pk.value)

        try:
            builder = build_retry_workchain_builder(
                original_campaign=campaign,
                recovery_plan=self.outputs["recovery_plan"],
                retry_candidates=self.outputs["retry_candidates"],
                retry_resources=self.outputs["retry_resources"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            self.report(f"Retry builder validation failed: {exc}")
            return self.exit_codes.ERROR_INVALID_CAMPAIGN

        labels = list(builder.candidates.get_dict())

        if not builder.retry_only.value:
            self.report("Retry builder is not in retry-only mode.")
            return self.exit_codes.ERROR_INVALID_CAMPAIGN

        self.ctx.retry_builder_validated = True

        self.report(
            f"Retry builder prepared: {labels}; "
            f"walltime={builder.resources.get_dict()['max_wallclock_seconds']}s; "
            "submission disabled."
        )

    def submit_targeted_retries(self):
        """Evaluate both gates; keep actual submission disabled."""
        requested = self.inputs.submit_retries.value
        enabled = self.inputs.execution_enabled.value

        authorised = retry_execution_authorised(
            requested,
            enabled,
        )

        if authorised:
            self.report(
                "Both execution gates enabled, but recovery "
                "submission is not yet implemented."
            )
            return self.exit_codes.ERROR_SUBMISSION_NOT_IMPLEMENTED

        if requested or enabled:
            self.report(
                "Recovery submission blocked: both execution "
                "gates are required."
            )
            return self.exit_codes.ERROR_SUBMISSION_NOT_IMPLEMENTED

        self.report(
            "Recovery execution remains disabled. "
            "No retry WorkChain submitted."
        )
        return

    def validate_original_campaign(self):
        if self.inputs.retry_walltime.value <= 0:
            return self.exit_codes.ERROR_INVALID_WALLTIME

        try:
            campaign = load_node(
                self.inputs.original_campaign_pk.value
            )

            if not isinstance(campaign, WorkChainNode):
                raise ValueError(
                    "Original campaign must be an AiiDA WorkChain."
                )

            plan = validate_recovery_provenance(campaign)

        except (ValueError, KeyError, TypeError) as exc:
            self.report(f"Recovery validation failed: {exc}")
            return self.exit_codes.ERROR_INVALID_CAMPAIGN

        recovery_plan = create_recovery_plan(
            original_summary=campaign.outputs.summary,
            original_candidates=campaign.inputs.candidates,
            campaign_pk=self.inputs.original_campaign_pk,
            retry_walltime=self.inputs.retry_walltime,
            reusable=Dict(dict={"labels": list(plan.reusable)}),
            retry=Dict(dict={"labels": list(plan.retry)}),
        )

        retry_payload = build_retry_candidates(
            campaign.inputs.candidates.get_dict(),
            plan.retry,
        )

        retry_candidates = create_retry_candidates(
            original_candidates=campaign.inputs.candidates,
            recovery_plan=recovery_plan,
            selected=Dict(dict=retry_payload),
        )

        resources_payload = build_retry_resources(
            campaign.inputs.resources.get_dict(),
            self.inputs.retry_walltime.value,
        )

        retry_resources = create_retry_resources(
            original_resources=campaign.inputs.resources,
            recovery_plan=recovery_plan,
            selected=Dict(dict=resources_payload),
        )

        self.out("recovery_plan", recovery_plan)
        self.out("retry_candidates", retry_candidates)
        self.out("retry_resources", retry_resources)

        self.report(
            f"Recovery validated: {len(plan.reusable)} reusable, "
            f"{len(plan.retry)} requiring retry. "
            "No calculations submitted."
        )
