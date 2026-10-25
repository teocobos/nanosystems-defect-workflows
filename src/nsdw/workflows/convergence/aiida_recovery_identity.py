"""Read-only AiiDA recovery process identity validation.

This module does not submit, restart or modify AiiDA processes.
UUID resolution and process-type checks are preliminary checks,
not proof of complete recovery provenance.
"""

from collections.abc import Callable

from aiida.common.links import LinkType
from aiida.orm import WorkChainNode, load_node

from nsdw.workflows.convergence.aiida_recovery_workchain import (
    CP2KConvergenceRecoveryWorkChain,
)

from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


def validate_tracking_identity(
    metadata: RecoveryExecutionMetadata,
    *,
    process_loader: Callable = load_node,
    expected_recovery_parent: WorkChainNode | None = None,
    original_campaign: WorkChainNode | None = None,
):
    """Resolve and perform preliminary checks on an AiiDA retry process.

    A successful return does not authorise submission, resubmission
    or acceptance of scientific results.
    """
    if not isinstance(metadata, RecoveryExecutionMetadata):
        raise TypeError("Expected RecoveryExecutionMetadata.")

    if metadata.engine != "aiida":
        raise ValueError("Expected AiiDA execution engine.")

    if metadata.process_uuid is None:
        raise ValueError(
            "Missing process UUID; reconciliation required."
        )

    try:
        process = process_loader(metadata.process_uuid)
    except Exception as exc:
        raise ValueError(
            "Unable to resolve AiiDA process UUID; "
            "reconciliation required."
        ) from exc

    if process is None:
        raise ValueError(
            "Unable to resolve AiiDA process UUID; "
            "reconciliation required."
        )

    if str(getattr(process, "uuid", "")) != metadata.process_uuid:
        raise ValueError("Resolved process UUID mismatch.")

    expected_type = CP2KConvergenceWorkChain.build_process_type()

    if getattr(process, "process_type", None) != expected_type:
        raise ValueError(
            "Unexpected AiiDA recovery child process type."
        )

    if not isinstance(process, WorkChainNode):
        raise ValueError(
            "Resolved process is not an AiiDA WorkChainNode."
        )

    if not isinstance(expected_recovery_parent, WorkChainNode):
        raise ValueError(
            "Expected recovery parent WorkChainNode is required."
        )

    expected_parent_type = (
        CP2KConvergenceRecoveryWorkChain.build_process_type()
    )

    if expected_recovery_parent.process_type != expected_parent_type:
        raise ValueError(
            "Unexpected recovery parent process type."
        )

    parents = [
        link.node
        for link in process.base.links.get_incoming(
            link_type=LinkType.CALL_WORK
        ).all()
    ]

    if (
        len(parents) != 1
        or parents[0].uuid != expected_recovery_parent.uuid
    ):
        raise ValueError(
            "Recovery child parent provenance mismatch."
        )

    if not isinstance(original_campaign, WorkChainNode):
        raise ValueError("Original campaign WorkChainNode is required.")

    if process.uuid == original_campaign.uuid:
        raise ValueError(
            "Original campaign cannot be a recovery child."
        )

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        build_retry_workchain_builder,
        validate_recovery_child_import_boundary,
    )

    required = (
        "recovery_plan",
        "retry_candidates",
        "retry_resources",
    )

    for name in required:
        if name not in expected_recovery_parent.outputs:
            raise ValueError(
                f"Missing recovery parent output: {name}"
            )

    expected = build_retry_workchain_builder(
        original_campaign,
        expected_recovery_parent.outputs.recovery_plan,
        expected_recovery_parent.outputs.retry_candidates,
        expected_recovery_parent.outputs.retry_resources,
    )

    validate_recovery_child_import_boundary(process)

    if process.inputs.code.uuid != expected.code.uuid:
        raise ValueError("Recovery child CP2K code mismatch.")

    if process.inputs.structure.uuid != expected.structure.uuid:
        raise ValueError("Recovery child structure mismatch.")

    actual_candidates = process.inputs.candidates.get_dict()
    expected_candidates = expected.candidates.get_dict()

    if (
        actual_candidates != expected_candidates
        or list(actual_candidates) != list(expected_candidates)
    ):
        raise ValueError("Recovery child candidates mismatch.")

    if (
        process.inputs.resources.get_dict()
        != expected.resources.get_dict()
    ):
        raise ValueError("Recovery child resources mismatch.")

    if (
        "retry_only" not in process.inputs
        or process.inputs.retry_only.value is not True
    ):
        raise ValueError("Recovery child must be retry-only.")

    expected_parameter = expected.get("parameter", None)
    actual_parameter = process.inputs.get("parameter", None)

    if (expected_parameter is None) != (actual_parameter is None):
        raise ValueError("Recovery child parameter mismatch.")

    if expected_parameter is not None:
        if actual_parameter.value != expected_parameter.value:
            raise ValueError("Recovery child parameter mismatch.")

    computer = expected.code.computer

    if computer is None:
        raise ValueError("Validated CP2K code has no computer.")

    if str(computer.uuid) != metadata.computer_uuid:
        raise ValueError(
            "Recovery execution computer UUID mismatch."
        )

    return process


def validate_aiida_execution_fingerprint(
    metadata: RecoveryExecutionMetadata,
    *,
    computer_uuid: str,
    resources: dict,
    attempt_number: int,
) -> bool:
    """Verify an AiiDA execution fingerprint against supplied inputs.

    This function does not resolve AiiDA nodes, access reservations,
    submit calculations, or certify scientific results.

    The caller must independently establish the provenance of the
    computer, resources and attempt number.
    """
    from nsdw.workflows.convergence.recovery_identity import (
        verify_execution_attempt_identity,
    )

    if not isinstance(metadata, RecoveryExecutionMetadata):
        raise TypeError("Expected RecoveryExecutionMetadata.")

    if metadata.engine != "aiida":
        raise ValueError("Expected AiiDA execution engine.")

    if (
        not isinstance(computer_uuid, str)
        or computer_uuid != metadata.computer_uuid
    ):
        raise ValueError(
            "Recovery execution computer UUID mismatch."
        )

    return verify_execution_attempt_identity(
        recorded_identity=metadata.execution_identity,
        scientific_identity=metadata.scientific_identity,
        backend="aiida",
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=attempt_number,
    )
