"""Read-only recovery identity boundary tests."""

from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_recovery_identity import (
    validate_tracking_identity,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


def make_metadata(**changes):
    values = {
        "scientific_identity": "science-test",
        "execution_identity": "attempt-test",
        "engine": "aiida",
        "computer_uuid": str(uuid4()),
        "process_uuid": str(uuid4()),
    }
    values.update(changes)
    return RecoveryExecutionMetadata(**values)


def test_wrong_execution_engine_is_rejected():
    with pytest.raises(ValueError, match="AiiDA execution engine"):
        validate_tracking_identity(
            make_metadata(engine="another-engine"),
            process_loader=lambda _: None,
        )


def test_wrong_resolved_uuid_is_rejected():
    from types import SimpleNamespace

    fake = SimpleNamespace(uuid=str(uuid4()))

    with pytest.raises(ValueError, match="UUID mismatch"):
        validate_tracking_identity(
            make_metadata(),
            process_loader=lambda _: fake,
        )


@pytest.mark.aiida_integration
def test_missing_recovery_parent_is_rejected():
    from aiida import load_profile
    from aiida.orm import load_node

    load_profile("nsdw-dev")
    original = load_node(268)

    metadata = make_metadata(
        process_uuid=str(original.uuid),
        computer_uuid=str(original.inputs.code.computer.uuid),
    )

    with pytest.raises(
        ValueError,
        match="Expected recovery parent WorkChainNode is required",
    ):
        validate_tracking_identity(
            metadata,
            process_loader=lambda _: original,
        )
