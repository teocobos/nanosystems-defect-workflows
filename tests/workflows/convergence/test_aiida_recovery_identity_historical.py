"""Historical AiiDA process identity boundary tests.

These tests use existing stored nodes and never submit calculations.
"""

import pytest
from aiida import load_profile
from aiida.orm import load_node

from nsdw.workflows.convergence.aiida_recovery_identity import (
    validate_tracking_identity,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


@pytest.mark.aiida_integration
def test_original_campaign_cannot_be_verified_as_retry_child():
    load_profile("nsdw-dev")

    original = load_node(268)

    metadata = RecoveryExecutionMetadata(
        scientific_identity="historical-science",
        execution_identity="historical-attempt",
        engine="aiida",
        computer_uuid=str(original.inputs.code.computer.uuid),
        process_uuid=str(original.uuid),
    )

    # An original campaign has the correct WorkChain type but
    # lacks the expected recovery-child parent provenance.
    with pytest.raises(ValueError, match="Recovery child parent provenance mismatch"):
        validate_tracking_identity(
            metadata,
            expected_recovery_parent=load_node(605),
            original_campaign=original,
        )


@pytest.mark.aiida_integration
def test_recovery_controller_is_not_a_retry_child():
    load_profile("nsdw-dev")

    recovery = load_node(605)
    original = load_node(268)

    metadata = RecoveryExecutionMetadata(
        scientific_identity="historical-science",
        execution_identity="historical-attempt",
        engine="aiida",
        computer_uuid=str(original.inputs.code.computer.uuid),
        process_uuid=str(recovery.uuid),
    )

    with pytest.raises(ValueError, match="Unexpected AiiDA recovery child process type"):
        validate_tracking_identity(
            metadata,
            expected_recovery_parent=recovery,
            original_campaign=original,
        )
