"""Requirements for the read-only AiiDA recovery identity adapter."""

from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


def metadata(process_uuid=None):
    return RecoveryExecutionMetadata(
        scientific_identity="science-1",
        execution_identity="attempt-1",
        engine="aiida",
        computer_uuid=str(uuid4()),
        process_uuid=process_uuid,
    )


def test_missing_process_uuid_requires_reconciliation():
    from nsdw.workflows.convergence.aiida_recovery_identity import (
        validate_tracking_identity,
    )

    with pytest.raises(ValueError, match="UUID"):
        validate_tracking_identity(
            metadata(),
            process_loader=lambda _: None,
        )


def test_unresolvable_process_requires_reconciliation():
    from nsdw.workflows.convergence.aiida_recovery_identity import (
        validate_tracking_identity,
    )

    with pytest.raises(ValueError, match="resolve"):
        validate_tracking_identity(
            metadata(str(uuid4())),
            process_loader=lambda _: None,
        )


def test_wrong_process_type_is_rejected():
    from types import SimpleNamespace

    from nsdw.workflows.convergence.aiida_recovery_identity import (
        validate_tracking_identity,
    )

    process_uuid = str(uuid4())

    fake = SimpleNamespace(
        uuid=process_uuid,
        process_type="unrelated.process",
    )

    with pytest.raises(ValueError, match="process type"):
        validate_tracking_identity(
            metadata(process_uuid),
            process_loader=lambda _: fake,
        )
