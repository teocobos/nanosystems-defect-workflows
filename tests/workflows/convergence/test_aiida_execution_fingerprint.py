"""AiiDA execution fingerprint adapter tests."""

from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_identity import (
    execution_attempt_identity,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)
from nsdw.workflows.convergence.aiida_recovery_identity import (
    validate_aiida_execution_fingerprint,
)


def make_case():
    computer_uuid = str(uuid4())
    resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 1800,
        "queue_name": "small",
        "account": "project-example",
    }

    scientific_identity = "scientific-example"

    fingerprint = execution_attempt_identity(
        scientific_identity=scientific_identity,
        backend="aiida",
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=1,
    )

    metadata = RecoveryExecutionMetadata(
        scientific_identity=scientific_identity,
        execution_identity=fingerprint,
        engine="aiida",
        computer_uuid=computer_uuid,
    )

    return metadata, computer_uuid, resources


def test_matching_aiida_execution_identity():
    metadata, computer_uuid, resources = make_case()

    assert validate_aiida_execution_fingerprint(
        metadata,
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=1,
    ) is True


def test_wrong_resources_rejected():
    metadata, computer_uuid, resources = make_case()

    changed = dict(resources)
    changed["max_wallclock_seconds"] = 600

    with pytest.raises(ValueError, match="Execution identity mismatch"):
        validate_aiida_execution_fingerprint(
            metadata,
            computer_uuid=computer_uuid,
            resources=changed,
            attempt_number=1,
        )


def test_wrong_computer_rejected():
    metadata, _, resources = make_case()

    with pytest.raises(ValueError, match="computer UUID mismatch"):
        validate_aiida_execution_fingerprint(
            metadata,
            computer_uuid=str(uuid4()),
            resources=resources,
            attempt_number=1,
        )


def test_wrong_attempt_number_rejected():
    metadata, computer_uuid, resources = make_case()

    with pytest.raises(ValueError, match="Execution identity mismatch"):
        validate_aiida_execution_fingerprint(
            metadata,
            computer_uuid=computer_uuid,
            resources=resources,
            attempt_number=2,
        )


def test_wrong_engine_rejected():
    metadata, computer_uuid, resources = make_case()

    wrong = RecoveryExecutionMetadata(
        scientific_identity=metadata.scientific_identity,
        execution_identity=metadata.execution_identity,
        engine="another-engine",
        computer_uuid=metadata.computer_uuid,
    )

    with pytest.raises(ValueError, match="AiiDA execution engine"):
        validate_aiida_execution_fingerprint(
            wrong,
            computer_uuid=computer_uuid,
            resources=resources,
            attempt_number=1,
        )
