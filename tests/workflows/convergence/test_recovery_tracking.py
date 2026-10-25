"""Tests for portable recovery execution metadata."""

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
    attach_process_uuid,
    requires_process_reconciliation,
)


def metadata(computer="lumi-computer"):
    return RecoveryExecutionMetadata(
        scientific_identity="scientific-hash",
        execution_identity="execution-hash",
        engine="aiida",
        computer_uuid=computer,
    )


def test_metadata_has_no_process_before_submission():
    assert metadata().process_uuid is None


def test_process_uuid_can_be_attached():
    original = metadata()
    process_uuid = str(uuid4())
    updated = attach_process_uuid(original, process_uuid)

    assert updated.process_uuid == process_uuid
    assert original.process_uuid is None


def test_same_process_binding_is_idempotent():
    process_uuid = str(uuid4())
    attached = attach_process_uuid(metadata(), process_uuid)

    assert attach_process_uuid(attached, process_uuid) == attached


def test_different_process_binding_is_rejected():
    attached = attach_process_uuid(metadata(), str(uuid4()))

    with pytest.raises(ValueError, match="different process"):
        attach_process_uuid(attached, str(uuid4()))


def test_invalid_process_uuid_is_rejected():
    with pytest.raises(ValueError, match="Invalid process UUID"):
        attach_process_uuid(metadata(), "not-a-uuid")


def test_uncertain_submission_requires_reconciliation():
    assert requires_process_reconciliation(metadata(), True)
    assert not requires_process_reconciliation(metadata(), False)


def test_known_process_does_not_have_missing_uuid():
    attached = attach_process_uuid(metadata(), str(uuid4()))
    assert not requires_process_reconciliation(attached, True)


def test_metadata_is_immutable():
    original = metadata()

    with pytest.raises(FrozenInstanceError):
        original.engine = "different-engine"


def test_tracking_is_cluster_independent():
    lumi = metadata("lumi-computer")
    cirrus = metadata("cirrus-computer")
    isambard = metadata("isambard-computer")

    assert len({
        lumi.computer_uuid,
        cirrus.computer_uuid,
        isambard.computer_uuid,
    }) == 3

    assert all(
        item.engine == "aiida"
        for item in (lumi, cirrus, isambard)
    )


def test_missing_identity_is_rejected():
    with pytest.raises(ValueError):
        RecoveryExecutionMetadata(
            scientific_identity="",
            execution_identity="attempt",
            engine="aiida",
            computer_uuid="computer",
        )
