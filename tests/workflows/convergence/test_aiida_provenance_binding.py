"""Tests for AiiDA provenance binding validation."""

from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_campaign_specification import (
    specification_digest,
)
from nsdw.workflows.convergence.aiida_provenance_binding import (
    validate_provenance_binding,
)


def make_binding():
    uuid = str(uuid4())

    specification = {
        "schema_version": 1,
        "campaign_name": "test-campaign",
        "expected_cutoffs": [300, 400],
    }

    digest = specification_digest(specification)

    inputs = {
        "specification": specification,
        "specification_sha256": digest,
        "candidate_cutoffs": {
            "candidate_a": 300,
            "candidate_b": 400,
        },
    }

    return uuid, digest, inputs


def test_valid_provenance_binding():
    uuid, digest, inputs = make_binding()

    result = validate_provenance_binding(
        registered_workchain_uuid=uuid,
        registered_specification_sha256=digest,
        actual_workchain_uuid=uuid,
        workchain_specification_input=inputs,
    )

    assert result.workchain_uuid == uuid
    assert result.specification_sha256 == digest


def test_workchain_uuid_mismatch_rejected():
    uuid, digest, inputs = make_binding()

    with pytest.raises(ValueError, match="UUID mismatch"):
        validate_provenance_binding(
            registered_workchain_uuid=uuid,
            registered_specification_sha256=digest,
            actual_workchain_uuid=str(uuid4()),
            workchain_specification_input=inputs,
        )


def test_registry_digest_mismatch_rejected():
    uuid, digest, inputs = make_binding()

    with pytest.raises(ValueError, match="Registry"):
        validate_provenance_binding(
            registered_workchain_uuid=uuid,
            registered_specification_sha256="0" * 64,
            actual_workchain_uuid=uuid,
            workchain_specification_input=inputs,
        )


def test_workchain_digest_mismatch_rejected():
    uuid, digest, inputs = make_binding()
    inputs["specification_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="WorkChain"):
        validate_provenance_binding(
            registered_workchain_uuid=uuid,
            registered_specification_sha256=digest,
            actual_workchain_uuid=uuid,
            workchain_specification_input=inputs,
        )


def test_missing_workchain_input_rejected():
    uuid, digest, inputs = make_binding()
    del inputs["specification"]

    with pytest.raises(ValueError, match="fields"):
        validate_provenance_binding(
            registered_workchain_uuid=uuid,
            registered_specification_sha256=digest,
            actual_workchain_uuid=uuid,
            workchain_specification_input=inputs,
        )


def test_invalid_uuid_rejected():
    uuid, digest, inputs = make_binding()

    with pytest.raises(ValueError, match="UUID"):
        validate_provenance_binding(
            registered_workchain_uuid="not-a-uuid",
            registered_specification_sha256=digest,
            actual_workchain_uuid=uuid,
            workchain_specification_input=inputs,
        )
