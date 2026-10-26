"""Tests for reporting registry schema v2."""

import json
from uuid import uuid4

import pytest

from nsdw.workflows.convergence import aiida_report_service as service


def create_registry(tmp_path, uuid=None, digest=None):
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    registry = tmp_path / "registry.json"

    service.register_campaign(
        registry,
        name="sio2",
        campaign_directory=campaign,
        report_directory=tmp_path / "report",
        expected_cutoffs=(300, 400),
        workchain_uuid=uuid,
        specification_sha256=digest,
    )
    return registry


def test_v2_provenance_roundtrip(tmp_path):
    uuid = str(uuid4())
    registry = create_registry(tmp_path, uuid, "a" * 64)

    payload = json.loads(registry.read_text())
    assert payload["schema_version"] == 2

    entry = service.load_registry(registry)[0]
    assert entry.workchain_uuid == uuid
    assert entry.specification_sha256 == "a" * 64


def test_v1_registry_still_loads(tmp_path):
    registry = create_registry(tmp_path)
    payload = json.loads(registry.read_text())
    payload["schema_version"] = 1
    registry.write_text(json.dumps(payload))

    entry = service.load_registry(registry)[0]
    assert entry.workchain_uuid is None


def test_append_preserves_existing_provenance(tmp_path):
    uuid = str(uuid4())
    registry = create_registry(tmp_path, uuid, "b" * 64)

    second = tmp_path / "second"
    second.mkdir()

    service.add_registered_campaign(
        registry,
        name="ta2o5",
        campaign_directory=second,
        report_directory=tmp_path / "second-report",
        expected_cutoffs=(500, 600),
    )

    entries = {item.name: item for item in service.load_registry(registry)}
    assert entries["sio2"].workchain_uuid == uuid
    assert entries["sio2"].specification_sha256 == "b" * 64


def test_duplicate_uuid_rejected_without_mutation(tmp_path):
    uuid = str(uuid4())
    registry = create_registry(tmp_path, uuid, "c" * 64)

    second = tmp_path / "second"
    second.mkdir()

    before = registry.read_bytes()

    with pytest.raises(ValueError, match="already registered"):
        service.add_registered_campaign(
            registry,
            name="duplicate",
            campaign_directory=second,
            report_directory=tmp_path / "different",
            expected_cutoffs=(300,),
            workchain_uuid=uuid,
            specification_sha256="c" * 64,
        )

    assert registry.read_bytes() == before


def test_partial_provenance_rejected(tmp_path):
    with pytest.raises(ValueError, match="together"):
        create_registry(tmp_path, str(uuid4()), None)


def test_invalid_digest_rejected(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        create_registry(tmp_path, str(uuid4()), "invalid")
