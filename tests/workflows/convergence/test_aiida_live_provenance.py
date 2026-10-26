"""Offline tests for live AiiDA WorkChain provenance verification."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from aiida.orm import WorkChainNode

from nsdw.workflows.convergence.aiida_campaign_specification import (
    specification_digest,
)
from nsdw.workflows.convergence.aiida_live_provenance import (
    verify_live_workchain_provenance,
)


class FakeDict:
    def __init__(self, payload):
        self.payload = payload

    def get_dict(self):
        return self.payload


def evidence():
    uuid = str(uuid4())
    specification = {
        "schema_version": 1,
        "campaign_name": "sio2",
        "expected_cutoffs": [300, 400],
    }
    digest = specification_digest(specification)

    scientific = {
        "specification": specification,
        "specification_sha256": digest,
        "candidate_cutoffs": {
            "candidate_300": 300,
            "candidate_400": 400,
        },
    }
    candidates = {
        "candidate_300": {
            "FORCE_EVAL": {
                "DFT": {"MGRID": {"CUTOFF": 300}}
            }
        },
        "candidate_400": {
            "FORCE_EVAL": {
                "DFT": {"MGRID": {"CUTOFF": 400}}
            }
        },
    }

    node = object.__new__(WorkChainNode)
    return uuid, digest, scientific, candidates, node


def verify(monkeypatch, *, mutate=None, expected=(300, 400)):
    uuid, digest, scientific, candidates, node = evidence()

    if mutate is not None:
        mutate(scientific, candidates)

    # Test the storage adapter without a database or daemon.
    class FakeWorkChainNode:
        pass

    fake = FakeWorkChainNode()
    fake.uuid = uuid
    fake.inputs = SimpleNamespace(
        scientific_specification=FakeDict(scientific),
        candidates=FakeDict(candidates),
    )

    import aiida.orm

    monkeypatch.setattr(
        aiida.orm, "WorkChainNode", FakeWorkChainNode
    )

    return verify_live_workchain_provenance(
        registered_workchain_uuid=uuid,
        registered_specification_sha256=digest,
        expected_cutoffs=expected,
        campaign_name="sio2",
        node_loader=lambda _: fake,
    )


def test_valid_workchain_passes(monkeypatch):
    assert verify(monkeypatch) is None


def test_registry_cutoff_mismatch_blocked(monkeypatch):
    with pytest.raises(ValueError, match="Registry cutoffs"):
        verify(monkeypatch, expected=(300, 500))


def test_actual_cp2k_cutoff_mismatch_blocked(monkeypatch):
    def mutate(scientific, candidates):
        candidates["candidate_400"]["FORCE_EVAL"]["DFT"][
            "MGRID"
        ]["CUTOFF"] = 500

    with pytest.raises(ValueError, match="CP2K cutoff mismatch"):
        verify(monkeypatch, mutate=mutate)


def test_candidate_mapping_mismatch_blocked(monkeypatch):
    def mutate(scientific, candidates):
        scientific["candidate_cutoffs"]["candidate_400"] = 500

    with pytest.raises(ValueError, match="scientific specification"):
        verify(monkeypatch, mutate=mutate)


def test_missing_scientific_input_blocked(monkeypatch):
    uuid, digest, scientific, candidates, _ = evidence()

    class FakeWorkChainNode:
        pass

    fake = FakeWorkChainNode()
    fake.uuid = uuid
    fake.inputs = SimpleNamespace(
        candidates=FakeDict(candidates)
    )

    import aiida.orm

    monkeypatch.setattr(
        aiida.orm, "WorkChainNode", FakeWorkChainNode
    )

    with pytest.raises(ValueError, match="missing required"):
        verify_live_workchain_provenance(
            registered_workchain_uuid=uuid,
            registered_specification_sha256=digest,
            expected_cutoffs=(300, 400),
            campaign_name="sio2",
            node_loader=lambda _: fake,
        )
