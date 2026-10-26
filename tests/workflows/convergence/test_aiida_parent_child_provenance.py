"""Offline tests for parent-child AiiDA provenance verification."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_parent_child_provenance import (
    verify_parent_child_provenance,
)


def setup_campaign(tmp_path):
    parent_uuid = str(uuid4())
    child_300 = str(uuid4())
    child_400 = str(uuid4())

    records = [
        {
            "cutoff_ry": 300,
            "aiida_pk": 101,
            "uuid": child_300,
        },
        {
            "cutoff_ry": 400,
            "aiida_pk": 102,
            "uuid": child_400,
        },
    ]

    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "campaign": "sio2",
            "calculations": {},
        })
    )
    (tmp_path / "cutoff_results.json").write_text(
        json.dumps({
            "results": [
                {
                    "cutoff_ry": record["cutoff_ry"],
                    "aiida_pk": record["aiida_pk"],
                    "finished_ok": True,
                }
                for record in records
            ]
        })
    )
    (tmp_path / "validation_summary.json").write_text(
        json.dumps({
            "calculations": [
                {
                    **record,
                    "exit_status": 0,
                }
                for record in records
            ]
        })
    )

    return parent_uuid, records


def run_check(
    monkeypatch,
    tmp_path,
    *,
    change=None,
    expected=(300, 400),
):
    import aiida.orm

    parent_uuid, records = setup_campaign(tmp_path)

    class FakeWorkChainNode:
        pass

    class FakeCalcJobNode:
        pass

    children = []

    for record in records:
        child = FakeCalcJobNode()
        child.pk = record["aiida_pk"]
        child.uuid = record["uuid"]
        child.inputs = SimpleNamespace(
            parameters=SimpleNamespace(
                get_dict=lambda cutoff=record["cutoff_ry"]: {
                    "FORCE_EVAL": {
                        "DFT": {
                            "MGRID": {"CUTOFF": cutoff}
                        }
                    }
                }
            )
        )
        children.append(child)

    if change is not None:
        change(children, records)

    parent = FakeWorkChainNode()
    parent.uuid = parent_uuid
    parent.base = SimpleNamespace(
        links=SimpleNamespace(
            get_outgoing=lambda **kwargs: SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(node=child)
                    for child in children
                ]
            )
        )
    )

    monkeypatch.setattr(
        aiida.orm, "WorkChainNode", FakeWorkChainNode
    )
    monkeypatch.setattr(
        aiida.orm, "CalcJobNode", FakeCalcJobNode
    )

    return verify_parent_child_provenance(
        registered_workchain_uuid=parent_uuid,
        campaign_directory=str(tmp_path),
        expected_cutoffs=expected,
        node_loader=lambda _: parent,
    )


def test_valid_parent_child_provenance(monkeypatch, tmp_path):
    assert run_check(monkeypatch, tmp_path) is None


def test_missing_child_blocked(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="child count"):
        run_check(
            monkeypatch,
            tmp_path,
            change=lambda children, records: children.pop(),
        )


def test_wrong_child_pk_blocked(monkeypatch, tmp_path):
    def change(children, records):
        children[1].pk = 999

    with pytest.raises(ValueError, match="PK mismatch"):
        run_check(monkeypatch, tmp_path, change=change)


def test_wrong_child_uuid_blocked(monkeypatch, tmp_path):
    def change(children, records):
        children[1].uuid = str(uuid4())

    with pytest.raises(ValueError, match="UUID mismatch"):
        run_check(monkeypatch, tmp_path, change=change)


def test_registry_cutoff_mismatch_blocked(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="cutoff coverage"):
        run_check(monkeypatch, tmp_path, expected=(300, 500))


def test_duplicate_child_cutoff_blocked(monkeypatch, tmp_path):
    def change(children, records):
        children[1].inputs.parameters.get_dict = (
            lambda: {
                "FORCE_EVAL": {
                    "DFT": {"MGRID": {"CUTOFF": 300}}
                }
            }
        )

    with pytest.raises(ValueError, match="Duplicate CP2K cutoff"):
        run_check(monkeypatch, tmp_path, change=change)
