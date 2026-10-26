"""Verify parent-child enforcement in the report service."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from nsdw.workflows.convergence import (
    aiida_live_provenance as live,
    aiida_parent_child_provenance as parent_child,
    aiida_report_service as service,
)


def create_registry(tmp_path, *, bound=True):
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    registry = tmp_path / "registry.json"

    kwargs = {}
    if bound:
        kwargs = {
            "workchain_uuid": str(uuid4()),
            "specification_sha256": "a" * 64,
        }

    service.register_campaign(
        registry,
        name="sio2",
        campaign_directory=campaign,
        report_directory=tmp_path / "report",
        expected_cutoffs=(300, 400),
        **kwargs,
    )

    return registry


def successful_report(*args, **kwargs):
    return SimpleNamespace(
        state=SimpleNamespace(value="REUSED"),
        reasons=(),
    )


def test_both_gates_precede_reporting(monkeypatch, tmp_path):
    registry = create_registry(tmp_path)
    calls = []

    monkeypatch.setattr(
        live,
        "verify_live_workchain_provenance",
        lambda **kwargs: calls.append("scientific"),
    )
    monkeypatch.setattr(
        parent_child,
        "verify_parent_child_provenance",
        lambda **kwargs: calls.append("parent_child"),
    )
    monkeypatch.setattr(
        service,
        "process_completed_aiida_campaign",
        lambda *args, **kwargs: (
            calls.append("report") or successful_report()
        ),
    )

    results = service.run_report_service_once(registry)

    assert results[0].state == "REUSED"
    assert calls == ["scientific", "parent_child", "report"]


def test_parent_child_failure_blocks_reporting(
    monkeypatch, tmp_path
):
    registry = create_registry(tmp_path)

    monkeypatch.setattr(
        live,
        "verify_live_workchain_provenance",
        lambda **kwargs: None,
    )

    def fail(**kwargs):
        raise ValueError("AiiDA CALL_CALC mismatch.")

    monkeypatch.setattr(
        parent_child,
        "verify_parent_child_provenance",
        fail,
    )

    monkeypatch.setattr(
        service,
        "process_completed_aiida_campaign",
        lambda *args, **kwargs: pytest.fail(
            "Reporting must not run after provenance failure."
        ),
    )

    results = service.run_report_service_once(registry)

    assert results[0].state == "BLOCKED"
    assert "CALL_CALC mismatch" in results[0].detail


def test_scientific_failure_skips_parent_child(
    monkeypatch, tmp_path
):
    registry = create_registry(tmp_path)

    def fail(**kwargs):
        raise ValueError("Scientific specification mismatch.")

    monkeypatch.setattr(
        live,
        "verify_live_workchain_provenance",
        fail,
    )

    monkeypatch.setattr(
        parent_child,
        "verify_parent_child_provenance",
        lambda **kwargs: pytest.fail(
            "Parent-child verification must not run."
        ),
    )

    results = service.run_report_service_once(registry)

    assert results[0].state == "BLOCKED"


def test_legacy_campaign_skips_both_gates(
    monkeypatch, tmp_path
):
    registry = create_registry(tmp_path, bound=False)

    monkeypatch.setattr(
        live,
        "verify_live_workchain_provenance",
        lambda **kwargs: pytest.fail(
            "Legacy campaign must not invoke scientific gate."
        ),
    )

    monkeypatch.setattr(
        parent_child,
        "verify_parent_child_provenance",
        lambda **kwargs: pytest.fail(
            "Legacy campaign must not invoke parent-child gate."
        ),
    )

    monkeypatch.setattr(
        service,
        "process_completed_aiida_campaign",
        successful_report,
    )

    results = service.run_report_service_once(registry)

    assert results[0].state == "REUSED"


def test_parent_child_receives_registry_identity(
    monkeypatch, tmp_path
):
    registry = create_registry(tmp_path)
    received = []

    monkeypatch.setattr(
        live,
        "verify_live_workchain_provenance",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        parent_child,
        "verify_parent_child_provenance",
        lambda **kwargs: received.append(kwargs),
    )
    monkeypatch.setattr(
        service,
        "process_completed_aiida_campaign",
        successful_report,
    )

    service.run_report_service_once(registry)

    campaign = service.load_registry(registry)[0]

    assert len(received) == 1
    assert received[0]["registered_workchain_uuid"] == (
        campaign.workchain_uuid
    )
    assert received[0]["campaign_directory"] == str(
        campaign.campaign_directory
    )
    assert received[0]["expected_cutoffs"] == (300, 400)
