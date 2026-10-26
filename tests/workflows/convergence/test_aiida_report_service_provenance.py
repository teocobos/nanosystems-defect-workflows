"""Reporting service provenance enforcement tests."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from nsdw.workflows.convergence import (
    aiida_report_service as service,
)
from nsdw.workflows.convergence import (
    aiida_live_provenance as provenance,
)


def registry(tmp_path, *, bound=True):
    campaign = tmp_path / "campaign"
    campaign.mkdir()

    path = tmp_path / "registry.json"

    kwargs = {}
    if bound:
        kwargs = {
            "workchain_uuid": str(uuid4()),
            "specification_sha256": "a" * 64,
        }

    service.register_campaign(
        path,
        name="sio2",
        campaign_directory=campaign,
        report_directory=tmp_path / "report",
        expected_cutoffs=(300, 400),
        **kwargs,
    )
    return path


def success(*args, **kwargs):
    return SimpleNamespace(
        state=SimpleNamespace(value="REUSED"),
        reasons=(),
    )


def test_verified_campaign_reaches_reporting(monkeypatch, tmp_path):
    path = registry(tmp_path)
    calls = []

    def verify(**kwargs):
        calls.append(("verify", kwargs))

    def process(*args, **kwargs):
        calls.append(("process", kwargs))
        return success()

    monkeypatch.setattr(
        provenance, "verify_live_workchain_provenance", verify
    )

    from nsdw.workflows.convergence import (
        aiida_parent_child_provenance as parent_child,
    )

    monkeypatch.setattr(
        parent_child,
        "verify_parent_child_provenance",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        service, "process_completed_aiida_campaign", process
    )

    result = service.run_report_service_once(path)

    assert result[0].state == "REUSED"
    assert [call[0] for call in calls] == ["verify", "process"]
    assert calls[0][1]["expected_cutoffs"] == (300, 400)


def test_mismatched_provenance_blocks_reporting(
    monkeypatch, tmp_path
):
    path = registry(tmp_path)

    def fail(**kwargs):
        raise ValueError("WorkChain UUID mismatch.")

    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not run after provenance failure.")

    monkeypatch.setattr(
        provenance, "verify_live_workchain_provenance", fail
    )
    monkeypatch.setattr(
        service, "process_completed_aiida_campaign", forbidden
    )

    result = service.run_report_service_once(path)

    assert result[0].state == "BLOCKED"
    assert "WorkChain UUID mismatch" in result[0].detail


def test_missing_aiida_node_blocks_reporting(monkeypatch, tmp_path):
    path = registry(tmp_path)

    def fail(**kwargs):
        raise RuntimeError("AiiDA node unavailable.")

    monkeypatch.setattr(
        provenance, "verify_live_workchain_provenance", fail
    )
    monkeypatch.setattr(
        service,
        "process_completed_aiida_campaign",
        lambda *args, **kwargs: pytest.fail(
            "Reporting must not run without the AiiDA node."
        ),
    )

    result = service.run_report_service_once(path)

    assert result[0].state == "BLOCKED"
    assert "AiiDA node unavailable" in result[0].detail


def test_legacy_campaign_retains_reporting(
    monkeypatch, tmp_path
):
    path = registry(tmp_path, bound=False)

    monkeypatch.setattr(
        provenance,
        "verify_live_workchain_provenance",
        lambda **kwargs: pytest.fail(
            "Legacy campaign must not invoke bound verification."
        ),
    )
    monkeypatch.setattr(
        service, "process_completed_aiida_campaign", success
    )

    result = service.run_report_service_once(path)
    assert result[0].state == "REUSED"


def test_verification_receives_registered_digest(
    monkeypatch, tmp_path
):
    path = registry(tmp_path)
    received = []

    monkeypatch.setattr(
        provenance,
        "verify_live_workchain_provenance",
        lambda **kwargs: received.append(kwargs),
    )
    monkeypatch.setattr(
        service, "process_completed_aiida_campaign", success
    )

    service.run_report_service_once(path)

    assert len(received) == 1
    assert received[0]["registered_specification_sha256"] == (
        "a" * 64
    )
    assert received[0]["registered_workchain_uuid"] == (
        service.load_registry(path)[0].workchain_uuid
    )
