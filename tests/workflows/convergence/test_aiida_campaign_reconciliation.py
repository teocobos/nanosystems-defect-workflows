
import json

import pytest

from nsdw.workflows.convergence.aiida_campaign_reconciliation import (
    reconcile_aiida_campaign,
)


def _campaign(tmp_path):
    manifest = {
        "campaign": "test-cutoff-campaign",
        "calculations": {
            "300": {
                "pk": 101,
                "uuid": "uuid-300",
                "cutoff_ry": 300,
            }
        },
    }

    results = {
        "results": [
            {
                "cutoff_ry": 300,
                "aiida_pk": 101,
                "finished_ok": True,
            },
            {
                "cutoff_ry": 400,
                "aiida_pk": 102,
                "finished_ok": True,
            },
        ]
    }

    summary = {
        "calculations": [
            {
                "cutoff_ry": 300,
                "aiida_pk": 101,
                "uuid": "uuid-300",
                "exit_status": 0,
            },
            {
                "cutoff_ry": 400,
                "aiida_pk": 102,
                "uuid": "uuid-400",
                "exit_status": 0,
            },
        ]
    }

    for filename, data in (
        ("manifest.json", manifest),
        ("cutoff_results.json", results),
        ("validation_summary.json", summary),
    ):
        (tmp_path / filename).write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    return tmp_path


def _edit(path, callback):
    data = json.loads(path.read_text())
    callback(data)
    path.write_text(json.dumps(data))


def test_reconciles_missing_manifest_entry(tmp_path):
    root = _campaign(tmp_path)

    campaign = reconcile_aiida_campaign(root)

    assert campaign.cutoffs == (300, 400)
    assert campaign.missing_manifest_cutoffs == (400,)
    assert campaign.calculations[300]["pk"] == 101
    assert campaign.calculations[400]["uuid"] == "uuid-400"


def test_rejects_conflicting_pk(tmp_path):
    root = _campaign(tmp_path)

    _edit(
        root / "manifest.json",
        lambda data: data["calculations"]["300"].update(
            pk=999
        ),
    )

    with pytest.raises(
        ValueError, match="inconsistent AiiDA PK"
    ):
        reconcile_aiida_campaign(root)


def test_rejects_conflicting_uuid(tmp_path):
    root = _campaign(tmp_path)

    _edit(
        root / "manifest.json",
        lambda data: data["calculations"]["300"].update(
            uuid="different-uuid"
        ),
    )

    with pytest.raises(
        ValueError, match="inconsistent AiiDA UUID"
    ):
        reconcile_aiida_campaign(root)


def test_rejects_missing_result(tmp_path):
    root = _campaign(tmp_path)

    _edit(
        root / "cutoff_results.json",
        lambda data: data["results"].pop(),
    )

    with pytest.raises(
        ValueError, match="Cutoff coverage differs"
    ):
        reconcile_aiida_campaign(root)


def test_rejects_failed_calculation(tmp_path):
    root = _campaign(tmp_path)

    _edit(
        root / "cutoff_results.json",
        lambda data: data["results"][0].update(
            finished_ok=False
        ),
    )

    with pytest.raises(
        ValueError, match="not recorded as successfully"
    ):
        reconcile_aiida_campaign(root)


def test_rejects_duplicate_cutoff(tmp_path):
    root = _campaign(tmp_path)

    _edit(
        root / "cutoff_results.json",
        lambda data: data["results"].append(
            dict(data["results"][0])
        ),
    )

    with pytest.raises(
        ValueError, match="Duplicate cutoff"
    ):
        reconcile_aiida_campaign(root)


def test_rejects_unknown_manifest_cutoff(tmp_path):
    root = _campaign(tmp_path)

    _edit(
        root / "manifest.json",
        lambda data: data["calculations"].update(
            {
                "500": {
                    "pk": 103,
                    "uuid": "uuid-500",
                }
            }
        ),
    )

    with pytest.raises(
        ValueError, match="absent from validated results"
    ):
        reconcile_aiida_campaign(root)
