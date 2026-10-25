"""Regression tests for atomic AiiDA multigrid reporting."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from nsdw.workflows.convergence import (
    aiida_multigrid_reporting as reporting,
)


def setup_campaign(monkeypatch, *, plot_mode="complete"):
    campaign = SimpleNamespace(
        calculations={300: {"pk": 101, "uuid": "test-uuid"}},
        cutoffs=(300,),
        missing_manifest_cutoffs=(),
        campaign_name="test-campaign",
    )

    monkeypatch.setattr(
        reporting, "reconcile_aiida_campaign",
        lambda directory: campaign,
    )

    def extract(calculations, destination):
        destination.write_text("[]", encoding="utf-8")
        return destination

    monkeypatch.setattr(
        reporting, "extract_aiida_multigrid_dataset",
        extract,
    )
    monkeypatch.setattr(
        reporting, "load_multigrid_dataset",
        lambda path: [],
    )

    def render(dataset, directory, **kwargs):
        assert kwargs["formats"] == ("png", "pdf", "svg")
        assert kwargs["dpi"] == 600

        if plot_mode == "fail":
            raise RuntimeError("Simulated plotting failure")

        if plot_mode == "missing":
            return

        for name in (
            "gaussian_grid_counts",
            "gaussian_grid_distribution",
            "multigrid_cutoff_levels",
        ):
            for ext in ("png", "pdf", "svg"):
                (directory / f"{name}.{ext}").write_bytes(
                    b"test-figure"
                )

    monkeypatch.setattr(
        reporting, "render_multigrid_figures", render,
    )


def test_generates_report(tmp_path, monkeypatch):
    setup_campaign(monkeypatch)
    output = tmp_path / "report"

    result = reporting.generate_aiida_multigrid_report(
        tmp_path, output
    )

    assert result.dataset == output / "multigrid_data.json"
    assert result.dataset.is_file()
    assert result.cutoffs == (300,)
    assert result.missing_manifest_cutoffs == ()

    provenance = json.loads(
        (output / "provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert provenance["cutoffs_ry"] == [300]
    assert provenance["calculations"]["300"]["pk"] == 101
    assert provenance["calculations"]["300"]["uuid"] == "test-uuid"
    assert len(provenance["sha256"]) == 10

    for filename, expected in provenance["sha256"].items():
        actual = hashlib.sha256(
            (output / filename).read_bytes()
        ).hexdigest()
        assert actual == expected

    assert not list(tmp_path.glob(".report-staging-*"))


def test_preserves_existing_directory(tmp_path, monkeypatch):
    setup_campaign(monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel.txt").write_text("preserve")

    with pytest.raises(FileExistsError):
        reporting.generate_aiida_multigrid_report(
            tmp_path, output
        )

    assert (output / "sentinel.txt").read_text() == "preserve"


def test_atomic_report_cleans_up_on_plot_failure(
    tmp_path, monkeypatch
):
    setup_campaign(monkeypatch, plot_mode="fail")
    output = tmp_path / "report"

    with pytest.raises(
        RuntimeError, match="Simulated plotting failure"
    ):
        reporting.generate_aiida_multigrid_report(
            tmp_path, output
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".report-staging-*"))


def test_atomic_report_rejects_missing_figures(
    tmp_path, monkeypatch
):
    setup_campaign(monkeypatch, plot_mode="missing")
    output = tmp_path / "report"

    with pytest.raises(
        RuntimeError, match="Incomplete AiiDA multigrid report"
    ):
        reporting.generate_aiida_multigrid_report(
            tmp_path, output
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".report-staging-*"))
