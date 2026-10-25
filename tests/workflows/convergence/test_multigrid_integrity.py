
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nsdw.workflows.convergence.models import (
    ConvergenceParameter,
)
from nsdw.workflows.convergence.multigrid_extraction import (
    extract_multigrid_dataset,
)


def _setup(tmp_path, monkeypatch, *, available=("300", "400")):
    candidates = []

    for label in ("300", "400"):
        directory = tmp_path / label
        directory.mkdir()

        if label in available:
            (directory / "calculation.out").write_text(
                "CP2K output", encoding="utf-8"
            )

        candidates.append(
            SimpleNamespace(
                label=label,
                directory=label,
                input_file="calculation.inp",
                value=SimpleNamespace(
                    value=float(label),
                    unit="Ry",
                ),
            )
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.multigrid_extraction."
        "load_convergence_manifest",
        lambda path: SimpleNamespace(
            calculator="cp2k",
            parameter=ConvergenceParameter.CUTOFF,
            candidates=candidates,
        ),
    )

    def parse(path):
        return SimpleNamespace(
            multigrid=SimpleNamespace(
                levels=[
                    SimpleNamespace(
                        grid_number=1,
                        count=80,
                        cutoff_au=150.0,
                    ),
                    SimpleNamespace(
                        grid_number=2,
                        count=20,
                        cutoff_au=50.0,
                    ),
                ],
                total_gridlevel_count=100,
            )
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.multigrid_extraction."
        "parse_cp2k_output",
        parse,
    )

    return tmp_path


def test_no_outputs_skips_multigrid(tmp_path, monkeypatch):
    campaign = _setup(
        tmp_path, monkeypatch, available=()
    )
    assert extract_multigrid_dataset(campaign) is None


def test_partial_outputs_rejected(tmp_path, monkeypatch):
    campaign = _setup(
        tmp_path, monkeypatch, available=("300",)
    )
    with pytest.raises(
        ValueError, match="Incomplete CP2K output coverage"
    ):
        extract_multigrid_dataset(campaign)


def test_identical_dataset_can_be_reused(
    tmp_path, monkeypatch
):
    campaign = _setup(tmp_path, monkeypatch)

    first = extract_multigrid_dataset(campaign)
    second = extract_multigrid_dataset(campaign)

    assert first == second


def test_changed_dataset_requires_overwrite(
    tmp_path, monkeypatch
):
    campaign = _setup(tmp_path, monkeypatch)
    output = extract_multigrid_dataset(campaign)

    records = json.loads(output.read_text())
    records[0]["counts"] = [70, 30]
    output.write_text(
        json.dumps(records), encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="differs from current"
    ):
        extract_multigrid_dataset(campaign)

    restored = extract_multigrid_dataset(
        campaign, overwrite=True
    )

    assert restored == output
    assert json.loads(output.read_text())[0]["counts"] == [80, 20]
