
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nsdw.workflows.convergence.multigrid_extraction import (
    extract_multigrid_dataset,
)


def _campaign(tmp_path, monkeypatch, *, parameter="cutoff",
              missing=()):
    candidates = []

    for label, cutoff in (
        ("300-Ry", 300),
        ("400-Ry", 400),
    ):
        directory = tmp_path / label
        directory.mkdir()

        output = directory / "calculation.out"
        output.write_text("CP2K output", encoding="utf-8")

        candidates.append(
            SimpleNamespace(
                label=label,
                directory=label,
                input_file="calculation.inp",
                value=SimpleNamespace(value=cutoff, unit="Ry"),
            )
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.multigrid_extraction."
        "load_convergence_manifest",
        lambda path: SimpleNamespace(
            calculator="cp2k",
            parameter=(
                __import__(
                    "nsdw.workflows.convergence.models",
                    fromlist=["ConvergenceParameter"],
                ).ConvergenceParameter.CUTOFF
                if parameter == "cutoff"
                else __import__(
                    "nsdw.workflows.convergence.models",
                    fromlist=["ConvergenceParameter"],
                ).ConvergenceParameter.RELATIVE_CUTOFF
            ),
            candidates=candidates,
        ),
    )

    def fake_parse(path):
        label = Path(path).parent.name

        if label in missing:
            return SimpleNamespace(multigrid=None)

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
        fake_parse,
    )

    return tmp_path


def test_extracts_dataset(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path, monkeypatch)

    output = extract_multigrid_dataset(campaign)

    assert output.is_file()
    records = json.loads(output.read_text())
    assert len(records) == 2
    assert records[0]["cutoff_ry"] == 300


def test_skips_non_cutoff_studies(tmp_path, monkeypatch):
    campaign = _campaign(
        tmp_path, monkeypatch, parameter="relative"
    )

    assert extract_multigrid_dataset(campaign) is None


def test_skips_when_no_multigrid_available(
    tmp_path, monkeypatch
):
    campaign = _campaign(
        tmp_path,
        monkeypatch,
        missing=("300-Ry", "400-Ry"),
    )

    assert extract_multigrid_dataset(campaign) is None


def test_rejects_partial_multigrid_data(
    tmp_path, monkeypatch
):
    campaign = _campaign(
        tmp_path,
        monkeypatch,
        missing=("400-Ry",),
    )

    with pytest.raises(ValueError, match="Incomplete"):
        extract_multigrid_dataset(campaign)


def test_reuses_identical_dataset_without_overwriting(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path, monkeypatch)

    output = extract_multigrid_dataset(campaign)
    original_content = output.read_bytes()

    reused = extract_multigrid_dataset(campaign)

    assert reused == output
    assert output.read_bytes() == original_content
