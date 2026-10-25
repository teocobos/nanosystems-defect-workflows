
"""Tests for standalone multigrid plotting."""

import json

import pytest

from nsdw.workflows.convergence.multigrid_plotting import (
    load_multigrid_dataset,
    render_multigrid_figures,
)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "multigrid.json"
    path.write_text(
        json.dumps([
            {
                "cutoff_ry": 300,
                "grid_numbers": [1, 2],
                "counts": [80, 20],
                "grid_cutoffs_au": [150.0, 50.0],
                "reported_total": 100,
            },
            {
                "cutoff_ry": 400,
                "grid_numbers": [1, 2],
                "counts": [60, 40],
                "grid_cutoffs_au": [200.0, 66.67],
                "reported_total": 100,
            },
        ]),
        encoding="utf-8",
    )
    return path


def test_load_multigrid_dataset(dataset):
    records = load_multigrid_dataset(dataset)
    assert len(records) == 2
    assert records[0]["percentages"] == [80.0, 20.0]


def test_render_all_formats(dataset, tmp_path):
    paths = render_multigrid_figures(
        dataset,
        tmp_path / "figures",
        dpi=150,
    )

    assert len(paths) == 9
    assert all(path.is_file() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_reject_inconsistent_total(dataset):
    records = json.loads(dataset.read_text())
    records[0]["reported_total"] = 999
    dataset.write_text(json.dumps(records))

    with pytest.raises(ValueError, match="Reported total"):
        load_multigrid_dataset(dataset)


def test_reject_duplicate_candidate(dataset):
    records = json.loads(dataset.read_text())
    records[1]["cutoff_ry"] = 300
    dataset.write_text(json.dumps(records))

    with pytest.raises(ValueError, match="Duplicate"):
        load_multigrid_dataset(dataset)


def test_reject_invalid_dpi(dataset, tmp_path):
    with pytest.raises(ValueError, match="DPI"):
        render_multigrid_figures(
            dataset,
            tmp_path / "figures",
            dpi=0,
        )
