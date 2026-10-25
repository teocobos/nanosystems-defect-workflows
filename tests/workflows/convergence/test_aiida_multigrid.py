
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nsdw.workflows.convergence.aiida_multigrid import (
    extract_aiida_multigrid_dataset,
)


def _node(pk, uuid="test-uuid", finished=True):
    repository = Mock()
    repository.list_object_names.return_value = [
        "aiida.out"
    ]
    repository.get_object_content.return_value = (
        b"mock CP2K output"
    )

    return SimpleNamespace(
        pk=pk,
        uuid=uuid,
        is_finished_ok=finished,
        outputs=SimpleNamespace(
            retrieved=SimpleNamespace(
                base=SimpleNamespace(
                    repository=repository
                )
            )
        ),
    )


def _patch(monkeypatch, node):
    from aiida import orm

    monkeypatch.setattr(
        orm,
        "load_node",
        lambda pk: node,
    )

    parsed = SimpleNamespace(
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
        "nsdw.workflows.convergence.aiida_multigrid."
        "parse_cp2k_output",
        lambda path: parsed,
    )


def test_extracts_aiida_multigrid_dataset(
    tmp_path, monkeypatch
):
    _patch(monkeypatch, _node(2027))

    output = extract_aiida_multigrid_dataset(
        {300: {"pk": 2027, "uuid": "test-uuid"}},
        tmp_path / "multigrid.json",
    )

    records = json.loads(output.read_text())

    assert records[0]["cutoff_ry"] == 300.0
    assert records[0]["aiida_pk"] == 2027
    assert records[0]["counts"] == [80, 20]


def test_rejects_uuid_mismatch(
    tmp_path, monkeypatch
):
    _patch(monkeypatch, _node(2027))

    with pytest.raises(ValueError, match="UUID mismatch"):
        extract_aiida_multigrid_dataset(
            {300: {"pk": 2027, "uuid": "wrong"}},
            tmp_path / "multigrid.json",
        )


def test_rejects_failed_calculation(
    tmp_path, monkeypatch
):
    _patch(
        monkeypatch,
        _node(2027, finished=False),
    )

    with pytest.raises(
        ValueError, match="did not finish successfully"
    ):
        extract_aiida_multigrid_dataset(
            {300: {"pk": 2027}},
            tmp_path / "multigrid.json",
        )


def test_rejects_empty_campaign(tmp_path):
    with pytest.raises(
        ValueError, match="No AiiDA calculations"
    ):
        extract_aiida_multigrid_dataset(
            {},
            tmp_path / "multigrid.json",
        )
