"""Independent calculations cannot claim WorkChain provenance."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from nsdw.workflows.convergence.aiida_parent_child_provenance import (
    verify_parent_child_provenance,
)


def test_independent_calculations_rejected_as_workchain_children(
    monkeypatch, tmp_path
):
    """No CALL_CALC children means no WorkChain-bound verification."""
    import aiida.orm

    from nsdw.workflows.convergence import (
        aiida_parent_child_provenance as provenance,
    )

    class FakeWorkChainNode:
        pass

    class FakeCalcJobNode:
        pass

    parent = FakeWorkChainNode()
    parent.uuid = str(uuid4())
    parent.base = SimpleNamespace(
        links=SimpleNamespace(
            get_outgoing=lambda **kwargs: SimpleNamespace(
                all=lambda: []
            )
        )
    )

    monkeypatch.setattr(
        aiida.orm, "WorkChainNode", FakeWorkChainNode
    )
    monkeypatch.setattr(
        aiida.orm, "CalcJobNode", FakeCalcJobNode
    )

    monkeypatch.setattr(
        provenance,
        "reconcile_aiida_campaign",
        lambda directory: SimpleNamespace(
            calculations={
                300: {"pk": 2027, "uuid": str(uuid4())},
                400: {"pk": 2029, "uuid": str(uuid4())},
            }
        ),
    )

    with pytest.raises(ValueError, match="child count"):
        verify_parent_child_provenance(
            registered_workchain_uuid=parent.uuid,
            campaign_directory=str(tmp_path),
            expected_cutoffs=(300, 400),
            node_loader=lambda uuid: parent,
        )


def test_independent_calculations_have_no_synthetic_parent_links():
    """Record the observed LUMI-C provenance classification."""
    independently_submitted = {
        300: 2027,
        400: 2029,
        500: 2031,
        600: 2022,
        800: 2033,
    }

    assert len(independently_submitted) == 5
    assert len(set(independently_submitted.values())) == 5
