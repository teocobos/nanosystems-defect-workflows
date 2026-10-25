"""Real AiiDA provenance with persistent NSDW recovery reservations."""

from uuid import uuid4

import pytest

pytest_plugins = ("aiida.tools.pytest_fixtures",)


@pytest.mark.aiida_integration
def test_real_aiida_parent_child_survives_reservation_restart(
    aiida_profile_clean,
    tmp_path,
):
    """Persist genuine AiiDA process UUIDs across SQLite store reopening."""
    from aiida.common.links import LinkType
    from aiida.engine import WorkChain, run_get_node
    from aiida.orm import Dict, load_node

    from nsdw.workflows.convergence.recovery_reservations import (
        ReservationStore,
    )
    from nsdw.workflows.convergence.recovery_state import (
        RecoveryRecord,
    )

    class ChildProbe(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("value", valid_type=Dict)
            spec.output("result", valid_type=Dict)
            spec.outline(cls.execute)

        def execute(self):
            self.out("result", self.inputs.value)

    class ParentProbe(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("value", valid_type=Dict)
            spec.outline(cls.launch, cls.inspect)

        def launch(self):
            child = self.submit(
                ChildProbe,
                value=self.inputs.value,
            )
            return self.to_context(child=child)

        def inspect(self):
            assert self.ctx.child.is_finished_ok

    _, parent = run_get_node(
        ParentProbe,
        value=Dict(dict={"test": "sqlite-restart"}),
    )

    assert parent.is_finished_ok

    children = parent.base.links.get_outgoing(
        link_type=LinkType.CALL_WORK,
    ).all()

    assert len(children) == 1

    child = children[0].node

    assert child.is_finished_ok

    database = tmp_path / "recovery.sqlite"

    scientific_identity = f"science-{uuid4()}"
    execution_identity = f"execution-{uuid4()}"

    store = ReservationStore(database)

    record = RecoveryRecord(
        scientific_identity=scientific_identity,
        execution_identity=execution_identity,
    )

    reserved = store.reserve(record)

    # Submission tracking must be durably claimed before UUID binding.
    claimed = store.claim_submission(
        reserved,
        expected_version=store.get_tracking(
            scientific_identity
        )["version"],
    )

    store.bind_parent_process(claimed, str(parent.uuid))
    store.bind_process(claimed, str(child.uuid))

    before = store.get_tracking(scientific_identity)

    assert before["parent_process_uuid"] == str(parent.uuid)
    assert before["process_uuid"] == str(child.uuid)
    assert before["submission_started"] is True

    # Reopen the SQLite store to simulate an application restart.
    del store

    restarted = ReservationStore(database)

    after = restarted.get_tracking(scientific_identity)

    assert after == before

    recovered_parent = load_node(after["parent_process_uuid"])
    recovered_child = load_node(after["process_uuid"])

    assert recovered_parent.uuid == parent.uuid
    assert recovered_child.uuid == child.uuid

    incoming = recovered_child.base.links.get_incoming(
        link_type=LinkType.CALL_WORK,
    ).all()

    assert len(incoming) == 1
    assert incoming[0].node.uuid == recovered_parent.uuid

    # Recovery inspection must not alter the durable reservation.
    assert restarted.get_tracking(scientific_identity) == before


@pytest.mark.aiida_integration
@pytest.mark.parametrize("wrong_parent", [False, True])
def test_real_aiida_reconciliation_rejects_invalid_provenance(
    aiida_profile_clean,
    tmp_path,
    wrong_parent,
):
    """Real AiiDA nodes must not bypass production recovery validation."""
    from aiida.engine import WorkChain, run_get_node
    from aiida.orm import Dict

    from nsdw.workflows.convergence.aiida_reconciliation import (
        reconcile_aiida_reservation,
    )
    from nsdw.workflows.convergence.recovery_reconciliation import (
        ReconciliationDecision,
    )
    from nsdw.workflows.convergence.recovery_reservations import (
        ReservationStore,
    )
    from nsdw.workflows.convergence.recovery_state import (
        RecoveryRecord,
    )
    from nsdw.workflows.convergence.recovery_tracking import (
        RecoveryExecutionMetadata,
    )

    class ProbeChild(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("value", valid_type=Dict)
            spec.outline(cls.execute)

        def execute(self):
            pass

    class ProbeParent(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("value", valid_type=Dict)
            spec.outline(cls.launch, cls.inspect)

        def launch(self):
            child = self.submit(
                ProbeChild,
                value=self.inputs.value,
            )
            return self.to_context(child=child)

        def inspect(self):
            assert self.ctx.child.is_finished_ok

    from aiida.common.links import LinkType

    _, parent = run_get_node(
        ProbeParent,
        value=Dict(dict={"probe": "reconciliation"}),
    )

    children = parent.base.links.get_outgoing(
        link_type=LinkType.CALL_WORK,
    ).all()

    assert len(children) == 1
    child = children[0].node

    assert parent.is_finished_ok
    assert child.is_finished_ok

    if wrong_parent:
        _, supplied_parent = run_get_node(
            ProbeParent,
            value=Dict(dict={"probe": "different-parent"}),
        )
    else:
        supplied_parent = parent

    database = tmp_path / "recovery.sqlite"
    store = ReservationStore(database)

    scientific_identity = f"science-{uuid4()}"
    execution_identity = f"execution-{uuid4()}"

    record = RecoveryRecord(
        scientific_identity=scientific_identity,
        execution_identity=execution_identity,
    )

    reserved = store.reserve(record)

    claimed = store.claim_submission(
        reserved,
        expected_version=store.get_tracking(
            scientific_identity
        )["version"],
    )

    store.bind_parent_process(claimed, str(parent.uuid))
    store.bind_process(claimed, str(child.uuid))

    before = store.get_tracking(scientific_identity)

    del store
    restarted = ReservationStore(database)

    metadata = RecoveryExecutionMetadata(
        scientific_identity=scientific_identity,
        execution_identity=execution_identity,
        engine="aiida",
        computer_uuid=str(uuid4()),
        process_uuid=str(child.uuid),
    )

    # An independent WorkChain is sufficient as the original campaign
    # for this rejection-only test. It is not a valid CP2K campaign.
    _, original_campaign = run_get_node(
        ProbeChild,
        value=Dict(dict={"probe": "original"}),
    )

    result = reconcile_aiida_reservation(
        restarted,
        metadata,
        expected_recovery_parent=supplied_parent,
        original_campaign=original_campaign,
    )

    assert (
        result.decision
        == ReconciliationDecision.RECONCILIATION_REQUIRED
    )

    assert restarted.get_tracking(scientific_identity) == before
