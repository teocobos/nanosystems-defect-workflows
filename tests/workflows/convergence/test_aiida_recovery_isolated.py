
"""Isolated AiiDA recovery integration tests."""

import pytest

pytest_plugins = ("aiida.tools.pytest_fixtures",)


def test_recovery_isolated_profile(aiida_profile_clean):
    """Verify an isolated AiiDA test profile is available."""
    from aiida.manage import get_manager
    from aiida.orm import Dict, QueryBuilder

    profile = get_manager().get_profile()

    assert profile is not None
    assert profile.is_test_profile

    marker = Dict(
        dict={
            "test": "nsdw_recovery_isolation",
            "execution_enabled": False,
        }
    ).store()

    assert marker.pk is not None

    query = QueryBuilder()
    query.append(Dict, filters={"id": marker.pk})

    assert query.count() == 1


def test_recovery_process_runner(aiida_profile_clean):
    """Exercise genuine AiiDA process provenance without HPC execution."""
    from aiida.common.links import LinkType
    from aiida.engine import WorkChain, calcfunction, run_get_node
    from aiida.orm import Dict

    @calcfunction
    def create_probe_result(value):
        return Dict(dict={"probe": value.get_dict()["probe"]})

    class RecoveryRunnerProbe(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("value", valid_type=Dict)
            spec.output("result", valid_type=Dict)
            spec.outline(cls.execute_probe)

        def execute_probe(self):
            result = create_probe_result(self.inputs.value)
            self.out("result", result)

    _, node = run_get_node(
        RecoveryRunnerProbe,
        value=Dict(dict={"probe": "nsdw-isolated"}),
    )

    assert node.is_finished_ok

    result = node.outputs.result
    assert result.get_dict() == {"probe": "nsdw-isolated"}

    creators = result.base.links.get_incoming(
        link_type=LinkType.CREATE
    ).all()

    assert len(creators) == 1

    creator = creators[0].node

    callers = creator.base.links.get_incoming(
        link_type=LinkType.CALL_CALC
    ).all()

    assert len(callers) == 1
    assert callers[0].node.uuid == node.uuid

    returned = result.base.links.get_incoming(
        link_type=LinkType.RETURN
    ).all()

    assert any(link.node.uuid == node.uuid for link in returned)


def test_recovery_nested_workchain(aiida_profile_clean):
    """Verify genuine CALL_WORK provenance in an isolated profile."""
    from aiida.common.links import LinkType
    from aiida.engine import WorkChain, run_get_node
    from aiida.orm import Dict

    class RecoveryChildProbe(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("value", valid_type=Dict)
            spec.output("result", valid_type=Dict)
            spec.outline(cls.execute_child)

        def execute_child(self):
            self.out("result", self.inputs.value)

    class RecoveryParentProbe(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)
            spec.input("value", valid_type=Dict)
            spec.output("result", valid_type=Dict)
            spec.outline(
                cls.submit_child,
                cls.inspect_child,
            )

        def submit_child(self):
            child = self.submit(
                RecoveryChildProbe,
                value=self.inputs.value,
            )
            return self.to_context(child=child)

        def inspect_child(self):
            assert self.ctx.child.is_finished_ok
            self.out(
                "result",
                self.ctx.child.outputs.result,
            )

    _, parent = run_get_node(
        RecoveryParentProbe,
        value=Dict(dict={"probe": "nested-recovery"}),
    )

    assert parent.is_finished_ok

    children = parent.base.links.get_outgoing(
        link_type=LinkType.CALL_WORK
    ).all()

    assert len(children) == 1

    child = children[0].node

    assert child.is_finished_ok

    incoming = child.base.links.get_incoming(
        link_type=LinkType.CALL_WORK
    ).all()

    assert len(incoming) == 1
    assert incoming[0].node.uuid == parent.uuid

    assert (
        parent.outputs.result.get_dict()
        == {"probe": "nested-recovery"}
    )
