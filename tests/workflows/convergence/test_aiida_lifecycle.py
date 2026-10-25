"""Opt-in local lifecycle test for the NSDW AiiDA WorkChain."""

import pytest

from aiida.engine import ToContext, WorkChain, calcfunction, run_get_node
from aiida.manage.configuration import load_profile
from aiida.orm import Dict, load_node

from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)


@calcfunction
def create_fake_cp2k_output() -> Dict:
    """Create a provenance-tracked deterministic CP2K result."""
    return Dict(dict={
        "energy": -108.49635643498,
        "energy_units": "a.u.",
        "nwarnings": 0,
    })


class FakeCp2kWorkChain(WorkChain):
    """Deterministic local substitute for a CP2K calculation."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.outline(cls.produce_result)
        spec.output("output_parameters", valid_type=Dict)

    def produce_result(self):
        self.out(
            "output_parameters",
            create_fake_cp2k_output(),
        )


class LocalConvergenceWorkChain(CP2KConvergenceWorkChain):
    """Test parent using local AiiDA child processes."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.outline(
            cls.submit_candidates,
            cls.inspect_candidates,
        )

    def submit_candidates(self):
        candidates = self.inputs.candidates.get_dict()

        self.ctx.candidate_names = {}
        running = {}

        for index, name in enumerate(candidates):
            internal_id = f"candidate_{index:04d}"
            self.ctx.candidate_names[internal_id] = name
            running[internal_id] = self.submit(FakeCp2kWorkChain)

        return ToContext(**running)


@pytest.mark.aiida_integration
def test_complete_parent_workchain_lifecycle():
    """Run a full parent-child lifecycle without remote CP2K."""
    load_profile("nsdw-dev")

    original = load_node(112)

    inputs = {
        "code": original.inputs.code,
        "structure": original.inputs.structure,
        "candidates": original.inputs.candidates,
        "resources": original.inputs.resources,
    }

    outputs, node = run_get_node(
        LocalConvergenceWorkChain,
        **inputs,
    )

    assert node.is_finished_ok
    assert node.exit_status == 0
    assert "summary" in outputs
    assert "analysis" in outputs

    summary = outputs["summary"]
    analysis = outputs["analysis"]

    assert summary.is_stored
    assert analysis.is_stored

    analysis_data = analysis.get_dict()
    assert analysis_data["validation_scope"] == "standard_prefix"
    assert analysis_data["production_certified"] is False
    assert len(analysis_data["report"]["candidates"]) == 5

    entries = summary.get_dict()
    assert len(entries) == len(inputs["candidates"].get_dict())

    for entry in entries.values():
        assert entry["finished_ok"] is True
        assert entry["energy_hartree"] == -108.49635643498
