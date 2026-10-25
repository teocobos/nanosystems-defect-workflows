"""Opt-in AiiDA two-stage lifecycle integration test."""

import pytest

from aiida.engine import ToContext, run_get_node
from aiida.manage.configuration import load_profile
from aiida.orm import Dict, load_node

from nsdw.workflows.convergence.aiida_two_stage import (
    CP2KTwoStageConvergenceWorkChain,
)
from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)

from test_aiida_lifecycle import FakeCp2kWorkChain


class LocalConvergenceWorkChain(CP2KConvergenceWorkChain):
    """Single convergence stage with simulated CP2K children."""

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


class LocalTwoStageWorkChain(CP2KTwoStageConvergenceWorkChain):
    """Two-stage controller using local convergence WorkChains."""

    def submit_cutoff(self):
        process = self.submit(
            LocalConvergenceWorkChain,
            code=self.inputs.code,
            structure=self.inputs.structure,
            candidates=self.inputs.cutoff_candidates,
            resources=self.inputs.resources,
        )
        return ToContext(cutoff_process=process)

    def submit_relative_cutoff(self):
        from aiida.orm import Int, Str
        from nsdw.workflows.convergence.aiida_two_stage import (
            prepare_relative_candidates,
        )

        submitted = self.inputs.cutoff_candidates.get_dict()
        base_parameters = next(iter(submitted.values()))

        count = (
            self.inputs.relative_candidate_count
            if "relative_candidate_count" in self.inputs
            else Int(5)
        )

        candidates = prepare_relative_candidates(
            base_parameters=Dict(dict=base_parameters),
            cutoff_analysis=self.ctx.cutoff_process.outputs.analysis,
            candidate_count=count,
        )

        self.out("relative_candidates", candidates)

        process = self.submit(
            LocalConvergenceWorkChain,
            code=self.inputs.code,
            structure=self.inputs.structure,
            candidates=candidates,
            resources=self.inputs.resources,
            parameter=Str("relative_cutoff"),
        )

        return ToContext(relative_process=process)


@pytest.mark.aiida_integration
def test_complete_two_stage_lifecycle():
    """Execute both stages through the AiiDA engine, without remote CP2K."""
    load_profile("nsdw-dev")

    original = load_node(112)

    outputs, node = run_get_node(
        LocalTwoStageWorkChain,
        code=original.inputs.code,
        structure=original.inputs.structure,
        cutoff_candidates=original.inputs.candidates,
        resources=original.inputs.resources,
    )

    assert node.is_finished_ok, (
        f"Parent PK {node.pk} failed with status {node.exit_status}"
    )

    assert node.exit_status == 0

    expected = (
        "cutoff_summary",
        "cutoff_analysis",
        "relative_candidates",
        "relative_summary",
        "relative_analysis",
    )

    for name in expected:
        assert name in outputs
        assert outputs[name].is_stored

    cutoff = outputs["cutoff_analysis"].get_dict()
    relative = outputs["relative_analysis"].get_dict()
    candidates = outputs["relative_candidates"].get_dict()

    assert cutoff["report"]["parameter"] == "cutoff"
    assert relative["report"]["parameter"] == "relative_cutoff"

    assert cutoff["production_certified"] is False
    assert relative["production_certified"] is False

    assert len(candidates) == 5
    assert list(candidates) == [
        "10-Ry", "20-Ry", "30-Ry", "40-Ry", "50-Ry"
    ]

    for index, parameters in enumerate(candidates.values(), start=1):
        mgrid = parameters["FORCE_EVAL"]["DFT"]["MGRID"]
        assert mgrid["CUTOFF"] == 400.0
        assert mgrid["REL_CUTOFF"] == index * 10.0

    print(
        f"PASS: Two-stage AiiDA lifecycle PK {node.pk}; "
        f"five relative-cutoff candidates."
    )
