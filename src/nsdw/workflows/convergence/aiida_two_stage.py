"""Unattended, provisional two-stage AiiDA CP2K convergence."""

from __future__ import annotations

from aiida.engine import ToContext, WorkChain, calcfunction
from aiida.orm import Code, Dict, Int, Str, StructureData

from nsdw.workflows.convergence.aiida_stage_candidates import (
    build_relative_cutoff_candidates,
)
from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
    validate_candidate_payload,
)


@calcfunction
def prepare_relative_candidates(
    base_parameters: Dict,
    cutoff_analysis: Dict,
    candidate_count: Int,
) -> Dict:
    """Generate Stage 2 inputs with explicit AiiDA provenance."""
    candidates = build_relative_cutoff_candidates(
        base_parameters=base_parameters.get_dict(),
        cutoff_analysis=cutoff_analysis.get_dict(),
        candidate_count=candidate_count.value,
    )
    return Dict(dict=candidates)


class CP2KTwoStageConvergenceWorkChain(WorkChain):
    """Run cutoff, then relative-cutoff convergence via AiiDA.

    This validation workflow is deliberately provisional. It does not
    certify production convergence or perform final joint verification.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("cutoff_candidates", valid_type=Dict)
        spec.input("resources", valid_type=Dict)
        spec.input(
            "relative_candidate_count",
            valid_type=Int,
            required=False,
        )

        spec.outline(
            cls.submit_cutoff,
            cls.inspect_cutoff,
            cls.submit_relative_cutoff,
            cls.inspect_relative_cutoff,
        )

        spec.output("cutoff_summary", valid_type=Dict)
        spec.output("cutoff_analysis", valid_type=Dict)
        spec.output("relative_candidates", valid_type=Dict)
        spec.output("relative_summary", valid_type=Dict)
        spec.output("relative_analysis", valid_type=Dict)

        spec.exit_code(
            410,
            "ERROR_INVALID_INPUT",
            message="Invalid two-stage convergence inputs.",
        )
        spec.exit_code(
            411,
            "ERROR_CUTOFF_FAILED",
            message="Cutoff convergence campaign failed.",
        )
        spec.exit_code(
            412,
            "ERROR_STAGE_PREPARATION_FAILED",
            message="Could not prepare relative-cutoff candidates.",
        )
        spec.exit_code(
            413,
            "ERROR_RELATIVE_FAILED",
            message="Relative-cutoff convergence campaign failed.",
        )

    def submit_cutoff(self):
        candidates = self.inputs.cutoff_candidates.get_dict()
        resources = self.inputs.resources.get_dict()

        try:
            validate_candidate_payload(candidates, resources)
        except ValueError as exc:
            self.report(f"Invalid cutoff inputs: {exc}")
            return self.exit_codes.ERROR_INVALID_INPUT

        self.report("Submitting cutoff convergence campaign.")

        process = self.submit(
            CP2KConvergenceWorkChain,
            code=self.inputs.code,
            structure=self.inputs.structure,
            candidates=self.inputs.cutoff_candidates,
            resources=self.inputs.resources,
            parameter=Str("cutoff"),
        )

        return ToContext(cutoff_process=process)

    def inspect_cutoff(self):
        process = self.ctx.cutoff_process

        if "summary" in process.outputs:
            self.out("cutoff_summary", process.outputs.summary)

        if not process.is_finished_ok or "analysis" not in process.outputs:
            self.report(
                f"Cutoff campaign failed: PK {process.pk}, "
                f"exit status {process.exit_status}."
            )
            return self.exit_codes.ERROR_CUTOFF_FAILED

        self.out("cutoff_analysis", process.outputs.analysis)

        self.report(
            f"Cutoff campaign completed: PK {process.pk}."
        )

    def submit_relative_cutoff(self):
        submitted = self.inputs.cutoff_candidates.get_dict()

        try:
            base_parameters = next(iter(submitted.values()))
            count = (
                self.inputs.relative_candidate_count
                if "relative_candidate_count" in self.inputs
                else Int(5)
            )
            candidate_node = prepare_relative_candidates(
                base_parameters=Dict(dict=base_parameters),
                cutoff_analysis=self.ctx.cutoff_process.outputs.analysis,
                candidate_count=count,
            )
            candidates = candidate_node.get_dict()
            validate_candidate_payload(
                candidates,
                self.inputs.resources.get_dict(),
            )
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            self.report(f"Relative-cutoff preparation failed: {exc}")
            return self.exit_codes.ERROR_STAGE_PREPARATION_FAILED

        self.out("relative_candidates", candidate_node)

        self.report(
            f"Submitting {len(candidates)} relative-cutoff candidates."
        )

        process = self.submit(
            CP2KConvergenceWorkChain,
            code=self.inputs.code,
            structure=self.inputs.structure,
            candidates=candidate_node,
            resources=self.inputs.resources,
            parameter=Str("relative_cutoff"),
        )

        return ToContext(relative_process=process)

    def inspect_relative_cutoff(self):
        process = self.ctx.relative_process

        if "summary" in process.outputs:
            self.out("relative_summary", process.outputs.summary)

        if not process.is_finished_ok or "analysis" not in process.outputs:
            self.report(
                f"Relative-cutoff campaign failed: PK {process.pk}, "
                f"exit status {process.exit_status}."
            )
            return self.exit_codes.ERROR_RELATIVE_FAILED

        self.out("relative_analysis", process.outputs.analysis)

        self.report(
            "Both stages completed. Results remain provisional; "
            "full-grid convergence and joint verification are required."
        )
