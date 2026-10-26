"""Persistent AiiDA orchestration for CP2K convergence candidates."""

from __future__ import annotations

import math

from aiida.engine import calcfunction

from aiida.engine import ToContext, WorkChain
from aiida.orm import Bool, Code, Dict, Str, StructureData
from aiida.plugins import CalculationFactory
from aiida.common.extendeddicts import AttributeDict
from nsdw.workflows.convergence.aiida_analysis import (
    analyse_aiida_convergence_campaign,
)


Cp2kCalculation = CalculationFactory("cp2k")


def validate_candidate_payload(candidates, resources):
    """Validate inputs before submitting any remote calculations."""

    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("At least one CP2K candidate is required.")

    for name, parameters in candidates.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Candidate names must be nonempty strings.")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(
                f"Candidate {name!r} requires CP2K parameters."
            )
        if "FORCE_EVAL" not in parameters:
            raise ValueError(
                f"Candidate {name!r} is missing FORCE_EVAL."
            )

    if not isinstance(resources, dict):
        raise ValueError("Resources must be a dictionary.")

    for field in (
        "num_machines",
        "num_mpiprocs_per_machine",
        "max_wallclock_seconds",
    ):
        value = resources.get(field)
        if type(value) is not int or value < 1:
            raise ValueError(
                f"{field} must be a positive integer."
            )

    return True




def valid_cp2k_energy(energy, units):
    """Require a finite CP2K total energy in Hartree."""
    return (
        type(energy) in (int, float)
        and math.isfinite(energy)
        and units == "a.u."
    )


def assess_candidate_result(
    *,
    pk,
    uuid,
    finished_ok,
    exit_status,
    output_parameters=None,
):
    """Assess a CP2K candidate without accessing AiiDA storage."""
    entry = {
        "pk": pk,
        "uuid": str(uuid),
        "finished_ok": bool(finished_ok),
        "exit_status": exit_status,
    }

    if not finished_ok:
        return entry, False

    if output_parameters is None:
        entry["missing_output_parameters"] = True
        return entry, False

    entry["output_parameters"] = output_parameters

    energy = output_parameters.get("energy")
    units = output_parameters.get("energy_units")

    if not valid_cp2k_energy(energy, units):
        entry["invalid_energy"] = True
        return entry, False

    entry["energy_hartree"] = float(energy)
    return entry, True


@calcfunction
def build_convergence_summary(entries: Dict, **output_nodes: Dict) -> Dict:
    """Create a provenance-tracked summary from CP2K parser outputs.

    The named output nodes establish explicit AiiDA provenance links.
    """
    return Dict(dict=entries.get_dict())


def validate_parser_summary_consistency(
    candidates, summary, output_nodes
):
    """Reject missing, inconsistent or invalid CP2K parser evidence."""
    submitted = candidates.get_dict()
    entries = summary.get_dict()

    expected_keys = {
        f"candidate_{index:04d}"
        for index in range(len(submitted))
    }

    if set(output_nodes) != expected_keys:
        raise ValueError(
            "CP2K parser nodes do not match the submitted candidates."
        )

    if set(entries) != set(submitted):
        raise ValueError(
            "Summary candidates do not match submitted candidates."
        )

    for index, name in enumerate(submitted):
        parser = output_nodes[f"candidate_{index:04d}"].get_dict()
        entry = entries[name]

        parser_energy = parser.get("energy")
        parser_units = parser.get("energy_units")
        summary_energy = entry.get("energy_hartree")

        if not valid_cp2k_energy(parser_energy, parser_units):
            raise ValueError(
                f"Invalid parser energy for {name}."
            )

        if (
            type(summary_energy) not in (int, float)
            or not math.isfinite(summary_energy)
            or not math.isclose(
                parser_energy,
                summary_energy,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ):
            raise ValueError(
                f"Parser and summary energy mismatch for {name}."
            )



@calcfunction
def build_convergence_analysis(
    structure: StructureData,
    candidates: Dict,
    summary: Dict,
    parameter: Str | None = None,
    **output_nodes: Dict,
) -> Dict:
    """Build an NSDW report with explicit AiiDA input provenance."""

    class AnalysisParent:
        is_finished_ok = True

        def __init__(self):
            self.inputs = type("Inputs", (), {
                "structure": structure,
                "candidates": candidates,
            })()
            self.outputs = AttributeDict(
                {"summary": summary}
            )

    validate_parser_summary_consistency(
        candidates, summary, output_nodes
    )

    parameter_name = parameter.value if parameter is not None else "cutoff"
    report = analyse_aiida_convergence_campaign(
        AnalysisParent(),
        parameter=parameter_name,
    )

    standard_count = 10
    candidate_count = len(candidates.get_dict())

    return Dict(dict={
        "report": report.to_dict(),
        "validation_scope": (
            "full_standard_grid"
            if candidate_count == standard_count
            else "standard_prefix"
        ),
        "production_certified": False,
        "provisional": True,
    })


class CP2KConvergenceWorkChain(WorkChain):
    """Submit CP2K candidates and preserve their process outcomes."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("candidates", valid_type=Dict)
        spec.input(
            "scientific_specification",
            valid_type=Dict,
            required=False,
        )
        spec.input("resources", valid_type=Dict)
        spec.input(
            "parameter",
            valid_type=Str,
            required=False,
        )
        spec.input(
            "retry_only",
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
        )
        spec.outline(
            cls.submit_candidates,
            cls.inspect_candidates,
        )

        spec.output("summary", valid_type=Dict)
        spec.output("analysis", valid_type=Dict, required=False)

        spec.exit_code(
            400,
            "ERROR_CHILD_FAILED",
            message="One or more CP2K candidates failed.",
        )
        spec.exit_code(
            401,
            "ERROR_INVALID_INPUT",
            message="Invalid CP2K candidate or resource configuration.",
        )
        spec.exit_code(
            402,
            "ERROR_ANALYSIS_FAILED",
            message="CP2K scientific convergence analysis failed.",
        )

    def submit_candidates(self):
        candidates = self.inputs.candidates.get_dict()
        resources = self.inputs.resources.get_dict()

        try:
            validate_candidate_payload(candidates, resources)

            if "scientific_specification" in self.inputs:
                from nsdw.workflows.convergence.aiida_campaign_specification import (
                    _validate_payload,
                    specification_digest,
                    CampaignSpecification,
                )
                from nsdw.workflows.convergence.aiida_specification_binding import (
                    validate_specification_binding,
                    validate_cp2k_cutoff_parameters,
                )

                binding_payload = (
                    self.inputs.scientific_specification.get_dict()
                )

                if set(binding_payload) != {
                    "specification",
                    "specification_sha256",
                    "candidate_cutoffs",
                }:
                    raise ValueError(
                        "Unexpected scientific specification fields."
                    )

                specification_payload = binding_payload[
                    "specification"
                ]
                name, cutoffs = _validate_payload(
                    specification_payload
                )
                digest = specification_digest(
                    specification_payload
                )

                if binding_payload["specification_sha256"] != digest:
                    raise ValueError(
                        "Scientific specification digest mismatch."
                    )

                specification = CampaignSpecification(
                    campaign_name=name,
                    expected_cutoffs=cutoffs,
                    sha256=digest,
                )

                candidate_cutoffs = binding_payload[
                    "candidate_cutoffs"
                ]

                validate_specification_binding(
                    specification,
                    candidates,
                    candidate_cutoffs,
                )
                validate_cp2k_cutoff_parameters(
                    candidates,
                    candidate_cutoffs,
                )

        except (ValueError, TypeError, KeyError) as exc:
            self.report(f"Invalid convergence input: {exc}")
            return self.exit_codes.ERROR_INVALID_INPUT

        running = {}
        self.ctx.candidate_names = {}

        for index, (name, parameters) in enumerate(
            candidates.items()
        ):
            internal_id = f"candidate_{index:04d}"

            builder = Cp2kCalculation.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.inputs.structure
            builder.parameters = Dict(dict=parameters)

            builder.metadata.label = name
            builder.metadata.options.resources = {
                "num_machines": resources["num_machines"],
                "num_mpiprocs_per_machine": (
                    resources["num_mpiprocs_per_machine"]
                ),
            }
            builder.metadata.options.max_wallclock_seconds = (
                resources["max_wallclock_seconds"]
            )
            builder.metadata.options.withmpi = True

            for key in (
                "queue_name",
                "account",
                "environment_variables",
                "parser_name",
            ):
                if resources.get(key) is not None:
                    builder.metadata.options[key] = resources[key]

            self.ctx.candidate_names[internal_id] = name
            running[internal_id] = self.submit(builder)

        return ToContext(**running)

    def inspect_candidates(self):
        entries = {}
        provenance_inputs = {}
        failed = False

        for internal_id, name in self.ctx.candidate_names.items():
            process = self.ctx[internal_id]

            output_node = (
                process.outputs.output_parameters
                if "output_parameters" in process.outputs
                else None
            )

            entry, valid = assess_candidate_result(
                pk=process.pk,
                uuid=process.uuid,
                finished_ok=process.is_finished_ok,
                exit_status=process.exit_status,
                output_parameters=(
                    output_node.get_dict()
                    if output_node is not None
                    else None
                ),
            )

            entry["imported"] = (
                process.base.attributes.get(
                    process.IMMIGRATED_KEY,
                    False,
                ) is True
            )

            if not valid:
                failed = True

            if output_node is not None:
                provenance_inputs[internal_id] = output_node

            entries[name] = entry

        summary = build_convergence_summary(
            entries=Dict(dict=entries),
            **provenance_inputs,
        )
        self.out("summary", summary)

        if failed:
            return self.exit_codes.ERROR_CHILD_FAILED

        if self.inputs.retry_only.value:
            self.report(
                "Retry-only execution completed. "
                "Scientific convergence analysis deferred."
            )
            return

        if any(
            entry.get("imported", False)
            for entry in entries.values()
        ):
            self.report(
                "Imported CP2K results cannot undergo "
                "scientific convergence analysis."
            )
            return self.exit_codes.ERROR_ANALYSIS_FAILED

        try:
            analysis = build_convergence_analysis(
                structure=self.inputs.structure,
                candidates=self.inputs.candidates,
                summary=summary,
                parameter=(
                    self.inputs.parameter
                    if "parameter" in self.inputs
                    else Str("cutoff")
                ),
                **provenance_inputs,
            )
        except (ValueError, KeyError, TypeError) as exc:
            self.report(f"Scientific analysis failed: {exc}")
            return self.exit_codes.ERROR_ANALYSIS_FAILED

        self.out("analysis", analysis)
