"""Candidate selection for provenance-preserving AiiDA recovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryPlan:
    """Partition a convergence campaign into reusable and retry candidates."""

    reusable: tuple[str, ...]
    retry: tuple[str, ...]


def build_recovery_plan(candidates: dict, summary: dict) -> RecoveryPlan:
    """Identify candidates requiring recovery without accepting partial results.

    A candidate is reusable only when its original calculation finished
    successfully and produced a finite, valid CP2K total energy.
    """
    from nsdw.workflows.convergence.aiida_workchain import (
        valid_cp2k_energy,
    )

    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Candidates must be a nonempty dictionary.")

    if not isinstance(summary, dict):
        raise ValueError("Summary must be a dictionary.")

    if set(candidates) != set(summary):
        raise ValueError("Candidate and summary labels do not match.")

    reusable = []
    retry = []

    for name in candidates:
        entry = summary[name]

        if not isinstance(entry, dict):
            raise ValueError(f"Invalid summary entry for {name}.")

        output = entry.get("output_parameters")

        valid = (
            entry.get("finished_ok") is True
            and entry.get("exit_status") == 0
            and isinstance(output, dict)
            and valid_cp2k_energy(
                output.get("energy"),
                output.get("energy_units"),
            )
            and entry.get("energy_hartree") == output.get("energy")
        )

        if valid:
            reusable.append(name)
        else:
            retry.append(name)

    return RecoveryPlan(
        reusable=tuple(reusable),
        retry=tuple(retry),
    )


def validate_recovery_provenance(campaign):
    """Verify original candidate calculations against their AiiDA inputs.

    Returns a RecoveryPlan only after all original candidate identities,
    input parameters, and reusable parser results have been checked.
    """
    from aiida.common.links import LinkType
    from aiida.orm import load_node
    from nsdw.workflows.convergence.aiida_workchain import (
        valid_cp2k_energy,
    )

    if "summary" not in campaign.outputs:
        raise ValueError("Original campaign has no summary.")

    candidates = campaign.inputs.candidates.get_dict()
    summary = campaign.outputs.summary.get_dict()
    plan = build_recovery_plan(candidates, summary)

    for name in candidates:
        entry = summary[name]

        if type(entry.get("pk")) is not int:
            raise ValueError(f"Missing calculation PK for {name}.")

        node = load_node(entry["pk"])

        caller = node.base.links.get_incoming(
            link_type=LinkType.CALL_CALC
        ).one().node

        if caller.uuid != campaign.uuid:
            raise ValueError(
                f"Calculation {name} does not belong to the "
                "original campaign."
            )

        if node.label != name:
            raise ValueError(f"Calculation label mismatch for {name}.")

        if str(node.uuid) != str(entry.get("uuid")):
            raise ValueError(f"Calculation UUID mismatch for {name}.")

        if node.inputs.parameters.get_dict() != candidates[name]:
            raise ValueError(f"CP2K parameter mismatch for {name}.")

        if node.inputs.structure.uuid != campaign.inputs.structure.uuid:
            raise ValueError(f"Structure mismatch for {name}.")

        if node.inputs.code.uuid != campaign.inputs.code.uuid:
            raise ValueError(f"CP2K code mismatch for {name}.")

        if node.exit_status != entry.get("exit_status"):
            raise ValueError(f"Exit-status mismatch for {name}.")

        if node.is_finished_ok != entry.get("finished_ok"):
            raise ValueError(f"Process-state mismatch for {name}.")

        if name in plan.reusable:
            if "output_parameters" not in node.outputs:
                raise ValueError(f"Missing parser output for {name}.")

            parser = node.outputs.output_parameters.get_dict()

            if not valid_cp2k_energy(
                parser.get("energy"),
                parser.get("energy_units"),
            ):
                raise ValueError(f"Invalid parser energy for {name}.")

            if parser != entry.get("output_parameters"):
                raise ValueError(f"Parser mismatch for {name}.")

            if parser["energy"] != entry.get("energy_hartree"):
                raise ValueError(f"Energy mismatch for {name}.")

    return plan



def build_retry_candidates(
    candidates: dict,
    retry_labels: tuple[str, ...],
) -> dict:
    """Select retry candidates without changing CP2K parameters.

    Preserve original campaign ordering and reject unknown or
    duplicate labels.
    """
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Original candidates must be non-empty.")

    if not isinstance(retry_labels, tuple):
        raise TypeError("Retry labels must be a tuple.")

    if len(set(retry_labels)) != len(retry_labels):
        raise ValueError("Duplicate retry labels are not allowed.")

    unknown = set(retry_labels) - set(candidates)

    if unknown:
        raise ValueError(
            f"Unknown retry candidate labels: {sorted(unknown)}"
        )

    return {
        name: parameters
        for name, parameters in candidates.items()
        if name in retry_labels
    }



def build_retry_resources(
    original_resources: dict,
    retry_walltime: int,
) -> dict:
    """Preserve HPC resources while increasing the retry walltime."""
    if not isinstance(original_resources, dict):
        raise TypeError("Original resources must be a dictionary.")

    if type(retry_walltime) is not int or retry_walltime <= 0:
        raise ValueError("Retry walltime must be a positive integer.")

    original_walltime = original_resources.get(
        "max_wallclock_seconds"
    )

    if type(original_walltime) is not int or original_walltime <= 0:
        raise ValueError("Original walltime must be a positive integer.")

    if retry_walltime <= original_walltime:
        raise ValueError(
            "Retry walltime must exceed the original walltime."
        )

    required = (
        "num_machines",
        "num_mpiprocs_per_machine",
    )

    for key in required:
        value = original_resources.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid original resource: {key}")

    updated = original_resources.copy()
    updated["max_wallclock_seconds"] = retry_walltime

    return updated
