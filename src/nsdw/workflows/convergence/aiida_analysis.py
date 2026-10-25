"""Analyse AiiDA CP2K cutoff campaigns using NSDW scientific analysers."""

from __future__ import annotations

from hashlib import sha256
import json
import math

from nsdw.workflows.convergence.analyser import (
    analyse_convergence_energy,
    analyse_convergence_against_reference,
    analyse_convergence_tail_stability,
)
from nsdw.workflows.convergence.models import ConvergenceObservation
from nsdw.workflows.convergence.recipes import (
    build_standard_cp2k_cutoff_study,
    build_standard_cp2k_relative_cutoff_study,
)
from nsdw.workflows.convergence.reporting import (
    build_convergence_report,
)

HARTREE_TO_EV = 27.211386245988


def analyse_aiida_convergence_campaign(
    parent,
    *,
    parameter="cutoff",
):
    """Analyse a completed AiiDA cutoff or relative-cutoff campaign.

    Returns an NSDW ConvergenceReport. A selection from a subset of
    the standard cutoff grid is provisional, not production-certified.
    """
    if not parent.is_finished_ok:
        raise ValueError("The AiiDA parent WorkChain must finish successfully.")

    if "summary" not in parent.outputs:
        raise ValueError("The parent WorkChain has no summary output.")

    results = parent.outputs.summary.get_dict()
    submitted = parent.inputs.candidates.get_dict()

    study_builders = {
        "cutoff": (
            build_standard_cp2k_cutoff_study,
            "CUTOFF",
        ),
        "relative_cutoff": (
            build_standard_cp2k_relative_cutoff_study,
            "REL_CUTOFF",
        ),
    }

    if parameter not in study_builders:
        raise ValueError(
            f"Unsupported convergence parameter: {parameter!r}"
        )

    study_builder, cp2k_key = study_builders[parameter]
    standard = study_builder()
    labels = list(submitted)

    expected = [candidate.label for candidate in standard.candidates]

    if not labels or labels != expected[:len(labels)]:
        raise ValueError(
            f"Submitted {parameter} values must be an ordered prefix "
            "of the standard NSDW recipe."
        )

    if len(labels) < 2:
        raise ValueError("At least two cutoff candidates are required.")

    study = standard.model_copy(
        update={"candidates": standard.candidates[:len(labels)]}
    )

    atoms = parent.inputs.structure.get_ase()

    structure_hash = sha256(
        json.dumps(
            {
                "symbols": atoms.get_chemical_symbols(),
                "positions": atoms.positions.tolist(),
                "cell": atoms.cell.array.tolist(),
                "pbc": atoms.pbc.tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    observations = []

    # Remove the studied parameter from each candidate's CP2K input.
    # All remaining parameters must be identical.
    from copy import deepcopy

    fixed_parameters = None

    for name in labels:
        candidate_parameters = deepcopy(submitted[name])

        try:
            candidate_parameters["FORCE_EVAL"]["DFT"]["MGRID"].pop(
                cp2k_key
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Missing {cp2k_key} for candidate {name}."
            ) from exc

        if fixed_parameters is None:
            fixed_parameters = candidate_parameters
        elif candidate_parameters != fixed_parameters:
            raise ValueError(
                "Fixed CP2K parameters differ between candidates."
            )

    if set(results) != set(submitted):
        raise ValueError("Summary candidates do not match submitted candidates.")

    for candidate in study.candidates:
        entry = results[candidate.label]
        parameters = submitted[candidate.label]

        if not entry.get("finished_ok") or entry.get("exit_status") != 0:
            raise ValueError(f"Candidate {candidate.label} did not succeed.")

        actual_value = parameters["FORCE_EVAL"]["DFT"]["MGRID"][cp2k_key]

        if actual_value != candidate.value.value:
            raise ValueError(
                f"{parameter.replace('_', ' ').capitalize()} mismatch "
                f"for candidate {candidate.label}."
            )

        energy = entry.get("energy_hartree")

        if (
            type(energy) not in (int, float)
            or not math.isfinite(energy)
        ):
            raise ValueError(
                f"Candidate {candidate.label} has an invalid energy."
            )

        observations.append(
            ConvergenceObservation(
                candidate=candidate,
                total_energy_ev=energy * HARTREE_TO_EV,
                n_atoms=len(atoms),
                structure_hash=structure_hash,
            )
        )

    adjacent = analyse_convergence_energy(study, observations)
    reference = analyse_convergence_against_reference(study, observations)
    tail = analyse_convergence_tail_stability(study, observations)

    return build_convergence_report(
        study=study,
        observations=observations,
        adjacent_analysis=adjacent,
        reference_analysis=reference,
        tail_analysis=tail,
    )



def analyse_aiida_cutoff_campaign(parent):
    """Backward-compatible cutoff campaign analyser."""
    return analyse_aiida_convergence_campaign(
        parent,
        parameter="cutoff",
    )
