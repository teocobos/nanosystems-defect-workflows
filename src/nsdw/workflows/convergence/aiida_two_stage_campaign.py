"""Submit the provisional two-stage CP2K campaign to LUMI-C."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Dict, Int, StructureData, load_code
from pymatgen.io.ase import AseAtomsAdaptor

from nsdw.workflows.convergence.aiida_cutoff_campaign import (
    build_cutoff_candidates,
)
from nsdw.workflows.convergence.aiida_two_stage import (
    CP2KTwoStageConvergenceWorkChain,
)
from nsdw.workflows.convergence.aiida_workchain import (
    validate_candidate_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run provisional two-stage CP2K convergence on LUMI-C."
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path(
            "tests/data/sio2/alpha_quartz_cod1526860.cif"
        ),
    )
    parser.add_argument(
        "--relative-candidates",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit to AiiDA; otherwise perform a dry run.",
    )

    args = parser.parse_args()

    structure, candidates = build_cutoff_candidates(args.structure)

    resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 600,
        "queue_name": "small",
        "account": "project_465003407",
    }

    validate_candidate_payload(candidates, resources)

    if not 2 <= args.relative_candidates <= 10:
        parser.error("--relative-candidates must be between 2 and 10.")

    print("=== NSDW TWO-STAGE CONVERGENCE ===")
    print("Structure:", args.structure)
    print("Cutoff candidates:", ", ".join(candidates))
    print("Relative-cutoff candidates:", args.relative_candidates)
    print("Target: LUMI-C")
    print("Scientific status: PROVISIONAL")

    if not args.submit:
        print("DRY RUN: no AiiDA process submitted.")
        return

    load_profile("nsdw-dev")

    builder = CP2KTwoStageConvergenceWorkChain.get_builder()
    builder.code = load_code("cp2k-lumi-c@lumi-c")
    builder.structure = StructureData(
        ase=AseAtomsAdaptor.get_atoms(structure)
    )
    builder.cutoff_candidates = Dict(dict=candidates)
    builder.resources = Dict(dict=resources)
    builder.relative_candidate_count = Int(args.relative_candidates)

    node = submit(builder)

    print("Submitted parent PK:", node.pk)
    print("Parent UUID:", node.uuid)
    print("Monitor: verdi process show", node.pk)
    print("Report: verdi process report", node.pk)


if __name__ == "__main__":
    main()
