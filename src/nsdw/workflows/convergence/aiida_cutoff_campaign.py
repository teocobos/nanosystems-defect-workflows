"""Submit a controlled SiO2 cutoff campaign to LUMI-C through AiiDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Dict, StructureData, load_code
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from nsdw.workflows.convergence.aiida_smoke import (
    build_smoke_test_inputs,
)
from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
    validate_candidate_payload,
)
from nsdw.workflows.convergence.recipes import (
    build_standard_cp2k_cutoff_study,
)


VALIDATION_CANDIDATES = 5


def build_cutoff_candidates(structure_path: Path):
    """Build a subset of the standard NSDW cutoff study."""

    structure, base_parameters = build_smoke_test_inputs(
        structure_path
    )

    study = build_standard_cp2k_cutoff_study()

    candidates = {}

    for candidate in study.candidates[:VALIDATION_CANDIDATES]:
        cutoff = candidate.value.value

        # Copy the nested input to avoid modifying other candidates.
        from copy import deepcopy

        parameters = deepcopy(base_parameters)

        parameters["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = cutoff

        candidates[candidate.label] = parameters

    return structure, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--structure",
        type=Path,
        default=Path(
            "tests/data/sio2/alpha_quartz_cod1526860.cif"
        ),
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit real CP2K calculations to LUMI-C.",
    )

    args = parser.parse_args()

    structure, candidates = build_cutoff_candidates(
        args.structure
    )

    resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 600,
        "queue_name": "small",
        "account": "project_465003407",
    }

    validate_candidate_payload(candidates, resources)

    print("Material: alpha-quartz SiO2")
    print(f"Atoms: {len(structure)}")
    print("Target: LUMI-C")
    print("Functional: PBE")
    print("Basis: TZV2P-MOLOPT-GTH")
    print("Relative cutoff: 30 Ry")
    print("Sampling: Gamma-only")
    print("Cutoff candidates:")

    for name in candidates:
        print(f"  {name}")

    print(f"Number of CP2K calculations: {len(candidates)}")

    if not args.submit:
        print("DRY RUN ONLY - no calculations submitted.")
        return

    load_profile("nsdw-dev")

    builder = CP2KConvergenceWorkChain.get_builder()

    builder.code = load_code("cp2k-lumi-c@lumi-c")
    builder.structure = StructureData(
        ase=AseAtomsAdaptor.get_atoms(structure)
    )
    builder.candidates = Dict(dict=candidates)
    builder.resources = Dict(dict=resources)

    node = submit(builder)

    print(f"Submitted WorkChain PK: {node.pk}")
    print(f"UUID: {node.uuid}")
    print(f"Monitor: verdi process show {node.pk}")


if __name__ == "__main__":
    main()
