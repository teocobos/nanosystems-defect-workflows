"""Submit a small SiO2 CP2K smoke test through AiiDA on LUMI-C."""

from __future__ import annotations

from pathlib import Path

from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Dict, StructureData, load_code
from pymatgen.core import Structure

from nsdw.calculators.cp2k.aiida import build_aiida_cp2k_parameters
from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KInputConfig,
    CP2KKindConfig,
    CP2KSCFConfig,
)
from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)


CP2K_DATA = (
    "/users/cobosfre/EasyBuild/SW/LUMI-25.03/C/"
    "CP2K/2024.3-cpeGNU-25.03-CPU/data"
)


def build_smoke_test_inputs(structure_path: Path):
    """Build one scientifically explicit SiO2 test candidate."""

    structure = Structure.from_file(str(structure_path))

    if len(structure) != 9:
        raise ValueError(
            f"Expected nine-atom alpha-quartz SiO2, found {len(structure)} atoms."
        )

    if structure.composition.reduced_formula != "SiO2":
        raise ValueError("Expected SiO2 composition.")

    config = CP2KInputConfig(
        project_name="nsdw-sio2-lumi-smoke",
        functional="PBE",
        run_type="ENERGY",
        charge=0,
        multiplicity=1,
        cutoff_ry=400.0,
        relative_cutoff_ry=30.0,
        scf=CP2KSCFConfig(
            solver="OT",
            eps_scf=1.0e-6,
            max_scf=100,
        ),
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file=f"{CP2K_DATA}/BASIS_MOLOPT",
            potential_file=f"{CP2K_DATA}/GTH_POTENTIALS",
            kinds=(
                CP2KKindConfig(
                    element="Si",
                    basis_set="TZV2P-MOLOPT-GTH-q4",
                    potential="GTH-PBE-q4",
                ),
                CP2KKindConfig(
                    element="O",
                    basis_set="TZV2P-MOLOPT-GTH-q6",
                    potential="GTH-PBE-q6",
                ),
            ),
        ),
    )

    parameters = build_aiida_cp2k_parameters(
        config=config,
        structure=structure,
    )

    return structure, parameters


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("tests/data/sio2/alpha_quartz_cod1526860.cif"),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the AiiDA WorkChain.",
    )
    args = parser.parse_args()

    structure, parameters = build_smoke_test_inputs(args.structure)

    print(f"Structure: {len(structure)} atoms")
    print("Functional: PBE")
    print("Cutoff: 400 Ry")
    print("Relative cutoff: 30 Ry")
    print("Sampling: Gamma-only")
    print("Basis: TZV2P-MOLOPT-GTH")
    print("Remote target: LUMI-C")
    print("Resources: 1 node, 1 MPI rank, 600 seconds")

    if not args.submit:
        print("DRY RUN ONLY — no AiiDA process submitted.")
        return

    load_profile("nsdw-dev")

    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)

    code = load_code("cp2k-lumi-c@lumi-c")

    builder = CP2KConvergenceWorkChain.get_builder()
    builder.code = code
    builder.structure = StructureData(ase=atoms)
    builder.candidates = Dict(
        dict={"sio2_smoke": parameters}
    )
    builder.resources = Dict(
        dict={
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 600,
            "queue_name": "small",
            "account": "project_465003407",
        }
    )

    node = submit(builder)

    print(f"Submitted WorkChain PK: {node.pk}")
    print(f"UUID: {node.uuid}")
    print(f"Monitor: verdi process show {node.pk}")


if __name__ == "__main__":
    main()
