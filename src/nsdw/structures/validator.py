from pymatgen.core import Structure

from nsdw.config.models import LatticeSummary, StructureSummary


def summarise_structure(structure: Structure) -> StructureSummary:
    """
    Generate a validated summary of a periodic structure.
    """

    lattice = structure.lattice

    lattice_summary = LatticeSummary(
        a=lattice.a,
        b=lattice.b,
        c=lattice.c,
        alpha=lattice.alpha,
        beta=lattice.beta,
        gamma=lattice.gamma,
        volume=lattice.volume,
    )

    return StructureSummary(
        formula=structure.composition.formula,
        reduced_formula=structure.composition.reduced_formula,
        num_sites=len(structure),
        density=structure.density,
        lattice=lattice_summary,
    )
