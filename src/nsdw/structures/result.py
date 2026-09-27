"""Build calculator-independent structural results."""

from __future__ import annotations

import hashlib
import json

from pymatgen.core import Structure

from nsdw.models.structure import StructureResult
from nsdw.structures.validator import summarise_structure


STRUCTURE_HASH_DECIMALS = 10


def calculate_structure_hash(
    structure: Structure,
    *,
    decimals: int = STRUCTURE_HASH_DECIMALS,
) -> str:
    """
    Calculate a deterministic SHA-256 hash for a periodic structure.

    The hash represents the computational geometry rather than the
    bytes of the source structure file. It includes:

    - lattice vectors,
    - ordered site species,
    - fractional coordinates.

    Numerical values are rounded before hashing so insignificant
    floating-point representation noise does not change the identity.

    Site order is intentionally preserved. Symmetry-equivalent or
    reordered representations are therefore not treated as identical.
    """

    if decimals < 0:
        raise ValueError(
            "Structure-hash decimal precision must be non-negative."
        )

    if not structure.is_ordered:
        raise ValueError(
            "Cannot calculate a deterministic structure hash for "
            "a disordered structure."
        )

    lattice = [
        [
            round(float(value), decimals)
            for value in vector
        ]
        for vector in structure.lattice.matrix
    ]

    sites = []

    for site in structure:
        sites.append(
            {
                "species": site.specie.symbol,
                "frac_coords": [
                    round(float(value), decimals)
                    for value in site.frac_coords
                ],
            }
        )

    payload = {
        "lattice": lattice,
        "sites": sites,
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def build_structure_result(
    structure: Structure,
) -> StructureResult:
    """
    Convert a pymatgen Structure into an NSDW StructureResult.

    The resulting model contains calculator-independent structural
    metadata suitable for persistent scientific results and workflow
    comparisons.
    """

    summary = summarise_structure(structure)

    return StructureResult(
        formula=summary.formula,
        n_atoms=summary.num_sites,
        periodic=True,
        structure_hash=calculate_structure_hash(structure),
        volume=float(summary.lattice.volume),
        density=float(summary.density),
    )
