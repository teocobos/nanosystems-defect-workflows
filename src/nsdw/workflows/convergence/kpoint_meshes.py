"""Automatic reciprocal-space k-point mesh generation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pymatgen.core import Structure


def mesh_from_spacing(
    structure: Structure,
    spacing_inv_angstrom: float,
) -> tuple[int, int, int]:
    """Generate an anisotropic mesh from reciprocal-vector lengths.

    Spacing is measured in inverse angstroms and uses the conventional
    reciprocal lattice (including the 2*pi factor).
    """
    if not math.isfinite(spacing_inv_angstrom) or spacing_inv_angstrom <= 0:
        raise ValueError("K-point spacing must be positive and finite.")

    lengths = structure.lattice.reciprocal_lattice.abc

    return tuple(
        max(1, math.ceil(length / spacing_inv_angstrom))
        for length in lengths
    )


def generate_mesh_sequence(
    structure: Structure,
    spacings_inv_angstrom: Sequence[float],
) -> tuple[tuple[int, int, int], ...]:
    """Generate unique meshes in strictly increasing cost order."""
    if len(spacings_inv_angstrom) < 2:
        raise ValueError("At least two k-point spacings are required.")

    meshes = [
        mesh_from_spacing(structure, spacing)
        for spacing in spacings_inv_angstrom
    ]

    unique_meshes = list(dict.fromkeys(meshes))

    if len(unique_meshes) < 2:
        raise ValueError(
            "K-point spacings produced fewer than two distinct meshes."
        )

    costs = [
        math.prod(mesh)
        for mesh in unique_meshes
    ]

    if any(
        later <= earlier
        for earlier, later in zip(costs, costs[1:])
    ):
        raise ValueError(
            "K-point spacings must produce strictly increasing mesh costs."
        )

    return tuple(unique_meshes)


def generate_automatic_mesh_sequence(
    structure: Structure,
    *,
    number_of_meshes: int = 4,
    initial_spacing_inv_angstrom: float = 1.5,
    refinement_factor: float = 0.7,
    max_mesh_points: int = 1000,
) -> tuple[tuple[int, int, int], ...]:
    """Plan increasingly dense anisotropic k-point meshes.

    This generates candidate meshes only. It does not establish
    energy convergence or run electronic-structure calculations.
    """
    if number_of_meshes < 2:
        raise ValueError("At least two meshes are required.")

    if (
        not math.isfinite(initial_spacing_inv_angstrom)
        or initial_spacing_inv_angstrom <= 0
    ):
        raise ValueError("Initial spacing must be positive and finite.")

    if (
        not math.isfinite(refinement_factor)
        or not 0 < refinement_factor < 1
    ):
        raise ValueError("Refinement factor must be between zero and one.")

    if max_mesh_points < 1:
        raise ValueError("Maximum mesh points must be positive.")

    meshes: list[tuple[int, int, int]] = []
    spacing = initial_spacing_inv_angstrom

    for _ in range(100):
        mesh = mesh_from_spacing(structure, spacing)

        if math.prod(mesh) > max_mesh_points:
            raise ValueError(
                "Automatic mesh generation exceeded the "
                "maximum allowed mesh points before producing "
                "the requested number of candidates."
            )

        if mesh not in meshes:
            meshes.append(mesh)

            if len(meshes) == number_of_meshes:
                return tuple(meshes)

        spacing *= refinement_factor

    raise ValueError(
        "Unable to generate enough distinct k-point meshes "
        "within the refinement limit."
    )
