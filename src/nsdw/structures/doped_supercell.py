"""Doped-backed supercell selection for NSDW."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure

from nsdw.structures.supercell import get_minimum_lattice_translation
from nsdw.structures.supercell_selection import (
    SelectedSupercell,
    SupercellSelectionError,
)


def select_doped_supercell(
    structure: Structure,
    *,
    min_atoms: int = 50,
    max_atoms: int = 250,
    min_image_distance: float = 10.0,
    ideal_threshold: float = 0.0,
) -> SelectedSupercell:
    """Select a supercell using doped's non-diagonal optimisation."""

    if not 1 <= min_atoms <= max_atoms:
        raise SupercellSelectionError(
            "Require 1 <= min_atoms <= max_atoms."
        )

    if not np.isfinite(min_image_distance) or min_image_distance <= 0:
        raise SupercellSelectionError(
            "Minimum image distance must be positive."
        )

    if not np.isfinite(ideal_threshold) or ideal_threshold < 0:
        raise SupercellSelectionError(
            "Ideal threshold must be non-negative."
        )

    try:
        from doped.generation import get_ideal_supercell_matrix
    except ImportError as exc:
        raise ImportError(
            "doped is required for this supercell engine. "
            "Install NSDW with: pip install -e '.[doped]'"
        ) from exc

    matrix = get_ideal_supercell_matrix(
        structure,
        min_image_distance=min_image_distance,
        min_atoms=min_atoms,
        force_diagonal=False,
        ideal_threshold=ideal_threshold,
    )

    if matrix is None:
        raise SupercellSelectionError(
            "doped could not identify a suitable supercell."
        )

    matrix = np.asarray(matrix)

    if matrix.shape != (3, 3):
        raise SupercellSelectionError(
            "doped returned an invalid transformation matrix."
        )

    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix, np.rint(matrix), atol=1e-8, rtol=0
    ):
        raise SupercellSelectionError(
            "doped returned a non-integer transformation matrix."
        )

    matrix = np.rint(matrix).astype(int)

    determinant = int(round(np.linalg.det(matrix)))

    if determinant <= 0:
        raise SupercellSelectionError(
            "doped returned a transformation with a non-positive determinant."
        )

    num_atoms = len(structure) * determinant

    if num_atoms > max_atoms:
        raise SupercellSelectionError(
            f"doped selected {num_atoms} atoms, exceeding "
            f"max_atoms={max_atoms}. No supercell within the "
            "specified atom budget was established."
        )

    supercell = structure.copy()
    supercell.make_supercell(matrix)

    distance = get_minimum_lattice_translation(
        supercell,
        image_range=2,
    )

    if distance + 1e-6 < min_image_distance:
        raise SupercellSelectionError(
            f"Selected supercell has minimum-image distance "
            f"{distance:.3f} Å, below the requested "
            f"{min_image_distance:.3f} Å."
        )

    transformation_matrix = tuple(
        tuple(int(value) for value in row)
        for row in matrix
    )

    return SelectedSupercell(
        engine="doped",
        transformation_matrix=transformation_matrix,
        structure=supercell,
        primitive_num_atoms=len(structure),
        num_atoms=len(supercell),
        minimum_image_distance_angstrom=distance,
    )
