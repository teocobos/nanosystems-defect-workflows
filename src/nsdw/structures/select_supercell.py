"""Unified supercell selection across supported engines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pymatgen.core import Structure

from nsdw.structures.doped_supercell import select_doped_supercell
from nsdw.structures.supercell import (
    get_minimum_lattice_translation,
    search_supercells,
)
from nsdw.structures.supercell_selection import (
    SelectedSupercell,
    SupercellEngine,
    SupercellSelectionError,
    validate_supercell_matrix,
)



def select_supercell(
    structure: Structure,
    *,
    engine: SupercellEngine = "doped",
    matrix: Sequence[Sequence[int]] | None = None,
    min_atoms: int = 50,
    max_atoms: int = 250,
    min_image_distance: float = 10.0,
    max_scale: int = 4,
    image_range: int = 2,
    ideal_threshold: float = 0.0,
) -> SelectedSupercell:
    """Select a supercell using doped, manual, or native methods."""

    if engine not in ("doped", "manual", "native"):
        raise SupercellSelectionError(
            f"Unsupported supercell engine: {engine}"
        )

    if len(structure) < 1:
        raise SupercellSelectionError(
            "Input structure contains no atomic sites."
        )

    if image_range < 1:
        raise SupercellSelectionError(
            "image_range must be at least 1."
        )

    # ---------------------------------------------------------
    # Engine 1: doped (default)
    # ---------------------------------------------------------
    if engine == "doped":
        if matrix is not None:
            raise SupercellSelectionError(
                "A manual matrix cannot be supplied "
                "with engine='doped'."
            )

        return select_doped_supercell(
            structure,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_image_distance=min_image_distance,
            ideal_threshold=ideal_threshold,
        )

    # ---------------------------------------------------------
    # Engine 2: manual transformation matrix
    # ---------------------------------------------------------
    if engine == "manual":
        if matrix is None:
            raise SupercellSelectionError(
                "engine='manual' requires a transformation matrix."
            )

        # Validate shape, integer entries and exact determinant.
        validated_matrix = validate_supercell_matrix(matrix)

        integer_matrix = np.asarray(
            validated_matrix,
            dtype=int,
        )

        selected_structure = structure.copy()
        selected_structure.make_supercell(integer_matrix)

        distance = get_minimum_lattice_translation(
            selected_structure,
            image_range=image_range,
        )

        return SelectedSupercell(
            engine="manual",
            transformation_matrix=validated_matrix,
            structure=selected_structure,
            primitive_num_atoms=len(structure),
            num_atoms=len(selected_structure),
            minimum_image_distance_angstrom=distance,
        )

    # ---------------------------------------------------------
    # Engine 3: native diagonal search
    # ---------------------------------------------------------
    if matrix is not None:
        raise SupercellSelectionError(
            "A manual matrix cannot be supplied "
            "with engine='native'."
        )

    search = search_supercells(
        structure,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        min_image_distance=min_image_distance,
        max_scale=max_scale,
        image_range=image_range,
    )

    candidate = search.selected_candidate

    if candidate is None:
        raise SupercellSelectionError(
            "Native search found no supercell satisfying "
            "the specified constraints."
        )

    scaling = candidate.scaling

    selected_structure = structure.copy()
    selected_structure.make_supercell(scaling)

    transformation_matrix = (
        (scaling[0], 0, 0),
        (0, scaling[1], 0),
        (0, 0, scaling[2]),
    )

    return SelectedSupercell(
        engine="native",
        transformation_matrix=transformation_matrix,
        structure=selected_structure,
        primitive_num_atoms=len(structure),
        num_atoms=len(selected_structure),
        minimum_image_distance_angstrom=(
            candidate.minimum_image_distance_angstrom
        ),
    )
