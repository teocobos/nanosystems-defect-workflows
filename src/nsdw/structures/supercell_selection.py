"""Backend-independent supercell selection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pymatgen.core import Structure


SupercellEngine = Literal["doped", "manual", "native"]

SupercellMatrix = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]


class SupercellSelectionError(ValueError):
    """Raised when a selected supercell is invalid."""


def validate_supercell_matrix(matrix) -> SupercellMatrix:
    """Validate a 3x3 integer matrix with positive determinant."""

    try:
        rows = tuple(tuple(row) for row in matrix)
    except (TypeError, ValueError) as exc:
        raise SupercellSelectionError(
            "Supercell matrix must be a 3x3 sequence."
        ) from exc

    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise SupercellSelectionError(
            "Supercell matrix must be a 3x3 matrix."
        )

    if any(
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        for row in rows
        for value in row
    ):
        raise SupercellSelectionError(
            "Supercell matrix entries must be integers."
        )

    a, b, c = rows

    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )

    if determinant == 0:
        raise SupercellSelectionError(
            "Supercell transformation must be nonsingular."
        )

    if determinant < 0:
        raise SupercellSelectionError(
            "Supercell matrix must have a positive determinant."
        )

    return tuple(
        tuple(int(value) for value in row)
        for row in rows
    )


@dataclass(frozen=True)
class SelectedSupercell:
    """A validated supercell selected by any supported engine."""

    engine: SupercellEngine
    transformation_matrix: SupercellMatrix
    structure: Structure
    primitive_num_atoms: int
    num_atoms: int
    minimum_image_distance_angstrom: float

    def __post_init__(self) -> None:
        if self.engine not in ("doped", "manual", "native"):
            raise SupercellSelectionError(
                f"Unsupported supercell engine: {self.engine}"
            )

        matrix = validate_supercell_matrix(
            self.transformation_matrix
        )

        determinant = self._integer_determinant(matrix)

        if self.primitive_num_atoms < 1:
            raise SupercellSelectionError(
                "Primitive atom count must be positive."
            )

        expected_atoms = self.primitive_num_atoms * determinant

        if self.num_atoms != expected_atoms:
            raise SupercellSelectionError(
                "Supercell atom count does not match "
                "the transformation determinant."
            )

        if len(self.structure) != self.num_atoms:
            raise SupercellSelectionError(
                "Structure atom count does not match "
                "the declared supercell atom count."
            )

        if (
            not np.isfinite(self.minimum_image_distance_angstrom)
            or self.minimum_image_distance_angstrom <= 0
        ):
            raise SupercellSelectionError(
                "Minimum image distance must be positive."
            )

    @staticmethod
    def _integer_determinant(matrix) -> int:
        """Calculate the exact determinant of a 3x3 integer matrix."""
        a, b, c = matrix

        return (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )

    @property
    def determinant(self) -> int:
        """Number of input cells represented by the supercell."""
        matrix = validate_supercell_matrix(
            self.transformation_matrix
        )
        return self._integer_determinant(matrix)
