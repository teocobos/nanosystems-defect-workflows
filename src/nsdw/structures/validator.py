import math

import numpy as np
from pymatgen.core import Structure

from nsdw.config.models import (
    LatticeSummary,
    StructureSummary,
    StructureValidationResult,
    ValidationCheck,
)


DEFAULT_MIN_DISTANCE = 0.5


def summarise_structure(structure: Structure) -> StructureSummary:
    """Generate a summary of a periodic structure."""

    lattice = structure.lattice

    return StructureSummary(
        formula=structure.composition.formula,
        reduced_formula=structure.composition.reduced_formula,
        num_sites=len(structure),
        density=structure.density,
        lattice=LatticeSummary(
            a=lattice.a,
            b=lattice.b,
            c=lattice.c,
            alpha=lattice.alpha,
            beta=lattice.beta,
            gamma=lattice.gamma,
            volume=lattice.volume,
        ),
    )


def get_minimum_distance(structure: Structure) -> float | None:
    """
    Return the shortest periodic distance between distinct sites.
    """

    if len(structure) < 2:
        return None

    distance_matrix = np.array(structure.distance_matrix, dtype=float)

    # Ignore the zero-distance diagonal.
    np.fill_diagonal(distance_matrix, np.inf)

    minimum = float(np.min(distance_matrix))

    if not math.isfinite(minimum):
        return None

    return minimum


def validate_structure(
    structure: Structure,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> StructureValidationResult:
    """
    Perform basic physical and crystallographic validation.

    This does not determine whether a structure is converged or
    scientifically appropriate for a particular DFT methodology.
    """

    checks: list[ValidationCheck] = []
    warnings: list[str] = []
    errors: list[str] = []

    # Lattice
    volume = float(structure.volume)

    lattice_ok = math.isfinite(volume) and volume > 0.0

    checks.append(
        ValidationCheck(
            name="Valid lattice",
            passed=lattice_ok,
            value=f"{volume:.6f} Å³",
        )
    )

    if not lattice_ok:
        errors.append("The structure has an invalid lattice or volume.")

    # Coordinates
    frac_coords = np.asarray(structure.frac_coords, dtype=float)
    coordinates_ok = bool(np.all(np.isfinite(frac_coords)))

    checks.append(
        ValidationCheck(
            name="Finite coordinates",
            passed=coordinates_ok,
        )
    )

    if not coordinates_ok:
        errors.append("One or more atomic coordinates are non-finite.")

    # Composition
    composition_ok = len(structure.composition.elements) > 0

    checks.append(
        ValidationCheck(
            name="Valid composition",
            passed=composition_ok,
            value=structure.composition.formula,
        )
    )

    if not composition_ok:
        errors.append("No valid chemical composition was detected.")

    # Ordered/disordered structure
    ordered = bool(structure.is_ordered)

    checks.append(
        ValidationCheck(
            name="Fully ordered sites",
            passed=ordered,
            value="ordered" if ordered else "disordered",
        )
    )

    if not ordered:
        warnings.append(
            "Structure contains partial occupancies or disordered sites. "
            "An ordered model may be required before DFT calculations."
        )

    # Minimum periodic interatomic distance
    minimum_distance = get_minimum_distance(structure)

    if minimum_distance is None:
        distance_ok = len(structure) == 1
        distance_value = "N/A"
    else:
        distance_ok = minimum_distance >= min_distance
        distance_value = f"{minimum_distance:.6f} Å"

    checks.append(
        ValidationCheck(
            name="Minimum interatomic distance",
            passed=distance_ok,
            value=distance_value,
            message=f"Threshold: {min_distance:.3f} Å",
        )
    )

    if not distance_ok:
        errors.append(
            f"Minimum interatomic distance is below the "
            f"{min_distance:.3f} Å validation threshold."
        )

    valid = len(errors) == 0

    return StructureValidationResult(
        valid=valid,
        minimum_distance=minimum_distance,
        ordered=ordered,
        checks=checks,
        warnings=warnings,
        errors=errors,
    )
