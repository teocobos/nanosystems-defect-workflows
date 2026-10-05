"""AiiDA input adaptation for NSDW CP2K configurations."""

from __future__ import annotations

from typing import Any

from pymatgen.core import Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
    CP2KXCFunctional,
)


class CP2KAiiDAAdapterError(RuntimeError):
    """Raised when an NSDW CP2K config cannot be adapted for AiiDA."""


def _build_aiida_xc(
    functional: CP2KXCFunctional,
) -> dict[str, Any]:
    """Build the aiida-cp2k XC section."""

    if functional == CP2KXCFunctional.PBE:
        return {
            "XC_FUNCTIONAL": {
                "_": "PBE",
            },
        }

    if functional == CP2KXCFunctional.PBESOL:
        return {
            "XC_FUNCTIONAL": {
                "GGA_X_PBE_SOL": {},
                "GGA_C_PBE_SOL": {},
            },
        }

    raise CP2KAiiDAAdapterError(
        f"Unsupported XC functional: {functional}"
    )


def _build_aiida_cell(
    structure: Structure,
) -> dict[str, str]:
    """Build CP2K CELL vectors from a pymatgen structure."""

    matrix = structure.lattice.matrix

    if matrix.shape != (3, 3):
        raise CP2KAiiDAAdapterError(
            "Structure lattice must contain three cell vectors."
        )

    def vector(values: Any) -> str:
        return " ".join(
            f"{float(value):.12f}"
            for value in values
        )

    return {
        "A": vector(matrix[0]),
        "B": vector(matrix[1]),
        "C": vector(matrix[2]),
        "PERIODIC": "XYZ",
    }


def _validate_kind_coverage(
    *,
    config: CP2KInputConfig,
    structure: Structure,
) -> None:
    """Validate that configured KIND entries match structure species."""

    basis_potential = config.basis_potential

    if basis_potential is None:
        raise CP2KAiiDAAdapterError(
            "A basis/potential configuration is required."
        )

    structure_elements = {
        element.symbol
        for element in structure.composition.elements
    }

    configured_elements = {
        kind.element
        for kind in basis_potential.kinds
    }

    missing_elements = (
        structure_elements - configured_elements
    )

    extra_elements = (
        configured_elements - structure_elements
    )

    if missing_elements:
        missing = ", ".join(
            sorted(missing_elements)
        )

        raise CP2KAiiDAAdapterError(
            "Missing CP2K KIND definitions for structure "
            f"elements: {missing}"
        )

    if extra_elements:
        extra = ", ".join(
            sorted(extra_elements)
        )

        raise CP2KAiiDAAdapterError(
            "CP2K KIND definitions contain elements not "
            f"present in the structure: {extra}"
        )


def build_aiida_cp2k_parameters(
    *,
    config: CP2KInputConfig,
    structure: Structure,
) -> dict[str, Any]:
    """Convert an NSDW CP2K config into aiida-cp2k parameters.

    Project naming and atomic coordinates are intentionally omitted.
    ``aiida-cp2k`` manages its own project name and writes coordinates
    from the supplied AiiDA ``StructureData`` node.
    """

    basis_potential = config.basis_potential

    if basis_potential is None:
        raise CP2KAiiDAAdapterError(
            "A basis/potential configuration is required."
        )

    if len(structure) == 0:
        raise CP2KAiiDAAdapterError(
            "Cannot generate AiiDA CP2K input for an empty structure."
        )

    _validate_kind_coverage(
        config=config,
        structure=structure,
    )

    scf = config.scf

    dft: dict[str, Any] = {
        "BASIS_SET_FILE_NAME": (
            basis_potential.basis_set_file
        ),
        "POTENTIAL_FILE_NAME": (
            basis_potential.potential_file
        ),
        "CHARGE": config.charge,
        "MULTIPLICITY": config.multiplicity,
        "QS": {
            "METHOD": "GPW",
        },
        "MGRID": {
            "CUTOFF": config.cutoff_ry,
            "REL_CUTOFF": config.relative_cutoff_ry,
        },
        "SCF": {
            "SCF_GUESS": scf.scf_guess,
            "EPS_SCF": scf.eps_scf,
            "MAX_SCF": scf.max_scf,
            "OT": {
                "MINIMIZER": scf.ot_minimizer,
                "PRECONDITIONER": scf.ot_preconditioner,
                "ENERGY_GAP": scf.energy_gap,
            },
            "OUTER_SCF": {
                "MAX_SCF": scf.outer_scf_max,
            },
        },
        "XC": _build_aiida_xc(
            config.functional
        ),
    }

    if config.k_points is not None:
        kx, ky, kz = config.k_points

        if min(kx, ky, kz) <= 0:
            raise CP2KAiiDAAdapterError(
                "CP2K k-point values must all be positive."
            )

        dft["KPOINTS"] = {
            "SCHEME": (
                f"MONKHORST-PACK {kx} {ky} {kz}"
            ),
        }

    kinds = [
        {
            "_": kind.element,
            "ELEMENT": kind.element,
            "BASIS_SET": kind.basis_set,
            "POTENTIAL": kind.potential,
        }
        for kind in basis_potential.kinds
    ]

    return {
        "GLOBAL": {
            "RUN_TYPE": config.run_type,
            "PRINT_LEVEL": "MEDIUM",
        },
        "FORCE_EVAL": {
            "METHOD": "Quickstep",
            "DFT": dft,
            "SUBSYS": {
                "CELL": _build_aiida_cell(
                    structure
                ),
                "KIND": kinds,
            },
        },
    }
