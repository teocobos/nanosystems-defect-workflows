"""CP2K input rendering from validated NSDW generation models."""

from __future__ import annotations

from pymatgen.core import Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KCoordinateMode,
    CP2KInputConfig,
    CP2KXCFunctional,
)


class CP2KGenerationError(RuntimeError):
    """Raised when a CP2K input cannot be generated."""


def _render_xc(
    functional: CP2KXCFunctional,
) -> list[str]:
    """Render the CP2K exchange-correlation section."""

    if functional == CP2KXCFunctional.PBE:
        return [
            "    &XC",
            "      &XC_FUNCTIONAL PBE",
            "      &END XC_FUNCTIONAL",
            "    &END XC",
        ]

    if functional == CP2KXCFunctional.PBESOL:
        return [
            "    &XC",
            "      &XC_FUNCTIONAL",
            "        &GGA_X_PBE_SOL",
            "        &END GGA_X_PBE_SOL",
            "        &GGA_C_PBE_SOL",
            "        &END GGA_C_PBE_SOL",
            "      &END XC_FUNCTIONAL",
            "    &END XC",
        ]

    raise CP2KGenerationError(
        f"Unsupported XC functional: {functional}"
    )


def _render_scf(
    config: CP2KInputConfig,
) -> list[str]:
    """Render the SCF section."""

    scf = config.scf

    if config.k_points is not None and scf.solver == "OT":
        raise CP2KGenerationError(
            "Explicit k-point sampling requires "
            "SCF solver DIAGONALIZATION; OT is unsupported."
        )

    lines = [
        "    &SCF",
        f"      SCF_GUESS {scf.scf_guess}",
        f"      EPS_SCF {scf.eps_scf:.8g}",
        f"      MAX_SCF {scf.max_scf}",
    ]

    if scf.solver == "OT":
        lines.extend([
            "      &OT",
            f"        MINIMIZER {scf.ot_minimizer}",
            f"        PRECONDITIONER {scf.ot_preconditioner}",
            f"        ENERGY_GAP {scf.energy_gap:.8g}",
            "      &END OT",
            "      &OUTER_SCF",
            f"        MAX_SCF {scf.outer_scf_max}",
            "      &END OUTER_SCF",
        ])
    elif scf.solver == "DIAGONALIZATION":
        lines.extend([
            "      &DIAGONALIZATION",
            "        ALGORITHM STANDARD",
            "      &END DIAGONALIZATION",
        ])
    else:
        raise CP2KGenerationError(
            f"Unsupported SCF solver: {scf.solver}"
        )

    lines.append("    &END SCF")
    return lines


def _render_kpoints(
    config: CP2KInputConfig,
) -> list[str]:
    """Render k-points when explicitly configured."""

    if config.k_points is None:
        return []

    kx, ky, kz = config.k_points

    if min(kx, ky, kz) <= 0:
        raise CP2KGenerationError(
            "CP2K k-point values must all be positive."
        )

    return [
        "    &KPOINTS",
        (
            "      SCHEME MONKHORST-PACK "
            f"{kx} {ky} {kz}"
        ),
        "    &END KPOINTS",
    ]


def _render_cell(
    structure: Structure,
) -> list[str]:
    """Render CP2K CELL vectors from a pymatgen Structure."""

    matrix = structure.lattice.matrix

    if matrix.shape != (3, 3):
        raise CP2KGenerationError(
            "Structure lattice must contain three cell vectors."
        )

    a_vector = matrix[0]
    b_vector = matrix[1]
    c_vector = matrix[2]

    return [
        "    &CELL",
        (
            "      A "
            f"{a_vector[0]:.12f} "
            f"{a_vector[1]:.12f} "
            f"{a_vector[2]:.12f}"
        ),
        (
            "      B "
            f"{b_vector[0]:.12f} "
            f"{b_vector[1]:.12f} "
            f"{b_vector[2]:.12f}"
        ),
        (
            "      C "
            f"{c_vector[0]:.12f} "
            f"{c_vector[1]:.12f} "
            f"{c_vector[2]:.12f}"
        ),
        "      PERIODIC XYZ",
        "    &END CELL",
    ]


def _render_topology(
    config: CP2KInputConfig,
) -> list[str]:
    """Render external-coordinate topology settings."""

    if config.coordinate_mode != CP2KCoordinateMode.EXTERNAL_XYZ:
        raise CP2KGenerationError(
            "Embedded CP2K coordinates are not yet supported."
        )

    if not config.coordinate_file:
        raise CP2KGenerationError(
            "External XYZ mode requires coordinate_file."
        )

    return [
        "    &TOPOLOGY",
        f"      COORD_FILE_NAME {config.coordinate_file}",
        (
            "      COORD_FILE_FORMAT "
            f"{config.coordinate_file_format}"
        ),
        "    &END TOPOLOGY",
    ]


def _render_kinds(
    config: CP2KInputConfig,
) -> list[str]:
    """Render species-specific KIND sections."""

    basis_potential = config.basis_potential

    if basis_potential is None:
        raise CP2KGenerationError(
            "A basis/potential configuration is required."
        )

    lines: list[str] = []

    for kind in basis_potential.kinds:
        lines.extend(
            [
                f"    &KIND {kind.element}",
                f"      ELEMENT {kind.element}",
                f"      BASIS_SET {kind.basis_set}",
                f"      POTENTIAL {kind.potential}",
                "    &END KIND",
            ]
        )

    return lines

def _validate_kind_coverage(
    config: CP2KInputConfig,
    structure: Structure,
) -> None:
    """Validate that CP2K KIND definitions match structure species."""

    basis_potential = config.basis_potential

    if basis_potential is None:
        raise CP2KGenerationError(
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

        raise CP2KGenerationError(
            "Missing CP2K KIND definitions for structure "
            f"elements: {missing}"
        )

    if extra_elements:
        extra = ", ".join(
            sorted(extra_elements)
        )

        raise CP2KGenerationError(
            "CP2K KIND definitions contain elements not "
            f"present in the structure: {extra}"
        )

def render_cp2k_input(
    config: CP2KInputConfig,
    structure: Structure,
) -> str:
    """Render a CP2K input from validated configuration and structure."""

    basis_potential = config.basis_potential

    if basis_potential is None:
        raise CP2KGenerationError(
            "A basis/potential configuration is required."
        )

    if len(structure) == 0:
        raise CP2KGenerationError(
            "Cannot generate CP2K input for an empty structure."
        )

    _validate_kind_coverage(
        config=config,
        structure=structure,
    )
    
    lines = [
        "&GLOBAL",
        f"  PROJECT {config.project_name}",
        f"  RUN_TYPE {config.run_type}",
        "  PRINT_LEVEL MEDIUM",
        "&END GLOBAL",
        "",
        "&FORCE_EVAL",
        "  METHOD QUICKSTEP",
        "",
        "  &DFT",
        (
            "    BASIS_SET_FILE_NAME "
            f"{basis_potential.basis_set_file}"
        ),
        (
            "    POTENTIAL_FILE_NAME "
            f"{basis_potential.potential_file}"
        ),
        f"    CHARGE {config.charge}",
        f"    MULTIPLICITY {config.multiplicity}",
        "",
        "    &QS",
        "      METHOD GPW",
        "    &END QS",
        "",
        "    &MGRID",
        f"      CUTOFF {config.cutoff_ry:.8g}",
        f"      REL_CUTOFF {config.relative_cutoff_ry:.8g}",
        "    &END MGRID",
        "",
    ]

    lines.extend(
        _render_scf(config)
    )

    kpoint_lines = _render_kpoints(config)

    if kpoint_lines:
        lines.append("")
        lines.extend(kpoint_lines)

    lines.append("")
    lines.extend(
        _render_xc(config.functional)
    )

    lines.extend(
        [
            "  &END DFT",
            "",
            "  &SUBSYS",
        ]
    )

    lines.extend(
        _render_cell(structure)
    )

    lines.append("")

    lines.extend(
        _render_topology(config)
    )

    lines.append("")

    lines.extend(
        _render_kinds(config)
    )

    lines.extend(
        [
            "  &END SUBSYS",
            "&END FORCE_EVAL",
            "",
        ]
    )

    return "\n".join(lines)