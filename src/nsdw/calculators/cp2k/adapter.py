"""Adapter from parsed CP2K output to NSDW scientific results."""

from __future__ import annotations

from pathlib import Path

from nsdw.calculators.cp2k.models import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KInput,
    ParsedCP2KResult,
)
from nsdw.models.calculation import (
    Backend,
    CalculationMetadata,
    CalculationStatus,
    CalculationType,
)
from nsdw.models.provenance import (
    CP2KKindSettings,
    CP2KSettings,
    ExecutionPlatform,
    ExecutionProvenance,
    ProvenanceResult,
    SchedulerType,
    SoftwareProvenance,
)
from nsdw.models.quantity import Quantity
from nsdw.models.result import (
    EnergyResult,
    NSDWResult,
)

from nsdw.provenance.files import (
    build_file_reference,
)

HARTREE_TO_EV = 27.211386245988


class CP2KAdapterError(RuntimeError):
    """Raised when parsed CP2K data cannot be mapped to NSDW."""


_RUN_TYPE_MAP = {
    CP2KRunType.ENERGY: CalculationType.SINGLE_POINT,
    CP2KRunType.GEO_OPT: CalculationType.GEOMETRY_OPTIMISATION,
    CP2KRunType.MD: CalculationType.MOLECULAR_DYNAMICS,
}


def _map_calculation_type(
    run_type: CP2KRunType | None,
) -> CalculationType:
    """Map a CP2K run type onto the NSDW calculation taxonomy."""

    if run_type is None:
        raise CP2KAdapterError(
            "CP2K run type is missing"
        )

    try:
        return _RUN_TYPE_MAP[run_type]

    except KeyError as exc:
        raise CP2KAdapterError(
            "Unsupported CP2K run type for NSDW adapter: "
            f"{run_type.value}"
        ) from exc


def _map_calculation_status(
    parsed: ParsedCP2KResult,
) -> CalculationStatus:
    """Determine scientific calculation status."""

    if not parsed.normal_termination:
        return CalculationStatus.FAILED

    if parsed.scf.status != CP2KSCFStatus.CONVERGED:
        return CalculationStatus.FAILED

    return CalculationStatus.COMPLETED


def _build_energy(
    parsed: ParsedCP2KResult,
) -> EnergyResult | None:
    """Convert the final CP2K total energy into canonical NSDW units."""

    energy_hartree = (
        parsed.energy.total_energy_hartree
    )

    if energy_hartree is None:
        return None

    return EnergyResult(
        total=Quantity(
            value=(
                energy_hartree
                * HARTREE_TO_EV
            ),
            unit="eV",
        )
    )


def adapt_cp2k_result(
    parsed: ParsedCP2KResult,
    *,
    input_settings: ParsedCP2KInput | None = None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    calculation_id: str | None = None,
    nsdw_version: str = "0.1.0",
    platform: ExecutionPlatform = ExecutionPlatform.LOCAL,
    scheduler: SchedulerType = SchedulerType.LOCAL,
) -> NSDWResult:
    """Convert parsed CP2K output into a calculator-independent NSDW result."""
    _validate_input_output_consistency(
        parsed,
        input_settings,
    )

    calculation_type = _map_calculation_type(
        parsed.run_type
    )

    calculation_status = _map_calculation_status(
        parsed
    )

    result_id = (
        calculation_id
        or parsed.project_name
    )

    if not result_id:
        raise CP2KAdapterError(
            "Calculation ID could not be determined"
        )

    if not parsed.cp2k_version:
        raise CP2KAdapterError(
            "CP2K version is required for provenance"
        )

    input_files = (
        (
            build_file_reference(
                input_path,
                format="cp2k-input",
            ),
        )
        if input_path is not None
        else ()
    )

    output_files = (
        (
            build_file_reference(
                output_path,
                format="cp2k-output",
            ),
        )
        if output_path is not None
        else ()
    )

    return NSDWResult(
        calculation=CalculationMetadata(
            id=result_id,
            type=calculation_type,
            status=calculation_status,
            backend=Backend.CP2K,
        ),
        energy=_build_energy(parsed),
        provenance=ProvenanceResult(
            software=SoftwareProvenance(
                calculator="CP2K",
                calculator_version=(
                    parsed.cp2k_version
                ),
                nsdw_version=nsdw_version,
            ),
            execution=ExecutionProvenance(
                platform=platform,
                scheduler=scheduler,
            ),
            input_files=input_files,
            output_files=output_files,
            cp2k=_build_cp2k_settings(
                parsed,
                input_settings,
            ),
        ),
    )

def _validate_input_output_consistency(
    parsed: ParsedCP2KResult,
    input_settings: ParsedCP2KInput | None,
) -> None:
    """Ensure CP2K input and output describe the same calculation."""

    if input_settings is None:
        return

    if (
        parsed.project_name is not None
        and input_settings.project_name is not None
        and parsed.project_name
        != input_settings.project_name
    ):
        raise CP2KAdapterError(
            "CP2K input/output project-name mismatch: "
            f"input={input_settings.project_name!r}, "
            f"output={parsed.project_name!r}"
        )

    if (
        parsed.run_type is not None
        and input_settings.run_type is not None
        and parsed.run_type
        != input_settings.run_type
    ):
        raise CP2KAdapterError(
            "CP2K input/output run-type mismatch: "
            f"input={input_settings.run_type.value!r}, "
            f"output={parsed.run_type.value!r}"
        )

    if (
        parsed.charge is not None
        and input_settings.charge is not None
        and parsed.charge
        != input_settings.charge
    ):
        raise CP2KAdapterError(
            "CP2K input/output charge mismatch: "
            f"input={input_settings.charge}, "
            f"output={parsed.charge}"
        )

    if (
        parsed.multiplicity is not None
        and input_settings.multiplicity is not None
        and parsed.multiplicity
        != input_settings.multiplicity
    ):
        raise CP2KAdapterError(
            "CP2K input/output multiplicity mismatch: "
            f"input={input_settings.multiplicity}, "
            f"output={parsed.multiplicity}"
        )

def _build_cp2k_settings(
    parsed: ParsedCP2KResult,
    input_settings: ParsedCP2KInput | None,
) -> CP2KSettings:
    """Build CP2K provenance from parsed input and output."""

    if input_settings is None:
        return CP2KSettings(
            charge=(
                parsed.charge
                if parsed.charge is not None
                else 0
            ),
            multiplicity=(
                parsed.multiplicity
                if parsed.multiplicity is not None
                else 1
            ),
        )

    return CP2KSettings(
        xc_functional=(
            input_settings.xc_functional
        ),
        kinds=tuple(
            CP2KKindSettings(
                kind=kind.kind,
                element=kind.element,
                basis_set=kind.basis_set,
                potential=kind.potential,
            )
            for kind in input_settings.kinds
        ),
        basis_set_file=(
            input_settings.basis_set_file
        ),
        potential_file=(
            input_settings.potential_file
        ),
        cutoff=(
            Quantity(
                value=input_settings.cutoff_ry,
                unit="Ry",
            )
            if input_settings.cutoff_ry is not None
            else None
        ),
        relative_cutoff=(
            Quantity(
                value=input_settings.relative_cutoff_ry,
                unit="Ry",
            )
            if input_settings.relative_cutoff_ry
            is not None
            else None
        ),
        charge=(
            input_settings.charge
            if input_settings.charge is not None
            else (
                parsed.charge
                if parsed.charge is not None
                else 0
            )
        ),
        multiplicity=(
            input_settings.multiplicity
            if input_settings.multiplicity is not None
            else (
                parsed.multiplicity
                if parsed.multiplicity is not None
                else 1
            )
        ),
        eps_scf=input_settings.eps_scf,
        k_points=input_settings.k_points,
        admm=input_settings.admm,
    )