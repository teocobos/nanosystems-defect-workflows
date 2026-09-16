from pathlib import Path

from nsdw.config.models import (
    StructureSummary,
    StructureValidationResult,
)
from nsdw.output.models import (
    CheckOutput,
    LatticeOutput,
    ParserOutput,
    SourceInfo,
    StructureOutput,
    StructureValidationOutput,
    ValidationOutput,
)


OUTPUT_PRECISION = 8


def _round_float(
    value: float | None,
    digits: int = OUTPUT_PRECISION,
) -> float | None:
    """
    Round a floating-point value for deterministic exported output.

    Scientific calculations should continue to use the original
    full-precision values.
    """

    if value is None:
        return None

    return round(float(value), digits)


def build_structure_validation_output(
    *,
    source_path: str | Path,
    summary: StructureSummary,
    validation: StructureValidationResult,
    parser_warnings: list[str],
    nsdw_version: str,
    minimum_distance_threshold: float,
) -> StructureValidationOutput:
    """
    Build the canonical machine-readable result for structure validation.
    """

    path = Path(source_path).expanduser().resolve()

    return StructureValidationOutput(
        nsdw_version=nsdw_version,
        source=SourceInfo(
            path=str(path),
            format=path.suffix.lower().lstrip("."),
        ),
        structure=StructureOutput(
            formula=summary.formula,
            reduced_formula=summary.reduced_formula,
            num_sites=summary.num_sites,
            density_g_cm3=_round_float(summary.density),
            lattice=LatticeOutput(
                a_angstrom=_round_float(summary.lattice.a),
                b_angstrom=_round_float(summary.lattice.b),
                c_angstrom=_round_float(summary.lattice.c),
                alpha_deg=_round_float(summary.lattice.alpha),
                beta_deg=_round_float(summary.lattice.beta),
                gamma_deg=_round_float(summary.lattice.gamma),
                volume_angstrom3=_round_float(
                    summary.lattice.volume
                ),
            ),
        ),
        parser=ParserOutput(
            warnings=parser_warnings,
        ),
        validation=ValidationOutput(
            valid=validation.valid,
            ordered=validation.ordered,
            minimum_distance_angstrom=_round_float(
                validation.minimum_distance
            ),
            minimum_distance_threshold_angstrom=_round_float(
                minimum_distance_threshold
            ),
            checks=[
                CheckOutput(
                    name=check.name,
                    passed=check.passed,
                    value=check.value,
                    message=check.message,
                )
                for check in validation.checks
            ],
            warnings=validation.warnings,
            errors=validation.errors,
        ),
    )