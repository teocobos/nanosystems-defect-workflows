"""Standard convergence recipes for NSDW workflows."""

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
)
from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)


STANDARD_CP2K_CUTOFF_RY: tuple[float, ...] = (
    400.0,
    450.0,
    500.0,
    560.0,
    600.0,
    650.0,
    700.0,
    750.0,
    800.0,
    850.0,
)

STANDARD_CP2K_RELATIVE_CUTOFF_RY: tuple[float, ...] = (
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
)


def _build_ry_study(
    *,
    parameter: ConvergenceParameter,
    values: tuple[float, ...],
) -> ConvergenceStudyDefinition:
    """Build a convergence study from an ordered grid in Ry."""

    candidates = [
        ConvergenceCandidate(
            label=f"{value:g}-Ry",
            order=order,
            value=Quantity(
                value=value,
                unit="Ry",
            ),
        )
        for order, value in enumerate(values)
    ]

    return ConvergenceStudyDefinition(
        parameter=parameter,
        candidates=candidates,
    )


def build_standard_cp2k_cutoff_study(
) -> ConvergenceStudyDefinition:
    """Build the standard NSDW CP2K cutoff convergence study."""

    return _build_ry_study(
        parameter=ConvergenceParameter.CUTOFF,
        values=STANDARD_CP2K_CUTOFF_RY,
    )


def build_standard_cp2k_relative_cutoff_study(
) -> ConvergenceStudyDefinition:
    """Build the standard NSDW CP2K relative-cutoff convergence study."""

    return _build_ry_study(
        parameter=ConvergenceParameter.RELATIVE_CUTOFF,
        values=STANDARD_CP2K_RELATIVE_CUTOFF_RY,
    )



class ConvergenceRecipeError(RuntimeError):
    """Raised when a standard convergence recipe cannot proceed."""


def selected_ry_value(
    *,
    study: ConvergenceStudyDefinition,
    analysis,
) -> float:
    """Return the selected convergence parameter value in Ry."""

    selected_label = analysis.selected_candidate_label

    if selected_label is None:
        raise ConvergenceRecipeError(
            "Convergence analysis did not select a candidate."
        )

    selected_candidate = next(
        (
            candidate
            for candidate in study.candidates
            if candidate.label == selected_label
        ),
        None,
    )

    if selected_candidate is None:
        raise ConvergenceRecipeError(
            "Selected convergence candidate "
            f"{selected_label!r} is not present in the study."
        )

    value = selected_candidate.value

    if not isinstance(value, Quantity) or value.unit != "Ry":
        raise ConvergenceRecipeError(
            "Selected convergence candidate must define "
            "a Quantity value in Ry."
        )

    return value.value


def apply_selected_cutoff(
    *,
    base_config: CP2KInputConfig,
    study: ConvergenceStudyDefinition,
    analysis,
) -> CP2KInputConfig:
    """Apply the selected cutoff to the configuration for the next stage."""

    if study.parameter != ConvergenceParameter.CUTOFF:
        raise ConvergenceRecipeError(
            "Selected cutoff can only be applied from a cutoff study."
        )

    selected_cutoff_ry = selected_ry_value(
        study=study,
        analysis=analysis,
    )

    return base_config.model_copy(
        update={
            "cutoff_ry": selected_cutoff_ry,
        }
    )