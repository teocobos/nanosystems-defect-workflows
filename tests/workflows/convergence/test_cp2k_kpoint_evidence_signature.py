"""Scientific signature tests for CP2K k-point sampling."""

from nsdw.calculators.cp2k.generation_models import (
    CP2KKPointConfig,
    CP2KWeightedKPoint,
)
from nsdw.calculators.cp2k.models import ParsedCP2KInput
from nsdw.workflows.convergence.cp2k_evidence import _input_signature


def parsed(sampling=None, mesh=None):
    return ParsedCP2KInput(
        k_points=mesh,
        kpoint_sampling=sampling,
        scf_solver="DIAGONALIZATION",
    )


def test_implicit_and_explicit_gamma_differ():
    implicit = parsed()
    explicit = parsed(CP2KKPointConfig(scheme="GAMMA"))

    assert _input_signature(implicit, "cutoff") != (
        _input_signature(explicit, "cutoff")
    )


def test_macdonald_shifts_differ():
    first = parsed(
        CP2KKPointConfig(
            scheme="MACDONALD",
            mesh=(4, 4, 4),
            shift=(0.0, 0.0, 0.0),
        )
    )
    second = parsed(
        CP2KKPointConfig(
            scheme="MACDONALD",
            mesh=(4, 4, 4),
            shift=(0.25, 0.25, 0.25),
        )
    )

    assert _input_signature(first, "relative_cutoff") != (
        _input_signature(second, "relative_cutoff")
    )


def test_general_weights_differ():
    first = parsed(
        CP2KKPointConfig(
            scheme="GENERAL",
            points=(
                CP2KWeightedKPoint(
                    coordinates=(0.0, 0.0, 0.0),
                    weight=1.0,
                ),
            ),
        )
    )
    second = parsed(
        CP2KKPointConfig(
            scheme="GENERAL",
            points=(
                CP2KWeightedKPoint(
                    coordinates=(0.0, 0.0, 0.0),
                    weight=0.5,
                ),
            ),
        )
    )

    assert _input_signature(first, "basis") != (
        _input_signature(second, "basis")
    )


def test_kpoint_campaign_may_vary_sampling():
    gamma = parsed(CP2KKPointConfig(scheme="GAMMA"))
    monkhorst = parsed(
        CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(2, 2, 2),
        )
    )

    assert _input_signature(gamma, "kpoints") == (
        _input_signature(monkhorst, "kpoints")
    )


def test_kpoint_campaign_must_preserve_solver():
    gamma = parsed(CP2KKPointConfig(scheme="GAMMA"))
    monkhorst = ParsedCP2KInput(
        kpoint_sampling=CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(2, 2, 2),
        ),
        scf_solver="OT",
    )

    assert _input_signature(gamma, "kpoints") != (
        _input_signature(monkhorst, "kpoints")
    )


def test_legacy_mesh_remains_part_of_signature():
    first = parsed(mesh=(2, 2, 2))
    second = parsed(mesh=(3, 3, 3))

    assert _input_signature(first, "cutoff") != (
        _input_signature(second, "cutoff")
    )
