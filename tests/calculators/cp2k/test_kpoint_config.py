"""Validation tests for structured CP2K k-point configurations."""

import pytest
from pydantic import ValidationError

from nsdw.calculators.cp2k.generation_models import (
    CP2KKPointConfig,
    CP2KKPointScheme,
    CP2KWeightedKPoint,
)


def test_default_is_implicit_gamma():
    config = CP2KKPointConfig()
    assert config.scheme == CP2KKPointScheme.NONE
    assert config.mesh is None


@pytest.mark.parametrize(
    "scheme",
    ["NONE", "GAMMA"],
)
def test_gamma_modes(scheme):
    assert CP2KKPointConfig(scheme=scheme).scheme == scheme


def test_monkhorst_pack():
    config = CP2KKPointConfig(
        scheme="MONKHORST-PACK",
        mesh=(4, 4, 4),
    )
    assert config.mesh == (4, 4, 4)


def test_macdonald():
    config = CP2KKPointConfig(
        scheme="MACDONALD",
        mesh=(4, 4, 4),
        shift=(0.25, 0.25, 0.25),
    )
    assert config.shift == (0.25, 0.25, 0.25)


def test_general():
    config = CP2KKPointConfig(
        scheme="GENERAL",
        points=(
            CP2KWeightedKPoint(
                coordinates=(0.0, 0.0, 0.0),
                weight=1.0,
            ),
        ),
    )
    assert len(config.points) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheme": "MONKHORST-PACK"},
        {"scheme": "MONKHORST-PACK", "mesh": (0, 2, 2)},
        {"scheme": "MACDONALD", "mesh": (2, 2, 2)},
        {"scheme": "GENERAL"},
        {"scheme": "GAMMA", "mesh": (1, 1, 1)},
        {"scheme": "NONE", "shift": (0.0, 0.0, 0.0)},
    ],
)
def test_invalid_scheme_combinations(kwargs):
    with pytest.raises(ValidationError):
        CP2KKPointConfig(**kwargs)


@pytest.mark.parametrize(
    "coordinate",
    [float("nan"), float("inf"), -float("inf")],
)
def test_nonfinite_coordinates_rejected(coordinate):
    with pytest.raises(ValidationError):
        CP2KKPointConfig(
            scheme="GENERAL",
            points=(
                CP2KWeightedKPoint(
                    coordinates=(coordinate, 0.0, 0.0),
                    weight=1.0,
                ),
            ),
        )


def test_nonfinite_weight_rejected():
    with pytest.raises(ValidationError):
        CP2KKPointConfig(
            scheme="GENERAL",
            points=(
                CP2KWeightedKPoint(
                    coordinates=(0.0, 0.0, 0.0),
                    weight=float("inf"),
                ),
            ),
        )


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        CP2KKPointConfig(scheme="GAMMA", unknown_option=True)


def test_input_config_preserves_legacy_kpoints():
    from nsdw.calculators.cp2k.generation_models import CP2KInputConfig

    config = CP2KInputConfig(
        project_name="legacy",
        k_points=(2, 2, 2),
    )

    assert config.k_points == (2, 2, 2)
    assert config.kpoint_sampling is None


def test_input_config_accepts_structured_kpoints():
    from nsdw.calculators.cp2k.generation_models import CP2KInputConfig

    config = CP2KInputConfig(
        project_name="structured",
        kpoint_sampling=CP2KKPointConfig(
            scheme="MACDONALD",
            mesh=(4, 4, 4),
            shift=(0.25, 0.25, 0.25),
        ),
    )

    assert config.k_points is None
    assert config.kpoint_sampling.scheme == CP2KKPointScheme.MACDONALD


def test_input_config_rejects_conflicting_kpoints():
    from nsdw.calculators.cp2k.generation_models import CP2KInputConfig

    with pytest.raises(ValidationError):
        CP2KInputConfig(
            project_name="conflict",
            k_points=(2, 2, 2),
            kpoint_sampling=CP2KKPointConfig(
                scheme="MONKHORST-PACK",
                mesh=(2, 2, 2),
            ),
        )


def test_input_config_default_remains_gamma_only():
    from nsdw.calculators.cp2k.generation_models import CP2KInputConfig

    config = CP2KInputConfig(project_name="default")

    assert config.k_points is None
    assert config.kpoint_sampling is None
