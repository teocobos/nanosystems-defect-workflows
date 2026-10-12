"""Regression tests for structured CP2K k-point rendering."""

import pytest

from nsdw.calculators.cp2k.generation_models import (
    CP2KInputConfig,
    CP2KKPointConfig,
    CP2KSCFConfig,
    CP2KWeightedKPoint,
)
from nsdw.calculators.cp2k.generator import (
    CP2KGenerationError,
    _render_kpoints,
    _render_scf,
)


def config(sampling=None, solver="DIAGONALIZATION"):
    return CP2KInputConfig(
        project_name="kpoint-rendering",
        kpoint_sampling=sampling,
        scf=CP2KSCFConfig(solver=solver),
    )


def test_default_omits_kpoints():
    assert _render_kpoints(config()) == []


def test_explicit_gamma():
    text = "\n".join(_render_kpoints(
        config(CP2KKPointConfig(scheme="GAMMA"))
    ))
    assert "SCHEME GAMMA" in text


def test_monkhorst_pack():
    text = "\n".join(_render_kpoints(
        config(CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(4, 4, 4),
        ))
    ))
    assert "SCHEME MONKHORST-PACK 4 4 4" in text


def test_macdonald():
    text = "\n".join(_render_kpoints(
        config(CP2KKPointConfig(
            scheme="MACDONALD",
            mesh=(4, 4, 4),
            shift=(0.25, 0.25, 0.25),
        ))
    ))
    assert "SCHEME MACDONALD 4 4 4 0.25 0.25 0.25" in text


def test_general_weighted():
    text = "\n".join(_render_kpoints(
        config(CP2KKPointConfig(
            scheme="GENERAL",
            points=(
                CP2KWeightedKPoint(
                    coordinates=(0.0, 0.0, 0.0),
                    weight=1.0,
                ),
            ),
        ))
    ))
    assert "SCHEME GENERAL" in text
    assert "KPOINT 0 0 0 1" in text


def test_none_preserves_implicit_gamma():
    assert _render_kpoints(
        config(CP2KKPointConfig(scheme="NONE"))
    ) == []


def test_structured_sampling_requires_diagonalization():
    with pytest.raises(
        CP2KGenerationError,
        match="DIAGONALIZATION",
    ):
        _render_scf(
            config(
                CP2KKPointConfig(
                    scheme="MONKHORST-PACK",
                    mesh=(2, 2, 2),
                ),
                solver="OT",
            )
        )


def test_legacy_and_structured_conflict_even_after_model_copy():
    original = CP2KInputConfig(
        project_name="conflict",
        k_points=(2, 2, 2),
        scf=CP2KSCFConfig(solver="DIAGONALIZATION"),
    )

    conflicting = original.model_copy(
        update={
            "kpoint_sampling": CP2KKPointConfig(
                scheme="MONKHORST-PACK",
                mesh=(2, 2, 2),
            )
        }
    )

    with pytest.raises(CP2KGenerationError):
        _render_kpoints(conflicting)
