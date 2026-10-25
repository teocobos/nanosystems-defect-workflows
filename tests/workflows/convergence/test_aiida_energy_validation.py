"""Unit tests for CP2K energy validation."""

import math

import pytest

from nsdw.workflows.convergence.aiida_workchain import (
    valid_cp2k_energy,
)


@pytest.mark.parametrize(
    "energy,units,expected",
    [
        (-108.49635643498, "a.u.", True),
        (0.0, "a.u.", True),
        (1.5, "a.u.", True),
        (None, "a.u.", False),
        ("-108.5", "a.u.", False),
        (float("nan"), "a.u.", False),
        (float("inf"), "a.u.", False),
        (-108.5, "eV", False),
        (-108.5, None, False),
        (True, "a.u.", False),
    ],
)
def test_valid_cp2k_energy(energy, units, expected):
    assert valid_cp2k_energy(energy, units) is expected
