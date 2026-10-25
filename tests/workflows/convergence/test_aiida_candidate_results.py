"""Tests for CP2K candidate result assessment."""

import pytest

from nsdw.workflows.convergence.aiida_workchain import (
    assess_candidate_result,
)


def assess(*, finished_ok=True, exit_status=0, parameters=None):
    return assess_candidate_result(
        pk=52,
        uuid="test-uuid",
        finished_ok=finished_ok,
        exit_status=exit_status,
        output_parameters=parameters,
    )


def test_successful_candidate():
    entry, valid = assess(
        parameters={
            "energy": -108.49635643498,
            "energy_units": "a.u.",
            "nwarnings": 0,
        }
    )

    assert valid
    assert entry["energy_hartree"] == -108.49635643498
    assert entry["pk"] == 52
    assert entry["exit_status"] == 0


def test_failed_child():
    entry, valid = assess(
        finished_ok=False,
        exit_status=400,
        parameters=None,
    )

    assert not valid
    assert entry["finished_ok"] is False
    assert entry["exit_status"] == 400


def test_missing_parser_output():
    entry, valid = assess(parameters=None)

    assert not valid
    assert entry["missing_output_parameters"] is True


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"energy": None, "energy_units": "a.u."},
        {"energy": float("nan"), "energy_units": "a.u."},
        {"energy": -108.5, "energy_units": "eV"},
        {"energy": "-108.5", "energy_units": "a.u."},
    ],
)
def test_invalid_energy(parameters):
    entry, valid = assess(parameters=parameters)

    assert not valid
    assert entry["invalid_energy"] is True
