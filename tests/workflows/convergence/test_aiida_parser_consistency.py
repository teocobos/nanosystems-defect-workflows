"""Pure tests for CP2K parser-to-summary consistency."""

from copy import deepcopy

import pytest

from nsdw.workflows.convergence.aiida_workchain import (
    validate_parser_summary_consistency,
)


class FakeDict:
    def __init__(self, data):
        self.data = data

    def get_dict(self):
        return deepcopy(self.data)


def make_inputs():
    candidates = FakeDict({
        "400-Ry": {},
        "450-Ry": {},
    })

    summary = FakeDict({
        "400-Ry": {"energy_hartree": -100.0},
        "450-Ry": {"energy_hartree": -100.1},
    })

    parsers = {
        "candidate_0000": FakeDict({
            "energy": -100.0,
            "energy_units": "a.u.",
        }),
        "candidate_0001": FakeDict({
            "energy": -100.1,
            "energy_units": "a.u.",
        }),
    }

    return candidates, summary, parsers


def test_matching_parser_energies_pass():
    candidates, summary, parsers = make_inputs()
    validate_parser_summary_consistency(candidates, summary, parsers)


def test_missing_parser_rejected():
    candidates, summary, parsers = make_inputs()
    del parsers["candidate_0001"]

    with pytest.raises(ValueError, match="parser nodes"):
        validate_parser_summary_consistency(candidates, summary, parsers)


def test_extra_parser_rejected():
    candidates, summary, parsers = make_inputs()
    parsers["candidate_0002"] = FakeDict({
        "energy": -100.2,
        "energy_units": "a.u.",
    })

    with pytest.raises(ValueError, match="parser nodes"):
        validate_parser_summary_consistency(candidates, summary, parsers)


def test_energy_mismatch_rejected():
    candidates, summary, parsers = make_inputs()
    parsers["candidate_0000"] = FakeDict({
        "energy": -99.0,
        "energy_units": "a.u.",
    })

    with pytest.raises(ValueError, match="energy mismatch"):
        validate_parser_summary_consistency(candidates, summary, parsers)


def test_wrong_units_rejected():
    candidates, summary, parsers = make_inputs()
    parsers["candidate_0000"] = FakeDict({
        "energy": -100.0,
        "energy_units": "eV",
    })

    with pytest.raises(ValueError, match="Invalid parser energy"):
        validate_parser_summary_consistency(candidates, summary, parsers)


def test_nonfinite_parser_energy_rejected():
    candidates, summary, parsers = make_inputs()
    parsers["candidate_0000"] = FakeDict({
        "energy": float("nan"),
        "energy_units": "a.u.",
    })

    with pytest.raises(ValueError, match="Invalid parser energy"):
        validate_parser_summary_consistency(candidates, summary, parsers)


def test_missing_summary_candidate_rejected():
    candidates, summary, parsers = make_inputs()
    summary.data.pop("450-Ry")

    with pytest.raises(ValueError, match="Summary candidates"):
        validate_parser_summary_consistency(candidates, summary, parsers)
