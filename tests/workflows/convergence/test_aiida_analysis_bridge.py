"""Regression tests for AiiDA-to-NSDW convergence analysis."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from aiida.common.extendeddicts import AttributeDict

from nsdw.workflows.convergence.aiida_analysis import (
    analyse_aiida_cutoff_campaign,
)
from nsdw.workflows.convergence.recipes import (
    build_standard_cp2k_cutoff_study,
)


class FakeDict:
    def __init__(self, value):
        self.value = value

    def get_dict(self):
        return deepcopy(self.value)


class FakeStructure:
    def get_ase(self):
        from ase import Atoms

        return Atoms(
            "SiO2",
            positions=[
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            cell=[5, 5, 5],
            pbc=True,
        )


def make_parent():
    study = build_standard_cp2k_cutoff_study()
    candidates = study.candidates[:5]

    submitted = {}
    results = {}

    for index, candidate in enumerate(candidates):
        submitted[candidate.label] = {
            "FORCE_EVAL": {
                "DFT": {
                    "MGRID": {
                        "CUTOFF": candidate.value.value
                    }
                }
            }
        }

        results[candidate.label] = {
            "finished_ok": True,
            "exit_status": 0,
            "energy_hartree": -100.0 + index * 0.00001,
        }

    return SimpleNamespace(
        is_finished_ok=True,
        inputs=SimpleNamespace(
            structure=FakeStructure(),
            candidates=FakeDict(submitted),
        ),
        outputs=AttributeDict(
            {"summary": FakeDict(results)}
        ),
    )


def test_successful_analysis():
    report = analyse_aiida_cutoff_campaign(make_parent())

    assert len(report.candidates) == 5
    assert report.energy_tolerance_satisfied


def test_failed_parent_rejected():
    parent = make_parent()
    parent.is_finished_ok = False

    with pytest.raises(ValueError, match="finish successfully"):
        analyse_aiida_cutoff_campaign(parent)


def test_missing_candidate_rejected():
    parent = make_parent()
    results = parent.outputs.summary.get_dict()
    results.pop("450-Ry")
    parent.outputs.summary = FakeDict(results)

    with pytest.raises(ValueError, match="do not match"):
        analyse_aiida_cutoff_campaign(parent)


def test_invalid_energy_rejected():
    parent = make_parent()
    results = parent.outputs.summary.get_dict()
    results["400-Ry"]["energy_hartree"] = float("nan")
    parent.outputs.summary = FakeDict(results)

    with pytest.raises(ValueError, match="invalid energy"):
        analyse_aiida_cutoff_campaign(parent)


def test_cutoff_mismatch_rejected():
    parent = make_parent()
    submitted = parent.inputs.candidates.get_dict()
    submitted["400-Ry"]["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = 500
    parent.inputs.candidates = FakeDict(submitted)

    with pytest.raises(ValueError, match="Cutoff mismatch"):
        analyse_aiida_cutoff_campaign(parent)


def test_attribute_dict_supports_analysis_interface():
    outputs = AttributeDict({"summary": FakeDict({})})

    assert "summary" in outputs
    assert outputs.summary.get_dict() == {}
