"""Relative-cutoff AiiDA scientific analysis tests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from aiida.common.extendeddicts import AttributeDict

from nsdw.workflows.convergence.aiida_analysis import (
    analyse_aiida_convergence_campaign,
)


class FakeDict:
    def __init__(self, data):
        self.data = deepcopy(data)

    def get_dict(self):
        return deepcopy(self.data)


class FakeStructure:
    def get_ase(self):
        from ase import Atoms
        return Atoms(
            "SiO2",
            positions=[
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            cell=[5, 5, 5],
            pbc=True,
        )


def make_parent():
    candidates = {}
    summary = {}

    energies = [
        -108.4960,
        -108.4961,
        -108.49615,
        -108.49616,
        -108.49617,
    ]

    for index, relative in enumerate((10, 20, 30, 40, 50)):
        name = f"{relative}-Ry"

        candidates[name] = {
            "FORCE_EVAL": {
                "DFT": {
                    "MGRID": {
                        "CUTOFF": 400.0,
                        "REL_CUTOFF": float(relative),
                    }
                }
            }
        }

        summary[name] = {
            "finished_ok": True,
            "exit_status": 0,
            "energy_hartree": energies[index],
        }

    return SimpleNamespace(
        is_finished_ok=True,
        inputs=SimpleNamespace(
            candidates=FakeDict(candidates),
            structure=FakeStructure(),
        ),
        outputs=AttributeDict({
            "summary": FakeDict(summary),
        }),
    )


def test_relative_cutoff_analysis():
    parent = make_parent()

    report = analyse_aiida_convergence_campaign(
        parent,
        parameter="relative_cutoff",
    )

    assert report.parameter.value == "relative_cutoff"
    assert len(report.candidates) == 5
    assert report.reference_candidate_label == "50-Ry"


def test_fixed_cutoff_mismatch_rejected():
    parent = make_parent()

    candidates = parent.inputs.candidates.get_dict()
    candidates["20-Ry"]["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = 600
    parent.inputs.candidates = FakeDict(candidates)

    with pytest.raises(ValueError, match="Fixed CP2K parameters"):
        analyse_aiida_convergence_campaign(
            parent,
            parameter="relative_cutoff",
        )


def test_other_fixed_parameter_mismatch_rejected():
    parent = make_parent()

    candidates = parent.inputs.candidates.get_dict()
    candidates["30-Ry"]["FORCE_EVAL"]["DFT"]["SCF"] = {
        "EPS_SCF": 1e-5
    }
    parent.inputs.candidates = FakeDict(candidates)

    with pytest.raises(ValueError, match="Fixed CP2K parameters"):
        analyse_aiida_convergence_campaign(
            parent,
            parameter="relative_cutoff",
        )


def test_invalid_parameter_rejected():
    parent = make_parent()

    with pytest.raises(ValueError, match="Unsupported convergence"):
        analyse_aiida_convergence_campaign(
            parent,
            parameter="basis",
        )
