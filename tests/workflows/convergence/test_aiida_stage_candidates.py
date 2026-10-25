"""Tests for automatic AiiDA convergence stage preparation."""

from copy import deepcopy

import pytest

from nsdw.workflows.convergence.aiida_stage_candidates import (
    build_relative_cutoff_candidates,
)


def make_inputs():
    base = {
        "FORCE_EVAL": {
            "DFT": {
                "MGRID": {
                    "CUTOFF": 600.0,
                    "REL_CUTOFF": 30.0,
                },
            },
        },
    }

    analysis = {
        "report": {
            "parameter": "cutoff",
            "energy_tolerance_satisfied": True,
            "selected_candidate_label": "400-Ry",
        },
        "validation_scope": "standard_prefix",
        "production_certified": False,
        "provisional": True,
    }

    return base, analysis


def test_relative_cutoff_candidates_use_selected_cutoff():
    base, analysis = make_inputs()
    original = deepcopy(base)

    candidates = build_relative_cutoff_candidates(
        base_parameters=base,
        cutoff_analysis=analysis,
    )

    assert list(candidates) == [
        "10-Ry", "20-Ry", "30-Ry", "40-Ry", "50-Ry"
    ]

    for index, parameters in enumerate(candidates.values(), start=1):
        mgrid = parameters["FORCE_EVAL"]["DFT"]["MGRID"]
        assert mgrid["CUTOFF"] == 400.0
        assert mgrid["REL_CUTOFF"] == index * 10.0

    assert base == original


def test_no_selected_cutoff_rejected():
    base, analysis = make_inputs()
    analysis["report"]["selected_candidate_label"] = None

    with pytest.raises(ValueError, match="selected no candidate"):
        build_relative_cutoff_candidates(
            base_parameters=base,
            cutoff_analysis=analysis,
        )


def test_failed_tolerance_rejected():
    base, analysis = make_inputs()
    analysis["report"]["energy_tolerance_satisfied"] = False

    with pytest.raises(ValueError, match="tolerance"):
        build_relative_cutoff_candidates(
            base_parameters=base,
            cutoff_analysis=analysis,
        )


def test_wrong_previous_parameter_rejected():
    base, analysis = make_inputs()
    analysis["report"]["parameter"] = "relative_cutoff"

    with pytest.raises(ValueError, match="cutoff study"):
        build_relative_cutoff_candidates(
            base_parameters=base,
            cutoff_analysis=analysis,
        )


def test_invalid_candidate_count_rejected():
    base, analysis = make_inputs()

    for count in (0, 1, 11, True):
        with pytest.raises(ValueError, match="candidate count"):
            build_relative_cutoff_candidates(
                base_parameters=base,
                cutoff_analysis=analysis,
                candidate_count=count,
            )


def test_independent_candidate_dictionaries():
    base, analysis = make_inputs()

    candidates = build_relative_cutoff_candidates(
        base_parameters=base,
        cutoff_analysis=analysis,
    )

    candidates["10-Ry"]["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"] = 999

    assert (
        candidates["20-Ry"]["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"]
        == 400.0
    )
