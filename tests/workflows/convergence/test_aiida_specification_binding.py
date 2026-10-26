"""Tests for scientific specification-to-candidate binding."""

import pytest

from nsdw.workflows.convergence.aiida_campaign_specification import (
    create_campaign_specification,
)
from nsdw.workflows.convergence.aiida_specification_binding import (
    validate_specification_binding,
)


def make_specification(tmp_path):
    return create_campaign_specification(
        tmp_path / "specification.json",
        campaign_name="sio2",
        expected_cutoffs=(300, 400, 500),
    )


def test_exact_candidate_coverage(tmp_path):
    spec = make_specification(tmp_path)

    result = validate_specification_binding(
        spec,
        candidates={
            "candidate_a": {},
            "candidate_b": {},
            "candidate_c": {},
        },
        candidate_cutoffs={
            "candidate_a": 300,
            "candidate_b": 400,
            "candidate_c": 500,
        },
    )

    assert result.expected_cutoffs == (300, 400, 500)
    assert result.specification_sha256 == spec.sha256


def test_missing_candidate_rejected(tmp_path):
    spec = make_specification(tmp_path)

    with pytest.raises(ValueError, match="do not match"):
        validate_specification_binding(
            spec,
            candidates={"a": {}, "b": {}},
            candidate_cutoffs={"a": 300, "b": 400},
        )


def test_unexpected_candidate_rejected(tmp_path):
    spec = make_specification(tmp_path)

    with pytest.raises(ValueError, match="do not match"):
        validate_specification_binding(
            spec,
            candidates={
                "a": {}, "b": {}, "c": {}, "d": {},
            },
            candidate_cutoffs={
                "a": 300,
                "b": 400,
                "c": 500,
                "d": 600,
            },
        )


def test_duplicate_cutoffs_rejected(tmp_path):
    spec = make_specification(tmp_path)

    with pytest.raises(ValueError, match="unique"):
        validate_specification_binding(
            spec,
            candidates={"a": {}, "b": {}, "c": {}},
            candidate_cutoffs={
                "a": 300,
                "b": 300,
                "c": 500,
            },
        )


def test_missing_mapping_entry_rejected(tmp_path):
    spec = make_specification(tmp_path)

    with pytest.raises(ValueError, match="mapping"):
        validate_specification_binding(
            spec,
            candidates={"a": {}, "b": {}, "c": {}},
            candidate_cutoffs={"a": 300, "b": 400},
        )


def test_boolean_cutoff_rejected(tmp_path):
    spec = make_specification(tmp_path)

    with pytest.raises(ValueError, match="unique"):
        validate_specification_binding(
            spec,
            candidates={"a": {}, "b": {}, "c": {}},
            candidate_cutoffs={
                "a": 300,
                "b": True,
                "c": 500,
            },
        )


def test_cp2k_actual_cutoffs_match():
    from nsdw.workflows.convergence.aiida_specification_binding import (
        validate_cp2k_cutoff_parameters,
    )

    candidates = {
        "a": {"FORCE_EVAL": {"DFT": {
            "MGRID": {"CUTOFF": 300}
        }}},
        "b": {"FORCE_EVAL": {"DFT": {
            "MGRID": {"CUTOFF": 400.0}
        }}},
    }

    validate_cp2k_cutoff_parameters(
        candidates,
        {"a": 300, "b": 400},
    )


def test_cp2k_actual_cutoff_mismatch_rejected():
    from nsdw.workflows.convergence.aiida_specification_binding import (
        validate_cp2k_cutoff_parameters,
    )

    candidates = {
        "a": {"FORCE_EVAL": {"DFT": {
            "MGRID": {"CUTOFF": 500}
        }}},
    }

    with pytest.raises(ValueError, match="mismatch"):
        validate_cp2k_cutoff_parameters(
            candidates,
            {"a": 300},
        )


@pytest.mark.parametrize(
    "actual",
    [None, "300", True, -300, float("nan"), float("inf")],
)
def test_invalid_cp2k_cutoff_rejected(actual):
    from nsdw.workflows.convergence.aiida_specification_binding import (
        validate_cp2k_cutoff_parameters,
    )

    candidates = {
        "a": {"FORCE_EVAL": {"DFT": {
            "MGRID": {"CUTOFF": actual}
        }}},
    }

    with pytest.raises(ValueError):
        validate_cp2k_cutoff_parameters(
            candidates,
            {"a": 300},
        )


def test_missing_cp2k_cutoff_rejected():
    from nsdw.workflows.convergence.aiida_specification_binding import (
        validate_cp2k_cutoff_parameters,
    )

    with pytest.raises(ValueError, match="Missing"):
        validate_cp2k_cutoff_parameters(
            {"a": {"FORCE_EVAL": {"DFT": {"MGRID": {}}}}},
            {"a": 300},
        )
