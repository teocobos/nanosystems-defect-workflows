"""Production CP2K k-point mesh evidence tests."""

import pytest

from nsdw.calculators.cp2k.generation_models import CP2KKPointConfig
from nsdw.calculators.cp2k.models import ParsedCP2KInput
from nsdw.workflows.convergence.cp2k_evidence import (
    _resolve_kpoint_evidence_mesh,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


def resolved_mesh(parsed):
    return _resolve_kpoint_evidence_mesh(parsed, "candidate")


def test_legacy_mesh():
    parsed = ParsedCP2KInput(k_points=(2, 2, 2))
    assert resolved_mesh(parsed) == (2, 2, 2)


def test_structured_monkhorst_pack():
    parsed = ParsedCP2KInput(
        kpoint_sampling=CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(3, 3, 3),
        )
    )
    assert resolved_mesh(parsed) == (3, 3, 3)


def test_consistent_legacy_and_structured_mesh():
    parsed = ParsedCP2KInput(
        k_points=(3, 3, 3),
        kpoint_sampling=CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(3, 3, 3),
        ),
    )
    assert resolved_mesh(parsed) == (3, 3, 3)


def test_conflicting_meshes_rejected():
    parsed = ParsedCP2KInput(
        k_points=(2, 2, 2),
        kpoint_sampling=CP2KKPointConfig(
            scheme="MONKHORST-PACK",
            mesh=(3, 3, 3),
        ),
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Conflicting k-point mesh evidence",
    ):
        resolved_mesh(parsed)


@pytest.mark.parametrize("scheme", ["NONE", "GAMMA"])
def test_non_mesh_sampling_rejected(scheme):
    parsed = ParsedCP2KInput(
        kpoint_sampling=CP2KKPointConfig(scheme=scheme)
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Unsupported k-point sampling",
    ):
        resolved_mesh(parsed)


def test_macdonald_sampling_rejected():
    parsed = ParsedCP2KInput(
        kpoint_sampling=CP2KKPointConfig(
            scheme="MACDONALD",
            mesh=(3, 3, 3),
            shift=(0.25, 0.25, 0.25),
        )
    )
    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Unsupported k-point sampling",
    ):
        resolved_mesh(parsed)
