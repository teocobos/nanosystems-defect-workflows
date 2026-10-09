"""Adversarial tests for CP2K final-energy provenance."""

from pathlib import Path

import pytest

from nsdw.workflows.convergence.cp2k_output_evidence import (
    validate_cp2k_single_point_output,
)
from nsdw.workflows.convergence.evidence_validation import (
    ConvergenceEvidenceValidationError,
)


FIXTURES = Path("tests/calculators/cp2k/fixtures")


@pytest.mark.parametrize(
    "name",
    [
        "energy_complete.out",
        "igzo_ordered_003_sp_real.out",
    ],
)
def test_accepts_existing_cp2k_fixtures(name):
    energy = validate_cp2k_single_point_output(
        FIXTURES / name
    )
    assert energy < 0


def make_output(tmp_path, events):
    path = tmp_path / "calculation.out"
    header = [
        "CP2K| version string: CP2K version 2025.2",
        "GLOBAL| Project name test",
        "GLOBAL| Run type ENERGY",
        "DFT| Multiplicity 1",
        "DFT| Charge 0",
    ]
    path.write_text(
        "\n".join(header + events) + "\n",
        encoding="utf-8",
    )
    return path


SUCCESS = "*** SCF run converged in 3 steps ***"
FAILURE = "*** SCF run NOT converged ***"
ENERGY = (
    "ENERGY| Total FORCE_EVAL ( QS ) "
    "energy [a.u.]: -101.234567890000"
)
END = "PROGRAM ENDED AT 2026-10-09"


@pytest.mark.parametrize(
    "events",
    [
        [SUCCESS, END],
        [ENERGY, END],
        [SUCCESS, ENERGY],
        [ENERGY, SUCCESS, END],
        [SUCCESS, END, ENERGY],
        [SUCCESS, ENERGY, END, END],
        [SUCCESS, ENERGY, SUCCESS, END],
        [SUCCESS, ENERGY, FAILURE, END],
        [FAILURE, SUCCESS, ENERGY, END],
        [SUCCESS, ENERGY, ENERGY, END],
        [SUCCESS, ENERGY, END, SUCCESS],
    ],
)
def test_rejects_ambiguous_output_histories(
    tmp_path,
    events,
):
    path = make_output(tmp_path, events)

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="Ambiguous CP2K single-point event sequence",
    ):
        validate_cp2k_single_point_output(path)


def test_accepts_unambiguous_single_point(tmp_path):
    path = make_output(
        tmp_path,
        [SUCCESS, ENERGY, END],
    )

    assert validate_cp2k_single_point_output(
        path
    ) == pytest.approx(-101.23456789)


def test_rejects_nonfinite_energy(tmp_path):
    path = make_output(
        tmp_path,
        [
            SUCCESS,
            ENERGY.replace(
                "-101.234567890000",
                "1e309",
            ),
            END,
        ],
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="not finite",
    ):
        validate_cp2k_single_point_output(path)


def test_rejects_wrong_run_type(tmp_path):
    path = make_output(
        tmp_path,
        [SUCCESS, ENERGY, END],
    )

    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "Run type ENERGY",
            "Run type GEO_OPT",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConvergenceEvidenceValidationError,
        match="ENERGY run type",
    ):
        validate_cp2k_single_point_output(path)
