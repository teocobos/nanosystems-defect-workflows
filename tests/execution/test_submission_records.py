"""Tests for persistent SLURM submission record validation."""

import pytest

from nsdw.execution.submission_records import (
    SlurmSubmissionRecord,
    SubmissionRecordError,
    load_slurm_submission_record,
    sha256_file,
)


def _load(directory):
    return load_slurm_submission_record(
        record_path="submission.json",
        candidate_directory=directory,
        calculation_id="10-Ry",
        target="archer2",
        input_file="calculation.inp",
    )


def _valid_record(directory):
    input_path = directory / "calculation.inp"
    input_path.write_text("&GLOBAL\n&END GLOBAL\n")

    record = SlurmSubmissionRecord(
        target="archer2",
        calculation_id="10-Ry",
        job_id="12345678",
        submitted_at="2026-10-12T18:00:00+01:00",
        input_file="calculation.inp",
        input_sha256=sha256_file(input_path),
    )

    (directory / "submission.json").write_text(
        record.model_dump_json(indent=2)
    )


def test_missing_record_returns_none(tmp_path):
    assert _load(tmp_path) is None


def test_valid_record_loads(tmp_path):
    _valid_record(tmp_path)

    record = _load(tmp_path)

    assert record is not None
    assert record.job_id == "12345678"


def test_modified_input_rejected(tmp_path):
    _valid_record(tmp_path)
    (tmp_path / "calculation.inp").write_text("modified")

    with pytest.raises(
        SubmissionRecordError,
        match="SHA-256 mismatch",
    ):
        _load(tmp_path)


def test_wrong_calculation_rejected(tmp_path):
    _valid_record(tmp_path)

    with pytest.raises(
        SubmissionRecordError,
        match="calculation ID mismatch",
    ):
        load_slurm_submission_record(
            record_path="submission.json",
            candidate_directory=tmp_path,
            calculation_id="20-Ry",
            target="archer2",
            input_file="calculation.inp",
        )


def test_wrong_target_rejected(tmp_path):
    _valid_record(tmp_path)

    with pytest.raises(
        SubmissionRecordError,
        match="HPC target mismatch",
    ):
        load_slurm_submission_record(
            record_path="submission.json",
            candidate_directory=tmp_path,
            calculation_id="10-Ry",
            target="other-cluster",
            input_file="calculation.inp",
        )


def test_malformed_record_rejected(tmp_path):
    (tmp_path / "submission.json").write_text("{invalid")

    with pytest.raises(
        SubmissionRecordError,
        match="Invalid SLURM submission record",
    ):
        _load(tmp_path)


def test_record_symlink_escape_rejected(tmp_path):
    candidate = tmp_path / "candidate"
    outside = tmp_path / "outside"

    candidate.mkdir()
    outside.mkdir()

    (candidate / "submission.json").symlink_to(
        outside / "submission.json"
    )

    with pytest.raises(
        SubmissionRecordError,
        match="escapes candidate directory",
    ):
        _load(candidate)
