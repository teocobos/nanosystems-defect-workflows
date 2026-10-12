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
        match="symlink",
    ):
        _load(candidate)


def _claimed_record(directory):
    from nsdw.execution.submission_claims import (
        SubmissionClaim,
        create_submission_claim,
    )

    input_path = directory / "calculation.inp"
    input_path.write_text("&GLOBAL\n&END GLOBAL\n")

    digest = sha256_file(input_path)

    create_submission_claim(
        candidate_directory=directory,
        claim=SubmissionClaim(
            target="archer2",
            calculation_id="10-Ry",
            input_sha256=digest,
        ),
    )

    return SlurmSubmissionRecord(
        target="archer2",
        calculation_id="10-Ry",
        job_id="12345678",
        submitted_at="2026-10-12T19:00:00+01:00",
        input_file="calculation.inp",
        input_sha256=digest,
    )


def test_publish_submission_record(tmp_path):
    from nsdw.execution.submission_records import (
        publish_slurm_submission_record,
    )

    record = _claimed_record(tmp_path)

    path = publish_slurm_submission_record(
        candidate_directory=tmp_path,
        record=record,
    )

    assert path.is_file()
    assert _load(tmp_path) == record


def test_publish_never_overwrites_record(tmp_path):
    from nsdw.execution.submission_records import (
        publish_slurm_submission_record,
    )

    record = _claimed_record(tmp_path)

    path = publish_slurm_submission_record(
        candidate_directory=tmp_path,
        record=record,
    )

    original = path.read_bytes()

    with pytest.raises(
        SubmissionRecordError,
        match="already exists",
    ):
        publish_slurm_submission_record(
            candidate_directory=tmp_path,
            record=record,
        )

    assert path.read_bytes() == original


def test_publish_rejects_missing_claim(tmp_path):
    from nsdw.execution.submission_records import (
        publish_slurm_submission_record,
    )

    input_path = tmp_path / "calculation.inp"
    input_path.write_text("input")

    record = SlurmSubmissionRecord(
        target="archer2",
        calculation_id="10-Ry",
        job_id="12345678",
        submitted_at="2026-10-12T19:00:00+01:00",
        input_file="calculation.inp",
        input_sha256=sha256_file(input_path),
    )

    with pytest.raises(
        SubmissionRecordError,
        match="without an existing claim",
    ):
        publish_slurm_submission_record(
            candidate_directory=tmp_path,
            record=record,
        )


def test_publish_rejects_modified_input(tmp_path):
    from nsdw.execution.submission_records import (
        publish_slurm_submission_record,
    )

    record = _claimed_record(tmp_path)

    (tmp_path / "calculation.inp").write_text("modified")

    with pytest.raises(
        SubmissionRecordError,
        match="SHA-256 mismatch",
    ):
        publish_slurm_submission_record(
            candidate_directory=tmp_path,
            record=record,
        )


def test_publish_rejects_partial_existing_record(tmp_path):
    from nsdw.execution.submission_records import (
        publish_slurm_submission_record,
    )

    record = _claimed_record(tmp_path)

    path = tmp_path / "submission.json"
    path.write_text("{partial")

    with pytest.raises(
        SubmissionRecordError,
        match="already exists",
    ):
        publish_slurm_submission_record(
            candidate_directory=tmp_path,
            record=record,
        )

    assert path.read_text() == "{partial"


@pytest.mark.parametrize("dangling", [False, True])
def test_inspect_rejects_record_symlinks(tmp_path, dangling):
    from pathlib import Path

    target = tmp_path / "target.json"

    if not dangling:
        target.write_text("{}")

    (tmp_path / "submission.json").symlink_to(target)

    input_path = tmp_path / "calculation.inp"
    input_path.write_text("input")

    with pytest.raises(
        SubmissionRecordError,
        match="symlink",
    ):
        load_slurm_submission_record(
            record_path=Path("submission.json"),
            candidate_directory=tmp_path,
            calculation_id="10-Ry",
            target="archer2",
            input_file=Path("calculation.inp"),
        )
