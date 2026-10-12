"""Tests for exclusive HPC submission claims."""

import json

import pytest

from nsdw.execution.submission_claims import (
    SubmissionClaim,
    SubmissionClaimError,
    create_submission_claim,
)


def _claim():
    return SubmissionClaim(
        target="archer2",
        calculation_id="10-Ry",
        input_sha256="a" * 64,
    )


def test_creates_submission_claim(tmp_path):
    path = create_submission_claim(
        candidate_directory=tmp_path,
        claim=_claim(),
    )

    data = json.loads(path.read_text())

    assert data["state"] == "claimed"
    assert data["target"] == "archer2"
    assert data["calculation_id"] == "10-Ry"
    assert data["input_sha256"] == "a" * 64


def test_existing_claim_is_never_overwritten(tmp_path):
    first = create_submission_claim(
        candidate_directory=tmp_path,
        claim=_claim(),
    )
    original = first.read_bytes()

    with pytest.raises(
        SubmissionClaimError,
        match="already exists",
    ):
        create_submission_claim(
            candidate_directory=tmp_path,
            claim=_claim(),
        )

    assert first.read_bytes() == original


def test_partial_claim_blocks_recreation(tmp_path):
    path = tmp_path / "submission-claim.json"
    path.write_text("{partial")

    with pytest.raises(
        SubmissionClaimError,
        match="already exists",
    ):
        create_submission_claim(
            candidate_directory=tmp_path,
            claim=_claim(),
        )

    assert path.read_text() == "{partial"


@pytest.mark.parametrize(
    "filename",
    ["../outside.json", "subdir/claim.json", "", ".."],
)
def test_rejects_unsafe_filename(tmp_path, filename):
    with pytest.raises(
        SubmissionClaimError,
        match="simple filename",
    ):
        create_submission_claim(
            candidate_directory=tmp_path,
            claim=_claim(),
            filename=filename,
        )


def test_missing_candidate_directory_rejected(tmp_path):
    with pytest.raises(
        SubmissionClaimError,
        match="does not exist",
    ):
        create_submission_claim(
            candidate_directory=tmp_path / "missing",
            claim=_claim(),
        )


def test_inspect_missing_claim(tmp_path):
    from nsdw.execution.submission_claims import inspect_submission_claim

    assert inspect_submission_claim(
        candidate_directory=tmp_path
    ) is None


def test_inspect_valid_claim(tmp_path):
    from nsdw.execution.submission_claims import inspect_submission_claim

    create_submission_claim(
        candidate_directory=tmp_path,
        claim=_claim(),
    )

    claim = inspect_submission_claim(
        candidate_directory=tmp_path
    )

    assert claim is not None
    assert claim.calculation_id == "10-Ry"


def test_inspect_malformed_claim(tmp_path):
    from nsdw.execution.submission_claims import inspect_submission_claim

    (tmp_path / "submission-claim.json").write_text("{partial")

    with pytest.raises(
        SubmissionClaimError,
        match="malformed",
    ):
        inspect_submission_claim(
            candidate_directory=tmp_path
        )


def test_inspect_symlink_escape(tmp_path):
    from nsdw.execution.submission_claims import inspect_submission_claim

    candidate = tmp_path / "candidate"
    outside = tmp_path / "outside"

    candidate.mkdir()
    outside.mkdir()

    (candidate / "submission-claim.json").symlink_to(
        outside / "claim.json"
    )

    with pytest.raises(
        SubmissionClaimError,
        match="escapes candidate directory",
    ):
        inspect_submission_claim(
            candidate_directory=candidate
        )
