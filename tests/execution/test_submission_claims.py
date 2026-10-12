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
        match="symlink",
    ):
        inspect_submission_claim(
            candidate_directory=candidate
        )


def test_claim_persistence_syncs_file_and_directory(
    monkeypatch, tmp_path
):
    import os

    from nsdw.execution.submission_claims import (
        create_submission_claim,
    )

    real_fsync = os.fsync
    synced = []

    def tracking_fsync(fd):
        synced.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracking_fsync)

    create_submission_claim(
        candidate_directory=tmp_path,
        claim=_claim(),
    )

    import stat

    assert len(synced) >= 2
    assert stat.S_ISREG(synced[0])
    assert stat.S_ISDIR(synced[1])


@pytest.mark.parametrize("dangling", [False, True])
def test_inspect_rejects_claim_symlinks(tmp_path, dangling):
    from nsdw.execution.submission_claims import (
        inspect_submission_claim,
    )

    target = tmp_path / "target.json"

    if not dangling:
        target.write_text("{}")

    (tmp_path / "submission-claim.json").symlink_to(target)

    with pytest.raises(
        SubmissionClaimError,
        match="symlink",
    ):
        inspect_submission_claim(
            candidate_directory=tmp_path,
        )


def test_generate_submission_tokens_are_unique():
    from nsdw.execution.submission_claims import (
        generate_submission_token,
    )

    tokens = {
        generate_submission_token()
        for _ in range(100)
    }

    assert len(tokens) == 100
    assert all(len(token) == 32 for token in tokens)


def test_tokenised_submission_claim_validates_token():
    from pydantic import ValidationError

    from nsdw.execution.submission_claims import (
        TokenisedSubmissionClaim,
    )

    claim = TokenisedSubmissionClaim(
        target="archer2",
        calculation_id="10-Ry",
        input_sha256="a" * 64,
        submission_token="b" * 32,
    )

    assert claim.schema_version == 2
    assert claim.submission_token == "b" * 32

    with pytest.raises(ValidationError):
        TokenisedSubmissionClaim(
            target="archer2",
            calculation_id="10-Ry",
            input_sha256="a" * 64,
            submission_token="not-a-valid-token",
        )


def test_slurm_name_for_submission_token():
    from nsdw.execution.submission_claims import (
        slurm_name_for_token,
    )

    token = "a" * 32

    assert slurm_name_for_token(token) == f"nsdw-{token}"


@pytest.mark.parametrize(
    "token",
    ["", "abc", "G" * 32, "a" * 31, "a" * 33],
)
def test_slurm_name_rejects_invalid_token(token):
    from nsdw.execution.submission_claims import (
        slurm_name_for_token,
    )

    with pytest.raises(ValueError, match="32 lowercase hex"):
        slurm_name_for_token(token)


def test_inspect_tokenised_submission_claim(tmp_path):
    from nsdw.execution.submission_claims import (
        TokenisedSubmissionClaim,
        inspect_submission_claim,
    )

    claim = TokenisedSubmissionClaim(
        target="archer2",
        calculation_id="10-Ry",
        input_sha256="a" * 64,
        submission_token="b" * 32,
    )

    (tmp_path / "submission-claim.json").write_text(
        claim.model_dump_json(),
        encoding="utf-8",
    )

    inspected = inspect_submission_claim(
        candidate_directory=tmp_path,
    )

    assert isinstance(inspected, TokenisedSubmissionClaim)
    assert inspected.schema_version == 2
    assert inspected.submission_token == "b" * 32


@pytest.mark.parametrize(
    "schema_version",
    [0, 3, "2", None, True],
)
def test_inspect_rejects_unsupported_claim_schema(
    tmp_path, schema_version
):
    import json

    from nsdw.execution.submission_claims import (
        inspect_submission_claim,
    )

    payload = {
        "schema_version": schema_version,
        "state": "claimed",
        "target": "archer2",
        "calculation_id": "10-Ry",
        "input_sha256": "a" * 64,
    }

    (tmp_path / "submission-claim.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        SubmissionClaimError,
        match="malformed",
    ):
        inspect_submission_claim(
            candidate_directory=tmp_path,
        )


def test_inspect_preserves_legacy_claim_compatibility(tmp_path):
    from nsdw.execution.submission_claims import (
        SubmissionClaim,
        create_submission_claim,
        inspect_submission_claim,
    )

    original = _claim()

    create_submission_claim(
        candidate_directory=tmp_path,
        claim=original,
    )

    inspected = inspect_submission_claim(
        candidate_directory=tmp_path,
    )

    assert isinstance(inspected, SubmissionClaim)
    assert inspected == original
