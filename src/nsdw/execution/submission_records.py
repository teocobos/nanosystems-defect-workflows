"""Read-only validation of persistent SLURM submission records."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SlurmSubmissionRecord(BaseModel):
    """Persisted identity of a submitted SLURM calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    backend: Literal["slurm"] = "slurm"

    target: str = Field(min_length=1)
    calculation_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    submitted_at: str = Field(min_length=1)

    input_file: str = Field(min_length=1)
    input_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )


class SubmissionRecordError(ValueError):
    """A submission record cannot be trusted."""


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a calculation input."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_slurm_submission_record(
    *,
    record_path: Path,
    candidate_directory: Path,
    calculation_id: str,
    target: str,
    input_file: Path,
) -> SlurmSubmissionRecord | None:
    """Load a record and verify its calculation identity.

    A missing record returns None, which must never be
    interpreted as authorisation to submit a job.
    """

    directory = Path(candidate_directory).resolve()
    record_path = Path(record_path)
    input_file = Path(input_file)

    if record_path.is_absolute() or input_file.is_absolute():
        raise SubmissionRecordError(
            "Record and input paths must be relative."
        )

    raw_record = directory / record_path

    if raw_record.is_symlink():
        raise SubmissionRecordError(
            "Submission record symlink is not trusted."
        )

    record = raw_record.resolve()
    input_path = (directory / input_file).resolve()

    for path in (record, input_path):
        if not path.is_relative_to(directory):
            raise SubmissionRecordError(
                "Submission record or input escapes candidate directory."
            )

    if not record.exists():
        return None

    if not record.is_file():
        raise SubmissionRecordError(
            "Submission record is not a regular file."
        )

    try:
        data = SlurmSubmissionRecord.model_validate_json(
            record.read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as exc:
        raise SubmissionRecordError(
            "Invalid SLURM submission record."
        ) from exc

    if data.calculation_id != calculation_id:
        raise SubmissionRecordError(
            "Submission calculation ID mismatch."
        )

    if data.target != target:
        raise SubmissionRecordError(
            "Submission HPC target mismatch."
        )

    if Path(data.input_file) != input_file:
        raise SubmissionRecordError(
            "Submission input filename mismatch."
        )

    try:
        actual_digest = sha256_file(input_path)
    except OSError as exc:
        raise SubmissionRecordError(
            "Cannot read submission input file."
        ) from exc

    if data.input_sha256 != actual_digest:
        raise SubmissionRecordError(
            "Submission input SHA-256 mismatch."
        )

    return data


def publish_slurm_submission_record(
    *,
    candidate_directory: Path,
    record: SlurmSubmissionRecord,
    filename: str = "submission.json",
) -> Path:
    """Exclusively publish an accepted SLURM job record.

    Requires an existing, matching submission claim.
    Never overwrites existing submission evidence.
    """
    import os

    from nsdw.execution.submission_claims import (
        SubmissionClaimError,
        inspect_submission_claim,
    )

    directory = Path(candidate_directory).resolve()

    if not directory.is_dir():
        raise SubmissionRecordError(
            "Candidate directory does not exist."
        )

    if (
        not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise SubmissionRecordError(
            "Record filename must be a simple filename."
        )

    if (
        Path(record.input_file).is_absolute()
        or Path(record.input_file).name != record.input_file
    ):
        raise SubmissionRecordError(
            "Record input filename must be a simple filename."
        )

    input_path = (directory / record.input_file).resolve()

    if not input_path.is_relative_to(directory):
        raise SubmissionRecordError(
            "Submission input escapes candidate directory."
        )

    try:
        claim = inspect_submission_claim(
            candidate_directory=directory,
        )
    except SubmissionClaimError as exc:
        raise SubmissionRecordError(
            f"Untrusted submission claim: {exc}"
        ) from exc

    if claim is None:
        raise SubmissionRecordError(
            "Cannot publish submission without an existing claim."
        )

    if (
        claim.target != record.target
        or claim.calculation_id != record.calculation_id
        or claim.input_sha256 != record.input_sha256
    ):
        raise SubmissionRecordError(
            "Submission claim and record identity mismatch."
        )

    try:
        actual_hash = sha256_file(input_path)
    except OSError as exc:
        raise SubmissionRecordError(
            "Cannot read submission input file."
        ) from exc

    if actual_hash != record.input_sha256:
        raise SubmissionRecordError(
            "Submission input SHA-256 mismatch."
        )

    destination = directory / filename
    payload = (record.model_dump_json(indent=2) + "\n").encode("utf-8")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise SubmissionRecordError(
            "Submission record already exists; refusing overwrite."
        ) from exc
    except OSError as exc:
        raise SubmissionRecordError(
            f"Cannot create submission record: {exc}"
        ) from exc

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    except OSError as exc:
        raise SubmissionRecordError(
            "Submission record persistence failed; "
            "manual reconciliation required."
        ) from exc

    return destination
