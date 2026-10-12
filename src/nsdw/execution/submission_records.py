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

    record = (directory / record_path).resolve()
    input_path = (directory / input_file).resolve()

    for path in (record, input_path):
        if not path.is_relative_to(directory):
            raise SubmissionRecordError(
                "Submission record or input escapes candidate directory."
            )

    if not record.is_file():
        return None

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
