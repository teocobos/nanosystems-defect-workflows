"""Exclusive, persistent claims for HPC submission attempts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class SubmissionClaimError(RuntimeError):
    """A submission claim cannot be safely created."""


class SubmissionClaim(BaseModel):
    """Persistent evidence that submission may have been attempted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    state: Literal["claimed"] = "claimed"

    target: str = Field(min_length=1)
    calculation_id: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def create_submission_claim(
    *,
    candidate_directory: Path,
    claim: SubmissionClaim,
    filename: str = "submission-claim.json",
) -> Path:
    """Create an exclusive claim without overwriting existing evidence."""

    directory = Path(candidate_directory).resolve()

    if not directory.is_dir():
        raise SubmissionClaimError(
            f"Candidate directory does not exist: {directory}"
        )

    if (
        not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise SubmissionClaimError(
            "Claim filename must be a simple filename."
        )

    destination = directory / filename

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    payload = (
        json.dumps(claim.model_dump(), indent=2) + "\n"
    ).encode("utf-8")

    try:
        fd = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise SubmissionClaimError(
            "Submission claim already exists; refusing duplicate claim."
        ) from exc
    except OSError as exc:
        raise SubmissionClaimError(
            f"Cannot create submission claim: {exc}"
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
        raise SubmissionClaimError(
            "Submission claim persistence failed; "
            "manual recovery required."
        ) from exc

    return destination


def inspect_submission_claim(
    *,
    candidate_directory: Path,
    filename: str = "submission-claim.json",
) -> SubmissionClaim | TokenisedSubmissionClaim | None:
    """Read a claim without modifying it.

    Missing claims return None. Malformed or unsafe claims raise
    SubmissionClaimError and must block submission.
    """

    directory = Path(candidate_directory).resolve()

    if (
        not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise SubmissionClaimError(
            "Claim filename must be a simple filename."
        )

    raw_path = directory / filename

    if raw_path.is_symlink():
        raise SubmissionClaimError(
            "Submission claim symlink is not trusted."
        )

    path = raw_path.resolve()

    if not path.is_relative_to(directory):
        raise SubmissionClaimError(
            "Claim path escapes candidate directory."
        )

    if not path.exists():
        return None

    if not path.is_file():
        raise SubmissionClaimError(
            "Submission claim is not a regular file."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise ValueError("Claim must be a JSON object.")

        version = data.get("schema_version")

        if version == 1 and type(version) is int:
            return SubmissionClaim.model_validate(data)

        if version == 2 and type(version) is int:
            return TokenisedSubmissionClaim.model_validate(data)

        raise ValueError("Unsupported submission claim schema.")

    except (ValueError, OSError) as exc:
        raise SubmissionClaimError(
            "Submission claim is malformed or unreadable."
        ) from exc


class TokenisedSubmissionClaim(BaseModel):
    """Version-2 claim with a unique scheduler submission token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    state: Literal["claimed"] = "claimed"

    target: str = Field(min_length=1)
    calculation_id: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission_token: str = Field(
        pattern=r"^[0-9a-f]{32}$"
    )


def generate_submission_token() -> str:
    """Generate a cryptographically random submission identifier."""
    import secrets

    return secrets.token_hex(16)


def slurm_name_for_token(token: str) -> str:
    """Return the deterministic SLURM name for a submission token."""
    import re

    if re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError(
            "Submission token must be 32 lowercase hex characters."
        )

    return f"nsdw-{token}"
