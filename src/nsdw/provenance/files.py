"""Utilities for recording reproducible file provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nsdw.models.provenance import FileReference


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 digest of a file."""

    file_path = Path(path).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_file_reference(
    path: str | Path,
    *,
    format: str | None = None,
) -> FileReference:
    """Build an NSDW provenance reference for a file."""

    file_path = Path(path).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return FileReference(
        path=str(file_path),
        sha256=sha256_file(file_path),
        format=format,
        size_bytes=file_path.stat().st_size,
    )
