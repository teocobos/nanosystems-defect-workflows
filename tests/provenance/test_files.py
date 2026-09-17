"""Tests for NSDW file-provenance utilities."""

import hashlib

import pytest

from nsdw.provenance.files import (
    build_file_reference,
    sha256_file,
)


def test_sha256_file(tmp_path):
    path = tmp_path / "example.inp"
    content = b"NSDW test content\n"

    path.write_bytes(content)

    expected = hashlib.sha256(
        content
    ).hexdigest()

    assert sha256_file(path) == expected


def test_build_file_reference(tmp_path):
    path = tmp_path / "calculation.inp"
    content = b"&GLOBAL\n&END GLOBAL\n"

    path.write_bytes(content)

    reference = build_file_reference(
        path,
        format="cp2k-input",
    )

    assert reference.path == str(
        path.resolve()
    )
    assert reference.sha256 == hashlib.sha256(
        content
    ).hexdigest()
    assert reference.size_bytes == len(content)
    assert reference.format == "cp2k-input"


def test_sha256_missing_file_rejected(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="File not found",
    ):
        sha256_file(
            tmp_path / "missing.inp"
        )


def test_reference_missing_file_rejected(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="File not found",
    ):
        build_file_reference(
            tmp_path / "missing.out"
        )
