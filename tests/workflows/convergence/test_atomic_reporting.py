"""Tests for exclusive atomic report publication."""

from pathlib import Path

import pytest

from nsdw.workflows.convergence.atomic_reporting import (
    publish_report_directory,
)


def test_publishes_complete_directory(tmp_path):
    staging = tmp_path / ".report-staging"
    staging.mkdir()
    (staging / "data.json").write_text('{"ok": true}')

    destination = tmp_path / "report"

    result = publish_report_directory(
        staging, destination
    )

    assert result == destination
    assert destination.is_dir()
    assert (destination / "data.json").read_text() == (
        '{"ok": true}'
    )
    assert not staging.exists()


def test_refuses_existing_destination(tmp_path):
    staging = tmp_path / ".report-staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new")

    destination = tmp_path / "report"
    destination.mkdir()
    (destination / "original.txt").write_text("original")

    with pytest.raises(FileExistsError):
        publish_report_directory(staging, destination)

    assert staging.is_dir()
    assert (destination / "original.txt").read_text() == (
        "original"
    )
    assert not (destination / "new.txt").exists()


def test_rejects_missing_staging(tmp_path):
    with pytest.raises(NotADirectoryError):
        publish_report_directory(
            tmp_path / "missing",
            tmp_path / "report",
        )


def test_rejects_different_parent(tmp_path):
    staging = tmp_path / "a" / "staging"
    staging.parent.mkdir()
    staging.mkdir()

    target_parent = tmp_path / "b"
    target_parent.mkdir()

    with pytest.raises(ValueError):
        publish_report_directory(
            staging,
            target_parent / "report",
        )
