"""Tests for NSDW project Git integration."""

from pathlib import Path
import subprocess

import pytest

from nsdw.project.git import (
    ProjectGitError,
    initialise_git_repository,
)


def test_initialise_git_repository_creates_repository(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-project"
    root.mkdir()

    result = initialise_git_repository(root)

    assert result == root.resolve()
    assert (root / ".git").is_dir()


def test_initialise_git_repository_uses_main_branch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-project"
    root.mkdir()

    initialise_git_repository(root)

    result = subprocess.run(
        [
            "git",
            "symbolic-ref",
            "--short",
            "HEAD",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "main"


def test_initialise_git_repository_does_not_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-project"
    root.mkdir()

    initialise_git_repository(root)

    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--verify",
            "HEAD",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_initialise_git_repository_rejects_existing_repository(
    tmp_path: Path,
) -> None:
    root = tmp_path / "igzo-project"
    root.mkdir()

    initialise_git_repository(root)

    with pytest.raises(
        ProjectGitError,
        match="Git repository already exists",
    ):
        initialise_git_repository(root)


def test_initialise_git_repository_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-project"

    with pytest.raises(
        ProjectGitError,
        match="Project directory does not exist",
    ):
        initialise_git_repository(root)
