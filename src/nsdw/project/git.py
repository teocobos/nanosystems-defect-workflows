"""Git integration for NSDW material projects."""

from pathlib import Path
import shutil
import subprocess


class ProjectGitError(RuntimeError):
    """Raised when Git operations for an NSDW project fail."""


def initialise_git_repository(
    root: Path,
) -> Path:
    """Initialise a Git repository for an NSDW project.

    The repository is initialised with ``main`` as its initial branch.
    This function does not stage, commit, configure a remote, or push
    project content.
    """

    root = root.expanduser().resolve()

    if not root.is_dir():
        raise ProjectGitError(
            f"Project directory does not exist: {root}"
        )

    if shutil.which("git") is None:
        raise ProjectGitError(
            "Git is not available on PATH."
        )

    git_directory = root / ".git"

    if git_directory.exists():
        raise ProjectGitError(
            f"Git repository already exists: {root}"
        )

    try:
        subprocess.run(
            [
                "git",
                "init",
                "-b",
                "main",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = (
            exc.stderr.strip()
            or exc.stdout.strip()
            or "git init failed"
        )

        raise ProjectGitError(
            f"Could not initialise Git repository: {message}"
        ) from exc

    return root