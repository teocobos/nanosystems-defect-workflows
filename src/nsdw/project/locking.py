"""Advisory project-level locking for Linux/HPC execution."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator


@contextmanager
def project_write_lock(
    root: str | Path,
) -> Iterator[None]:
    """Hold an exclusive lock for a project metadata transaction.

    The lock file is persistent so competing processes coordinate
    through the same inode. It must not be deleted after use.
    """
    project_root = Path(root).expanduser().resolve()

    if not project_root.is_dir():
        raise FileNotFoundError(
            f"Project directory not found: {project_root}"
        )

    lock_path = project_root / ".project.lock"

    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
