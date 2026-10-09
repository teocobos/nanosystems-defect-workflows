"""Regression tests for project-level file locking."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue

import pytest

from nsdw.project.locking import project_write_lock


def _lock_worker(root: str, events):
    """Acquire the project lock in a separate process."""
    with project_write_lock(root):
        events.put("acquired")


def test_lock_file_persists(tmp_path):
    with project_write_lock(tmp_path):
        assert (tmp_path / ".project.lock").is_file()

    assert (tmp_path / ".project.lock").is_file()


def test_lock_released_after_exception(tmp_path):
    with pytest.raises(RuntimeError, match="simulated"):
        with project_write_lock(tmp_path):
            raise RuntimeError("simulated")

    with project_write_lock(tmp_path):
        assert True


def test_second_process_waits_for_lock(tmp_path):
    context = mp.get_context("spawn")
    events = context.Queue()
    process = None

    try:
        with project_write_lock(tmp_path):
            process = context.Process(
                target=_lock_worker,
                args=(str(tmp_path), events),
            )
            process.start()

            # The child must not acquire the lock while
            # the parent holds it.
            with pytest.raises(queue.Empty):
                events.get(timeout=1.0)

            assert process.is_alive()

        # Releasing the parent lock allows the child through.
        assert events.get(timeout=10.0) == "acquired"

        process.join(timeout=10.0)
        assert process.exitcode == 0

    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

        events.close()
        events.join_thread()


def test_missing_project_directory_rejected(tmp_path):
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        with project_write_lock(missing):
            pass

    assert not missing.exists()
