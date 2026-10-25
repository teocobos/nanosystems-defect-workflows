"""Exclusive publication of completed report directories."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path


RENAME_NOREPLACE = 1


def publish_report_directory(
    staging: str | Path,
    destination: str | Path,
) -> Path:
    """Publish a staged directory without replacing existing data.

    On Linux, uses renameat2 with RENAME_NOREPLACE.
    """
    source = Path(staging).resolve()
    target = Path(destination).absolute()

    if not source.is_dir():
        raise NotADirectoryError(source)

    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    if source.parent != target.parent:
        raise ValueError(
            "Staging and destination must share a parent directory"
        )

    if os.name != "posix":
        raise RuntimeError(
            "Exclusive atomic publication requires POSIX"
        )

    libc = ctypes.CDLL(None, use_errno=True)

    renameat2 = getattr(libc, "renameat2", None)

    if renameat2 is None:
        raise RuntimeError(
            "renameat2 is unavailable; refusing unsafe fallback"
        )

    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int

    at_fdcwd = -100

    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        RENAME_NOREPLACE,
    )

    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                error,
                os.strerror(error),
                str(target),
            )
        raise OSError(
            error,
            os.strerror(error),
            str(target),
        )

    return target
