"""Discovery and validation of the local CP2K runtime environment."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence


class CP2KEnvironmentError(RuntimeError):
    """Base error for CP2K runtime-environment discovery."""


class CP2KDataDirectoryNotFoundError(CP2KEnvironmentError):
    """Raised when a usable CP2K data directory cannot be found."""


class CP2KDataFileNotFoundError(CP2KEnvironmentError):
    """Raised when a required CP2K data file is unavailable."""


def _normalise_directory(path: str | Path) -> Path:
    """Return an expanded, absolute path without requiring it to exist."""
    return Path(path).expanduser().resolve()


def validate_cp2k_data_dir(
    directory: str | Path,
    *,
    required_files: Sequence[str] = (),
) -> Path:
    """Validate a CP2K data directory and optional required resources.

    Parameters
    ----------
    directory
        Candidate CP2K data directory.
    required_files
        Logical CP2K resource filenames that must be present.

    Returns
    -------
    pathlib.Path
        The validated absolute data-directory path.

    Raises
    ------
    CP2KDataDirectoryNotFoundError
        If the directory does not exist.
    CP2KDataFileNotFoundError
        If one or more required files are absent.
    """
    path = _normalise_directory(directory)

    if not path.is_dir():
        raise CP2KDataDirectoryNotFoundError(
            f"CP2K data directory does not exist: {path}"
        )

    missing = tuple(
        filename
        for filename in required_files
        if not (path / filename).is_file()
    )

    if missing:
        formatted = ", ".join(missing)
        raise CP2KDataFileNotFoundError(
            "CP2K data directory "
            f"{path} is missing required file(s): {formatted}"
        )

    return path


def _executable_relative_candidates(
    executable: str | Path,
) -> tuple[Path, ...]:
    """Return plausible CP2K data locations relative to an executable."""
    executable_text = str(executable)

    resolved_executable: str | None

    executable_path = Path(executable_text).expanduser()

    if executable_path.parent != Path("."):
        resolved_executable = str(executable_path.resolve())
    else:
        resolved_executable = shutil.which(executable_text)

    if resolved_executable is None:
        return ()

    binary = Path(resolved_executable).resolve()
    prefix = binary.parent.parent

    return (
        prefix / "share" / "cp2k" / "data",
        prefix / "share" / "cp2k",
        prefix / "data",
        binary.parent / "data",
    )


def resolve_cp2k_data_dir(
    *,
    explicit: str | Path | None = None,
    executable: str | Path = "cp2k.psmp",
    environment: Mapping[str, str] | None = None,
    required_files: Sequence[str] = (),
) -> Path:
    """Resolve the CP2K data directory using deterministic precedence.

    Discovery order:

    1. explicit ``explicit`` argument;
    2. ``CP2K_DATA_DIR`` in the supplied/current environment;
    3. locations relative to the resolved CP2K executable.

    No filesystem-wide search is performed.
    """
    env = os.environ if environment is None else environment

    if explicit is not None:
        return validate_cp2k_data_dir(
            explicit,
            required_files=required_files,
        )

    env_data_dir = env.get("CP2K_DATA_DIR")

    if env_data_dir:
        return validate_cp2k_data_dir(
            env_data_dir,
            required_files=required_files,
        )

    for candidate in _executable_relative_candidates(executable):
        if candidate.is_dir():
            try:
                return validate_cp2k_data_dir(
                    candidate,
                    required_files=required_files,
                )
            except CP2KDataFileNotFoundError:
                continue

    raise CP2KDataDirectoryNotFoundError(
        "Could not locate a usable CP2K data directory. "
        "Provide one explicitly, set CP2K_DATA_DIR, or use a "
        "CP2K installation with a discoverable data directory."
    )


def build_cp2k_environment(
    *,
    explicit_data_dir: str | Path | None = None,
    executable: str | Path = "cp2k.psmp",
    environment: Mapping[str, str] | None = None,
    required_files: Sequence[str] = (),
) -> dict[str, str]:
    """Build an execution environment containing a resolved CP2K data path."""
    merged = dict(environment or {})

    data_dir = resolve_cp2k_data_dir(
        explicit=explicit_data_dir,
        executable=executable,
        environment=merged if environment is not None else None,
        required_files=required_files,
    )

    merged["CP2K_DATA_DIR"] = str(data_dir)

    return merged
