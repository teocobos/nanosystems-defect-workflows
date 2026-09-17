"""Local process execution for NSDW."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
import socket

from nsdw.execution.models import (
    ExecutionBackend,
    ExecutionRequest,
    ExecutionResult,
    ExecutionState,
)


class LocalExecutionError(RuntimeError):
    """Raised when a local execution cannot be started."""


def _timestamp() -> str:
    """Return the current local time as ISO 8601."""

    return datetime.now().astimezone().isoformat()


class LocalExecutor:
    """Execute an NSDW calculation on the local machine."""

    def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        """Execute a request and wait for it to finish."""

        working_directory = (
            Path(request.working_directory)
            .expanduser()
            .resolve()
        )

        if not working_directory.is_dir():
            raise LocalExecutionError(
                "Working directory does not exist: "
                f"{working_directory}"
            )

        environment = os.environ.copy()
        environment.update(request.environment)

        stdout_handle = None
        stderr_handle = None

        try:
            if request.stdout_file is not None:
                stdout_path = self._resolve_output_path(
                    working_directory,
                    request.stdout_file,
                )

                stdout_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                stdout_handle = stdout_path.open(
                    "w",
                    encoding="utf-8",
                )

            if request.stderr_file is not None:
                stderr_path = self._resolve_output_path(
                    working_directory,
                    request.stderr_file,
                )

                stderr_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                stderr_handle = stderr_path.open(
                    "w",
                    encoding="utf-8",
                )

            stdin_handle = None

            try:
                if request.stdin_file is not None:
                    stdin_path = self._resolve_input_path(
                        working_directory,
                        request.stdin_file,
                    )

                    stdin_handle = stdin_path.open(
                        "r",
                        encoding="utf-8",
                    )

                started_at = _timestamp()

                completed = subprocess.run(
                    list(request.command),
                    cwd=working_directory,
                    env=environment,
                    stdin=stdin_handle,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )

                completed_at = _timestamp()

            except OSError as exc:
                raise LocalExecutionError(
                    "Could not execute command: "
                    f"{request.command[0]}"
                ) from exc

            finally:
                if stdin_handle is not None:
                    stdin_handle.close()

        finally:
            if stdout_handle is not None:
                stdout_handle.close()

            if stderr_handle is not None:
                stderr_handle.close()

        state = (
            ExecutionState.COMPLETED
            if completed.returncode == 0
            else ExecutionState.FAILED
        )

        return ExecutionResult(
            calculation_id=request.calculation_id,
            backend=ExecutionBackend.LOCAL,
            state=state,
            return_code=completed.returncode,
            host=socket.gethostname(),
            command=request.command,
            started_at=started_at,
            completed_at=completed_at,
            stdout_file=(
                self._resolve_output_path(
                    working_directory,
                    request.stdout_file,
                )
                if request.stdout_file is not None
                else None
            ),
            stderr_file=(
                self._resolve_output_path(
                    working_directory,
                    request.stderr_file,
                )
                if request.stderr_file is not None
                else None
            ),
        )

    @staticmethod
    def _resolve_input_path(
        working_directory: Path,
        path: Path,
    ) -> Path:
        if path.is_absolute():
            resolved = path
        else:
            resolved = working_directory / path

        if not resolved.is_file():
            raise LocalExecutionError(
                f"Input file does not exist: {resolved}"
            )

        return resolved

    @staticmethod
    def _resolve_output_path(
        working_directory: Path,
        path: Path,
    ) -> Path:
        if path.is_absolute():
            return path

        return working_directory / path
