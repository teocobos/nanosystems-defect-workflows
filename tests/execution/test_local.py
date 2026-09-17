"""Tests for local NSDW execution."""

import sys

import pytest

from nsdw.execution import (
    ExecutionBackend,
    ExecutionRequest,
    ExecutionState,
    LocalExecutionError,
    LocalExecutor,
)


def test_local_executor_runs_command(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="local_test",
        command=(
            sys.executable,
            "-c",
            "print('NSDW execution works')",
        ),
        working_directory=tmp_path,
        stdout_file="stdout.txt",
        stderr_file="stderr.txt",
    )

    result = LocalExecutor().execute(
        request
    )

    assert (
        result.backend
        == ExecutionBackend.LOCAL
    )
    assert (
        result.state
        == ExecutionState.COMPLETED
    )
    assert result.return_code == 0
    assert result.host is not None
    assert result.command == request.command

    assert (
        tmp_path / "stdout.txt"
    ).read_text().strip() == (
        "NSDW execution works"
    )


def test_local_executor_records_timestamps(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="timestamp_test",
        command=(
            sys.executable,
            "-c",
            "pass",
        ),
        working_directory=tmp_path,
    )

    result = LocalExecutor().execute(
        request
    )

    assert result.started_at is not None
    assert result.completed_at is not None


def test_local_executor_failure_state(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="failure_test",
        command=(
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ),
        working_directory=tmp_path,
    )

    result = LocalExecutor().execute(
        request
    )

    assert (
        result.state
        == ExecutionState.FAILED
    )
    assert result.return_code == 7


def test_local_executor_environment(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="environment_test",
        command=(
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ['NSDW_TEST'])"
            ),
        ),
        working_directory=tmp_path,
        stdout_file="environment.txt",
        environment={
            "NSDW_TEST": "working",
        },
    )

    LocalExecutor().execute(request)

    assert (
        tmp_path / "environment.txt"
    ).read_text().strip() == "working"


def test_local_executor_stdin(
    tmp_path,
):
    input_path = tmp_path / "stdin.txt"

    input_path.write_text(
        "hello NSDW\n"
    )

    request = ExecutionRequest(
        calculation_id="stdin_test",
        command=(
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(sys.stdin.read().strip())"
            ),
        ),
        working_directory=tmp_path,
        stdin_file="stdin.txt",
        stdout_file="stdout.txt",
    )

    LocalExecutor().execute(request)

    assert (
        tmp_path / "stdout.txt"
    ).read_text().strip() == (
        "hello NSDW"
    )


def test_missing_working_directory_rejected(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="missing_directory",
        command=(
            sys.executable,
            "-c",
            "pass",
        ),
        working_directory=(
            tmp_path / "missing"
        ),
    )

    with pytest.raises(
        LocalExecutionError,
        match="Working directory",
    ):
        LocalExecutor().execute(request)


def test_missing_executable_rejected(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="missing_program",
        command=(
            "nsdw-program-that-does-not-exist",
        ),
        working_directory=tmp_path,
    )

    with pytest.raises(
        LocalExecutionError,
        match="Could not execute command",
    ):
        LocalExecutor().execute(request)
