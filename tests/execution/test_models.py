"""Tests for calculator-independent execution models."""

import pytest
from pydantic import ValidationError

from nsdw.execution import (
    ExecutionBackend,
    ExecutionRequest,
    ExecutionResult,
    ExecutionState,
)


def test_execution_request(tmp_path):
    request = ExecutionRequest(
        calculation_id="igzo_sp_001",
        command=(
            "cp2k.psmp",
            "-i",
            "calculation.inp",
            "-o",
            "calculation.out",
        ),
        working_directory=tmp_path,
        stdout_file=tmp_path / "calculation.out",
    )

    assert request.calculation_id == "igzo_sp_001"
    assert request.command[0] == "cp2k.psmp"
    assert request.working_directory == tmp_path


def test_execution_request_environment(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="test",
        command=("cp2k.psmp",),
        working_directory=tmp_path,
        environment={
            "OMP_NUM_THREADS": "1",
        },
    )

    assert (
        request.environment["OMP_NUM_THREADS"]
        == "1"
    )


def test_empty_calculation_id_rejected(
    tmp_path,
):
    with pytest.raises(
        ValidationError,
        match="calculation ID",
    ):
        ExecutionRequest(
            calculation_id="   ",
            command=("cp2k.psmp",),
            working_directory=tmp_path,
        )


def test_empty_command_rejected(
    tmp_path,
):
    with pytest.raises(
        ValidationError,
        match="command",
    ):
        ExecutionRequest(
            calculation_id="test",
            command=(),
            working_directory=tmp_path,
        )


def test_empty_command_argument_rejected(
    tmp_path,
):
    with pytest.raises(
        ValidationError,
        match="empty argument",
    ):
        ExecutionRequest(
            calculation_id="test",
            command=(
                "cp2k.psmp",
                "",
            ),
            working_directory=tmp_path,
        )


def test_execution_result():
    result = ExecutionResult(
        calculation_id="igzo_sp_001",
        backend=ExecutionBackend.LOCAL,
        state=ExecutionState.COMPLETED,
        return_code=0,
        started_at=(
            "2026-09-17T12:00:00+01:00"
        ),
        completed_at=(
            "2026-09-17T12:01:00+01:00"
        ),
    )

    assert (
        result.state
        == ExecutionState.COMPLETED
    )
    assert (
        result.backend
        == ExecutionBackend.LOCAL
    )
    assert result.return_code == 0


def test_execution_models_are_immutable(
    tmp_path,
):
    request = ExecutionRequest(
        calculation_id="test",
        command=("cp2k.psmp",),
        working_directory=tmp_path,
    )

    with pytest.raises(
        ValidationError
    ):
        request.calculation_id = "changed"
