from unittest.mock import patch

import pytest

from dataclasses import dataclass
from enum import StrEnum

from nsdw.execution.aiida import (
    AiiDAExecutionError,
    AiiDAUnavailableError,
    aiida_available,
    build_aiida_execution_result,
    map_aiida_process_state,
    require_aiida,
)
from nsdw.execution.models import ExecutionState


def test_aiida_available_when_package_exists() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=object(),
    ):
        assert aiida_available()


def test_aiida_unavailable_when_package_missing() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=None,
    ):
        assert not aiida_available()


def test_require_aiida_accepts_available_installation() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=object(),
    ):
        require_aiida()


def test_require_aiida_rejects_missing_installation() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=None,
    ):
        with pytest.raises(
            AiiDAUnavailableError,
            match="AiiDA support is not installed",
        ):
            require_aiida()


@pytest.mark.parametrize(
    ("aiida_state", "expected"),
    [
        ("created", ExecutionState.CREATED),
        ("waiting", ExecutionState.SUBMITTED),
        ("running", ExecutionState.RUNNING),
        ("finished", ExecutionState.COMPLETED),
        ("excepted", ExecutionState.FAILED),
        ("killed", ExecutionState.CANCELLED),
    ],
)
def test_map_aiida_process_state(
    aiida_state: str,
    expected: ExecutionState,
) -> None:
    assert map_aiida_process_state(aiida_state) == expected


def test_map_aiida_process_state_normalises_case() -> None:
    assert (
        map_aiida_process_state("RUNNING")
        == ExecutionState.RUNNING
    )


def test_map_aiida_process_state_rejects_unknown_state() -> None:
    with pytest.raises(
        AiiDAExecutionError,
        match="Unsupported AiiDA process state",
    ):
        map_aiida_process_state("paused")


class FakeProcessState(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class FakeAiiDAProcess:
    pk: int | None
    uuid: str
    process_state: FakeProcessState | str | None


def test_build_aiida_execution_result() -> None:
    process = FakeAiiDAProcess(
        pk=42,
        uuid="550e8400-e29b-41d4-a716-446655440000",
        process_state=FakeProcessState.RUNNING,
    )

    result = build_aiida_execution_result(
        calculation_id="igzo-aiida-001",
        process=process,
        job_id="987654",
        host="lumi",
    )

    assert result.calculation_id == "igzo-aiida-001"
    assert result.backend.value == "aiida"
    assert result.state == ExecutionState.RUNNING

    assert result.job_id == "987654"
    assert result.process_id == "42"
    assert (
        result.process_uuid
        == "550e8400-e29b-41d4-a716-446655440000"
    )

    assert result.host == "lumi"


def test_build_aiida_execution_result_without_pk() -> None:
    process = FakeAiiDAProcess(
        pk=None,
        uuid="550e8400-e29b-41d4-a716-446655440001",
        process_state=FakeProcessState.FINISHED,
    )

    result = build_aiida_execution_result(
        calculation_id="aiida-test",
        process=process,
    )

    assert result.process_id is None
    assert result.state == ExecutionState.COMPLETED


def test_build_aiida_execution_result_rejects_missing_state() -> None:
    process = FakeAiiDAProcess(
        pk=1,
        uuid="550e8400-e29b-41d4-a716-446655440002",
        process_state=None,
    )

    with pytest.raises(
        AiiDAExecutionError,
        match="no process state",
    ):
        build_aiida_execution_result(
            calculation_id="aiida-test",
            process=process,
        )


def test_build_aiida_execution_result_rejects_empty_uuid() -> None:
    process = FakeAiiDAProcess(
        pk=1,
        uuid="",
        process_state="running",
    )

    with pytest.raises(
        AiiDAExecutionError,
        match="UUID cannot be empty",
    ):
        build_aiida_execution_result(
            calculation_id="aiida-test",
            process=process,
        )