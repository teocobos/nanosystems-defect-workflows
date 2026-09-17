from unittest.mock import Mock, patch

import pytest

from nsdw.execution.monitor import (
    SlurmMonitorConfig,
    SlurmMonitorError,
    wait_for_slurm_job,
)
from nsdw.execution.slurm import (
    SlurmJobState,
    SlurmStatusResult,
)


def _status(
    state: SlurmJobState,
    *,
    raw_state: str | None = None,
    exit_code: str | None = None,
) -> SlurmStatusResult:
    return SlurmStatusResult(
        job_id="123456",
        state=state,
        raw_state=raw_state or state.value.upper(),
        exit_code=exit_code,
    )


def test_monitor_returns_completed_job():
    executor = Mock()
    executor.query_status.return_value = _status(
        SlurmJobState.COMPLETED,
        raw_state="COMPLETED",
        exit_code="0:0",
    )

    result = wait_for_slurm_job(
        executor,
        "123456",
        config=SlurmMonitorConfig(
            poll_interval=0.01,
        ),
    )

    assert result.state == SlurmJobState.COMPLETED
    assert result.exit_code == "0:0"
    executor.query_status.assert_called_once_with(
        "123456"
    )


@pytest.mark.parametrize(
    "state",
    [
        SlurmJobState.FAILED,
        SlurmJobState.CANCELLED,
        SlurmJobState.TIMEOUT,
        SlurmJobState.OUT_OF_MEMORY,
    ],
)
def test_monitor_returns_terminal_failure_state(state):
    executor = Mock()
    executor.query_status.return_value = _status(state)

    result = wait_for_slurm_job(
        executor,
        "123456",
        config=SlurmMonitorConfig(
            poll_interval=0.01,
        ),
    )

    assert result.state == state


@patch("nsdw.execution.monitor.sleep")
def test_monitor_polls_until_completed(mock_sleep):
    executor = Mock()

    executor.query_status.side_effect = [
        _status(
            SlurmJobState.PENDING,
            raw_state="PENDING",
        ),
        _status(
            SlurmJobState.RUNNING,
            raw_state="RUNNING",
        ),
        _status(
            SlurmJobState.COMPLETED,
            raw_state="COMPLETED",
            exit_code="0:0",
        ),
    ]

    result = wait_for_slurm_job(
        executor,
        "123456",
        config=SlurmMonitorConfig(
            poll_interval=5.0,
        ),
    )

    assert result.state == SlurmJobState.COMPLETED
    assert executor.query_status.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(5.0)


@patch("nsdw.execution.monitor.sleep")
def test_monitor_tolerates_missing_accounting_record(
    mock_sleep,
):
    executor = Mock()

    executor.query_status.side_effect = [
        None,
        _status(
            SlurmJobState.COMPLETED,
            raw_state="COMPLETED",
            exit_code="0:0",
        ),
    ]

    result = wait_for_slurm_job(
        executor,
        "123456",
        config=SlurmMonitorConfig(
            poll_interval=5.0,
        ),
    )

    assert result.state == SlurmJobState.COMPLETED
    assert executor.query_status.call_count == 2
    mock_sleep.assert_called_once_with(5.0)


def test_monitor_config_rejects_invalid_poll_interval():
    with pytest.raises(
        ValueError,
        match="poll_interval",
    ):
        SlurmMonitorConfig(
            poll_interval=0,
        )


def test_monitor_config_rejects_invalid_timeout():
    with pytest.raises(
        ValueError,
        match="timeout",
    ):
        SlurmMonitorConfig(
            timeout=0,
        )


def test_monitor_rejects_empty_job_id():
    executor = Mock()

    with pytest.raises(
        ValueError,
        match="job ID",
    ):
        wait_for_slurm_job(
            executor,
            " ",
        )

@patch("nsdw.execution.monitor.sleep")
@patch("nsdw.execution.monitor.monotonic")
def test_monitor_times_out(
    mock_monotonic,
    mock_sleep,
):
    executor = Mock()

    executor.query_status.return_value = _status(
        SlurmJobState.RUNNING,
        raw_state="RUNNING",
    )

    mock_monotonic.side_effect = [
        0.0,
        5.0,
        10.0,
    ]

    with pytest.raises(
        SlurmMonitorError,
        match="Timed out",
    ):
        wait_for_slurm_job(
            executor,
            "123456",
            config=SlurmMonitorConfig(
                poll_interval=5.0,
                timeout=10.0,
            ),
        )

    assert executor.query_status.call_count == 2
    assert mock_sleep.call_count == 1