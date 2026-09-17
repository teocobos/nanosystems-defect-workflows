from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep

from nsdw.execution.slurm import (
    SlurmExecutor,
    SlurmJobState,
    SlurmStatusResult,
)


class SlurmMonitorError(RuntimeError):
    """Raised when monitoring a SLURM job fails."""


@dataclass(frozen=True)
class SlurmMonitorConfig:
    """Configuration for SLURM job monitoring."""

    poll_interval: float = 30.0
    timeout: float | None = None

    def __post_init__(self) -> None:
        if self.poll_interval <= 0:
            raise ValueError(
                "poll_interval must be greater than zero"
            )

        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero"
            )


_TERMINAL_STATES = {
    SlurmJobState.COMPLETED,
    SlurmJobState.FAILED,
    SlurmJobState.CANCELLED,
    SlurmJobState.TIMEOUT,
    SlurmJobState.OUT_OF_MEMORY,
}


def wait_for_slurm_job(
    executor: SlurmExecutor,
    job_id: str,
    *,
    config: SlurmMonitorConfig | None = None,
) -> SlurmStatusResult:
    """Wait until a SLURM job reaches a terminal state."""

    if not job_id.strip():
        raise ValueError(
            "SLURM job ID cannot be empty"
        )

    monitor_config = config or SlurmMonitorConfig()

    started = monotonic()

    while True:
        status = executor.query_status(job_id)

        if status is not None:
            if status.state in _TERMINAL_STATES:
                return status

        if (
            monitor_config.timeout is not None
            and monotonic() - started
            >= monitor_config.timeout
        ):
            raise SlurmMonitorError(
                "Timed out while waiting for SLURM job "
                f"{job_id}"
            )

        sleep(monitor_config.poll_interval)
