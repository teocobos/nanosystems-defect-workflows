"""Optional AiiDA execution support for NSDW."""

from __future__ import annotations

from importlib.util import find_spec

from typing import Protocol

from nsdw.execution.models import (
    ExecutionBackend,
    ExecutionResult,
    ExecutionState,
)


class AiiDAUnavailableError(RuntimeError):
    """Raised when an AiiDA operation is requested without AiiDA installed."""


class AiiDAExecutionError(RuntimeError):
    """Raised when NSDW cannot interpret an AiiDA execution."""


class AiiDAProcessLike(Protocol):
    """Minimal process interface required by the NSDW AiiDA adapter."""

    pk: int | None
    uuid: object
    process_state: object


def aiida_available() -> bool:
    """Return whether the optional AiiDA dependency is installed."""

    return find_spec("aiida") is not None


def require_aiida() -> None:
    """Require AiiDA to be installed for the requested operation."""

    if not aiida_available():
        raise AiiDAUnavailableError(
            "AiiDA support is not installed. "
            "Install NSDW with the 'aiida' extra: "
            "pip install -e '.[aiida]'"
        )


def map_aiida_process_state(
    process_state: str,
) -> ExecutionState:
    """Map an AiiDA process state onto the NSDW lifecycle."""

    state = process_state.strip().lower()

    mapping = {
        "created": ExecutionState.CREATED,
        "waiting": ExecutionState.SUBMITTED,
        "running": ExecutionState.RUNNING,
        "finished": ExecutionState.COMPLETED,
        "excepted": ExecutionState.FAILED,
        "killed": ExecutionState.CANCELLED,
    }

    try:
        return mapping[state]
    except KeyError as exc:
        raise AiiDAExecutionError(
            "Unsupported AiiDA process state: "
            f"{process_state!r}"
        ) from exc


def build_aiida_execution_result(
    *,
    calculation_id: str,
    process: AiiDAProcessLike,
    job_id: str | None = None,
    host: str | None = None,
) -> ExecutionResult:
    """Build an NSDW execution result from an AiiDA process."""

    raw_state = process.process_state

    if raw_state is None:
        raise AiiDAExecutionError(
            "AiiDA process has no process state"
        )

    state_value = getattr(
        raw_state,
        "value",
        raw_state,
    )

    state = map_aiida_process_state(
        str(state_value)
    )

    process_id = (
        str(process.pk)
        if process.pk is not None
        else None
    )

    process_uuid = str(process.uuid)

    if not process_uuid.strip():
        raise AiiDAExecutionError(
            "AiiDA process UUID cannot be empty"
        )

    return ExecutionResult(
        calculation_id=calculation_id,
        backend=ExecutionBackend.AIIDA,
        state=state,
        job_id=job_id,
        process_id=process_id,
        process_uuid=process_uuid,
        host=host,
    )