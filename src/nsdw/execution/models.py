"""Calculator-independent execution models for NSDW."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class ExecutionState(StrEnum):
    """Lifecycle state of an NSDW execution."""

    CREATED = "created"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionBackend(StrEnum):
    """Mechanism used to execute a calculation."""

    LOCAL = "local"
    SLURM = "slurm"


class ExecutionRequest(BaseModel):
    """Description of a calculation to execute."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    calculation_id: str
    command: tuple[str, ...]
    working_directory: Path

    stdin_file: Path | None = None
    stdout_file: Path | None = None
    stderr_file: Path | None = None

    environment: dict[str, str] = Field(
        default_factory=dict,
    )

    @field_validator("calculation_id")
    @classmethod
    def calculation_id_not_empty(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "calculation ID must not be empty"
            )

        return value

    @field_validator("command")
    @classmethod
    def command_not_empty(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError(
                "execution command must not be empty"
            )

        if any(
            not part.strip()
            for part in value
        ):
            raise ValueError(
                "execution command contains "
                "an empty argument"
            )

        return value


class ExecutionResult(BaseModel):
    """Result of executing or submitting a calculation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    calculation_id: str
    backend: ExecutionBackend
    state: ExecutionState

    return_code: int | None = None
    job_id: str | None = None

    host: str | None = None
    command: tuple[str, ...] = ()

    started_at: str | None = None
    completed_at: str | None = None

    stdout_file: Path | None = None
    stderr_file: Path | None = None