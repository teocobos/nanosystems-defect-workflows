"""Generic HPC execution-profile models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from nsdw.execution.models import ExecutionBackend


class HPCScheduler(StrEnum):
    """Schedulers supported by NSDW HPC profiles."""

    SLURM = "slurm"


class HPCProfileResources(BaseModel):
    """Default compute resources associated with an HPC profile."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    machines: int = Field(
        default=1,
        ge=1,
    )

    mpi_per_machine: int = Field(
        default=1,
        ge=1,
    )

    omp_threads: int = Field(
        default=1,
        ge=1,
    )

    walltime_seconds: int = Field(
        default=600,
        ge=1,
    )

    queue: str | None = None
    account: str | None = None

    @field_validator(
        "queue",
        "account",
    )
    @classmethod
    def normalise_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalise optional scheduler strings."""

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        return value


class HPCProfile(BaseModel):
    """Reusable user-level configuration for an HPC execution target."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str = Field(
        min_length=1,
    )

    backend: ExecutionBackend = ExecutionBackend.AIIDA

    scheduler: HPCScheduler = HPCScheduler.SLURM

    aiida_profile: str | None = None

    computer: str = Field(
        min_length=1,
    )

    code: str = Field(
        min_length=1,
    )

    resources: HPCProfileResources = Field(
        default_factory=HPCProfileResources,
    )

    @field_validator(
        "name",
        "computer",
        "code",
    )
    @classmethod
    def required_strings_not_empty(
        cls,
        value: str,
    ) -> str:
        """Reject empty or whitespace-only required strings."""

        value = value.strip()

        if not value:
            raise ValueError(
                "HPC profile values must not be empty"
            )

        return value

    @field_validator("aiida_profile")
    @classmethod
    def normalise_aiida_profile(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalise the optional AiiDA profile name."""

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        return value
