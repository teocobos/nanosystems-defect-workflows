"""Reproducibility and execution provenance for NSDW scientific results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nsdw.models.quantity import Quantity


class SchedulerType(StrEnum):
    """Supported workload schedulers."""

    LOCAL = "local"
    SLURM = "slurm"


class ExecutionPlatform(StrEnum):
    """Known execution platforms."""

    LOCAL = "local"
    ARCHER2 = "archer2"
    CIRRUS = "cirrus"
    LUMI = "lumi"
    OTHER = "other"


class FileReference(BaseModel):
    """Reference to an input, output, or scientific artefact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    format: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)

    @field_validator("path", "sha256")
    @classmethod
    def required_strings_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value must not be empty")

        return value


class SoftwareProvenance(BaseModel):
    """Software identity associated with a calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calculator: str
    calculator_version: str
    nsdw_version: str
    git_commit: str | None = None


class CP2KSettings(BaseModel):
    """Scientifically relevant CP2K settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    xc_functional: str

    basis_sets: tuple[str, ...] = ()
    potentials: tuple[str, ...] = ()

    cutoff: Quantity | None = None
    relative_cutoff: Quantity | None = None

    charge: int = 0
    multiplicity: int = Field(default=1, gt=0)

    eps_scf: float | None = Field(default=None, gt=0)

    k_points: tuple[int, int, int] | None = None

    admm: bool = False


class ExecutionProvenance(BaseModel):
    """Information describing where and how a calculation ran."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: ExecutionPlatform
    scheduler: SchedulerType

    host: str | None = None
    job_id: str | None = None
    command: str | None = None

    started_at: str | None = None
    completed_at: str | None = None


class ProvenanceResult(BaseModel):
    """Complete reproducibility provenance associated with an NSDW result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    software: SoftwareProvenance
    execution: ExecutionProvenance

    input_files: tuple[FileReference, ...] = ()
    output_files: tuple[FileReference, ...] = ()
    artefacts: tuple[FileReference, ...] = ()

    cp2k: CP2KSettings | None = None
