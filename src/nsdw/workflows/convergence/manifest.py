"""Persistent manifest models for convergence studies."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.models import (
    ConvergenceCriterion,
    ConvergenceParameter,
)


class ConvergenceManifestCandidate(BaseModel):
    """One generated calculation recorded in a convergence manifest."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    label: str = Field(min_length=1)
    order: int = Field(ge=0)
    value: Quantity | str | tuple[int, int, int]
    directory: str = Field(min_length=1)
    input_file: str = Field(min_length=1)
    coordinate_file: str = Field(min_length=1)


class ConvergenceStudyManifest(BaseModel):
    """Portable description of a generated convergence study."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = Field(default=1, ge=1)
    calculator: str = Field(min_length=1)
    parameter: ConvergenceParameter
    criterion: ConvergenceCriterion
    candidates: tuple[ConvergenceManifestCandidate, ...]


def write_convergence_manifest(
    *,
    manifest: ConvergenceStudyManifest,
    path: str | Path,
) -> Path:
    """Write a convergence-study manifest as JSON."""

    path = Path(path)

    path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return path
