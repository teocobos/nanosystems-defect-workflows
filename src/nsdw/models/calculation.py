"""Calculation metadata for NSDW scientific results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class CalculationType(StrEnum):
    """Supported high-level calculation types."""

    GEOMETRY_OPTIMISATION = "geometry_optimisation"
    SINGLE_POINT = "single_point"
    ELECTRONIC_STRUCTURE = "electronic_structure"
    DOS = "dos"
    BAND_STRUCTURE = "band_structure"
    DEFECT_RELAXATION = "defect_relaxation"
    CHARGED_DEFECT = "charged_defect"
    NEB = "neb"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    VIBRATIONAL = "vibrational"
    CONSTRAINED_DFT = "constrained_dft"
    OPTICAL = "optical"
    MLMD = "mlmd"


class CalculationStatus(StrEnum):
    """Execution status of a calculation."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Backend(StrEnum):
    """Supported computational backends."""

    CP2K = "cp2k"
    VASP = "vasp"
    MACE = "mace"
    LAMMPS = "lammps"


class CalculationMetadata(BaseModel):
    """Calculator-independent metadata describing a calculation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: CalculationType
    status: CalculationStatus
    backend: Backend
