"""Tests for NSDW provenance models."""

import pytest
from pydantic import ValidationError

from nsdw.models import (
    CP2KSettings,
    ExecutionPlatform,
    ExecutionProvenance,
    FileReference,
    ProvenanceResult,
    Quantity,
    SchedulerType,
    SoftwareProvenance,
)


def test_file_reference():
    file = FileReference(
        path="outputs/igzo.out",
        sha256="abc123",
        format="cp2k-output",
        size_bytes=1024,
    )

    assert file.path == "outputs/igzo.out"
    assert file.size_bytes == 1024


def test_file_reference_rejects_empty_hash():
    with pytest.raises(ValidationError):
        FileReference(
            path="outputs/igzo.out",
            sha256="   ",
        )


def test_cp2k_settings():
    settings = CP2KSettings(
        xc_functional="PBE0-TC-LRC",
        basis_sets=(
            "TZV2P-MOLOPT-PBE-GTH-q13",
            "TZV2P-MOLOPT-PBE-GTH-q6",
        ),
        potentials=(
            "GTH-PBE-q13",
            "GTH-PBE-q6",
        ),
        cutoff=Quantity(
            value=600.0,
            unit="Ry",
        ),
        relative_cutoff=Quantity(
            value=60.0,
            unit="Ry",
        ),
        charge=0,
        multiplicity=1,
        eps_scf=1e-6,
        k_points=(2, 2, 1),
        admm=True,
    )

    assert settings.xc_functional == "PBE0-TC-LRC"
    assert settings.admm is True
    assert settings.k_points == (2, 2, 1)


def test_execution_on_archer2():
    execution = ExecutionProvenance(
        platform=ExecutionPlatform.ARCHER2,
        scheduler=SchedulerType.SLURM,
        host="archer2",
        job_id="1234567",
        command="srun cp2k.psmp input.inp",
    )

    assert execution.platform == ExecutionPlatform.ARCHER2
    assert execution.scheduler == SchedulerType.SLURM


def test_complete_provenance():
    provenance = ProvenanceResult(
        software=SoftwareProvenance(
            calculator="cp2k",
            calculator_version="2026.2",
            nsdw_version="0.1.0",
            git_commit="abc123",
        ),
        execution=ExecutionProvenance(
            platform=ExecutionPlatform.LOCAL,
            scheduler=SchedulerType.LOCAL,
            host="petersham",
        ),
        input_files=(
            FileReference(
                path="input.inp",
                sha256="inputhash",
                format="cp2k-input",
            ),
        ),
        output_files=(
            FileReference(
                path="output.out",
                sha256="outputhash",
                format="cp2k-output",
            ),
        ),
        cp2k=CP2KSettings(
            xc_functional="PBE",
            cutoff=Quantity(
                value=600.0,
                unit="Ry",
            ),
        ),
    )

    assert provenance.software.calculator == "cp2k"
    assert provenance.cp2k is not None
    assert provenance.cp2k.cutoff is not None
    assert provenance.cp2k.cutoff.value == 600.0
