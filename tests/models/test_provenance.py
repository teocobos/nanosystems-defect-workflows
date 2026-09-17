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
    CP2KKindSettings,
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
        kinds=(
            CP2KKindSettings(
                kind="In",
                element="In",
                basis_set="TZV2P-MOLOPT-PBE-GTH-q13",
                potential="GTH-PBE-q13",
            ),
            CP2KKindSettings(
                kind="O",
                element="O",
                basis_set="TZV2P-MOLOPT-PBE-GTH-q6",
                potential="GTH-PBE-q6",
            ),
        ),
        basis_set_file="BASIS_MOLOPT_UZH",
        potential_file="POTENTIAL_UZH",
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

    assert len(settings.kinds) == 2

    assert settings.kinds[0].kind == "In"
    assert settings.kinds[0].element == "In"
    assert (
        settings.kinds[0].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q13"
    )
    assert (
        settings.kinds[0].potential
        == "GTH-PBE-q13"
    )

    assert settings.kinds[1].kind == "O"
    assert settings.kinds[1].element == "O"

    assert (
        settings.basis_set_file
        == "BASIS_MOLOPT_UZH"
    )
    assert (
        settings.potential_file
        == "POTENTIAL_UZH"
    )

    assert settings.cutoff is not None
    assert settings.cutoff.value == 600.0
    assert settings.cutoff.unit == "Ry"

    assert settings.relative_cutoff is not None
    assert settings.relative_cutoff.value == 60.0
    assert settings.relative_cutoff.unit == "Ry"

    assert settings.charge == 0
    assert settings.multiplicity == 1
    assert settings.eps_scf == 1e-6
    assert settings.k_points == (2, 2, 1)
    assert settings.admm is True

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
