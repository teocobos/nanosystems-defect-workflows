import pytest
from pydantic import ValidationError

from nsdw.execution.models import ExecutionBackend
from nsdw.execution.profiles.models import (
    HPCProfile,
    HPCProfileResources,
    HPCScheduler,
)


def test_hpc_profile_minimal_defaults():
    profile = HPCProfile(
        name="lumi",
        computer="lumi-c",
        code="cp2k-lumi-c@lumi-c",
    )

    assert profile.name == "lumi"
    assert profile.backend == ExecutionBackend.AIIDA
    assert profile.scheduler == HPCScheduler.SLURM
    assert profile.aiida_profile is None
    assert profile.computer == "lumi-c"
    assert profile.code == "cp2k-lumi-c@lumi-c"

    assert profile.resources == HPCProfileResources()


def test_hpc_profile_lumi_configuration():
    profile = HPCProfile(
        name="lumi",
        aiida_profile="nsdw-dev",
        computer="lumi-c",
        code="cp2k-lumi-c@lumi-c",
        resources=HPCProfileResources(
            machines=1,
            mpi_per_machine=1,
            omp_threads=2,
            walltime_seconds=600,
            queue="debug",
            account="project_465003407",
        ),
    )

    assert profile.aiida_profile == "nsdw-dev"

    assert profile.resources.machines == 1
    assert profile.resources.mpi_per_machine == 1
    assert profile.resources.omp_threads == 2
    assert profile.resources.walltime_seconds == 600
    assert profile.resources.queue == "debug"
    assert profile.resources.account == "project_465003407"


def test_hpc_profile_strips_string_values():
    profile = HPCProfile(
        name="  lumi  ",
        aiida_profile="  nsdw-dev  ",
        computer="  lumi-c  ",
        code="  cp2k-lumi-c@lumi-c  ",
        resources=HPCProfileResources(
            queue="  debug  ",
            account="  project_465003407  ",
        ),
    )

    assert profile.name == "lumi"
    assert profile.aiida_profile == "nsdw-dev"
    assert profile.computer == "lumi-c"
    assert profile.code == "cp2k-lumi-c@lumi-c"
    assert profile.resources.queue == "debug"
    assert profile.resources.account == "project_465003407"


def test_hpc_profile_normalises_empty_optional_values():
    profile = HPCProfile(
        name="lumi",
        aiida_profile="   ",
        computer="lumi-c",
        code="cp2k-lumi-c@lumi-c",
        resources=HPCProfileResources(
            queue=" ",
            account=" ",
        ),
    )

    assert profile.aiida_profile is None
    assert profile.resources.queue is None
    assert profile.resources.account is None


@pytest.mark.parametrize(
    "field",
    (
        "name",
        "computer",
        "code",
    ),
)
def test_hpc_profile_rejects_empty_required_strings(field):
    values = {
        "name": "lumi",
        "computer": "lumi-c",
        "code": "cp2k-lumi-c@lumi-c",
    }

    values[field] = "   "

    with pytest.raises(ValidationError):
        HPCProfile(**values)


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    (
        ("machines", 0),
        ("mpi_per_machine", 0),
        ("omp_threads", 0),
        ("walltime_seconds", 0),
    ),
)
def test_hpc_profile_resources_require_positive_values(
    field,
    value,
):
    values = {
        "machines": 1,
        "mpi_per_machine": 1,
        "omp_threads": 1,
        "walltime_seconds": 600,
    }

    values[field] = value

    with pytest.raises(ValidationError):
        HPCProfileResources(**values)


def test_hpc_profile_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        HPCProfile(
            name="lumi",
            computer="lumi-c",
            code="cp2k-lumi-c@lumi-c",
            unexpected="value",
        )


def test_hpc_profile_is_frozen():
    profile = HPCProfile(
        name="lumi",
        computer="lumi-c",
        code="cp2k-lumi-c@lumi-c",
    )

    with pytest.raises(ValidationError):
        profile.name = "other"
