from pathlib import Path

import pytest
import yaml

from nsdw.execution.profiles.models import (
    HPCProfile,
    HPCProfileResources,
)
from nsdw.execution.profiles.storage import (
    HPC_PROFILE_SCHEMA_VERSION,
    HPCProfileStoreError,
    default_hpc_profiles_path,
    delete_hpc_profile,
    get_hpc_profile,
    load_hpc_profiles,
    save_hpc_profiles,
    upsert_hpc_profile,
)


def build_lumi_profile() -> HPCProfile:
    return HPCProfile(
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


def test_default_hpc_profiles_path_uses_xdg_config_home(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "XDG_CONFIG_HOME",
        str(tmp_path),
    )

    assert default_hpc_profiles_path() == (
        tmp_path
        / "nsdw"
        / "hpc-profiles.yaml"
    )


def test_load_missing_hpc_profile_file_returns_empty(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    assert load_hpc_profiles(path) == {}


def test_save_and_load_hpc_profiles(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"
    profile = build_lumi_profile()

    written_path = save_hpc_profiles(
        {
            "lumi": profile,
        },
        path,
    )

    assert written_path == path
    assert path.is_file()

    profiles = load_hpc_profiles(path)

    assert profiles == {
        "lumi": profile,
    }


def test_saved_hpc_profile_file_has_schema_version(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    save_hpc_profiles(
        {
            "lumi": build_lumi_profile(),
        },
        path,
    )

    raw_data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    assert raw_data["schema_version"] == (
        HPC_PROFILE_SCHEMA_VERSION
    )

    assert "lumi" in raw_data["profiles"]


def test_get_hpc_profile(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"
    profile = build_lumi_profile()

    save_hpc_profiles(
        {
            "lumi": profile,
        },
        path,
    )

    assert get_hpc_profile(
        "lumi",
        path,
    ) == profile


def test_get_hpc_profile_rejects_unknown_name(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    with pytest.raises(
        HPCProfileStoreError,
        match="HPC profile not found",
    ):
        get_hpc_profile(
            "missing",
            path,
        )


def test_upsert_hpc_profile_creates_profile(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"
    profile = build_lumi_profile()

    upsert_hpc_profile(
        profile,
        path,
    )

    assert get_hpc_profile(
        "lumi",
        path,
    ) == profile


def test_upsert_hpc_profile_replaces_profile(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    original = build_lumi_profile()

    updated = original.model_copy(
        update={
            "resources": original.resources.model_copy(
                update={
                    "walltime_seconds": 1200,
                }
            ),
        }
    )

    upsert_hpc_profile(
        original,
        path,
    )

    upsert_hpc_profile(
        updated,
        path,
    )

    assert get_hpc_profile(
        "lumi",
        path,
    ).resources.walltime_seconds == 1200


def test_delete_hpc_profile(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    upsert_hpc_profile(
        build_lumi_profile(),
        path,
    )

    delete_hpc_profile(
        "lumi",
        path,
    )

    assert load_hpc_profiles(path) == {}


def test_delete_unknown_hpc_profile_fails(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    with pytest.raises(
        HPCProfileStoreError,
        match="HPC profile not found",
    ):
        delete_hpc_profile(
            "missing",
            path,
        )


def test_load_rejects_unsupported_schema_version(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 999,
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HPCProfileStoreError,
        match="Unsupported HPC profile schema version",
    ):
        load_hpc_profiles(path)


def test_load_rejects_profile_key_name_mismatch(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profiles": {
                    "lumi": {
                        "name": "other",
                        "computer": "lumi-c",
                        "code": "cp2k-lumi-c@lumi-c",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        HPCProfileStoreError,
        match="does not match profile name",
    ):
        load_hpc_profiles(path)


def test_save_rejects_profile_key_name_mismatch(
    tmp_path,
):
    path = tmp_path / "hpc-profiles.yaml"

    with pytest.raises(
        HPCProfileStoreError,
        match="does not match profile name",
    ):
        save_hpc_profiles(
            {
                "wrong-name": build_lumi_profile(),
            },
            path,
        )
