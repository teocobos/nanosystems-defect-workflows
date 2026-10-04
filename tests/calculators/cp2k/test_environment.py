from pathlib import Path

import pytest

from nsdw.calculators.cp2k.environment import (
    CP2KDataDirectoryNotFoundError,
    CP2KDataFileNotFoundError,
    build_cp2k_environment,
    resolve_cp2k_data_dir,
    validate_cp2k_data_dir,
)


def test_validate_cp2k_data_dir_accepts_existing_directory(
    tmp_path: Path,
):
    result = validate_cp2k_data_dir(tmp_path)

    assert result == tmp_path.resolve()


def test_validate_cp2k_data_dir_rejects_missing_directory(
    tmp_path: Path,
):
    missing = tmp_path / "missing"

    with pytest.raises(
        CP2KDataDirectoryNotFoundError,
        match="does not exist",
    ):
        validate_cp2k_data_dir(missing)


def test_validate_cp2k_data_dir_accepts_required_files(
    tmp_path: Path,
):
    (tmp_path / "BASIS_MOLOPT").write_text("basis")
    (tmp_path / "GTH_POTENTIALS").write_text("potential")

    result = validate_cp2k_data_dir(
        tmp_path,
        required_files=(
            "BASIS_MOLOPT",
            "GTH_POTENTIALS",
        ),
    )

    assert result == tmp_path.resolve()


def test_validate_cp2k_data_dir_reports_missing_required_file(
    tmp_path: Path,
):
    (tmp_path / "BASIS_MOLOPT").write_text("basis")

    with pytest.raises(
        CP2KDataFileNotFoundError,
        match="GTH_POTENTIALS",
    ):
        validate_cp2k_data_dir(
            tmp_path,
            required_files=(
                "BASIS_MOLOPT",
                "GTH_POTENTIALS",
            ),
        )


def test_explicit_data_dir_takes_precedence_over_environment(
    tmp_path: Path,
):
    explicit = tmp_path / "explicit"
    from_env = tmp_path / "environment"

    explicit.mkdir()
    from_env.mkdir()

    result = resolve_cp2k_data_dir(
        explicit=explicit,
        environment={
            "CP2K_DATA_DIR": str(from_env),
        },
    )

    assert result == explicit.resolve()


def test_resolve_cp2k_data_dir_uses_environment_variable(
    tmp_path: Path,
):
    data_dir = tmp_path / "cp2k-data"
    data_dir.mkdir()

    result = resolve_cp2k_data_dir(
        environment={
            "CP2K_DATA_DIR": str(data_dir),
        },
    )

    assert result == data_dir.resolve()


def test_invalid_explicit_directory_does_not_fall_back(
    tmp_path: Path,
):
    valid_environment = tmp_path / "environment"
    valid_environment.mkdir()

    with pytest.raises(CP2KDataDirectoryNotFoundError):
        resolve_cp2k_data_dir(
            explicit=tmp_path / "missing",
            environment={
                "CP2K_DATA_DIR": str(valid_environment),
            },
        )


def test_build_cp2k_environment_preserves_existing_values(
    tmp_path: Path,
):
    data_dir = tmp_path / "cp2k-data"
    data_dir.mkdir()

    result = build_cp2k_environment(
        explicit_data_dir=data_dir,
        environment={
            "OMP_NUM_THREADS": "4",
        },
    )

    assert result["OMP_NUM_THREADS"] == "4"
    assert result["CP2K_DATA_DIR"] == str(data_dir.resolve())


def test_build_cp2k_environment_overrides_existing_cp2k_data_dir(
    tmp_path: Path,
):
    explicit = tmp_path / "explicit"
    existing = tmp_path / "existing"

    explicit.mkdir()
    existing.mkdir()

    result = build_cp2k_environment(
        explicit_data_dir=explicit,
        environment={
            "CP2K_DATA_DIR": str(existing),
        },
    )

    assert result["CP2K_DATA_DIR"] == str(explicit.resolve())


def test_build_cp2k_environment_inherits_process_cp2k_data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_dir = tmp_path / "process-data"
    data_dir.mkdir()

    monkeypatch.setenv(
        "CP2K_DATA_DIR",
        str(data_dir),
    )

    result = build_cp2k_environment(
        environment={
            "OMP_NUM_THREADS": "4",
        },
    )

    assert result["OMP_NUM_THREADS"] == "4"
    assert result["CP2K_DATA_DIR"] == str(
        data_dir.resolve()
    )


def test_caller_cp2k_data_dir_overrides_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    process_data_dir = tmp_path / "process-data"
    caller_data_dir = tmp_path / "caller-data"

    process_data_dir.mkdir()
    caller_data_dir.mkdir()

    monkeypatch.setenv(
        "CP2K_DATA_DIR",
        str(process_data_dir),
    )

    result = build_cp2k_environment(
        environment={
            "CP2K_DATA_DIR": str(caller_data_dir),
            "OMP_NUM_THREADS": "8",
        },
    )

    assert result["CP2K_DATA_DIR"] == str(
        caller_data_dir.resolve()
    )

    assert result["OMP_NUM_THREADS"] == "8"


def test_resolve_cp2k_data_dir_from_executable_installation(
    tmp_path: Path,
):
    prefix = tmp_path / "cp2k-install"
    bin_dir = prefix / "bin"
    data_dir = prefix / "share" / "cp2k" / "data"

    bin_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    executable = bin_dir / "cp2k.psmp"
    executable.write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )

    result = resolve_cp2k_data_dir(
        executable=executable,
        environment={},
    )

    assert result == data_dir.resolve()


def test_executable_relative_discovery_validates_required_files(
    tmp_path: Path,
):
    prefix = tmp_path / "cp2k-install"
    bin_dir = prefix / "bin"
    data_dir = prefix / "share" / "cp2k" / "data"

    bin_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    executable = bin_dir / "cp2k.psmp"
    executable.write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )

    (data_dir / "BASIS_MOLOPT").write_text(
        "basis",
        encoding="utf-8",
    )

    (data_dir / "GTH_POTENTIALS").write_text(
        "potential",
        encoding="utf-8",
    )

    result = resolve_cp2k_data_dir(
        executable=executable,
        environment={},
        required_files=(
            "BASIS_MOLOPT",
            "GTH_POTENTIALS",
        ),
    )

    assert result == data_dir.resolve()