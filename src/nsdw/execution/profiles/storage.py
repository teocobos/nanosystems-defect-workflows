"""Persistent user-level storage for NSDW HPC profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from nsdw.execution.profiles.models import HPCProfile


HPC_PROFILE_SCHEMA_VERSION = 1
HPC_PROFILE_FILENAME = "hpc-profiles.yaml"


class HPCProfileStoreError(RuntimeError):
    """Raised when persistent HPC profile storage cannot be used."""


def default_hpc_profiles_path() -> Path:
    """Return the default user-level NSDW HPC profile file."""

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")

    if xdg_config_home:
        config_root = Path(xdg_config_home).expanduser()
    else:
        config_root = Path.home() / ".config"

    return config_root / "nsdw" / HPC_PROFILE_FILENAME


def _resolve_path(
    path: Path | None,
) -> Path:
    """Resolve an explicit or default profile-store path."""

    if path is None:
        path = default_hpc_profiles_path()

    return path.expanduser()


def load_hpc_profiles(
    path: Path | None = None,
) -> dict[str, HPCProfile]:
    """Load all configured HPC profiles.

    A missing profile file represents an empty profile collection.
    """

    profile_path = _resolve_path(path)

    if not profile_path.exists():
        return {}

    if not profile_path.is_file():
        raise HPCProfileStoreError(
            f"HPC profile path is not a file: {profile_path}"
        )

    try:
        raw_data = yaml.safe_load(
            profile_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise HPCProfileStoreError(
            f"Could not read HPC profiles from {profile_path}: {exc}"
        ) from exc

    if raw_data is None:
        return {}

    if not isinstance(raw_data, Mapping):
        raise HPCProfileStoreError(
            "HPC profile file must contain a YAML mapping"
        )

    schema_version = raw_data.get(
        "schema_version",
        HPC_PROFILE_SCHEMA_VERSION,
    )

    if schema_version != HPC_PROFILE_SCHEMA_VERSION:
        raise HPCProfileStoreError(
            "Unsupported HPC profile schema version: "
            f"{schema_version!r}"
        )

    raw_profiles = raw_data.get("profiles", {})

    if not isinstance(raw_profiles, Mapping):
        raise HPCProfileStoreError(
            "'profiles' must contain a YAML mapping"
        )

    profiles: dict[str, HPCProfile] = {}

    for profile_name, profile_data in raw_profiles.items():
        if not isinstance(profile_name, str):
            raise HPCProfileStoreError(
                "HPC profile names must be strings"
            )

        if not isinstance(profile_data, Mapping):
            raise HPCProfileStoreError(
                f"HPC profile {profile_name!r} must be a mapping"
            )

        try:
            profile = HPCProfile.model_validate(
                dict(profile_data)
            )
        except ValidationError as exc:
            raise HPCProfileStoreError(
                f"Invalid HPC profile {profile_name!r}: {exc}"
            ) from exc

        if profile.name != profile_name:
            raise HPCProfileStoreError(
                f"HPC profile key {profile_name!r} does not match "
                f"profile name {profile.name!r}"
            )

        profiles[profile_name] = profile

    return profiles


def save_hpc_profiles(
    profiles: Mapping[str, HPCProfile],
    path: Path | None = None,
) -> Path:
    """Persist the supplied HPC profile collection."""

    profile_path = _resolve_path(path)

    serialised_profiles: dict[str, dict[str, Any]] = {}

    for profile_name in sorted(profiles):
        profile = profiles[profile_name]

        if profile.name != profile_name:
            raise HPCProfileStoreError(
                f"HPC profile key {profile_name!r} does not match "
                f"profile name {profile.name!r}"
            )

        serialised_profiles[profile_name] = profile.model_dump(
            mode="json",
        )

    payload = {
        "schema_version": HPC_PROFILE_SCHEMA_VERSION,
        "profiles": serialised_profiles,
    }

    try:
        profile_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = profile_path.with_suffix(
            profile_path.suffix + ".tmp"
        )

        temporary_path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(profile_path)

    except OSError as exc:
        raise HPCProfileStoreError(
            f"Could not write HPC profiles to {profile_path}: {exc}"
        ) from exc

    return profile_path


def get_hpc_profile(
    name: str,
    path: Path | None = None,
) -> HPCProfile:
    """Return one configured HPC profile."""

    name = name.strip()

    if not name:
        raise HPCProfileStoreError(
            "HPC profile name must not be empty"
        )

    profiles = load_hpc_profiles(path)

    try:
        return profiles[name]
    except KeyError as exc:
        raise HPCProfileStoreError(
            f"HPC profile not found: {name!r}"
        ) from exc


def upsert_hpc_profile(
    profile: HPCProfile,
    path: Path | None = None,
) -> Path:
    """Create or replace one HPC profile."""

    profiles = load_hpc_profiles(path)

    profiles[profile.name] = profile

    return save_hpc_profiles(
        profiles,
        path,
    )


def delete_hpc_profile(
    name: str,
    path: Path | None = None,
) -> Path:
    """Delete one HPC profile."""

    name = name.strip()

    if not name:
        raise HPCProfileStoreError(
            "HPC profile name must not be empty"
        )

    profiles = load_hpc_profiles(path)

    if name not in profiles:
        raise HPCProfileStoreError(
            f"HPC profile not found: {name!r}"
        )

    del profiles[name]

    return save_hpc_profiles(
        profiles,
        path,
    )
