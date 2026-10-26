"""Persistent registry and single-pass AiiDA reporting service."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .aiida_auto_postprocessing import (
    process_completed_aiida_campaign,
)


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RegisteredCampaign:
    name: str
    campaign_directory: Path
    report_directory: Path
    expected_cutoffs: tuple[int, ...]
    workchain_uuid: str | None = None
    specification_sha256: str | None = None


@dataclass(frozen=True)
class ServiceResult:
    name: str
    state: str
    detail: str


def _validate_cutoffs(values: Iterable[int]) -> tuple[int, ...]:
    values = tuple(values)
    if (
        not values
        or any(type(x) is not int or x <= 0 for x in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "Expected cutoffs must be unique positive integers."
        )
    return tuple(sorted(values))


def _validate_provenance_fields(
    workchain_uuid: str | None,
    specification_sha256: str | None,
) -> tuple[str | None, str | None]:
    from uuid import UUID
    import re

    if (workchain_uuid is None) != (specification_sha256 is None):
        raise ValueError(
            "WorkChain UUID and specification SHA-256 "
            "must be supplied together."
        )

    if workchain_uuid is None:
        return None, None

    if not isinstance(workchain_uuid, str):
        raise ValueError("Invalid WorkChain UUID.")

    try:
        canonical_uuid = str(UUID(workchain_uuid))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid WorkChain UUID.") from exc

    if (
        not isinstance(specification_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", specification_sha256) is None
    ):
        raise ValueError("Invalid specification SHA-256.")

    return canonical_uuid, specification_sha256


def _provenance_fields(
    workchain_uuid: str | None,
    specification_sha256: str | None,
) -> dict:
    if workchain_uuid is None:
        return {}
    return {
        "workchain_uuid": workchain_uuid,
        "specification_sha256": specification_sha256,
    }


def register_campaign(
    registry_path: str | Path,
    *,
    name: str,
    campaign_directory: str | Path,
    report_directory: str | Path,
    expected_cutoffs: Iterable[int],
    workchain_uuid: str | None = None,
    specification_sha256: str | None = None,
) -> Path:
    """Create a registry without overwriting an existing registry."""
    registry = Path(registry_path).expanduser().absolute()

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Campaign name must be nonempty.")

    cutoffs = _validate_cutoffs(expected_cutoffs)
    workchain_uuid, specification_sha256 = _validate_provenance_fields(
        workchain_uuid, specification_sha256
    )

    campaign = Path(campaign_directory).expanduser().resolve(
        strict=True
    )
    if not campaign.is_dir():
        raise NotADirectoryError(campaign)

    report = Path(report_directory).expanduser().absolute()

    if registry.exists() or registry.is_symlink():
        raise FileExistsError(
            f"Registry already exists: {registry}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaigns": [
            {
                "name": name,
                "campaign_directory": str(campaign),
                "report_directory": str(report),
                "expected_cutoffs": list(cutoffs),
                **_provenance_fields(
                    workchain_uuid, specification_sha256
                ),
            }
        ],
    }

    with registry.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return registry


def load_registry(
    registry_path: str | Path,
) -> tuple[RegisteredCampaign, ...]:
    """Load and validate an existing registry."""
    registry = Path(registry_path).expanduser().absolute()

    if registry.is_symlink():
        raise ValueError("Registry symlinks are not supported.")

    payload = json.loads(registry.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Registry must be a JSON object.")

    if set(payload) != {"schema_version", "campaigns"}:
        raise ValueError("Unexpected registry fields.")

    if type(payload["schema_version"]) is not int or (
        payload["schema_version"] not in (1, SCHEMA_VERSION)
    ):
        raise ValueError("Unsupported registry schema.")

    schema_version = payload["schema_version"]
    entries = payload["campaigns"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Registry campaigns must be nonempty.")

    campaigns = []
    names = set()
    destinations = set()
    workchain_uuids = set()

    for entry in entries:
        required = {
            "name",
            "campaign_directory",
            "report_directory",
            "expected_cutoffs",
        }
        optional = {"workchain_uuid", "specification_sha256"}

        if not isinstance(entry, dict):
            raise ValueError("Invalid campaign registry entry.")

        fields = set(entry)
        if schema_version == 1:
            valid = fields == required
        else:
            valid = required <= fields <= required | optional

        if not valid:
            raise ValueError("Invalid campaign registry entry.")

        uuid, digest = _validate_provenance_fields(
            entry.get("workchain_uuid"),
            entry.get("specification_sha256"),
        )

        if uuid is not None:
            if uuid in workchain_uuids:
                raise ValueError("Duplicate WorkChain UUID.")
            workchain_uuids.add(uuid)

        name = entry["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid campaign name.")
        if name in names:
            raise ValueError("Duplicate campaign name.")
        names.add(name)

        for field in ("campaign_directory", "report_directory"):
            if not isinstance(entry[field], str) or not entry[field]:
                raise ValueError(f"Invalid {field}.")

        if not isinstance(entry["expected_cutoffs"], list):
            raise ValueError("Expected cutoffs must be a list.")

        destination = Path(entry["report_directory"]).expanduser()
        if not destination.is_absolute():
            raise ValueError("Report directory must be absolute.")

        campaign_path = Path(
            entry["campaign_directory"]
        ).expanduser()
        if not campaign_path.is_absolute():
            raise ValueError("Campaign directory must be absolute.")

        destination_key = str(destination)
        if destination_key in destinations:
            raise ValueError("Duplicate report destination.")
        destinations.add(destination_key)

        campaigns.append(
            RegisteredCampaign(
                name=name,
                campaign_directory=campaign_path,
                report_directory=destination,
                expected_cutoffs=_validate_cutoffs(
                    entry["expected_cutoffs"]
                ),
                workchain_uuid=uuid,
                specification_sha256=digest,
            )
        )

    return tuple(campaigns)


def run_report_service_once(
    registry_path: str | Path,
) -> tuple[ServiceResult, ...]:
    """Check every registered campaign once.

    Campaign failures are isolated. No submission or retry occurs.
    """
    campaigns = load_registry(registry_path)
    results = []

    for campaign in campaigns:
        try:
            if campaign.workchain_uuid is not None:
                from nsdw.workflows.convergence.aiida_live_provenance import (
                    verify_live_workchain_provenance,
                )

                try:
                    verify_live_workchain_provenance(
                        registered_workchain_uuid=campaign.workchain_uuid,
                        registered_specification_sha256=(
                            campaign.specification_sha256
                        ),
                        expected_cutoffs=campaign.expected_cutoffs,
                    )

                    from nsdw.workflows.convergence.aiida_parent_child_provenance import (
                        verify_parent_child_provenance,
                    )

                    verify_parent_child_provenance(
                        registered_workchain_uuid=campaign.workchain_uuid,
                        campaign_directory=str(
                            campaign.campaign_directory
                        ),
                        expected_cutoffs=campaign.expected_cutoffs,
                    )
                except Exception as exc:
                    results.append(
                        ServiceResult(
                            name=campaign.name,
                            state="BLOCKED",
                            detail=(
                                "AiiDA provenance verification failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    )
                    continue

            result = process_completed_aiida_campaign(
                campaign.campaign_directory,
                campaign.report_directory,
                expected_cutoffs=campaign.expected_cutoffs,
            )
            results.append(
                ServiceResult(
                    name=campaign.name,
                    state=result.state.value,
                    detail="; ".join(result.reasons),
                )
            )
        except Exception as exc:
            results.append(
                ServiceResult(
                    name=campaign.name,
                    state="ERROR",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

    return tuple(results)


def add_registered_campaign(
    registry_path: str | Path,
    *,
    name: str,
    campaign_directory: str | Path,
    report_directory: str | Path,
    expected_cutoffs: Iterable[int],
    workchain_uuid: str | None = None,
    specification_sha256: str | None = None,
) -> Path:
    """Atomically add a campaign under an exclusive registry lock."""
    import fcntl
    import os
    import tempfile

    registry = Path(registry_path).expanduser().absolute()
    lock_path = registry.with_name(registry.name + ".lock")

    if registry.is_symlink():
        raise ValueError("Registry symlinks are not supported.")

    if not registry.is_file():
        raise FileNotFoundError(registry)

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Campaign name must be nonempty.")

    cutoffs = _validate_cutoffs(expected_cutoffs)
    workchain_uuid, specification_sha256 = _validate_provenance_fields(
        workchain_uuid, specification_sha256
    )

    campaign = Path(campaign_directory).expanduser().resolve(
        strict=True
    )
    if not campaign.is_dir():
        raise NotADirectoryError(campaign)

    destination = Path(report_directory).expanduser().absolute()

    if lock_path.is_symlink():
        raise ValueError("Registry lock symlinks are not supported.")

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        existing = load_registry(registry)

        if any(item.name == name for item in existing):
            raise ValueError(f"Campaign already registered: {name}")

        if any(
            item.report_directory == destination
            for item in existing
        ):
            raise ValueError(
                f"Report destination already registered: {destination}"
            )

        if workchain_uuid is not None and any(
            item.workchain_uuid == workchain_uuid
            for item in existing
        ):
            raise ValueError('WorkChain UUID already registered.')

        entries = [
            {
                "name": item.name,
                "campaign_directory": str(item.campaign_directory),
                "report_directory": str(item.report_directory),
                "expected_cutoffs": list(item.expected_cutoffs),
                **_provenance_fields(
                    item.workchain_uuid,
                    item.specification_sha256,
                ),
            }
            for item in existing
        ]

        entries.append(
            {
                "name": name,
                "campaign_directory": str(campaign),
                "report_directory": str(destination),
                "expected_cutoffs": list(cutoffs),
                **_provenance_fields(
                    workchain_uuid, specification_sha256
                ),
            }
        )

        payload = {
            "schema_version": SCHEMA_VERSION,
            "campaigns": entries,
        }

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=registry.parent,
                prefix=f".{registry.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary, registry)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return registry


def run_locked_report_service_once(
    registry_path: str | Path,
) -> tuple[ServiceResult, ...]:
    """Run one service pass with an exclusive nonblocking lock."""
    import fcntl

    registry = Path(registry_path).expanduser().absolute()
    lock_path = registry.with_name(registry.name + ".run.lock")

    if lock_path.is_symlink():
        raise ValueError("Service lock symlinks are not supported.")

    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another reporting service instance is running."
            ) from exc

        return run_report_service_once(registry)
