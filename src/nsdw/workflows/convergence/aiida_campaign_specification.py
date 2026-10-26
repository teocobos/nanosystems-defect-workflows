"""Versioned scientific specification for AiiDA campaigns."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CampaignSpecification:
    campaign_name: str
    expected_cutoffs: tuple[int, ...]
    sha256: str


def _validate_payload(payload: object) -> tuple[str, tuple[int, ...]]:
    if not isinstance(payload, dict):
        raise ValueError("Specification must be a JSON object.")

    if set(payload) != {
        "schema_version",
        "campaign_name",
        "expected_cutoffs",
    }:
        raise ValueError("Unexpected specification fields.")

    if type(payload["schema_version"]) is not int or (
        payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported specification schema.")

    name = payload["campaign_name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Invalid campaign name.")

    values = payload["expected_cutoffs"]
    if not isinstance(values, list) or not values:
        raise ValueError("Expected cutoffs must be a nonempty list.")

    if (
        any(type(value) is not int or value <= 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "Expected cutoffs must be unique positive integers."
        )

    return name, tuple(sorted(values))


def specification_digest(payload: object) -> str:
    """Hash canonical validated scientific specification content."""
    name, cutoffs = _validate_payload(payload)

    canonical = {
        "schema_version": SCHEMA_VERSION,
        "campaign_name": name,
        "expected_cutoffs": list(cutoffs),
    }

    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def load_campaign_specification(
    path: str | Path,
) -> CampaignSpecification:
    """Read and validate a scientific campaign specification."""
    source = Path(path).expanduser().absolute()

    if source.is_symlink():
        raise ValueError(
            "Campaign specification symlinks are not supported."
        )

    payload = json.loads(source.read_text(encoding="utf-8"))
    name, cutoffs = _validate_payload(payload)

    return CampaignSpecification(
        campaign_name=name,
        expected_cutoffs=cutoffs,
        sha256=specification_digest(payload),
    )


def create_campaign_specification(
    path: str | Path,
    *,
    campaign_name: str,
    expected_cutoffs: tuple[int, ...],
) -> CampaignSpecification:
    """Create a specification without overwriting existing data."""
    destination = Path(path).expanduser().absolute()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_name": campaign_name,
        "expected_cutoffs": list(expected_cutoffs),
    }

    _validate_payload(payload)

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    return load_campaign_specification(destination)
