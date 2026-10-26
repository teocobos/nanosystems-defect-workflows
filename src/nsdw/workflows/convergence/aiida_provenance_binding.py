"""Validate registered scientific provenance against AiiDA inputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from uuid import UUID

from .aiida_campaign_specification import (
    specification_digest,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProvenanceBinding:
    workchain_uuid: str
    specification_sha256: str


def validate_provenance_binding(
    *,
    registered_workchain_uuid: str,
    registered_specification_sha256: str,
    actual_workchain_uuid: str,
    workchain_specification_input: dict,
) -> ProvenanceBinding:
    """Validate registry identity against WorkChain input evidence.

    The caller must obtain the actual UUID and specification input
    from the AiiDA process node, not from the registry itself.
    """
    if not isinstance(registered_workchain_uuid, str):
        raise ValueError("Invalid registered WorkChain UUID.")

    if not isinstance(actual_workchain_uuid, str):
        raise ValueError("Invalid actual WorkChain UUID.")

    try:
        registered_uuid = str(UUID(registered_workchain_uuid))
        actual_uuid = str(UUID(actual_workchain_uuid))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid WorkChain UUID.") from exc

    if registered_uuid != actual_uuid:
        raise ValueError("WorkChain UUID mismatch.")

    if (
        not isinstance(registered_specification_sha256, str)
        or not _SHA256_PATTERN.fullmatch(
            registered_specification_sha256
        )
    ):
        raise ValueError("Invalid specification SHA-256.")

    if not isinstance(workchain_specification_input, dict):
        raise ValueError("Invalid WorkChain specification input.")

    if set(workchain_specification_input) != {
        "specification",
        "specification_sha256",
        "candidate_cutoffs",
    }:
        raise ValueError(
            "Unexpected WorkChain specification fields."
        )

    specification = workchain_specification_input[
        "specification"
    ]

    actual_digest = specification_digest(specification)

    if (
        workchain_specification_input[
            "specification_sha256"
        ] != actual_digest
    ):
        raise ValueError(
            "WorkChain specification digest mismatch."
        )

    if registered_specification_sha256 != actual_digest:
        raise ValueError(
            "Registry specification digest mismatch."
        )

    return ProvenanceBinding(
        workchain_uuid=actual_uuid,
        specification_sha256=actual_digest,
    )
