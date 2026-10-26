"""Verify registered campaigns against persisted AiiDA WorkChains."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nsdw.workflows.convergence.aiida_campaign_specification import (
    CampaignSpecification,
    _validate_payload,
    specification_digest,
)
from nsdw.workflows.convergence.aiida_provenance_binding import (
    validate_provenance_binding,
)
from nsdw.workflows.convergence.aiida_specification_binding import (
    validate_cp2k_cutoff_parameters,
    validate_specification_binding,
)


def verify_live_workchain_provenance(
    *,
    registered_workchain_uuid: str,
    registered_specification_sha256: str,
    expected_cutoffs: tuple[int, ...],
    campaign_name: str | None = None,
    node_loader: Callable[[str], Any] | None = None,
) -> None:
    """Verify a registered scientific campaign using AiiDA storage.

    The optional loader supports offline unit testing. Production uses
    aiida.orm.load_node. No remote calculation is submitted.
    """
    from uuid import UUID

    from aiida.orm import WorkChainNode

    try:
        canonical_uuid = str(UUID(registered_workchain_uuid))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid registered WorkChain UUID.") from exc

    if node_loader is None:
        from aiida.orm import load_node

        node_loader = load_node

    node = node_loader(canonical_uuid)

    if not isinstance(node, WorkChainNode):
        raise ValueError("Registered node is not an AiiDA WorkChain.")

    inputs = node.inputs

    try:
        scientific_node = inputs.scientific_specification
        candidates_node = inputs.candidates
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "WorkChain is missing required scientific inputs."
        ) from exc

    if not hasattr(scientific_node, "get_dict"):
        raise ValueError("Invalid scientific specification input.")

    if not hasattr(candidates_node, "get_dict"):
        raise ValueError("Invalid candidates input.")

    scientific = scientific_node.get_dict()
    candidates = candidates_node.get_dict()

    validate_provenance_binding(
        registered_workchain_uuid=canonical_uuid,
        registered_specification_sha256=(
            registered_specification_sha256
        ),
        actual_workchain_uuid=str(node.uuid),
        workchain_specification_input=scientific,
    )

    specification_payload = scientific["specification"]
    name, cutoffs = _validate_payload(specification_payload)
    digest = specification_digest(specification_payload)

    specification = CampaignSpecification(
        campaign_name=name,
        expected_cutoffs=cutoffs,
        sha256=digest,
    )

    if tuple(expected_cutoffs) != specification.expected_cutoffs:
        raise ValueError(
            "Registry cutoffs do not match WorkChain specification."
        )

    if campaign_name is not None and (
        campaign_name != specification.campaign_name
    ):
        raise ValueError(
            "Registry campaign name does not match specification."
        )

    candidate_cutoffs = scientific["candidate_cutoffs"]

    validate_specification_binding(
        specification,
        candidates,
        candidate_cutoffs,
    )
    validate_cp2k_cutoff_parameters(
        candidates,
        candidate_cutoffs,
    )
