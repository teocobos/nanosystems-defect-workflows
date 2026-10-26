"""Verify campaign calculations against AiiDA CALL_CALC provenance."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID

from nsdw.workflows.convergence.aiida_campaign_reconciliation import (
    reconcile_aiida_campaign,
)


def verify_parent_child_provenance(
    *,
    registered_workchain_uuid: str,
    campaign_directory: str,
    expected_cutoffs: tuple[int, ...],
    node_loader: Callable[[str], Any] | None = None,
) -> None:
    """Require exact agreement between campaign records and CALL_CALC links."""
    from aiida.common.links import LinkType
    from aiida.orm import CalcJobNode, WorkChainNode

    try:
        canonical_uuid = str(UUID(registered_workchain_uuid))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid registered WorkChain UUID.") from exc

    if node_loader is None:
        from aiida.orm import load_node

        node_loader = load_node

    parent = node_loader(canonical_uuid)

    if not isinstance(parent, WorkChainNode):
        raise ValueError("Registered node is not an AiiDA WorkChain.")

    if str(UUID(str(parent.uuid))) != canonical_uuid:
        raise ValueError("Loaded WorkChain UUID mismatch.")

    reconciled = reconcile_aiida_campaign(campaign_directory)
    records = reconciled.calculations

    if tuple(sorted(records)) != tuple(expected_cutoffs):
        raise ValueError(
            "Campaign cutoff coverage differs from registry."
        )

    outgoing = parent.base.links.get_outgoing(
        link_type=LinkType.CALL_CALC
    ).all()

    if len(outgoing) != len(records):
        raise ValueError(
            "WorkChain CALL_CALC child count differs from campaign."
        )

    children_by_cutoff = {}
    seen_pks = set()
    seen_uuids = set()

    for link in outgoing:
        child = link.node

        if not isinstance(child, CalcJobNode):
            raise ValueError(
                "WorkChain CALL_CALC child is not a CalcJobNode."
            )

        if child.pk in seen_pks or str(child.uuid) in seen_uuids:
            raise ValueError("Duplicate WorkChain calculation child.")

        seen_pks.add(child.pk)
        seen_uuids.add(str(child.uuid))

        try:
            parameters = child.inputs.parameters.get_dict()
            cutoff = parameters["FORCE_EVAL"]["DFT"][
                "MGRID"
            ]["CUTOFF"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(
                "Child calculation is missing CP2K cutoff evidence."
            ) from exc

        if type(cutoff) not in (int, float) or (
            cutoff not in records
        ):
            raise ValueError(
                "Child CP2K cutoff is not in campaign."
            )

        if cutoff in children_by_cutoff:
            raise ValueError(
                "Duplicate CP2K cutoff among WorkChain children."
            )

        children_by_cutoff[cutoff] = child

    if set(children_by_cutoff) != set(records):
        raise ValueError(
            "WorkChain child cutoff coverage differs from campaign."
        )

    for cutoff, record in records.items():
        child = children_by_cutoff[cutoff]

        if type(record["pk"]) is not int or (
            child.pk != record["pk"]
        ):
            raise ValueError(
                f"AiiDA PK mismatch at {cutoff} Ry."
            )

        try:
            record_uuid = str(UUID(record["uuid"]))
            child_uuid = str(UUID(str(child.uuid)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                f"Invalid AiiDA UUID at {cutoff} Ry."
            ) from exc

        if child_uuid != record_uuid:
            raise ValueError(
                f"AiiDA UUID mismatch at {cutoff} Ry."
            )
