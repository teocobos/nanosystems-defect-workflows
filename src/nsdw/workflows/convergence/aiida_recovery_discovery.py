
"""Read-only discovery of correlated AiiDA recovery processes."""

from dataclasses import dataclass
from enum import Enum

from aiida.orm import QueryBuilder, WorkChainNode

from nsdw.workflows.convergence.aiida_recovery_workchain import (
    CP2KConvergenceRecoveryWorkChain,
)

CORRELATION_LABEL_FIELD = "label"


class DiscoveryStatus(str, Enum):
    NOT_FOUND = "not_found"
    FOUND = "found"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class DiscoveryResult:
    status: DiscoveryStatus
    process_uuid: str | None
    match_count: int


def discover_recovery_parent(
    correlation_id: str,
    *,
    query_factory=QueryBuilder,
) -> DiscoveryResult:
    """Find a recovery parent by exact correlation metadata.

    Discovery does not validate scientific provenance or authorise
    submission, resubmission or acceptance of results.
    """

    if not isinstance(correlation_id, str) or not correlation_id:
        raise ValueError("A non-empty correlation ID is required.")

    query = query_factory()

    query.append(
        WorkChainNode,
        filters={
            "process_type": (
                CP2KConvergenceRecoveryWorkChain.build_process_type()
            ),
            CORRELATION_LABEL_FIELD: correlation_id,
        },
        project=["uuid"],
    )

    matches = query.all()

    if not matches:
        return DiscoveryResult(
            DiscoveryStatus.NOT_FOUND,
            None,
            0,
        )

    if len(matches) != 1:
        return DiscoveryResult(
            DiscoveryStatus.AMBIGUOUS,
            None,
            len(matches),
        )

    return DiscoveryResult(
        DiscoveryStatus.FOUND,
        str(matches[0][0]),
        1,
    )
