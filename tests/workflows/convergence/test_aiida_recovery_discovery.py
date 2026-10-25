
import pytest

from nsdw.workflows.convergence.aiida_recovery_discovery import (
    CORRELATION_LABEL_FIELD,
    DiscoveryStatus,
    discover_recovery_parent,
)


class FakeQuery:
    def __init__(self, matches):
        self.matches = matches
        self.filters = None
        self.projection = None

    def append(self, cls, *, filters, project):
        self.filters = filters
        self.projection = project

    def all(self):
        return self.matches


def factory(matches):
    return lambda: FakeQuery(matches)


def test_no_matching_process():
    result = discover_recovery_parent(
        "correlation",
        query_factory=factory([]),
    )

    assert result.status is DiscoveryStatus.NOT_FOUND
    assert result.process_uuid is None
    assert result.match_count == 0


def test_single_matching_process():
    result = discover_recovery_parent(
        "correlation",
        query_factory=factory([["process-uuid"]]),
    )

    assert result.status is DiscoveryStatus.FOUND
    assert result.process_uuid == "process-uuid"
    assert result.match_count == 1


def test_duplicate_matching_processes():
    result = discover_recovery_parent(
        "correlation",
        query_factory=factory([
            ["process-one"],
            ["process-two"],
        ]),
    )

    assert result.status is DiscoveryStatus.AMBIGUOUS
    assert result.process_uuid is None
    assert result.match_count == 2


def test_query_uses_exact_correlation_filter():
    query = FakeQuery([])

    discover_recovery_parent(
        "expected-correlation",
        query_factory=lambda: query,
    )

    assert query.filters[
        CORRELATION_LABEL_FIELD
    ] == "expected-correlation"
    from nsdw.workflows.convergence.aiida_recovery_workchain import CP2KConvergenceRecoveryWorkChain
    assert query.filters["process_type"] == (
        CP2KConvergenceRecoveryWorkChain.build_process_type()
    )

    assert query.projection == ["uuid"]


@pytest.mark.parametrize("value", ["", None, 123])
def test_invalid_correlation_rejected(value):
    with pytest.raises(ValueError):
        discover_recovery_parent(
            value,
            query_factory=factory([]),
        )
