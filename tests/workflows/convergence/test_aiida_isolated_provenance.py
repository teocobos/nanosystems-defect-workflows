"""Isolated AiiDA provenance integration tests."""

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.aiida_integration
def test_isolated_aiida_profile_can_store_and_reload_node(
    aiida_profile,
):
    """Validate real AiiDA storage without HPC execution."""
    from aiida.orm import Int, load_node

    node = Int(42).store()

    assert node.is_stored
    assert node.pk is not None

    reloaded = load_node(node.uuid)

    assert reloaded.uuid == node.uuid
    assert reloaded.value == 42
