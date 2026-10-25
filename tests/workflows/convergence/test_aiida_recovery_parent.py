
import pytest

from nsdw.workflows.convergence.aiida_recovery_parent import (
    build_correlated_recovery_parent,
)


@pytest.fixture
def _load_aiida_profile_for_builder_tests():
    """Load the configured AiiDA profile before constructing ORM nodes."""
    from aiida import load_profile

    load_profile()


@pytest.mark.aiida_integration
@pytest.mark.usefixtures("_load_aiida_profile_for_builder_tests")
def test_parent_builder_contains_correlation():
    builder = build_correlated_recovery_parent(
        original_campaign_pk=268,
        retry_walltime=1800,
        correlation_id="nsdw-recovery-test",
    )

    assert builder.metadata.label == "nsdw-recovery-test"
    assert builder.original_campaign_pk.value == 268
    assert builder.retry_walltime.value == 1800


@pytest.mark.aiida_integration
@pytest.mark.usefixtures("_load_aiida_profile_for_builder_tests")
def test_parent_builder_disables_execution():
    builder = build_correlated_recovery_parent(
        original_campaign_pk=268,
        retry_walltime=1800,
        correlation_id="nsdw-recovery-test",
    )

    assert builder.submit_retries.value is False
    assert builder.execution_enabled.value is False


@pytest.mark.parametrize(
    "campaign,walltime,correlation",
    [
        (0, 1800, "valid"),
        (268, 0, "valid"),
        (268, 1800, ""),
        (True, 1800, "valid"),
        (268, True, "valid"),
        (268, 1800, None),
    ],
)
def test_invalid_parent_inputs_rejected(
    campaign, walltime, correlation
):
    with pytest.raises(ValueError):
        build_correlated_recovery_parent(
            original_campaign_pk=campaign,
            retry_walltime=walltime,
            correlation_id=correlation,
        )
