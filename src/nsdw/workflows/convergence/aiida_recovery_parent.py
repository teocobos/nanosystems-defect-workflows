
"""Prepare correlated AiiDA recovery-parent builders."""

from aiida.orm import Bool, Int

from nsdw.workflows.convergence.aiida_recovery_workchain import (
    CP2KConvergenceRecoveryWorkChain,
)


def build_correlated_recovery_parent(
    *,
    original_campaign_pk: int,
    retry_walltime: int,
    correlation_id: str,
):
    """Build a correlated recovery parent without submitting it."""

    if type(original_campaign_pk) is not int or original_campaign_pk < 1:
        raise ValueError("original_campaign_pk must be positive.")

    if type(retry_walltime) is not int or retry_walltime < 1:
        raise ValueError("retry_walltime must be positive.")

    if not isinstance(correlation_id, str) or not correlation_id.strip():
        raise ValueError("A non-empty correlation ID is required.")

    builder = CP2KConvergenceRecoveryWorkChain.get_builder()

    builder.original_campaign_pk = Int(original_campaign_pk)
    builder.retry_walltime = Int(retry_walltime)

    # Both gates remain disabled at this stage.
    builder.submit_retries = Bool(False)
    builder.execution_enabled = Bool(False)

    builder.metadata.label = correlation_id
    builder.metadata.description = (
        "NSDW targeted CP2K convergence recovery parent. "
        "Correlation metadata is not provenance verification."
    )

    return builder
