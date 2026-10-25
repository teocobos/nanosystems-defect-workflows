"""Portable metadata for recovery execution tracking."""

from dataclasses import dataclass, replace
from uuid import UUID


@dataclass(frozen=True)
class RecoveryExecutionMetadata:
    """Snapshot of a recovery attempt's external process identity.

    This object does not submit, discover or validate an AiiDA process.
    """

    scientific_identity: str
    execution_identity: str
    engine: str
    computer_uuid: str
    process_uuid: str | None = None

    def __post_init__(self):
        for name in (
            "scientific_identity",
            "execution_identity",
            "engine",
            "computer_uuid",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        if self.process_uuid is not None:
            if not isinstance(self.process_uuid, str):
                raise TypeError("process_uuid must be a string.")
            try:
                parsed = UUID(self.process_uuid)
            except (ValueError, AttributeError) as exc:
                raise ValueError("Invalid process UUID.") from exc
            if str(parsed) != self.process_uuid:
                raise ValueError(
                    "Process UUID must use canonical lowercase format."
                )


def attach_process_uuid(
    metadata: RecoveryExecutionMetadata,
    process_uuid: str,
) -> RecoveryExecutionMetadata:
    """Bind a process UUID once; never silently replace it."""
    if not isinstance(metadata, RecoveryExecutionMetadata):
        raise TypeError("Expected RecoveryExecutionMetadata.")

    updated = replace(metadata, process_uuid=process_uuid)

    if metadata.process_uuid is not None:
        if metadata.process_uuid != updated.process_uuid:
            raise ValueError(
                "An execution attempt is already bound "
                "to a different process."
            )
        return metadata

    return updated


def requires_process_reconciliation(
    metadata: RecoveryExecutionMetadata,
    submission_started: bool,
) -> bool:
    """Flag a started submission without a recorded process UUID."""
    if not isinstance(metadata, RecoveryExecutionMetadata):
        raise TypeError("Expected RecoveryExecutionMetadata.")
    if type(submission_started) is not bool:
        raise TypeError("submission_started must be a Python bool.")

    return submission_started and metadata.process_uuid is None
