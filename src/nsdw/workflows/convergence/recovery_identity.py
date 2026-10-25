"""Deterministic, backend-independent recovery identities."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any


IDENTITY_SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> str:
    """Return a stable JSON representation, rejecting invalid values."""

    def validate(value: Any) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Non-finite values are not permitted.")
            return
        if isinstance(value, list):
            for item in value:
                validate(item)
            return
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Identity dictionary keys must be strings.")
            for item in value.values():
                validate(item)
            return
        raise TypeError(
            f"Unsupported identity value: {type(value).__name__}"
        )

    validate(payload)

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(payload: dict[str, Any]) -> str:
    return sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def scientific_recovery_identity(
    *,
    original_campaign_uuid: str,
    candidate_parameters: dict[str, Any],
    structure_uuid: str,
    code_input_identity: str,
) -> str:
    """Identify the scientific recovery independently of HPC resources.

    code_input_identity should describe the scientific method/input
    specification, not a cluster-specific AiiDA Code UUID.
    """
    if not original_campaign_uuid.strip():
        raise ValueError("Original campaign UUID is required.")
    if not structure_uuid.strip():
        raise ValueError("Structure UUID is required.")
    if not code_input_identity.strip():
        raise ValueError("Scientific code/input identity is required.")
    if not candidate_parameters:
        raise ValueError("Candidate parameters cannot be empty.")

    return _digest({
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "kind": "scientific_recovery",
        "original_campaign_uuid": original_campaign_uuid,
        "structure_uuid": structure_uuid,
        "code_input_identity": code_input_identity,
        "candidate_parameters": candidate_parameters,
    })


def execution_attempt_identity(
    *,
    scientific_identity: str,
    backend: str,
    computer_uuid: str,
    resources: dict[str, Any],
    attempt_number: int,
) -> str:
    """Identify a particular execution attempt on an HPC backend."""
    if not scientific_identity.strip():
        raise ValueError("Scientific identity is required.")
    if not backend.strip():
        raise ValueError("Backend is required.")
    if not computer_uuid.strip():
        raise ValueError("Computer UUID is required.")
    if not isinstance(attempt_number, int) or isinstance(
        attempt_number, bool
    ) or attempt_number < 1:
        raise ValueError("Attempt number must be a positive integer.")

    return _digest({
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "kind": "execution_attempt",
        "scientific_identity": scientific_identity,
        "backend": backend,
        "computer_uuid": computer_uuid,
        "resources": resources,
        "attempt_number": attempt_number,
    })


def verify_execution_attempt_identity(
    *,
    recorded_identity: str,
    scientific_identity: str,
    backend: str,
    computer_uuid: str,
    resources: dict[str, Any],
    attempt_number: int,
) -> bool:
    """Verify a recorded execution identity against explicit inputs.

    This is a pure comparison. Callers must independently establish
    that the supplied inputs correspond to the actual execution.
    """
    if (
        not isinstance(recorded_identity, str)
        or len(recorded_identity) != 64
        or any(character not in "0123456789abcdef"
               for character in recorded_identity)
    ):
        raise ValueError(
            "Invalid recorded identity: expected lowercase SHA-256 hex."
        )

    expected = execution_attempt_identity(
        scientific_identity=scientific_identity,
        backend=backend,
        computer_uuid=computer_uuid,
        resources=resources,
        attempt_number=attempt_number,
    )

    if recorded_identity != expected:
        raise ValueError("Execution identity mismatch.")

    return True
