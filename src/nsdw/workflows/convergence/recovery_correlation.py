
"""Deterministic correlation identifiers for recovery attempts."""

import hashlib
import json


def recovery_correlation_id(
    scientific_identity: str,
    execution_identity: str,
    attempt_number: int,
) -> str:
    """Identify one recovery attempt without submitting any process."""

    for name, value in (
        ("scientific_identity", scientific_identity),
        ("execution_identity", execution_identity),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("attempt_number must be a positive integer.")

    payload = {
        "schema": "nsdw-recovery-correlation-v1",
        "scientific_identity": scientific_identity,
        "execution_identity": execution_identity,
        "attempt_number": attempt_number,
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return "nsdw-recovery-v1-" + hashlib.sha256(encoded).hexdigest()
