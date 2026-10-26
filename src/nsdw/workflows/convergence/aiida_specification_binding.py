"""Validate scientific specifications against AiiDA candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .aiida_campaign_specification import CampaignSpecification


@dataclass(frozen=True)
class SpecificationBinding:
    specification_sha256: str
    expected_cutoffs: tuple[int, ...]
    candidate_names: tuple[str, ...]


def validate_specification_binding(
    specification: CampaignSpecification,
    candidates: Mapping[str, object],
    candidate_cutoffs: Mapping[str, int],
) -> SpecificationBinding:
    """Require exact agreement between planned cutoffs and candidates.

    This function does not submit calculations or access AiiDA storage.
    """
    if not isinstance(specification, CampaignSpecification):
        raise TypeError("Expected CampaignSpecification.")

    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("Candidates must be a nonempty mapping.")

    if not isinstance(candidate_cutoffs, Mapping):
        raise ValueError("Candidate cutoffs must be a mapping.")

    names = tuple(candidates)

    if any(
        not isinstance(name, str) or not name.strip()
        for name in names
    ):
        raise ValueError("Candidate names must be nonempty strings.")

    if set(names) != set(candidate_cutoffs):
        raise ValueError(
            "Candidate cutoff mapping does not match candidates."
        )

    values = tuple(candidate_cutoffs[name] for name in names)

    if (
        any(type(value) is not int or value <= 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "Candidate cutoffs must be unique positive integers."
        )

    if tuple(sorted(values)) != specification.expected_cutoffs:
        raise ValueError(
            "Candidate cutoffs do not match scientific specification."
        )

    return SpecificationBinding(
        specification_sha256=specification.sha256,
        expected_cutoffs=specification.expected_cutoffs,
        candidate_names=names,
    )


def validate_cp2k_cutoff_parameters(
    candidates: Mapping[str, object],
    candidate_cutoffs: Mapping[str, int],
) -> None:
    """Verify the actual CP2K MGRID/CUTOFF for every candidate.

    Reject missing, ambiguous, nonnumeric, nonfinite or inconsistent
    cutoff values. No AiiDA storage or remote calculations are used.
    """
    import math

    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("Candidates must be a nonempty mapping.")

    if not isinstance(candidate_cutoffs, Mapping):
        raise ValueError("Candidate cutoffs must be a mapping.")

    if set(candidates) != set(candidate_cutoffs):
        raise ValueError(
            "Candidate cutoff mapping does not match candidates."
        )

    for name, parameters in candidates.items():
        expected = candidate_cutoffs[name]

        if type(expected) is not int or expected <= 0:
            raise ValueError(
                f"Invalid declared cutoff for {name!r}."
            )

        if not isinstance(parameters, Mapping):
            raise ValueError(
                f"Invalid CP2K parameters for {name!r}."
            )

        try:
            actual = (
                parameters["FORCE_EVAL"]
                ["DFT"]["MGRID"]["CUTOFF"]
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Missing CP2K MGRID/CUTOFF for {name!r}."
            ) from exc

        if isinstance(actual, bool):
            raise ValueError(
                f"Invalid CP2K cutoff for {name!r}."
            )

        if type(actual) not in (int, float):
            raise ValueError(
                f"CP2K cutoff must be numeric for {name!r}."
            )

        if not math.isfinite(actual) or actual <= 0:
            raise ValueError(
                f"Invalid CP2K cutoff for {name!r}."
            )

        if actual != expected:
            raise ValueError(
                f"CP2K cutoff mismatch for {name!r}: "
                f"declared {expected} Ry, actual {actual} Ry."
            )
