"""Construct CP2K inputs for successive AiiDA convergence stages."""

from __future__ import annotations

from copy import deepcopy
import math

from nsdw.workflows.convergence.recipes import (
    build_standard_cp2k_cutoff_study,
    build_standard_cp2k_relative_cutoff_study,
)


def build_relative_cutoff_candidates(
    *,
    base_parameters: dict,
    cutoff_analysis: dict,
    candidate_count: int = 5,
) -> dict[str, dict]:
    """Build relative-cutoff candidates from a previous cutoff report.

    The selected cutoff is carried forward, but remains provisional
    unless independently certified.
    """
    report = cutoff_analysis["report"]

    if report["parameter"] != "cutoff":
        raise ValueError("Previous analysis must be a cutoff study.")

    if not report["energy_tolerance_satisfied"]:
        raise ValueError("Cutoff energy tolerance was not satisfied.")

    selected_label = report["selected_candidate_label"]

    if selected_label is None:
        raise ValueError("Cutoff analysis selected no candidate.")

    cutoff_study = build_standard_cp2k_cutoff_study()

    selected = next(
        (
            candidate
            for candidate in cutoff_study.candidates
            if candidate.label == selected_label
        ),
        None,
    )

    if selected is None:
        raise ValueError("Selected cutoff is not in the standard grid.")

    cutoff_ry = selected.value.value

    if not math.isfinite(cutoff_ry) or cutoff_ry <= 0:
        raise ValueError("Invalid selected cutoff.")

    relative_study = build_standard_cp2k_relative_cutoff_study()

    if (
        type(candidate_count) is not int
        or not 2 <= candidate_count <= len(relative_study.candidates)
    ):
        raise ValueError("Invalid relative-cutoff candidate count.")

    candidates = {}

    for candidate in relative_study.candidates[:candidate_count]:
        parameters = deepcopy(base_parameters)
        mgrid = parameters["FORCE_EVAL"]["DFT"]["MGRID"]

        mgrid["CUTOFF"] = cutoff_ry
        mgrid["REL_CUTOFF"] = candidate.value.value

        candidates[candidate.label] = parameters

    return candidates
