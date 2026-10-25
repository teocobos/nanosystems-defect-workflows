
"""Extract validated CP2K multigrid datasets from campaigns."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path

from nsdw.calculators.cp2k.parser import parse_cp2k_output
from nsdw.workflows.convergence.manifest import (
    load_convergence_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceParameter,
)
from nsdw.workflows.convergence.multigrid_plotting import (
    load_multigrid_dataset,
)


def extract_multigrid_dataset(
    campaign_directory: str | Path,
    *,
    destination: str | Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Export a CUTOFF campaign's multigrid data.

    Returns None when the campaign contains no multigrid data.
    Rejects partially available multigrid information rather
    than silently presenting an incomplete candidate sweep.
    """
    campaign = Path(campaign_directory).expanduser().resolve()

    manifest = load_convergence_manifest(
        campaign / "manifest.json"
    )

    if manifest.calculator != "cp2k":
        raise ValueError("Expected a CP2K campaign.")

    if manifest.parameter != ConvergenceParameter.CUTOFF:
        return None

    records = []
    missing = []
    missing_outputs = []
    available_outputs = 0

    for candidate in manifest.candidates:
        directory = campaign / candidate.directory
        derived = Path(candidate.input_file).with_suffix(".out")

        preferred = directory / derived
        legacy = directory / "calculation.out"

        if preferred.is_file():
            output = preferred
        elif legacy.is_file():
            output = legacy
        else:
            missing_outputs.append(candidate.label)
            continue

        available_outputs += 1
        parsed = parse_cp2k_output(output)

        if parsed.multigrid is None:
            missing.append(candidate.label)
            continue

        value = candidate.value
        if not hasattr(value, "value") or value.unit != "Ry":
            raise ValueError(
                f"Expected cutoff in Ry for {candidate.label}."
            )

        grid = parsed.multigrid

        records.append({
            "cutoff_ry": float(value.value),
            "grid_numbers": [
                level.grid_number for level in grid.levels
            ],
            "counts": [
                level.count for level in grid.levels
            ],
            "grid_cutoffs_au": [
                level.cutoff_au for level in grid.levels
            ],
            "reported_total": grid.total_gridlevel_count,
        })

    if available_outputs == 0:
        return None

    if missing_outputs:
        raise ValueError(
            "Incomplete CP2K output coverage. Missing candidates: "
            + ", ".join(missing_outputs)
        )

    if missing:
        if len(missing) == len(manifest.candidates):
            return None

        raise ValueError(
            "Incomplete multigrid coverage. Missing candidates: "
            + ", ".join(missing)
        )

    totals = {record["reported_total"] for record in records}

    if len(totals) > 1:
        warnings.warn(
            "CP2K reported different total grid-level counts "
            "across candidates. Raw counts may not be directly "
            "comparable; inspect calculation settings and "
            "parallel decomposition.",
            UserWarning,
            stacklevel=2,
        )

    output = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else campaign / "figures" / "multigrid_data.json"
    )

    if output.exists() and not overwrite:
        existing = load_multigrid_dataset(output)

        # Compare normalized scientific content, not JSON formatting.
        with tempfile.TemporaryDirectory() as temporary_directory:
            comparison = (
                Path(temporary_directory) / "candidate.json"
            )
            comparison.write_text(
                json.dumps(records, indent=2) + "\n",
                encoding="utf-8",
            )
            current = load_multigrid_dataset(comparison)

        if existing != current:
            raise ValueError(
                "Existing multigrid dataset differs from current "
                "CP2K outputs. Explicit overwrite is required."
            )

        return output

    output.parent.mkdir(parents=True, exist_ok=True)

    # Validate using the existing scientific dataset model
    # before replacing any destination file.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=".multigrid-",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(records, handle, indent=2)
        handle.write("\n")

    try:
        load_multigrid_dataset(temporary)

        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {output}"
            )

        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return output
