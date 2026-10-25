
"""Extract CP2K multigrid datasets from AiiDA provenance."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from nsdw.calculators.cp2k.parser import parse_cp2k_output
from nsdw.workflows.convergence.multigrid_plotting import (
    load_multigrid_dataset,
)


def extract_aiida_multigrid_dataset(
    calculations: Mapping[int, Mapping],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Extract a complete multigrid dataset from AiiDA nodes.

    The caller supplies cutoff-to-calculation mappings.
    AiiDA is loaded only when this function is invoked.
    """
    from aiida import orm

    if not calculations:
        raise ValueError("No AiiDA calculations supplied.")

    records = []

    for cutoff, metadata in sorted(calculations.items()):
        pk = metadata.get("pk")

        if pk is None:
            raise ValueError(
                f"Missing AiiDA PK for {cutoff} Ry."
            )

        node = orm.load_node(int(pk))

        if not node.is_finished_ok:
            raise ValueError(
                f"AiiDA PK {pk} did not finish successfully."
            )

        expected_uuid = metadata.get("uuid")

        if (
            expected_uuid is not None
            and str(node.uuid) != str(expected_uuid)
        ):
            raise ValueError(
                f"UUID mismatch for AiiDA PK {pk}."
            )

        retrieved = node.outputs.retrieved
        names = retrieved.base.repository.list_object_names()

        if "aiida.out" not in names:
            raise FileNotFoundError(
                f"Missing aiida.out for AiiDA PK {pk}."
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aiida.out"

            output.write_bytes(
                retrieved.base.repository.get_object_content(
                    "aiida.out",
                    mode="rb",
                )
            )

            parsed = parse_cp2k_output(output)

        if parsed.multigrid is None:
            raise ValueError(
                f"No CP2K multigrid data in PK {pk}."
            )

        grid = parsed.multigrid

        records.append({
            "cutoff_ry": float(cutoff),
            "aiida_pk": int(pk),
            "grid_numbers": [
                level.grid_number
                for level in grid.levels
            ],
            "counts": [
                level.count
                for level in grid.levels
            ],
            "grid_cutoffs_au": [
                level.cutoff_au
                for level in grid.levels
            ],
            "reported_total": grid.total_gridlevel_count,
        })

    destination = Path(destination).expanduser().resolve()

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=".aiida-multigrid-",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(records, handle, indent=2)
        handle.write("\n")

    try:
        load_multigrid_dataset(temporary)

        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {destination}"
            )

        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination
