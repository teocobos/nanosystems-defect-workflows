
"""Backend-independent CP2K multigrid figure rendering."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = (
    "#0B2545",
    "#008E99",
    "#5D86AD",
    "#A6C8CC",
    "#607D8B",
    "#8A9BA8",
)

SUPPORTED_FORMATS = ("png", "pdf", "svg")


def load_multigrid_dataset(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load and validate a previously exported multigrid dataset."""

    records = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty list of records.")

    validated = []
    reference_grids = None
    seen_candidates = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object.")

        cutoff = record.get("cutoff_ry")
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, (int, float))
            or not math.isfinite(cutoff)
            or cutoff <= 0
        ):
            raise ValueError("Invalid cutoff_ry.")

        if cutoff in seen_candidates:
            raise ValueError("Duplicate cutoff candidate.")
        seen_candidates.add(cutoff)

        grids = record.get("grid_numbers")
        counts = record.get("counts")
        grid_cutoffs = record.get("grid_cutoffs_au")

        if not all(
            isinstance(value, list)
            for value in (grids, counts, grid_cutoffs)
        ):
            raise ValueError("Missing multigrid arrays.")

        if not grids or not (
            len(grids) == len(counts) == len(grid_cutoffs)
        ):
            raise ValueError("Inconsistent multigrid arrays.")

        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in grids
            )
            or len(set(grids)) != len(grids)
        ):
            raise ValueError("Invalid grid numbers.")

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise ValueError("Invalid grid counts.")

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in grid_cutoffs
        ):
            raise ValueError("Invalid grid cutoff values.")

        if reference_grids is None:
            reference_grids = grids
        elif grids != reference_grids:
            raise ValueError(
                "Grid numbering differs across candidates."
            )

        total = sum(counts)
        reported = record.get("reported_total")

        if (
            total <= 0
            or isinstance(reported, bool)
            or not isinstance(reported, int)
            or reported != total
        ):
            raise ValueError(
                "Reported total does not match grid counts."
            )

        validated.append({
            "cutoff_ry": float(cutoff),
            "grid_numbers": list(grids),
            "counts": list(counts),
            "grid_cutoffs_au": list(grid_cutoffs),
            "reported_total": total,
            "percentages": [
                100.0 * value / total
                for value in counts
            ],
        })

    return sorted(
        validated,
        key=lambda item: item["cutoff_ry"],
    )


def render_multigrid_figures(
    dataset: str | Path,
    output_directory: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
    dpi: int = 600,
) -> list[Path]:
    """Render figures from validated data without HPC access."""

    if (
        isinstance(dpi, bool)
        or not isinstance(dpi, int)
        or dpi < 72
    ):
        raise ValueError("DPI must be an integer >= 72.")

    if (
        not formats
        or any(
            item not in SUPPORTED_FORMATS
            for item in formats
        )
    ):
        raise ValueError("Unsupported figure format.")

    records = load_multigrid_dataset(dataset)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    grid_numbers = records[0]["grid_numbers"]
    positions = np.arange(len(records))
    labels = [
        f'{record["cutoff_ry"]:g}'
        for record in records
    ]

    generated = []

    def save(fig, name):
        for extension in dict.fromkeys(formats):
            path = output / f"{name}.{extension}"
            fig.savefig(
                path,
                dpi=dpi if extension == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
            generated.append(path)
        plt.close(fig)

    def stacked(field, ylabel, name):
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        bottom = np.zeros(len(records))

        for index, grid in enumerate(grid_numbers):
            values = np.array([
                record[field][index]
                for record in records
            ])

            ax.bar(
                positions,
                values,
                bottom=bottom,
                color=COLORS[index % len(COLORS)],
                label=f"Grid {grid}",
                width=0.68,
            )
            bottom += values

        ax.set_xticks(positions, labels)
        ax.set_xlabel("CUTOFF (Ry)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        save(fig, name)

    stacked(
        "counts",
        "Reported grid-level count",
        "gaussian_grid_counts",
    )

    stacked(
        "percentages",
        "Distribution of reported counts (%)",
        "gaussian_grid_distribution",
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for index, grid in enumerate(grid_numbers):
        ax.plot(
            [
                record["cutoff_ry"]
                for record in records
            ],
            [
                record["grid_cutoffs_au"][index]
                for record in records
            ],
            marker="o",
            linewidth=2,
            color=COLORS[index % len(COLORS)],
            label=f"Grid {grid}",
        )

    ax.set_xlabel("CUTOFF (Ry)")
    ax.set_ylabel("Reported grid cutoff (a.u.)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "multigrid_cutoff_levels")

    return generated
