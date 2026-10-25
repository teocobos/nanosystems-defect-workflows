
"""Generate multigrid reports from completed AiiDA campaigns."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nsdw.workflows.convergence.aiida_campaign_reconciliation import (
    reconcile_aiida_campaign,
)
from nsdw.workflows.convergence.aiida_multigrid import (
    extract_aiida_multigrid_dataset,
)
from nsdw.workflows.convergence.multigrid_plotting import (
    load_multigrid_dataset,
    render_multigrid_figures,
)


@dataclass(frozen=True)
class AiidaMultigridReport:
    dataset: Path
    figure_directory: Path
    cutoffs: tuple[int, ...]
    missing_manifest_cutoffs: tuple[int, ...]



from nsdw.workflows.convergence.atomic_reporting import (
    publish_report_directory,
)

def generate_aiida_multigrid_report(
    campaign_directory: str | Path,
    output_directory: str | Path,
) -> AiidaMultigridReport:
    """Validate and atomically publish a completed AiiDA report."""
    campaign = reconcile_aiida_campaign(campaign_directory)
    destination = Path(output_directory).expanduser().absolute()

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Report destination already exists: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary)

        dataset = extract_aiida_multigrid_dataset(
            campaign.calculations,
            staging / "multigrid_data.json",
        )

        load_multigrid_dataset(dataset)

        render_multigrid_figures(
            dataset,
            staging,
            formats=("png", "pdf", "svg"),
            dpi=600,
        )

        expected = {
            "multigrid_data.json",
            *(
                f"{name}.{ext}"
                for name in (
                    "gaussian_grid_counts",
                    "gaussian_grid_distribution",
                    "multigrid_cutoff_levels",
                )
                for ext in ("png", "pdf", "svg")
            ),
        }

        missing = sorted(
            name for name in expected
            if not (staging / name).is_file()
        )
        if missing:
            raise RuntimeError(
                "Incomplete AiiDA multigrid report: "
                + ", ".join(missing)
            )

        hashes = {}
        for name in sorted(expected):
            digest = hashlib.sha256()
            with (staging / name).open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(1024 * 1024), b""
                ):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()

        provenance = {
            "schema_version": 1,
            "campaign": getattr(
                campaign, "campaign_name", None
            ),
            "cutoffs_ry": list(campaign.cutoffs),
            "missing_manifest_cutoffs_ry": list(
                campaign.missing_manifest_cutoffs
            ),
            "calculations": {
                str(cutoff): dict(identity)
                for cutoff, identity
                in campaign.calculations.items()
            },
            "figure_formats": ["png", "pdf", "svg"],
            "figure_dpi": 600,
            "sha256": hashes,
        }

        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )

        publish_report_directory(staging, destination)

    return AiidaMultigridReport(
        dataset=destination / "multigrid_data.json",
        figure_directory=destination,
        cutoffs=campaign.cutoffs,
        missing_manifest_cutoffs=(
            campaign.missing_manifest_cutoffs
        ),
    )
