
"""Reconcile AiiDA convergence campaign provenance records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReconciledAiidaCampaign:
    """Validated calculation identities and provenance findings."""

    campaign_directory: Path
    calculations: dict[int, dict[str, object]]
    missing_manifest_cutoffs: tuple[int, ...]
    campaign_name: str | None = None

    @property
    def cutoffs(self) -> tuple[int, ...]:
        return tuple(sorted(self.calculations))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required campaign record missing: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object: {path}"
        )

    return data


def _index_records(records, *, source: str) -> dict[int, dict]:
    if not isinstance(records, list):
        raise ValueError(
            f"Expected a list of calculations in {source}."
        )

    indexed = {}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid calculation record in {source}."
            )

        cutoff = int(record["cutoff_ry"])

        if cutoff <= 0:
            raise ValueError(
                f"Invalid cutoff {cutoff} Ry in {source}."
            )

        if cutoff in indexed:
            raise ValueError(
                f"Duplicate cutoff {cutoff} Ry in {source}."
            )

        indexed[cutoff] = record

    return indexed


def reconcile_aiida_campaign(
    campaign_directory: str | Path,
) -> ReconciledAiidaCampaign:
    """Reconcile three independent campaign evidence records.

    Results and validation summary must cover the same
    cutoff values. The manifest may omit entries, but any
    entries it contains must agree with those records.
    """
    root = Path(
        campaign_directory
    ).expanduser().resolve()

    manifest = _read_json(root / "manifest.json")
    results = _read_json(root / "cutoff_results.json")
    summary = _read_json(root / "validation_summary.json")

    manifest_records = manifest.get("calculations")

    if not isinstance(manifest_records, dict):
        raise ValueError(
            "Manifest calculations must be a mapping."
        )

    results_by_cutoff = _index_records(
        results.get("results"),
        source="cutoff_results.json",
    )

    summary_by_cutoff = _index_records(
        summary.get("calculations"),
        source="validation_summary.json",
    )

    if not results_by_cutoff:
        raise ValueError(
            "Campaign contains no calculation results."
        )

    if set(results_by_cutoff) != set(summary_by_cutoff):
        raise ValueError(
            "Cutoff coverage differs between results "
            "and validation summary."
        )

    manifest_by_cutoff = {}

    for key, record in manifest_records.items():
        cutoff = int(key)

        if cutoff in manifest_by_cutoff:
            raise ValueError(
                f"Duplicate manifest cutoff {cutoff} Ry."
            )

        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid manifest record at {cutoff} Ry."
            )

        if (
            "cutoff_ry" in record
            and int(record["cutoff_ry"]) != cutoff
        ):
            raise ValueError(
                f"Manifest cutoff mismatch at {cutoff} Ry."
            )

        manifest_by_cutoff[cutoff] = record

    unknown_manifest_cutoffs = (
        set(manifest_by_cutoff) - set(results_by_cutoff)
    )

    if unknown_manifest_cutoffs:
        raise ValueError(
            "Manifest contains cutoffs absent from "
            "validated results: "
            + ", ".join(
                str(value)
                for value in sorted(unknown_manifest_cutoffs)
            )
        )

    calculations = {}

    for cutoff in sorted(results_by_cutoff):
        result = results_by_cutoff[cutoff]
        validation = summary_by_cutoff[cutoff]
        manifest_record = manifest_by_cutoff.get(cutoff, {})

        pk_values = {
            int(value)
            for value in (
                result.get("aiida_pk"),
                validation.get("aiida_pk"),
                manifest_record.get("pk"),
            )
            if value is not None
        }

        if len(pk_values) != 1:
            raise ValueError(
                f"Missing or inconsistent AiiDA PK "
                f"at {cutoff} Ry: {sorted(pk_values)}"
            )

        pk = next(iter(pk_values))

        if pk <= 0:
            raise ValueError(
                f"Invalid AiiDA PK at {cutoff} Ry."
            )

        uuid_values = {
            str(value)
            for value in (
                validation.get("uuid"),
                manifest_record.get("uuid"),
            )
            if value
        }

        if len(uuid_values) != 1:
            raise ValueError(
                f"Missing or inconsistent AiiDA UUID "
                f"at {cutoff} Ry."
            )

        if result.get("finished_ok") is not True:
            raise ValueError(
                f"Calculation at {cutoff} Ry is not "
                "recorded as successfully completed."
            )

        if validation.get("exit_status") != 0:
            raise ValueError(
                f"Calculation at {cutoff} Ry has a "
                "nonzero or missing recorded exit status."
            )

        calculations[cutoff] = {
            "pk": pk,
            "uuid": next(iter(uuid_values)),
        }

    missing = tuple(
        sorted(
            set(calculations) - set(manifest_by_cutoff)
        )
    )

    return ReconciledAiidaCampaign(
        campaign_directory=root,
        calculations=calculations,
        missing_manifest_cutoffs=missing,
        campaign_name=manifest.get("campaign"),
    )
