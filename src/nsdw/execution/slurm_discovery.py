"""Read-only SLURM discovery for interrupted submissions."""

from __future__ import annotations

import re
import subprocess
from datetime import date
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from nsdw.execution.slurm import SlurmExecutor
from nsdw.execution.submission_claims import slurm_name_for_token


class DiscoveryState(str, Enum):
    UNIQUE = "unique"
    NONE = "none"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


class SlurmDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: DiscoveryState
    job_ids: tuple[str, ...] = ()
    reason: str


def _query(
    executable: str,
    arguments: list[str],
) -> str:
    """Run a read-only scheduler query."""

    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "SLURM discovery query unavailable."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            "SLURM discovery query failed."
        )

    return result.stdout


def _parse_jobs(
    output: str,
) -> dict[str, str]:
    """Parse scheduler job IDs and names without losing conflicts."""

    jobs: dict[str, str] = {}

    for line in output.splitlines():
        if not line.strip():
            continue

        fields = line.split("|")

        if len(fields) != 2:
            raise ValueError("Malformed SLURM discovery response.")

        job_id, job_name = (value.strip() for value in fields)

        if not job_name:
            raise ValueError("Missing SLURM job name.")

        if re.fullmatch(
            r"[0-9]+(?:_[0-9]+)?\.[A-Za-z0-9_]+",
            job_id,
        ):
            continue

        if not re.fullmatch(
            r"[0-9]+(?:_[0-9]+)?",
            job_id,
        ):
            raise ValueError("Malformed SLURM job ID.")

        previous = jobs.get(job_id)

        if previous is not None and previous != job_name:
            raise ValueError("Conflicting SLURM job names.")

        jobs[job_id] = job_name

    return jobs


def discover_slurm_submission(
    *,
    executor: SlurmExecutor,
    submission_token: str,
    username: str | None = None,
    accounting_start: str | None = None,
) -> SlurmDiscoveryResult:
    """Discover a submission without changing scheduler state."""

    expected_name = slurm_name_for_token(submission_token)

    if username is not None and (
        not username
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", username) is None
    ):
        raise ValueError("Invalid SLURM username.")

    if accounting_start is not None:
        if re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
            accounting_start,
        ) is None:
            raise ValueError("Invalid accounting start date.")

        try:
            date.fromisoformat(accounting_start)
        except ValueError as exc:
            raise ValueError(
                "Invalid accounting start date."
            ) from exc

    squeue_args = [
        "--noheader",
        "--format=%i|%j",
    ]

    sacct_args = [
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,JobName",
    ]

    if username is not None:
        squeue_args.extend(["--user", username])
        sacct_args.extend(["--user", username])

    if accounting_start is not None:
        sacct_args.extend(["--starttime", accounting_start])

    try:
        active_output = _query(
            executor.squeue_executable,
            squeue_args,
        )

        accounting_output = _query(
            executor.sacct_executable,
            sacct_args,
        )

        active = _parse_jobs(active_output)
        accounting = _parse_jobs(accounting_output)

        for job_id in active.keys() & accounting.keys():
            if active[job_id] != accounting[job_id]:
                raise ValueError(
                    "Conflicting scheduler job identity."
                )

    except (RuntimeError, ValueError):
        return SlurmDiscoveryResult(
            state=DiscoveryState.UNAVAILABLE,
            reason="Scheduler evidence could not be verified.",
        )

    job_ids = tuple(
        sorted(
            job_id
            for job_id, name in (active | accounting).items()
            if name == expected_name
        )
    )

    if not job_ids:
        return SlurmDiscoveryResult(
            state=DiscoveryState.NONE,
            reason=(
                "No matching job found; resubmission remains blocked."
            ),
        )

    if len(job_ids) > 1:
        return SlurmDiscoveryResult(
            state=DiscoveryState.AMBIGUOUS,
            job_ids=job_ids,
            reason="Multiple matching SLURM jobs found.",
        )

    return SlurmDiscoveryResult(
        state=DiscoveryState.UNIQUE,
        job_ids=job_ids,
        reason="One matching SLURM job identified.",
    )
