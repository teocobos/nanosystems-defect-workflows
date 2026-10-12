"""Tests for read-only token-based SLURM discovery."""

from unittest.mock import patch

import pytest

from nsdw.execution.slurm import SlurmExecutor
from nsdw.execution.slurm_discovery import (
    DiscoveryState,
    discover_slurm_submission,
)

TOKEN = "a" * 32
NAME = f"nsdw-{TOKEN}"


def _discover(active: str, accounting: str):
    outputs = [active, accounting]

    def fake_query(executable, arguments):
        return outputs.pop(0)

    with patch(
        "nsdw.execution.slurm_discovery._query",
        side_effect=fake_query,
    ):
        return discover_slurm_submission(
            executor=SlurmExecutor(),
            submission_token=TOKEN,
        )


def test_discovery_unique_active_job():
    result = _discover(f"123|{NAME}\n", "")

    assert result.state is DiscoveryState.UNIQUE
    assert result.job_ids == ("123",)


def test_discovery_unique_accounting_job():
    result = _discover("", f"123|{NAME}\n")

    assert result.state is DiscoveryState.UNIQUE
    assert result.job_ids == ("123",)


def test_discovery_deduplicates_same_job():
    result = _discover(
        f"123|{NAME}\n",
        f"123|{NAME}\n",
    )

    assert result.state is DiscoveryState.UNIQUE
    assert result.job_ids == ("123",)


def test_discovery_rejects_multiple_matches():
    result = _discover(
        f"123|{NAME}\n",
        f"456|{NAME}\n",
    )

    assert result.state is DiscoveryState.AMBIGUOUS
    assert result.job_ids == ("123", "456")


def test_discovery_no_match_remains_blocked():
    result = _discover("123|other-job\n", "")

    assert result.state is DiscoveryState.NONE
    assert result.job_ids == ()


def test_discovery_ignores_job_steps():
    result = _discover(
        "",
        f"123.batch|{NAME}\n123|{NAME}\n",
    )

    assert result.state is DiscoveryState.UNIQUE
    assert result.job_ids == ("123",)


@pytest.mark.parametrize(
    "output",
    ["invalid-line", "abc|job", "123|"],
)
def test_discovery_rejects_malformed_output(output):
    result = _discover(output, "")

    assert result.state is DiscoveryState.UNAVAILABLE


def test_discovery_handles_scheduler_failure():
    with patch(
        "nsdw.execution.slurm_discovery._query",
        side_effect=RuntimeError("scheduler unavailable"),
    ):
        result = discover_slurm_submission(
            executor=SlurmExecutor(),
            submission_token=TOKEN,
        )

    assert result.state is DiscoveryState.UNAVAILABLE


def test_discovery_rejects_cross_source_identity_conflict():
    result = _discover(
        f"123|{NAME}\n",
        "123|different-job\n",
    )

    assert result.state is DiscoveryState.UNAVAILABLE


def test_discovery_rejects_conflicting_names_in_one_source():
    result = _discover(
        f"123|{NAME}\n123|different-job\n",
        "",
    )

    assert result.state is DiscoveryState.UNAVAILABLE


def test_discovery_rejects_malformed_step_identity():
    result = _discover(
        "",
        "123.batch|",
    )

    assert result.state is DiscoveryState.UNAVAILABLE


def test_discovery_ignores_batch_and_extern_steps():
    result = _discover(
        "",
        (
            f"123.batch|{NAME}\n"
            f"123.extern|{NAME}\n"
            f"123|{NAME}\n"
        ),
    )

    assert result.state is DiscoveryState.UNIQUE
    assert result.job_ids == ("123",)


def test_discovery_scopes_scheduler_queries():
    from unittest.mock import call

    with patch(
        "nsdw.execution.slurm_discovery._query",
        side_effect=["", ""],
    ) as query:
        result = discover_slurm_submission(
            executor=SlurmExecutor(),
            submission_token=TOKEN,
            username="uccatco",
            accounting_start="2026-10-01",
        )

    assert result.state is DiscoveryState.NONE

    assert query.call_args_list == [
        call(
            "squeue",
            [
                "--noheader",
                "--format=%i|%j",
                "--user",
                "uccatco",
            ],
        ),
        call(
            "sacct",
            [
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,JobName",
                "--user",
                "uccatco",
                "--starttime",
                "2026-10-01",
            ],
        ),
    ]


@pytest.mark.parametrize(
    "username",
    ["", "bad user", "--allusers", "name;echo"],
)
def test_discovery_rejects_invalid_username(username):
    with pytest.raises(ValueError, match="username"):
        discover_slurm_submission(
            executor=SlurmExecutor(),
            submission_token=TOKEN,
            username=username,
        )


@pytest.mark.parametrize(
    "start",
    ["yesterday", "2026/10/01", "2026-1-1"],
)
def test_discovery_rejects_invalid_start_date(start):
    with pytest.raises(ValueError, match="start date"):
        discover_slurm_submission(
            executor=SlurmExecutor(),
            submission_token=TOKEN,
            accounting_start=start,
        )


@pytest.mark.parametrize(
    "start",
    [
        "2026-02-30",
        "2026-13-01",
        "2025-02-29",
        "2026-04-31",
    ],
)
def test_discovery_rejects_nonexistent_calendar_dates(start):
    with pytest.raises(ValueError, match="start date"):
        discover_slurm_submission(
            executor=SlurmExecutor(),
            submission_token=TOKEN,
            accounting_start=start,
        )


def test_discovery_accepts_valid_leap_day():
    with patch(
        "nsdw.execution.slurm_discovery._query",
        side_effect=["", ""],
    ):
        result = discover_slurm_submission(
            executor=SlurmExecutor(),
            submission_token=TOKEN,
            username="uccatco",
            accounting_start="2024-02-29",
        )

    assert result.state is DiscoveryState.NONE
