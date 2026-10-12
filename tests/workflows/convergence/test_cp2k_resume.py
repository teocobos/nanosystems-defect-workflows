"""Tests for conservative CP2K convergence resumption planning."""

from pathlib import Path

import pytest

from nsdw.workflows.convergence.cp2k_status import (
    CP2KCampaignStatus,
    CP2KCandidateState,
    CP2KCandidateStatus,
)
from nsdw.workflows.convergence.cp2k_resume import (
    CP2KResumeAction,
    plan_cp2k_convergence_resume,
)


@pytest.mark.parametrize(
    ("state", "expected_action"),
    [
        (CP2KCandidateState.COMPLETED, CP2KResumeAction.PRESERVE),
        (CP2KCandidateState.PENDING, CP2KResumeAction.BLOCK),
        (CP2KCandidateState.INCOMPLETE, CP2KResumeAction.BLOCK),
        (CP2KCandidateState.INVALID, CP2KResumeAction.BLOCK),
    ],
)
def test_resume_planner_actions(
    monkeypatch,
    tmp_path,
    state,
    expected_action,
):
    candidate = CP2KCandidateStatus(
        label="400-Ry",
        state=state,
        reason="Test scientific status",
        input_path=tmp_path / "400-Ry" / "calculation.inp",
        output_path=None,
    )

    status = CP2KCampaignStatus(
        campaign_directory=tmp_path,
        candidates=(candidate,),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda directory: status,
    )

    plan = plan_cp2k_convergence_resume(tmp_path)

    assert len(plan.candidates) == 1
    assert plan.candidates[0].scientific_state == state
    assert plan.candidates[0].action == expected_action
    assert plan.all_completed == (
        state == CP2KCandidateState.COMPLETED
    )


def test_empty_campaign_is_not_completed(monkeypatch, tmp_path):
    status = CP2KCampaignStatus(
        campaign_directory=tmp_path,
        candidates=(),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda directory: status,
    )

    plan = plan_cp2k_convergence_resume(tmp_path)

    assert not plan.all_completed
    assert not plan.requires_attention


def test_pending_candidate_requires_attention(monkeypatch, tmp_path):
    candidate = CP2KCandidateStatus(
        label="10-Ry",
        state=CP2KCandidateState.PENDING,
        reason="No output found",
        input_path=tmp_path / "10-Ry" / "calculation.inp",
        output_path=None,
    )

    status = CP2KCampaignStatus(
        campaign_directory=tmp_path,
        candidates=(candidate,),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda directory: status,
    )

    plan = plan_cp2k_convergence_resume(tmp_path)

    assert plan.requires_attention
    assert plan.candidates[0].action == CP2KResumeAction.BLOCK
    assert "Scheduler state" in plan.candidates[0].reason


def test_slurm_resume_monitors_recorded_running_job(
    monkeypatch, tmp_path
):
    from nsdw.execution.slurm import (
        SlurmJobState,
        SlurmStatusResult,
    )
    from nsdw.execution.submission_records import (
        SlurmSubmissionRecord,
        sha256_file,
    )
    from nsdw.workflows.convergence.cp2k_resume import (
        plan_cp2k_convergence_slurm_resume,
    )

    candidate_dir = tmp_path / "10-Ry"
    candidate_dir.mkdir()
    input_path = candidate_dir / "calculation.inp"
    input_path.write_text("test input")

    record = SlurmSubmissionRecord(
        target="archer2",
        calculation_id="10-Ry",
        job_id="12345",
        submitted_at="2026-10-12T18:00:00+01:00",
        input_file="calculation.inp",
        input_sha256=sha256_file(input_path),
    )
    (candidate_dir / "submission.json").write_text(
        record.model_dump_json()
    )

    from types import SimpleNamespace

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda _: CP2KCampaignStatus(
            campaign_directory=tmp_path,
            candidates=(
                CP2KCandidateStatus(
                    label="10-Ry",
                    state=CP2KCandidateState.PENDING,
                    reason="No output",
                    input_path=input_path,
                    output_path=None,
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.manifest."
        "load_convergence_manifest",
        lambda _: SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    label="10-Ry",
                    directory="10-Ry",
                    input_file="calculation.inp",
                ),
            ),
        ),
    )

    class FakeExecutor:
        def query_status(self, job_id):
            assert job_id == "12345"
            return SlurmStatusResult(
                job_id=job_id,
                state=SlurmJobState.RUNNING,
                raw_state="RUNNING",
            )

    plan = plan_cp2k_convergence_slurm_resume(
        tmp_path,
        target="archer2",
        executor=FakeExecutor(),
    )

    assert plan.candidates[0].action == CP2KResumeAction.MONITOR


def test_slurm_resume_blocks_without_record(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace
    from nsdw.workflows.convergence.cp2k_resume import (
        plan_cp2k_convergence_slurm_resume,
    )

    candidate_dir = tmp_path / "10-Ry"
    candidate_dir.mkdir()

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda _: CP2KCampaignStatus(
            campaign_directory=tmp_path,
            candidates=(
                CP2KCandidateStatus(
                    label="10-Ry",
                    state=CP2KCandidateState.PENDING,
                    reason="No output",
                    input_path=candidate_dir / "calculation.inp",
                    output_path=None,
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.manifest."
        "load_convergence_manifest",
        lambda _: SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    label="10-Ry",
                    directory="10-Ry",
                    input_file="calculation.inp",
                ),
            ),
        ),
    )

    class FakeExecutor:
        def query_status(self, job_id):
            pytest.fail("Scheduler must not be queried without a record.")

    plan = plan_cp2k_convergence_slurm_resume(
        tmp_path,
        target="archer2",
        executor=FakeExecutor(),
    )

    assert plan.candidates[0].action == CP2KResumeAction.BLOCK
    assert "No submission record" in plan.candidates[0].reason


@pytest.mark.parametrize(
    "raw_state",
    [
        "PENDING",
        "RUNNING",
        "COMPLETING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "UNKNOWN",
    ],
)
def test_slurm_resume_scheduler_state_policy(
    monkeypatch, tmp_path, raw_state
):
    from types import SimpleNamespace

    from nsdw.execution.slurm import (
        SlurmStatusResult,
        parse_slurm_state,
    )
    from nsdw.execution.submission_records import (
        SlurmSubmissionRecord,
        sha256_file,
    )
    from nsdw.workflows.convergence.cp2k_resume import (
        plan_cp2k_convergence_slurm_resume,
    )

    candidate_dir = tmp_path / "10-Ry"
    candidate_dir.mkdir()

    input_path = candidate_dir / "calculation.inp"
    input_path.write_text("test input")

    record = SlurmSubmissionRecord(
        target="archer2",
        calculation_id="10-Ry",
        job_id="12345",
        submitted_at="2026-10-12T18:00:00+01:00",
        input_file="calculation.inp",
        input_sha256=sha256_file(input_path),
    )
    (candidate_dir / "submission.json").write_text(
        record.model_dump_json()
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda _: CP2KCampaignStatus(
            campaign_directory=tmp_path,
            candidates=(
                CP2KCandidateStatus(
                    label="10-Ry",
                    state=CP2KCandidateState.PENDING,
                    reason="No output",
                    input_path=input_path,
                    output_path=None,
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.manifest."
        "load_convergence_manifest",
        lambda _: SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    label="10-Ry",
                    directory="10-Ry",
                    input_file="calculation.inp",
                ),
            ),
        ),
    )

    class FakeExecutor:
        def query_status(self, job_id):
            return SlurmStatusResult(
                job_id=job_id,
                state=parse_slurm_state(raw_state),
                raw_state=raw_state,
            )

    plan = plan_cp2k_convergence_slurm_resume(
        tmp_path,
        target="archer2",
        executor=FakeExecutor(),
    )

    expected = (
        CP2KResumeAction.MONITOR
        if raw_state in {
            "PENDING",
            "RUNNING",
            "COMPLETING",
        }
        else CP2KResumeAction.BLOCK
    )

    assert plan.candidates[0].action == expected


def test_slurm_resume_blocks_scheduler_job_id_mismatch(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from nsdw.execution.slurm import (
        SlurmJobState,
        SlurmStatusResult,
    )
    from nsdw.execution.submission_records import (
        SlurmSubmissionRecord,
        sha256_file,
    )
    from nsdw.workflows.convergence.cp2k_resume import (
        plan_cp2k_convergence_slurm_resume,
    )

    candidate_dir = tmp_path / "10-Ry"
    candidate_dir.mkdir()

    input_path = candidate_dir / "calculation.inp"
    input_path.write_text("test input")

    record = SlurmSubmissionRecord(
        target="archer2",
        calculation_id="10-Ry",
        job_id="12345",
        submitted_at="2026-10-12T18:00:00+01:00",
        input_file="calculation.inp",
        input_sha256=sha256_file(input_path),
    )
    (candidate_dir / "submission.json").write_text(
        record.model_dump_json()
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda _: CP2KCampaignStatus(
            campaign_directory=tmp_path,
            candidates=(
                CP2KCandidateStatus(
                    label="10-Ry",
                    state=CP2KCandidateState.PENDING,
                    reason="No output",
                    input_path=input_path,
                    output_path=None,
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.manifest."
        "load_convergence_manifest",
        lambda _: SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    label="10-Ry",
                    directory="10-Ry",
                    input_file="calculation.inp",
                ),
            ),
        ),
    )

    class FakeExecutor:
        def query_status(self, job_id):
            return SlurmStatusResult(
                job_id="99999",
                state=SlurmJobState.RUNNING,
                raw_state="RUNNING",
            )

    plan = plan_cp2k_convergence_slurm_resume(
        tmp_path,
        target="archer2",
        executor=FakeExecutor(),
    )

    assert plan.candidates[0].action == CP2KResumeAction.BLOCK


@pytest.mark.parametrize(
    "claim_mode",
    ["valid", "malformed", "wrong_target", "wrong_calculation", "wrong_hash"],
)
def test_slurm_resume_blocks_unreconciled_claim(
    monkeypatch, tmp_path, claim_mode
):
    from types import SimpleNamespace

    from nsdw.execution.submission_claims import (
        SubmissionClaim,
        create_submission_claim,
    )
    from nsdw.execution.submission_records import sha256_file
    from nsdw.workflows.convergence.cp2k_resume import (
        plan_cp2k_convergence_slurm_resume,
    )

    candidate_dir = tmp_path / "10-Ry"
    candidate_dir.mkdir()

    input_path = candidate_dir / "calculation.inp"
    input_path.write_text("test input")

    if claim_mode == "malformed":
        (candidate_dir / "submission-claim.json").write_text("{partial")
    else:
        claim = SubmissionClaim(
            target=(
                "other-cluster"
                if claim_mode == "wrong_target"
                else "archer2"
            ),
            calculation_id=(
                "20-Ry"
                if claim_mode == "wrong_calculation"
                else "10-Ry"
            ),
            input_sha256=(
                "a" * 64
                if claim_mode == "wrong_hash"
                else sha256_file(input_path)
            ),
        )
        create_submission_claim(
            candidate_directory=candidate_dir,
            claim=claim,
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda _: CP2KCampaignStatus(
            campaign_directory=tmp_path,
            candidates=(
                CP2KCandidateStatus(
                    label="10-Ry",
                    state=CP2KCandidateState.PENDING,
                    reason="No output",
                    input_path=input_path,
                    output_path=None,
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.manifest."
        "load_convergence_manifest",
        lambda _: SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    label="10-Ry",
                    directory="10-Ry",
                    input_file="calculation.inp",
                ),
            ),
        ),
    )

    class FakeExecutor:
        def query_status(self, job_id):
            pytest.fail(
                "SLURM must not be queried for unreconciled claims."
            )

    plan = plan_cp2k_convergence_slurm_resume(
        tmp_path,
        target="archer2",
        executor=FakeExecutor(),
    )

    assert plan.candidates[0].action == CP2KResumeAction.BLOCK


@pytest.mark.parametrize(
    "claim_version,record_version,matching_token,expected_action",
    [
        (1, 1, True, CP2KResumeAction.MONITOR),
        (2, 2, True, CP2KResumeAction.MONITOR),
        (2, 2, False, CP2KResumeAction.BLOCK),
        (2, 1, True, CP2KResumeAction.BLOCK),
        (1, 2, True, CP2KResumeAction.BLOCK),
    ],
)
def test_slurm_resume_requires_matching_claim_record_identity(
    monkeypatch,
    tmp_path,
    claim_version,
    record_version,
    matching_token,
    expected_action,
):
    from types import SimpleNamespace

    from nsdw.execution.slurm import SlurmJobState, SlurmStatusResult
    from nsdw.execution.submission_claims import (
        SubmissionClaim,
        TokenisedSubmissionClaim,
        create_submission_claim,
    )
    from nsdw.execution.submission_records import (
        SlurmSubmissionRecord,
        TokenisedSlurmSubmissionRecord,
        sha256_file,
    )
    from nsdw.workflows.convergence.cp2k_resume import (
        plan_cp2k_convergence_slurm_resume,
    )

    candidate_dir = tmp_path / "10-Ry"
    candidate_dir.mkdir()

    input_path = candidate_dir / "calculation.inp"
    input_path.write_text("test input", encoding="utf-8")

    identity = {
        "target": "archer2",
        "calculation_id": "10-Ry",
        "input_sha256": sha256_file(input_path),
    }

    if claim_version == 1:
        claim = SubmissionClaim(**identity)
    else:
        claim = TokenisedSubmissionClaim(
            **identity,
            submission_token="a" * 32,
        )

    create_submission_claim(
        candidate_directory=candidate_dir,
        claim=claim,
    )

    record_identity = {
        **identity,
        "job_id": "12345",
        "submitted_at": "2026-10-12T18:00:00+01:00",
        "input_file": "calculation.inp",
    }

    if record_version == 1:
        record = SlurmSubmissionRecord(**record_identity)
    else:
        record = TokenisedSlurmSubmissionRecord(
            **record_identity,
            submission_token=(
                "a" * 32 if matching_token else "b" * 32
            ),
        )

    # Deliberately write the record directly to test untrusted
    # on-disk evidence, bypassing the protected publisher.
    (candidate_dir / "submission.json").write_text(
        record.model_dump_json(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_resume."
        "inspect_cp2k_convergence_campaign",
        lambda _: CP2KCampaignStatus(
            campaign_directory=tmp_path,
            candidates=(
                CP2KCandidateStatus(
                    label="10-Ry",
                    state=CP2KCandidateState.PENDING,
                    reason="No output",
                    input_path=input_path,
                    output_path=None,
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.manifest."
        "load_convergence_manifest",
        lambda _: SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    label="10-Ry",
                    directory="10-Ry",
                    input_file="calculation.inp",
                ),
            ),
        ),
    )

    class FakeExecutor:
        def query_status(self, job_id):
            if expected_action == CP2KResumeAction.BLOCK:
                pytest.fail(
                    "Untrusted claim-record evidence must not "
                    "reach the scheduler."
                )

            assert job_id == "12345"

            return SlurmStatusResult(
                job_id=job_id,
                state=SlurmJobState.RUNNING,
                raw_state="RUNNING",
            )

    plan = plan_cp2k_convergence_slurm_resume(
        tmp_path,
        target="archer2",
        executor=FakeExecutor(),
    )

    assert plan.candidates[0].action == expected_action

    if expected_action == CP2KResumeAction.BLOCK:
        assert "Untrusted submission evidence" in (
            plan.candidates[0].reason
        )
