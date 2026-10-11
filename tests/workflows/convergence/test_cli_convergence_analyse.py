from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from nsdw.cli import app


runner = CliRunner()


def test_convergence_analyse_uses_existing_outputs(
    monkeypatch, tmp_path
):
    calls = []

    def fake_analyse(*, campaign_directory):
        calls.append(campaign_directory)
        return SimpleNamespace(
            parameter=SimpleNamespace(value="cutoff"),
            energy_tolerance_satisfied=True,
            selected_candidate_label="600-Ry",
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "analyse_and_report_existing_cp2k_convergence_campaign",
        fake_analyse,
    )

    result = runner.invoke(
        app,
        ["workflow", "convergence-analyse", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [tmp_path.resolve()]
    assert "600-Ry" in result.output


def test_convergence_analyse_rejects_missing_campaign(
    tmp_path,
):
    missing = tmp_path / "missing"

    result = runner.invoke(
        app,
        ["workflow", "convergence-analyse", str(missing)],
    )

    assert result.exit_code != 0
    assert "ERROR" in result.output
