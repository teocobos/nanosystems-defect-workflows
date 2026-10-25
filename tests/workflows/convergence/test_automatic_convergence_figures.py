
from pathlib import Path
from unittest.mock import Mock

from nsdw.workflows.convergence.cp2k_workflow import (
    _build_and_write_convergence_report,
)


def test_figures_generated_by_default(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "extract_multigrid_dataset",
        lambda *args, **kwargs: None,
    )

    report = Mock()
    report.write_plot = Mock()

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "analyse_convergence_energy",
        Mock(),
    )
    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "analyse_convergence_against_reference",
        Mock(),
    )
    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "analyse_convergence_tail_stability",
        Mock(),
    )
    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "build_convergence_report",
        lambda **kwargs: report,
    )

    _build_and_write_convergence_report(
        campaign_directory=tmp_path,
        study=Mock(),
        observations=[],
    )

    assert report.write_plot.call_count == 3
    assert (tmp_path / "figures").is_dir()


def test_figures_can_be_disabled(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "extract_multigrid_dataset",
        lambda *args, **kwargs: None,
    )

    report = Mock()

    for name in (
        "analyse_convergence_energy",
        "analyse_convergence_against_reference",
        "analyse_convergence_tail_stability",
    ):
        monkeypatch.setattr(
            "nsdw.workflows.convergence.cp2k_workflow."
            + name,
            Mock(),
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "build_convergence_report",
        lambda **kwargs: report,
    )

    _build_and_write_convergence_report(
        campaign_directory=tmp_path,
        study=Mock(),
        observations=[],
        generate_figures=False,
    )

    report.write_plot.assert_not_called()
    assert not (tmp_path / "figures").exists()


def test_multigrid_dataset_generated_without_figures(
    monkeypatch,
    tmp_path: Path,
):
    report = Mock()

    for name in (
        "analyse_convergence_energy",
        "analyse_convergence_against_reference",
        "analyse_convergence_tail_stability",
    ):
        monkeypatch.setattr(
            "nsdw.workflows.convergence.cp2k_workflow."
            + name,
            Mock(),
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "build_convergence_report",
        lambda **kwargs: report,
    )

    extraction = Mock(return_value=None)

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "extract_multigrid_dataset",
        extraction,
    )

    _build_and_write_convergence_report(
        campaign_directory=tmp_path,
        study=Mock(),
        observations=[],
        generate_figures=False,
    )

    extraction.assert_called_once()
    report.write_plot.assert_not_called()


def test_multigrid_figures_generated_when_available(
    monkeypatch,
    tmp_path: Path,
):
    report = Mock()

    for name in (
        "analyse_convergence_energy",
        "analyse_convergence_against_reference",
        "analyse_convergence_tail_stability",
    ):
        monkeypatch.setattr(
            "nsdw.workflows.convergence.cp2k_workflow."
            + name,
            Mock(),
        )

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "build_convergence_report",
        lambda **kwargs: report,
    )

    dataset = tmp_path / "multigrid_data.json"
    dataset.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "extract_multigrid_dataset",
        lambda *args, **kwargs: dataset,
    )

    render = Mock()

    monkeypatch.setattr(
        "nsdw.workflows.convergence.cp2k_workflow."
        "render_multigrid_figures",
        render,
    )

    _build_and_write_convergence_report(
        campaign_directory=tmp_path,
        study=Mock(),
        observations=[],
    )

    render.assert_called_once()
    report.write_plot.assert_called()
