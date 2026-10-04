"""Tests for standard NSDW convergence recipes."""

from pathlib import Path

import pytest

from pymatgen.core import Lattice, Structure

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KInputConfig,
    CP2KKindConfig,
)
from nsdw.workflows.convergence.cp2k_generation import (
    generate_cp2k_convergence_study,
)
from nsdw.workflows.convergence.models import (
    ConvergenceParameter,
)
from nsdw.workflows.convergence.recipes import (
    STANDARD_CP2K_CUTOFF_RY,
    STANDARD_CP2K_RELATIVE_CUTOFF_RY,
    ConvergenceRecipeError,
    apply_selected_cutoff,
    build_standard_cp2k_cutoff_study,
    build_standard_cp2k_relative_cutoff_study,
    selected_ry_value,
)
from nsdw.workflows.convergence.analyser import (
    ConvergenceTailAnalysis,
)



def test_standard_cp2k_cutoff_grid() -> None:
    assert STANDARD_CP2K_CUTOFF_RY == (
        400.0,
        450.0,
        500.0,
        560.0,
        600.0,
        650.0,
        700.0,
        750.0,
        800.0,
        850.0,
    )


def test_standard_cp2k_relative_cutoff_grid() -> None:
    assert STANDARD_CP2K_RELATIVE_CUTOFF_RY == (
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
    )



def test_build_standard_cp2k_cutoff_study() -> None:
    study = build_standard_cp2k_cutoff_study()

    assert study.parameter == ConvergenceParameter.CUTOFF
    assert len(study.candidates) == 10

    assert tuple(
        candidate.value.value
        for candidate in study.candidates
    ) == STANDARD_CP2K_CUTOFF_RY

    assert tuple(
        candidate.order
        for candidate in study.candidates
    ) == tuple(range(10))

    assert tuple(
        candidate.label
        for candidate in study.candidates
    ) == (
        "400-Ry",
        "450-Ry",
        "500-Ry",
        "560-Ry",
        "600-Ry",
        "650-Ry",
        "700-Ry",
        "750-Ry",
        "800-Ry",
        "850-Ry",
    )

    assert all(
        candidate.value.unit == "Ry"
        for candidate in study.candidates
    )


def test_build_standard_cp2k_relative_cutoff_study() -> None:
    study = build_standard_cp2k_relative_cutoff_study()

    assert (
        study.parameter
        == ConvergenceParameter.RELATIVE_CUTOFF
    )
    assert len(study.candidates) == 10

    assert tuple(
        candidate.value.value
        for candidate in study.candidates
    ) == STANDARD_CP2K_RELATIVE_CUTOFF_RY

    assert tuple(
        candidate.order
        for candidate in study.candidates
    ) == tuple(range(10))

    assert tuple(
        candidate.label
        for candidate in study.candidates
    ) == (
        "10-Ry",
        "20-Ry",
        "30-Ry",
        "40-Ry",
        "50-Ry",
        "60-Ry",
        "70-Ry",
        "80-Ry",
        "90-Ry",
        "100-Ry",
    )

    assert all(
        candidate.value.unit == "Ry"
        for candidate in study.candidates
    )


def test_standard_cp2k_studies_use_default_energy_tolerance() -> None:
    cutoff_study = build_standard_cp2k_cutoff_study()
    relative_cutoff_study = (
        build_standard_cp2k_relative_cutoff_study()
    )

    assert (
        cutoff_study.criterion.energy_tolerance_ev_per_atom
        == 1.0e-3
    )
    assert (
        relative_cutoff_study.criterion.energy_tolerance_ev_per_atom
        == 1.0e-3
    )


def test_selected_ry_value_resolves_tail_selection() -> None:
    study = build_standard_cp2k_cutoff_study()

    analysis = ConvergenceTailAnalysis(
        selected_candidate_label="560-Ry",
        comparisons=(),
        energy_tolerance_satisfied=True,
    )

    assert selected_ry_value(
        study=study,
        analysis=analysis,
    ) == 560.0


def test_selected_ry_value_rejects_no_selection() -> None:
    study = build_standard_cp2k_cutoff_study()

    analysis = ConvergenceTailAnalysis(
        selected_candidate_label=None,
        comparisons=(),
        energy_tolerance_satisfied=False,
    )

    with pytest.raises(
        ConvergenceRecipeError,
        match="did not select a candidate",
    ):
        selected_ry_value(
            study=study,
            analysis=analysis,
        )


def test_selected_ry_value_rejects_unknown_selection() -> None:
    study = build_standard_cp2k_cutoff_study()

    analysis = ConvergenceTailAnalysis(
        selected_candidate_label="999-Ry",
        comparisons=(),
        energy_tolerance_satisfied=True,
    )

    with pytest.raises(
        ConvergenceRecipeError,
        match="is not present in the study",
    ):
        selected_ry_value(
            study=study,
            analysis=analysis,
        )


def test_apply_selected_cutoff_updates_next_stage_config() -> None:
    base_config = CP2KInputConfig(
        project_name="igzo",
        run_type="ENERGY",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
    )
    study = build_standard_cp2k_cutoff_study()

    analysis = ConvergenceTailAnalysis(
        selected_candidate_label="560-Ry",
        comparisons=(),
        energy_tolerance_satisfied=True,
    )

    next_config = apply_selected_cutoff(
        base_config=base_config,
        study=study,
        analysis=analysis,
    )

    assert next_config.cutoff_ry == 560.0
    assert next_config.relative_cutoff_ry == 60.0
    assert next_config.project_name == "igzo"
    assert next_config.run_type == "ENERGY"

    # CP2KInputConfig is frozen, so the original configuration
    # must remain unchanged.
    assert base_config.cutoff_ry == 600.0


def test_selected_cutoff_is_preserved_in_relative_cutoff_packages(
    tmp_path: Path,
) -> None:
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["O"],
        coords=[[0.0, 0.0, 0.0]],
    )

    base_config = CP2KInputConfig(
        project_name="test",
        run_type="ENERGY",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
        basis_potential=CP2KBasisPotentialConfig(
            basis_set_file="BASIS_SET",
            potential_file="POTENTIAL",
            kinds=(
                CP2KKindConfig(
                    element="O",
                    basis_set="DZVP-MOLOPT-SR-GTH",
                    potential="GTH-PBE",
                ),
            ),
        ),
    )

    cutoff_study = build_standard_cp2k_cutoff_study()

    cutoff_analysis = ConvergenceTailAnalysis(
        selected_candidate_label="560-Ry",
        comparisons=(),
        energy_tolerance_satisfied=True,
    )

    next_config = apply_selected_cutoff(
        base_config=base_config,
        study=cutoff_study,
        analysis=cutoff_analysis,
    )

    relative_cutoff_study = (
        build_standard_cp2k_relative_cutoff_study()
    )

    output_directory = tmp_path / "relative-cutoff"

    packages = generate_cp2k_convergence_study(
        structure=structure,
        base_config=next_config,
        study=relative_cutoff_study,
        output_directory=output_directory,
    )

    assert len(packages) == 10

    for value in STANDARD_CP2K_RELATIVE_CUTOFF_RY:
        label = f"{value:g}-Ry"

        input_text = next(
            (output_directory / label).glob("*.inp")
        ).read_text()

        assert "CUTOFF 560" in input_text
        assert f"REL_CUTOFF {value:g}" in input_text


def test_apply_selected_cutoff_rejects_wrong_parameter() -> None:
    base_config = CP2KInputConfig(
        project_name="test",
        run_type="ENERGY",
        cutoff_ry=600.0,
        relative_cutoff_ry=60.0,
    )

    study = build_standard_cp2k_relative_cutoff_study()

    analysis = ConvergenceTailAnalysis(
        selected_candidate_label="40-Ry",
        comparisons=(),
        energy_tolerance_satisfied=True,
    )

    with pytest.raises(
        ConvergenceRecipeError,
        match="cutoff study",
    ):
        apply_selected_cutoff(
            base_config=base_config,
            study=study,
            analysis=analysis,
        )