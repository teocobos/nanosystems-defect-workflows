"""Tests for CP2K-specific convergence diagnostics."""

import pytest

from nsdw.calculators.cp2k.convergence import (
    CP2KConvergenceMode,
    assess_cp2k_multigrid,
)
from nsdw.calculators.cp2k.models import (
    ParsedCP2KMultigrid,
    ParsedCP2KMultigridLevel,
)
import math

from nsdw.calculators.cp2k.convergence import (
    assess_cp2k_calculation_validity,
)

from nsdw.calculators.cp2k.models import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KEnergy,
    ParsedCP2KResult,
    ParsedCP2KSCF,
)
from nsdw.calculators.cp2k.convergence import (
    assess_cp2k_input_consistency,
)

from nsdw.calculators.cp2k.models import (
    ParsedCP2KInput,
    ParsedCP2KKind,
)

def make_multigrid(
    counts=(65027435, 30531810, 17861146, 8080035),
    total=121500426,
):
    """Build multigrid data based on the real IGZO CP2K output."""

    return ParsedCP2KMultigrid(
        levels=tuple(
            ParsedCP2KMultigridLevel(
                grid_number=index,
                count=count,
                cutoff_au=cutoff,
            )
            for index, (count, cutoff) in enumerate(
                zip(
                    counts,
                    (200.0, 66.67, 22.22, 7.41),
                ),
                start=1,
            )
        ),
        total_gridlevel_count=total,
    )


def test_valid_igzo_multigrid():
    assessment = assess_cp2k_multigrid(make_multigrid())

    assert assessment.available
    assert assessment.internally_consistent
    assert assessment.grid_counts == (
        65027435,
        30531810,
        17861146,
        8080035,
    )
    assert sum(assessment.grid_fractions) == pytest.approx(1.0)
    assert assessment.issues == ()


def test_missing_multigrid():
    assessment = assess_cp2k_multigrid(None)

    assert not assessment.available
    assert not assessment.internally_consistent
    assert assessment.grid_counts == ()
    assert assessment.grid_fractions == ()
    assert assessment.issues


def test_inconsistent_total():
    assessment = assess_cp2k_multigrid(
        make_multigrid(total=100)
    )

    assert assessment.available
    assert not assessment.internally_consistent
    assert any(
        "does not match" in issue
        for issue in assessment.issues
    )


def test_zero_total():
    assessment = assess_cp2k_multigrid(
        make_multigrid(total=0)
    )

    assert not assessment.internally_consistent
    assert assessment.grid_fractions == ()


def test_missing_total():
    assessment = assess_cp2k_multigrid(
        make_multigrid(total=None)
    )

    assert not assessment.internally_consistent
    assert assessment.grid_fractions == ()


def test_empty_levels():
    assessment = assess_cp2k_multigrid(
        ParsedCP2KMultigrid(
            levels=(),
            total_gridlevel_count=0,
        )
    )

    assert not assessment.internally_consistent
    assert assessment.grid_counts == ()


def test_nonconsecutive_grid_numbers():
    multigrid = ParsedCP2KMultigrid(
        levels=(
            ParsedCP2KMultigridLevel(
                grid_number=1,
                count=50,
                cutoff_au=200.0,
            ),
            ParsedCP2KMultigridLevel(
                grid_number=3,
                count=50,
                cutoff_au=66.67,
            ),
        ),
        total_gridlevel_count=100,
    )

    assessment = assess_cp2k_multigrid(multigrid)

    assert not assessment.internally_consistent
    assert any(
        "consecutively numbered" in issue
        for issue in assessment.issues
    )


def test_convergence_modes():
    assert CP2KConvergenceMode.FULL_SCF == "full_scf"
    assert CP2KConvergenceMode.GRID_DIAGNOSTIC == "grid_diagnostic"

def make_cp2k_result(
    *,
    run_type=CP2KRunType.ENERGY,
    scf_status=CP2KSCFStatus.CONVERGED,
    normal_termination=True,
    energy=-100.0,
    multigrid=None,
):
    """Build a CP2K result for convergence-validity testing."""

    return ParsedCP2KResult(
        run_type=run_type,
        scf=ParsedCP2KSCF(
            status=scf_status,
        ),
        normal_termination=normal_termination,
        energy=ParsedCP2KEnergy(
            total_energy_hartree=energy,
        ),
        multigrid=multigrid,
    )


def test_full_scf_valid_calculation():
    parsed = make_cp2k_result()

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert assessment.valid
    assert assessment.issues == ()


def test_full_scf_rejects_unconverged_scf():
    parsed = make_cp2k_result(
        scf_status=CP2KSCFStatus.NOT_CONVERGED,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert not assessment.valid
    assert any("converged SCF" in issue for issue in assessment.issues)


def test_full_scf_rejects_unknown_scf():
    parsed = make_cp2k_result(
        scf_status=CP2KSCFStatus.UNKNOWN,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert not assessment.valid


def test_grid_diagnostic_accepts_unconverged_scf():
    parsed = make_cp2k_result(
        scf_status=CP2KSCFStatus.NOT_CONVERGED,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    )

    assert assessment.valid
    assert assessment.issues == ()


def test_grid_diagnostic_rejects_missing_multigrid():
    parsed = make_cp2k_result(
        multigrid=None,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    )

    assert not assessment.valid
    assert any("unavailable" in issue for issue in assessment.issues)


def test_grid_diagnostic_rejects_inconsistent_multigrid():
    parsed = make_cp2k_result(
        multigrid=make_multigrid(total=100),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    )

    assert not assessment.valid


@pytest.mark.parametrize(
    "mode",
    [
        CP2KConvergenceMode.FULL_SCF,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    ],
)
def test_both_modes_reject_abnormal_termination(mode):
    parsed = make_cp2k_result(
        normal_termination=False,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        mode,
    )

    assert not assessment.valid


@pytest.mark.parametrize(
    "mode",
    [
        CP2KConvergenceMode.FULL_SCF,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    ],
)
def test_both_modes_reject_missing_energy(mode):
    parsed = make_cp2k_result(
        energy=None,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        mode,
    )

    assert not assessment.valid


@pytest.mark.parametrize(
    "mode",
    [
        CP2KConvergenceMode.FULL_SCF,
        CP2KConvergenceMode.GRID_DIAGNOSTIC,
    ],
)
def test_both_modes_reject_nonfinite_energy(mode):
    parsed = make_cp2k_result(
        energy=math.nan,
        multigrid=make_multigrid(),
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        mode,
    )

    assert not assessment.valid


def test_convergence_rejects_geometry_optimisation():
    parsed = make_cp2k_result(
        run_type=CP2KRunType.GEO_OPT,
    )

    assessment = assess_cp2k_calculation_validity(
        parsed,
        CP2KConvergenceMode.FULL_SCF,
    )

    assert not assessment.valid
    assert any("ENERGY run" in issue for issue in assessment.issues)

def make_convergence_input(
    *,
    cutoff=400.0,
    relative_cutoff=60.0,
    basis="DZVP-MOLOPT-SR-GTH",
    potential="GTH-PBE-q6",
    xc_functional="PBE",
    eps_scf=1.0e-7,
):
    """Construct a CP2K input for convergence testing."""

    return ParsedCP2KInput(
        run_type=CP2KRunType.ENERGY,
        charge=0,
        multiplicity=1,
        xc_functional=xc_functional,
        cutoff_ry=cutoff,
        relative_cutoff_ry=relative_cutoff,
        eps_scf=eps_scf,
        k_points=(1, 1, 1),
        kinds=(
            ParsedCP2KKind(
                kind="O",
                element="O",
                basis_set=basis,
                potential=potential,
            ),
        ),
        basis_set_file="BASIS_MOLOPT",
        potential_file="GTH_POTENTIALS",
        admm=False,
    )


def test_cutoff_study_accepts_fixed_settings():
    """Only CUTOFF changes between valid candidates."""

    inputs = [
        make_convergence_input(cutoff=400.0),
        make_convergence_input(cutoff=500.0),
        make_convergence_input(cutoff=600.0),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "cutoff",
    )

    assert assessment.consistent
    assert assessment.issues == ()
    assert assessment.varied_parameter == "cutoff"


def test_cutoff_study_rejects_relative_cutoff_change():
    """REL_CUTOFF must remain fixed during a CUTOFF study."""

    inputs = [
        make_convergence_input(cutoff=400.0),
        make_convergence_input(
            cutoff=500.0,
            relative_cutoff=80.0,
        ),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "cutoff",
    )

    assert not assessment.consistent
    assert any(
        "relative_cutoff_ry differs" in issue
        for issue in assessment.issues
    )


def test_cutoff_study_rejects_functional_change():
    """The XC functional must remain fixed."""

    inputs = [
        make_convergence_input(cutoff=400.0),
        make_convergence_input(
            cutoff=500.0,
            xc_functional="PBE0",
        ),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "cutoff",
    )

    assert not assessment.consistent
    assert any(
        "xc_functional differs" in issue
        for issue in assessment.issues
    )


def test_cutoff_study_rejects_missing_settings():
    """Missing required parsed settings must be reported."""

    inputs = [
        make_convergence_input(cutoff=400.0),
        make_convergence_input(
            cutoff=500.0,
            eps_scf=None,
        ),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "cutoff",
    )

    assert not assessment.consistent
    assert any(
        "eps_scf is missing" in issue
        for issue in assessment.issues
    )


def test_cutoff_study_rejects_unchanged_cutoff():
    """A study must actually vary its selected parameter."""

    inputs = [
        make_convergence_input(cutoff=400.0),
        make_convergence_input(cutoff=400.0),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "cutoff",
    )

    assert not assessment.consistent
    assert any(
        "does not vary" in issue
        for issue in assessment.issues
    )


def test_basis_study_allows_basis_changes():
    """Basis convergence requires different basis assignments."""

    inputs = [
        make_convergence_input(
            basis="DZVP-MOLOPT-SR-GTH",
        ),
        make_convergence_input(
            basis="TZVP-MOLOPT-GTH",
        ),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "basis",
    )

    assert assessment.consistent


def test_basis_study_rejects_pseudopotential_changes():
    """Changing the basis must not silently change the potential."""

    inputs = [
        make_convergence_input(
            basis="DZVP-MOLOPT-SR-GTH",
            potential="GTH-PBE-q6",
        ),
        make_convergence_input(
            basis="TZVP-MOLOPT-GTH",
            potential="GTH-PBE-q4",
        ),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "basis",
    )

    assert not assessment.consistent
    assert any(
        "potential" in issue.lower()
        for issue in assessment.issues
    )


def test_input_consistency_requires_two_candidates():
    """A single input cannot establish a convergence study."""

    assessment = assess_cp2k_input_consistency(
        [make_convergence_input()],
        "cutoff",
    )

    assert not assessment.consistent
    assert any(
        "At least two" in issue
        for issue in assessment.issues
    )

def test_basis_study_rejects_unchanged_basis():
    """A basis study must actually change a basis assignment."""

    inputs = [
        make_convergence_input(
            basis="DZVP-MOLOPT-SR-GTH",
        ),
        make_convergence_input(
            basis="DZVP-MOLOPT-SR-GTH",
        ),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "basis",
    )

    assert not assessment.consistent
    assert any(
        "basis_set does not vary" in issue
        for issue in assessment.issues
    )


def test_basis_study_rejects_missing_potential():
    """Missing pseudopotentials cannot establish comparability."""

    inputs = [
        make_convergence_input(
            basis="DZVP-MOLOPT-SR-GTH",
        ),
        make_convergence_input(
            basis="TZVP-MOLOPT-GTH",
            potential=None,
        ),
    ]

    assessment = assess_cp2k_input_consistency(
        inputs,
        "basis",
    )

    assert not assessment.consistent
    assert any(
        "potential is missing" in issue
        for issue in assessment.issues
    )


def test_basis_study_rejects_missing_kind():
    """All candidates must contain the same KIND identities."""

    reference = make_convergence_input(
        basis="DZVP-MOLOPT-SR-GTH",
    )

    candidate = make_convergence_input(
        basis="TZVP-MOLOPT-GTH",
    ).model_copy(
        update={"kinds": ()}
    )

    assessment = assess_cp2k_input_consistency(
        [reference, candidate],
        "basis",
    )

    assert not assessment.consistent
    assert any(
        "KIND identities differ" in issue
        for issue in assessment.issues
    )


def test_basis_study_rejects_element_change():
    """A basis change must not alter a KIND's chemical element."""

    reference = make_convergence_input(
        basis="DZVP-MOLOPT-SR-GTH",
    )

    candidate = make_convergence_input(
        basis="TZVP-MOLOPT-GTH",
    ).model_copy(
        update={
            "kinds": (
                ParsedCP2KKind(
                    kind="O",
                    element="N",
                    basis_set="TZVP-MOLOPT-GTH",
                    potential="GTH-PBE-q6",
                ),
            )
        }
    )

    assessment = assess_cp2k_input_consistency(
        [reference, candidate],
        "basis",
    )

    assert not assessment.consistent
    assert any(
        "element differs" in issue
        for issue in assessment.issues
    )


def test_basis_study_rejects_duplicate_kind_names():
    """Duplicate KIND identities must not be silently overwritten."""

    reference = make_convergence_input()

    duplicate = ParsedCP2KKind(
        kind="O",
        element="O",
        basis_set="TZVP-MOLOPT-GTH",
        potential="GTH-PBE-q6",
    )

    candidate = make_convergence_input(
        basis="TZVP-MOLOPT-GTH",
    ).model_copy(
        update={
            "kinds": (
                duplicate,
                duplicate,
            )
        }
    )

    assessment = assess_cp2k_input_consistency(
        [reference, candidate],
        "basis",
    )

    assert not assessment.consistent
    assert any(
        "duplicate KIND names" in issue
        for issue in assessment.issues
    )


def test_input_consistency_rejects_unsupported_parameter():
    """Unsupported study parameters must raise a clear error."""

    inputs = [
        make_convergence_input(cutoff=400.0),
        make_convergence_input(cutoff=500.0),
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported convergence parameter",
    ):
        assess_cp2k_input_consistency(
            inputs,
            "unsupported_parameter",
        )

def test_integrated_cutoff_study_selects_stable_candidate():
    """Select the least costly candidate stable against all later ones."""
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceCriterion,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
        ConvergenceCandidate(label="800Ry", order=2),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=1.0e-3,
        ),
    )

    # All calculations use exactly the same geometry.
    structure_hash = "a" * 64

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash=structure_hash,
        )
        for candidate, energy in zip(
            candidates,
            (-100.0, -100.020, -100.025),
        )
    ]

    project_names = (
        "cutoff_400",
        "cutoff_600",
        "cutoff_800",
    )

    inputs = [
        make_convergence_input(
            cutoff=cutoff,
        ).model_copy(
            update={"project_name": project_name}
        )
        for cutoff, project_name in zip(
            (400.0, 600.0, 800.0),
            project_names,
        )
    ]

    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV

    results = [
        make_cp2k_result(
            energy=observation.total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_name}
        )
        for observation, project_name in zip(
            observations,
            project_names,
        )
    ]

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert assessment.valid
    assert assessment.converged
    assert assessment.selected_candidate_label == "600Ry"
    assert assessment.reference_candidate_label == "800Ry"
    assert assessment.input_consistency.consistent
    assert all(
        validity.valid
        for validity in assessment.calculation_validities
    )
    assert assessment.issues == ()

def test_integrated_study_rejects_unconverged_scf():
    """Reject energy convergence if any CP2K SCF did not converge."""
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
        ConvergenceCandidate(label="800Ry", order=2),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
    )

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash="a" * 64,
        )
        for candidate, energy in zip(
            candidates,
            (-100.0, -100.020, -100.025),
        )
    ]

    project_names = (
        "cutoff_400",
        "cutoff_600",
        "cutoff_800",
    )

    inputs = [
        make_convergence_input(
            cutoff=cutoff,
        ).model_copy(
            update={"project_name": project_name}
        )
        for cutoff, project_name in zip(
            (400.0, 600.0, 800.0),
            project_names,
        )
    ]

    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV

    results = [
        make_cp2k_result(
            energy=observations[0].total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_names[0]}
        ),
        make_cp2k_result(
            scf_status=CP2KSCFStatus.NOT_CONVERGED,
            energy=observations[1].total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_names[1]}
        ),
        make_cp2k_result(
            energy=observations[2].total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_names[2]}
        ),
    ]

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert not assessment.valid
    assert not assessment.converged
    assert assessment.selected_candidate_label is None

    assert not assessment.calculation_validities[1].valid

    assert any(
        "FULL_SCF mode requires converged SCF" in issue
        for issue in assessment.issues
    )

def test_integrated_study_rejects_structure_mismatch():
    """Reject convergence studies using different structures."""
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
        ConvergenceCandidate(label="800Ry", order=2),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
    )

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash=structure_hash,
        )
        for candidate, energy, structure_hash in zip(
            candidates,
            (-100.0, -100.020, -100.025),
            (
                "a" * 64,
                "b" * 64,  # Deliberately different geometry
                "a" * 64,
            ),
        )
    ]

    project_names = (
        "cutoff_400",
        "cutoff_600",
        "cutoff_800",
    )

    inputs = [
        make_convergence_input(
            cutoff=cutoff,
        ).model_copy(
            update={"project_name": project_name}
        )
        for cutoff, project_name in zip(
            (400.0, 600.0, 800.0),
            project_names,
        )
    ]

    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV

    results = [
        make_cp2k_result(
            energy=observation.total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_name}
        )
        for observation, project_name in zip(
            observations,
            project_names,
        )
    ]

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert not assessment.valid
    assert not assessment.converged
    assert assessment.selected_candidate_label is None

    assert any(
        "same computational structure" in issue
        for issue in assessment.issues
    )

def test_integrated_study_rejects_xc_functional_change():
    """Reject a cutoff study when the XC functional changes."""
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
        ConvergenceCandidate(label="800Ry", order=2),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
    )

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash="a" * 64,
        )
        for candidate, energy in zip(
            candidates,
            (-100.0, -100.020, -100.025),
        )
    ]

    project_names = (
        "cutoff_400",
        "cutoff_600",
        "cutoff_800",
    )

    inputs = [
        make_convergence_input(
            cutoff=400.0,
            xc_functional="PBE",
        ).model_copy(
            update={"project_name": project_names[0]}
        ),
        make_convergence_input(
            cutoff=600.0,
            xc_functional="PBEsol",  # Deliberate mismatch
        ).model_copy(
            update={"project_name": project_names[1]}
        ),
        make_convergence_input(
            cutoff=800.0,
            xc_functional="PBE",
        ).model_copy(
            update={"project_name": project_names[2]}
        ),
    ]

    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV

    results = [
        make_cp2k_result(
            energy=observation.total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_name}
        )
        for observation, project_name in zip(
            observations,
            project_names,
        )
    ]

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert not assessment.valid
    assert not assessment.converged
    assert assessment.selected_candidate_label is None

    assert not assessment.input_consistency.consistent

    assert any(
        "xc_functional differs" in issue
        for issue in assessment.issues
    )

def test_integrated_study_rejects_energy_mismatch():
    """Reject observations inconsistent with CP2K output energies."""
    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
        ConvergenceCandidate(label="800Ry", order=2),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
    )

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash="a" * 64,
        )
        for candidate, energy in zip(
            candidates,
            (-100.0, -100.020, -100.025),
        )
    ]

    project_names = (
        "cutoff_400",
        "cutoff_600",
        "cutoff_800",
    )

    inputs = [
        make_convergence_input(
            cutoff=cutoff,
        ).model_copy(
            update={"project_name": project_name}
        )
        for cutoff, project_name in zip(
            (400.0, 600.0, 800.0),
            project_names,
        )
    ]

    results = [
        make_cp2k_result(
            energy=observation.total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_name}
        )
        for observation, project_name in zip(
            observations,
            project_names,
        )
    ]

    # Deliberately change the second CP2K result by 0.1 eV.
    results[1] = make_cp2k_result(
        energy=(
            observations[1].total_energy_ev + 0.1
        ) / HARTREE_TO_EV,
    ).model_copy(
        update={"project_name": project_names[1]}
    )

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert not assessment.valid
    assert not assessment.converged
    assert assessment.selected_candidate_label is None

    assert any(
        "Candidate 2: observation energy does not match"
        in issue
        for issue in assessment.issues
    )

def test_integrated_study_rejects_project_name_mismatch():
    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
    )

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash="a" * 64,
        )
        for candidate, energy in zip(
            candidates,
            (-100.0, -100.001),
        )
    ]

    inputs = [
        make_convergence_input(
            cutoff=400.0,
        ).model_copy(
            update={"project_name": "cutoff_400"}
        ),
        make_convergence_input(
            cutoff=600.0,
        ).model_copy(
            update={"project_name": "cutoff_600"}
        ),
    ]

    results = [
        make_cp2k_result(
            energy=observation.total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_name}
        )
        for observation, project_name in zip(
            observations,
            ("cutoff_400", "wrong_project"),
        )
    ]

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert not assessment.valid
    assert not assessment.converged
    assert assessment.selected_candidate_label is None
    assert any(
        "project_name mismatch" in issue
        for issue in assessment.issues
    )

def test_integrated_study_rejects_missing_project_name():
    """Reject a study when an input or output project name is missing."""
    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
    )

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash="a" * 64,
        )
        for candidate, energy in zip(
            candidates,
            (-100.0, -100.001),
        )
    ]

    inputs = [
        make_convergence_input(
            cutoff=400.0,
        ).model_copy(
            update={"project_name": "cutoff_400"}
        ),
        make_convergence_input(
            cutoff=600.0,
        ).model_copy(
            update={"project_name": None}
        ),
    ]

    results = [
        make_cp2k_result(
            energy=observation.total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": project_name}
        )
        for observation, project_name in zip(
            observations,
            ("cutoff_400", "cutoff_600"),
        )
    ]

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert not assessment.valid
    assert not assessment.converged
    assert assessment.selected_candidate_label is None

    assert any(
        "Candidate 2: CP2K input project_name is missing"
        in issue
        for issue in assessment.issues
    )

def test_integrated_study_rejects_duplicate_project_names():
    """Reject convergence candidates sharing a CP2K project name."""
    from nsdw.calculators.cp2k.adapter import HARTREE_TO_EV
    from nsdw.calculators.cp2k.convergence import (
        assess_cp2k_convergence_study,
    )
    from nsdw.workflows.convergence.models import (
        ConvergenceCandidate,
        ConvergenceObservation,
        ConvergenceParameter,
        ConvergenceStudyDefinition,
    )

    candidates = [
        ConvergenceCandidate(label="400Ry", order=0),
        ConvergenceCandidate(label="600Ry", order=1),
    ]

    study = ConvergenceStudyDefinition(
        parameter=ConvergenceParameter.CUTOFF,
        candidates=candidates,
    )

    observations = [
        ConvergenceObservation(
            candidate=candidate,
            total_energy_ev=energy,
            n_atoms=10,
            structure_hash="a" * 64,
        )
        for candidate, energy in zip(
            candidates,
            (-100.0, -100.001),
        )
    ]

    # Deliberately reuse the same project name for both candidates.
    inputs = [
        make_convergence_input(
            cutoff=cutoff,
        ).model_copy(
            update={"project_name": "duplicate_project"}
        )
        for cutoff in (400.0, 600.0)
    ]

    results = [
        make_cp2k_result(
            energy=observation.total_energy_ev / HARTREE_TO_EV,
        ).model_copy(
            update={"project_name": "duplicate_project"}
        )
        for observation in observations
    ]

    assessment = assess_cp2k_convergence_study(
        study,
        observations,
        inputs,
        results,
    )

    assert not assessment.valid
    assert not assessment.converged
    assert assessment.selected_candidate_label is None

    assert any(
        "CP2K input project names must be unique"
        in issue
        for issue in assessment.issues
    )

    assert any(
        "CP2K output project names must be unique"
        in issue
        for issue in assessment.issues
    )