import pytest

from nsdw.models.calculation import (
    Backend,
    CalculationMetadata,
    CalculationStatus,
    CalculationType,
)
from nsdw.models.quantity import Quantity
from nsdw.models.result import (
    EnergyResult,
    NSDWResult,
)
from nsdw.models.structure import StructureResult
from nsdw.workflows.convergence.models import (
    ConvergenceCandidate,
)
from nsdw.workflows.convergence.observation import (
    ConvergenceObservationError,
    convergence_observation_from_result,
)


def test_convergence_observation_from_result() -> None:
    candidate = ConvergenceCandidate(
        label="600-Ry",
        order=1,
        value=Quantity(
            value=600.0,
            unit="Ry",
        ),
    )

    result = NSDWResult(
        calculation=CalculationMetadata(
            id="600-Ry",
            type=CalculationType.SINGLE_POINT,
            status=CalculationStatus.COMPLETED,
            backend=Backend.CP2K,
        ),
        structure=StructureResult(
            formula="SiO2",
            n_atoms=9,
            periodic=True,
            structure_hash="a" * 64,
        ),
        energy=EnergyResult(
            total=Quantity(
                value=-123.456,
                unit="eV",
            ),
        ),
    )

    observation = convergence_observation_from_result(
        candidate=candidate,
        result=result,
    )

    assert observation.candidate == candidate
    assert observation.total_energy_ev == -123.456
    assert observation.n_atoms == 9
    assert observation.structure_hash == "a" * 64
    assert observation.energy_ev_per_atom == -123.456 / 9

def test_convergence_observation_requires_structure() -> None:
    candidate = ConvergenceCandidate(
        label="600-Ry",
        order=1,
        value=Quantity(value=600.0, unit="Ry"),
    )

    result = NSDWResult(
        calculation=CalculationMetadata(
            id="600-Ry",
            type=CalculationType.SINGLE_POINT,
            status=CalculationStatus.COMPLETED,
            backend=Backend.CP2K,
        ),
        energy=EnergyResult(
            total=Quantity(
                value=-123.456,
                unit="eV",
            ),
        ),
    )

    with pytest.raises(
        ConvergenceObservationError,
        match="requires structural metadata",
    ):
        convergence_observation_from_result(
            candidate=candidate,
            result=result,
        )


def test_convergence_observation_requires_total_energy() -> None:
    candidate = ConvergenceCandidate(
        label="600-Ry",
        order=1,
        value=Quantity(value=600.0, unit="Ry"),
    )

    result = NSDWResult(
        calculation=CalculationMetadata(
            id="600-Ry",
            type=CalculationType.SINGLE_POINT,
            status=CalculationStatus.COMPLETED,
            backend=Backend.CP2K,
        ),
        structure=StructureResult(
            formula="SiO2",
            n_atoms=9,
            periodic=True,
            structure_hash="a" * 64,
        ),
        energy=EnergyResult(),
    )

    with pytest.raises(
        ConvergenceObservationError,
        match="requires a total energy",
    ):
        convergence_observation_from_result(
            candidate=candidate,
            result=result,
        )


def test_convergence_observation_rejects_non_ev_energy() -> None:
    candidate = ConvergenceCandidate(
        label="600-Ry",
        order=1,
        value=Quantity(value=600.0, unit="Ry"),
    )

    result = NSDWResult(
        calculation=CalculationMetadata(
            id="600-Ry",
            type=CalculationType.SINGLE_POINT,
            status=CalculationStatus.COMPLETED,
            backend=Backend.CP2K,
        ),
        structure=StructureResult(
            formula="SiO2",
            n_atoms=9,
            periodic=True,
            structure_hash="a" * 64,
        ),
        energy=EnergyResult(
            total=Quantity(
                value=-4.537,
                unit="Ha",
            ),
        ),
    )

    with pytest.raises(
        ConvergenceObservationError,
        match="requires total energy in eV",
    ):
        convergence_observation_from_result(
            candidate=candidate,
            result=result,
        )

def test_collect_convergence_observations_preserves_candidate_order() -> None:
    from nsdw.workflows.convergence.observation import (
        collect_convergence_observations,
    )

    candidates = (
        ConvergenceCandidate(
            label="400-Ry",
            order=0,
            value=Quantity(value=400.0, unit="Ry"),
        ),
        ConvergenceCandidate(
            label="600-Ry",
            order=1,
            value=Quantity(value=600.0, unit="Ry"),
        ),
    )

    results = (
        NSDWResult(
            calculation=CalculationMetadata(
                id="400-Ry",
                type=CalculationType.SINGLE_POINT,
                status=CalculationStatus.COMPLETED,
                backend=Backend.CP2K,
            ),
            structure=StructureResult(
                formula="SiO2",
                n_atoms=9,
                periodic=True,
                structure_hash="a" * 64,
            ),
            energy=EnergyResult(
                total=Quantity(
                    value=-123.400,
                    unit="eV",
                ),
            ),
        ),
        NSDWResult(
            calculation=CalculationMetadata(
                id="600-Ry",
                type=CalculationType.SINGLE_POINT,
                status=CalculationStatus.COMPLETED,
                backend=Backend.CP2K,
            ),
            structure=StructureResult(
                formula="SiO2",
                n_atoms=9,
                periodic=True,
                structure_hash="a" * 64,
            ),
            energy=EnergyResult(
                total=Quantity(
                    value=-123.450,
                    unit="eV",
                ),
            ),
        ),
    )

    observations = collect_convergence_observations(
        candidates=candidates,
        results=results,
    )

    assert len(observations) == 2

    assert observations[0].candidate == candidates[0]
    assert observations[0].total_energy_ev == -123.400

    assert observations[1].candidate == candidates[1]
    assert observations[1].total_energy_ev == -123.450

def test_collect_convergence_observations_requires_matching_counts() -> None:
    from nsdw.workflows.convergence.observation import (
        collect_convergence_observations,
    )

    candidates = (
        ConvergenceCandidate(
            label="400-Ry",
            order=0,
            value=Quantity(value=400.0, unit="Ry"),
        ),
        ConvergenceCandidate(
            label="600-Ry",
            order=1,
            value=Quantity(value=600.0, unit="Ry"),
        ),
    )

    results = (
        NSDWResult(
            calculation=CalculationMetadata(
                id="400-Ry",
                type=CalculationType.SINGLE_POINT,
                status=CalculationStatus.COMPLETED,
                backend=Backend.CP2K,
            ),
            structure=StructureResult(
                formula="SiO2",
                n_atoms=9,
                periodic=True,
                structure_hash="a" * 64,
            ),
            energy=EnergyResult(
                total=Quantity(
                    value=-123.400,
                    unit="eV",
                ),
            ),
        ),
    )

    with pytest.raises(
        ConvergenceObservationError,
        match="candidate and result counts must match",
    ):
        collect_convergence_observations(
            candidates=candidates,
            results=results,
        )

def test_collect_convergence_observations_rejects_mismatched_result_identity(
) -> None:
    from nsdw.workflows.convergence.observation import (
        collect_convergence_observations,
    )

    candidates = (
        ConvergenceCandidate(
            label="400-Ry",
            order=0,
            value=Quantity(value=400.0, unit="Ry"),
        ),
        ConvergenceCandidate(
            label="600-Ry",
            order=1,
            value=Quantity(value=600.0, unit="Ry"),
        ),
    )

    results = (
        NSDWResult(
            calculation=CalculationMetadata(
                id="600-Ry",
                type=CalculationType.SINGLE_POINT,
                status=CalculationStatus.COMPLETED,
                backend=Backend.CP2K,
            ),
            structure=StructureResult(
                formula="SiO2",
                n_atoms=9,
                periodic=True,
                structure_hash="a" * 64,
            ),
            energy=EnergyResult(
                total=Quantity(
                    value=-123.450,
                    unit="eV",
                ),
            ),
        ),
        NSDWResult(
            calculation=CalculationMetadata(
                id="400-Ry",
                type=CalculationType.SINGLE_POINT,
                status=CalculationStatus.COMPLETED,
                backend=Backend.CP2K,
            ),
            structure=StructureResult(
                formula="SiO2",
                n_atoms=9,
                periodic=True,
                structure_hash="a" * 64,
            ),
            energy=EnergyResult(
                total=Quantity(
                    value=-123.400,
                    unit="eV",
                ),
            ),
        ),
    )

    with pytest.raises(
        ConvergenceObservationError,
        match="does not match candidate",
    ):
        collect_convergence_observations(
            candidates=candidates,
            results=results,
        )