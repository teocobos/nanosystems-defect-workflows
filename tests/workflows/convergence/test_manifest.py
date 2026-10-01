from pathlib import Path

from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.manifest import (
    ConvergenceManifestCandidate,
    ConvergenceStudyManifest,
    load_convergence_manifest,
    write_convergence_manifest,
    convergence_study_from_manifest,
)
from nsdw.workflows.convergence.models import (
    ConvergenceCriterion,
    ConvergenceParameter,
)


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = ConvergenceStudyManifest(
        calculator="cp2k",
        parameter=ConvergenceParameter.CUTOFF,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=1.0e-3,
        ),
        candidates=(
            ConvergenceManifestCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(
                    value=400.0,
                    unit="Ry",
                ),
                directory="400-Ry",
                input_file="test-400-Ry.inp",
                coordinate_file="test-400-Ry.xyz",
            ),
            ConvergenceManifestCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(
                    value=600.0,
                    unit="Ry",
                ),
                directory="600-Ry",
                input_file="test-600-Ry.inp",
                coordinate_file="test-600-Ry.xyz",
            ),
        ),
    )

    path = tmp_path / "manifest.json"

    write_convergence_manifest(
        manifest=manifest,
        path=path,
    )

    loaded = load_convergence_manifest(path)

    assert loaded == manifest
    assert loaded.candidates[0].value == Quantity(
        value=400.0,
        unit="Ry",
    )
    assert loaded.candidates[1].directory == "600-Ry"

def test_convergence_study_from_manifest() -> None:
    manifest = ConvergenceStudyManifest(
        calculator="cp2k",
        parameter=ConvergenceParameter.CUTOFF,
        criterion=ConvergenceCriterion(
            energy_tolerance_ev_per_atom=1.0e-3,
        ),
        candidates=(
            ConvergenceManifestCandidate(
                label="400-Ry",
                order=0,
                value=Quantity(
                    value=400.0,
                    unit="Ry",
                ),
                directory="400-Ry",
                input_file="test-400-Ry.inp",
                coordinate_file="test-400-Ry.xyz",
            ),
            ConvergenceManifestCandidate(
                label="600-Ry",
                order=1,
                value=Quantity(
                    value=600.0,
                    unit="Ry",
                ),
                directory="600-Ry",
                input_file="test-600-Ry.inp",
                coordinate_file="test-600-Ry.xyz",
            ),
        ),
    )

    study = convergence_study_from_manifest(
        manifest
    )

    assert study.parameter == manifest.parameter
    assert study.criterion == manifest.criterion

    assert len(study.candidates) == 2

    assert study.candidates[0].label == "400-Ry"
    assert study.candidates[0].order == 0
    assert study.candidates[0].value == Quantity(
        value=400.0,
        unit="Ry",
    )

    assert study.candidates[1].label == "600-Ry"
    assert study.candidates[1].order == 1
    assert study.candidates[1].value == Quantity(
        value=600.0,
        unit="Ry",
    )