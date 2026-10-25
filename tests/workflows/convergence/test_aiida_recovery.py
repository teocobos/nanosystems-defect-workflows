"""Regression tests for AiiDA convergence recovery selection."""

import pytest

from nsdw.workflows.convergence.aiida_recovery import (
    build_recovery_plan,
)


def entry(energy=None, *, success=True, exit_status=0):
    result = {
        "finished_ok": success,
        "exit_status": exit_status,
    }

    if energy is not None:
        result["output_parameters"] = {
            "energy": energy,
            "energy_units": "a.u.",
        }
        result["energy_hartree"] = energy

    return result


def test_only_failed_candidates_are_retried():
    candidates = {
        "10-Ry": {},
        "20-Ry": {},
        "30-Ry": {},
        "40-Ry": {},
        "50-Ry": {},
    }

    summary = {
        "10-Ry": entry(-108.6187),
        "20-Ry": entry(-108.5012),
        "30-Ry": entry(-108.4963),
        "40-Ry": entry(success=False, exit_status=303),
        "50-Ry": entry(success=False, exit_status=303),
    }

    plan = build_recovery_plan(candidates, summary)

    assert plan.reusable == ("10-Ry", "20-Ry", "30-Ry")
    assert plan.retry == ("40-Ry", "50-Ry")


def test_failed_calculation_cannot_reuse_partial_energy():
    candidates = {"40-Ry": {}}

    summary = {
        "40-Ry": entry(
            -108.4961,
            success=False,
            exit_status=303,
        )
    }

    plan = build_recovery_plan(candidates, summary)

    assert plan.reusable == ()
    assert plan.retry == ("40-Ry",)


def test_missing_energy_requires_retry():
    plan = build_recovery_plan(
        {"40-Ry": {}},
        {"40-Ry": entry()},
    )

    assert plan.retry == ("40-Ry",)


def test_candidate_mismatch_is_rejected():
    with pytest.raises(ValueError, match="labels do not match"):
        build_recovery_plan(
            {"10-Ry": {}},
            {"20-Ry": entry(-108.5)},
        )


def test_nonfinite_energy_requires_retry():
    plan = build_recovery_plan(
        {"10-Ry": {}},
        {"10-Ry": entry(float("nan"))},
    )

    assert plan.retry == ("10-Ry",)



def test_build_retry_candidates():
    from nsdw.workflows.convergence.aiida_recovery import (
        build_retry_candidates,
    )

    candidates = {
        "10-Ry": {"MGRID": {"REL_CUTOFF": 10}},
        "20-Ry": {"MGRID": {"REL_CUTOFF": 20}},
        "30-Ry": {"MGRID": {"REL_CUTOFF": 30}},
        "40-Ry": {"MGRID": {"REL_CUTOFF": 40}},
        "50-Ry": {"MGRID": {"REL_CUTOFF": 50}},
    }

    selected = build_retry_candidates(
        candidates,
        ("40-Ry", "50-Ry"),
    )

    assert list(selected) == ["40-Ry", "50-Ry"]
    assert selected["40-Ry"] == candidates["40-Ry"]
    assert selected["50-Ry"] == candidates["50-Ry"]
    assert len(candidates) == 5


def test_build_retry_candidates_rejects_unknown_labels():
    import pytest

    from nsdw.workflows.convergence.aiida_recovery import (
        build_retry_candidates,
    )

    with pytest.raises(ValueError, match="Unknown"):
        build_retry_candidates(
            {"40-Ry": {"MGRID": {"REL_CUTOFF": 40}}},
            ("60-Ry",),
        )


def test_build_retry_candidates_rejects_duplicates():
    import pytest

    from nsdw.workflows.convergence.aiida_recovery import (
        build_retry_candidates,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_retry_candidates(
            {"40-Ry": {"MGRID": {"REL_CUTOFF": 40}}},
            ("40-Ry", "40-Ry"),
        )



def test_build_retry_resources():
    from nsdw.workflows.convergence.aiida_recovery import (
        build_retry_resources,
    )

    original = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 600,
        "queue_name": "small",
        "account": "project_465003407",
    }

    updated = build_retry_resources(original, 1800)

    assert updated == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 1800,
        "queue_name": "small",
        "account": "project_465003407",
    }

    assert original["max_wallclock_seconds"] == 600


def test_build_retry_resources_rejects_shorter_walltime():
    import pytest

    from nsdw.workflows.convergence.aiida_recovery import (
        build_retry_resources,
    )

    original = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 600,
    }

    with pytest.raises(ValueError, match="exceed"):
        build_retry_resources(original, 600)

    with pytest.raises(ValueError, match="exceed"):
        build_retry_resources(original, 300)


def test_build_retry_resources_rejects_invalid_resources():
    import pytest

    from nsdw.workflows.convergence.aiida_recovery import (
        build_retry_resources,
    )

    original = {
        "num_machines": 0,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 600,
    }

    with pytest.raises(ValueError, match="num_machines"):
        build_retry_resources(original, 1800)
