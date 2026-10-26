"""Runtime tests for pre-submission specification validation."""

from types import SimpleNamespace

import pytest

from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)
from nsdw.workflows.convergence.aiida_campaign_specification import (
    specification_digest,
)


class FakeDict:
    def __init__(self, payload=None, *, dict=None):
        if payload is not None and dict is not None:
            raise ValueError(
                "Specify either payload or dict, not both."
            )

        self.payload = (
            dict if dict is not None else payload
        )

    def get_dict(self):
        return self.payload


def make_inputs(*, actual_cutoff=300, declared_cutoff=300):
    specification = {
        "schema_version": 1,
        "campaign_name": "runtime-validation",
        "expected_cutoffs": [300],
    }

    binding = {
        "specification": specification,
        "specification_sha256": specification_digest(
            specification
        ),
        "candidate_cutoffs": {
            "candidate_a": declared_cutoff,
        },
    }

    return {
        "candidates": FakeDict({
            "candidate_a": {
                "FORCE_EVAL": {
                    "DFT": {
                        "MGRID": {
                            "CUTOFF": actual_cutoff,
                        }
                    }
                }
            }
        }),
        "resources": FakeDict({
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 300,
        }),
        "scientific_specification": FakeDict(binding),
    }


def invoke_without_submission(monkeypatch, inputs):
    submissions = []

    def forbidden_submit(*args, **kwargs):
        submissions.append((args, kwargs))
        raise AssertionError(
            "A remote calculation must not be submitted."
        )

    class FakeInputs(SimpleNamespace):
        def __contains__(self, key):
            return hasattr(self, key)

    fake = SimpleNamespace(
        inputs=FakeInputs(**inputs),
        report=lambda message: None,
        exit_codes=SimpleNamespace(
            ERROR_INVALID_INPUT="INVALID_INPUT"
        ),
        submit=forbidden_submit,
    )

    result = CP2KConvergenceWorkChain.submit_candidates(fake)

    return result, submissions


def test_actual_cutoff_mismatch_prevents_submission(monkeypatch):
    result, submissions = invoke_without_submission(
        monkeypatch,
        make_inputs(actual_cutoff=500),
    )

    assert result == "INVALID_INPUT"
    assert submissions == []


def test_declared_cutoff_mismatch_prevents_submission(monkeypatch):
    result, submissions = invoke_without_submission(
        monkeypatch,
        make_inputs(declared_cutoff=400),
    )

    assert result == "INVALID_INPUT"
    assert submissions == []


def test_modified_digest_prevents_submission(monkeypatch):
    inputs = make_inputs()
    payload = inputs["scientific_specification"].get_dict()
    payload["specification_sha256"] = "0" * 64

    result, submissions = invoke_without_submission(
        monkeypatch,
        inputs,
    )

    assert result == "INVALID_INPUT"
    assert submissions == []


def test_missing_cutoff_prevents_submission(monkeypatch):
    inputs = make_inputs()
    candidate = inputs["candidates"].get_dict()["candidate_a"]
    del candidate["FORCE_EVAL"]["DFT"]["MGRID"]["CUTOFF"]

    result, submissions = invoke_without_submission(
        monkeypatch,
        inputs,
    )

    assert result == "INVALID_INPUT"
    assert submissions == []



def test_valid_specification_reaches_mock_submission(monkeypatch):
    """A valid campaign must reach the mocked submission path."""
    from nsdw.workflows.convergence import aiida_workchain

    class FakeBuilder:
        def __init__(self):
            self.metadata = SimpleNamespace(
                label=None,
                options=SimpleNamespace(),
            )

    class FakeCalculation:
        @staticmethod
        def get_builder():
            return FakeBuilder()

    monkeypatch.setattr(
        aiida_workchain,
        "Cp2kCalculation",
        FakeCalculation,
    )
    monkeypatch.setattr(
        aiida_workchain,
        "Dict",
        FakeDict,
    )

    submitted = []

    def fake_submit(builder):
        submitted.append(builder)
        return object()

    class FakeInputs(SimpleNamespace):
        def __contains__(self, key):
            return hasattr(self, key)

    inputs = make_inputs()
    inputs["code"] = object()
    inputs["structure"] = object()

    fake = SimpleNamespace(
        inputs=FakeInputs(**inputs),
        ctx=SimpleNamespace(),
        report=lambda message: None,
        exit_codes=SimpleNamespace(
            ERROR_INVALID_INPUT="INVALID_INPUT",
        ),
        submit=fake_submit,
    )

    result = (
        CP2KConvergenceWorkChain.submit_candidates(fake)
    )

    assert len(submitted) == 1
    assert submitted[0].metadata.label == "candidate_a"
    assert submitted[0].parameters.get_dict()[
        "FORCE_EVAL"
    ]["DFT"]["MGRID"]["CUTOFF"] == 300

    assert result is not None
