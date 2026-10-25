"""Isolated retry-input validation tests; no AiiDA submissions."""

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from aiida.common.links import LinkType
from aiida.orm import WorkChainNode

from nsdw.workflows.convergence.aiida_recovery_identity import (
    validate_tracking_identity,
)
from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)
from nsdw.workflows.convergence.aiida_recovery_workchain import (
    CP2KConvergenceRecoveryWorkChain,
)
from nsdw.workflows.convergence.recovery_tracking import (
    RecoveryExecutionMetadata,
)


class Value:
    def __init__(self, value):
        self.value = value


class Mapping:
    def __init__(self, data):
        self.data = data

    def get_dict(self):
        return dict(self.data)


class Namespace(dict):
    def __getattr__(self, name):
        return self[name]


def node():
    return SimpleNamespace(uuid=str(uuid4()))


def make_case():
    code = node()
    structure = node()
    computer = node()
    code.computer = computer

    candidates = Mapping({"40-Ry": {"FORCE_EVAL": {}}})
    resources = Mapping({
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
        "max_wallclock_seconds": 1800,
    })

    inputs = Namespace(
        code=code,
        structure=structure,
        candidates=candidates,
        resources=resources,
        retry_only=Value(True),
    )

    process_uuid = str(uuid4())
    parent_uuid = str(uuid4())

    parent = SimpleNamespace(
        uuid=parent_uuid,
        process_type=(
            CP2KConvergenceRecoveryWorkChain.build_process_type()
        ),
        outputs=Namespace(
            recovery_plan=node(),
            retry_candidates=node(),
            retry_resources=node(),
        ),
    )

    class Incoming:
        def get_incoming(self, *, link_type):
            assert link_type is LinkType.CALL_WORK
            return self

        def all(self):
            return [SimpleNamespace(node=parent)]

    process = SimpleNamespace(
        uuid=process_uuid,
        process_type=CP2KConvergenceWorkChain.build_process_type(),
        inputs=inputs,
        base=SimpleNamespace(links=Incoming()),
    )

    original = SimpleNamespace(uuid=str(uuid4()))

    expected = Namespace(**inputs)

    metadata = RecoveryExecutionMetadata(
        scientific_identity="science",
        execution_identity="attempt",
        engine="aiida",
        computer_uuid=computer.uuid,
        process_uuid=process_uuid,
    )

    return metadata, process, parent, original, expected


def validate_case(metadata, process, parent, original, expected):
    with (
        patch(
            "nsdw.workflows.convergence.aiida_recovery_identity."
            "WorkChainNode",
            SimpleNamespace,
        ),
        patch(
            "nsdw.workflows.convergence.aiida_recovery_workchain."
            "build_retry_workchain_builder",
            return_value=expected,
        ),
    ):
        return validate_tracking_identity(
            metadata,
            process_loader=lambda _: process,
            expected_recovery_parent=parent,
            original_campaign=original,
        )


def test_matching_retry_inputs():
    case = make_case()
    assert validate_case(*case) is case[1]


@pytest.mark.parametrize(
    "field,expected_error",
    [
        ("code", "code mismatch"),
        ("structure", "structure mismatch"),
        ("candidates", "candidates mismatch"),
        ("resources", "resources mismatch"),
        ("retry_only", "retry-only"),
    ],
)
def test_mismatched_inputs_are_rejected(field, expected_error):
    metadata, process, parent, original, expected = make_case()

    if field in ("code", "structure"):
        process.inputs[field] = node()
    elif field == "candidates":
        process.inputs.candidates = Mapping({
            "50-Ry": {"FORCE_EVAL": {}},
        })
    elif field == "resources":
        process.inputs.resources = Mapping({
            "num_machines": 2,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 1800,
        })
    else:
        process.inputs.retry_only = Value(False)

    with pytest.raises(ValueError, match=expected_error):
        validate_case(metadata, process, parent, original, expected)


def test_wrong_computer_is_rejected():
    metadata, process, parent, original, expected = make_case()

    wrong = RecoveryExecutionMetadata(
        scientific_identity=metadata.scientific_identity,
        execution_identity=metadata.execution_identity,
        engine=metadata.engine,
        computer_uuid=str(uuid4()),
        process_uuid=metadata.process_uuid,
    )

    with pytest.raises(ValueError, match="computer UUID mismatch"):
        validate_case(wrong, process, parent, original, expected)


def test_imported_inputs_are_rejected():
    metadata, process, parent, original, expected = make_case()
    process.inputs["import_folders"] = node()

    with pytest.raises(ValueError, match="Imported CP2K"):
        validate_case(metadata, process, parent, original, expected)
