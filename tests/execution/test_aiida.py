from unittest.mock import MagicMock, patch

import pytest

from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
import sys

from nsdw.execution.aiida import (
    AiiDACp2kResources,
    AiiDAExecutionError,
    AiiDAUnavailableError,
    aiida_available,
    build_aiida_execution_result,
    map_aiida_process_state,
    require_aiida,
    submit_cp2k_aiida,
)
from nsdw.execution.models import ExecutionState


def test_aiida_available_when_package_exists() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=object(),
    ):
        assert aiida_available()


def test_aiida_unavailable_when_package_missing() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=None,
    ):
        assert not aiida_available()


def test_require_aiida_accepts_available_installation() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=object(),
    ):
        require_aiida()


def test_require_aiida_rejects_missing_installation() -> None:
    with patch(
        "nsdw.execution.aiida.find_spec",
        return_value=None,
    ):
        with pytest.raises(
            AiiDAUnavailableError,
            match="AiiDA support is not installed",
        ):
            require_aiida()


@pytest.mark.parametrize(
    ("aiida_state", "expected"),
    [
        ("created", ExecutionState.CREATED),
        ("waiting", ExecutionState.SUBMITTED),
        ("running", ExecutionState.RUNNING),
        ("finished", ExecutionState.COMPLETED),
        ("excepted", ExecutionState.FAILED),
        ("killed", ExecutionState.CANCELLED),
    ],
)
def test_map_aiida_process_state(
    aiida_state: str,
    expected: ExecutionState,
) -> None:
    assert map_aiida_process_state(aiida_state) == expected


def test_map_aiida_process_state_normalises_case() -> None:
    assert (
        map_aiida_process_state("RUNNING")
        == ExecutionState.RUNNING
    )


def test_map_aiida_process_state_rejects_unknown_state() -> None:
    with pytest.raises(
        AiiDAExecutionError,
        match="Unsupported AiiDA process state",
    ):
        map_aiida_process_state("paused")


class FakeProcessState(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class FakeAiiDAProcess:
    pk: int | None
    uuid: str
    process_state: FakeProcessState | str | None


def test_build_aiida_execution_result() -> None:
    process = FakeAiiDAProcess(
        pk=42,
        uuid="550e8400-e29b-41d4-a716-446655440000",
        process_state=FakeProcessState.RUNNING,
    )

    result = build_aiida_execution_result(
        calculation_id="igzo-aiida-001",
        process=process,
        job_id="987654",
        host="lumi",
    )

    assert result.calculation_id == "igzo-aiida-001"
    assert result.backend.value == "aiida"
    assert result.state == ExecutionState.RUNNING

    assert result.job_id == "987654"
    assert result.process_id == "42"
    assert (
        result.process_uuid
        == "550e8400-e29b-41d4-a716-446655440000"
    )

    assert result.host == "lumi"


def test_build_aiida_execution_result_without_pk() -> None:
    process = FakeAiiDAProcess(
        pk=None,
        uuid="550e8400-e29b-41d4-a716-446655440001",
        process_state=FakeProcessState.FINISHED,
    )

    result = build_aiida_execution_result(
        calculation_id="aiida-test",
        process=process,
    )

    assert result.process_id is None
    assert result.state == ExecutionState.COMPLETED


def test_build_aiida_execution_result_rejects_missing_state() -> None:
    process = FakeAiiDAProcess(
        pk=1,
        uuid="550e8400-e29b-41d4-a716-446655440002",
        process_state=None,
    )

    with pytest.raises(
        AiiDAExecutionError,
        match="no process state",
    ):
        build_aiida_execution_result(
            calculation_id="aiida-test",
            process=process,
        )


def test_build_aiida_execution_result_rejects_empty_uuid() -> None:
    process = FakeAiiDAProcess(
        pk=1,
        uuid="",
        process_state="running",
    )

    with pytest.raises(
        AiiDAExecutionError,
        match="UUID cannot be empty",
    ):
        build_aiida_execution_result(
            calculation_id="aiida-test",
            process=process,
        )

def test_aiida_cp2k_resources_defaults() -> None:
    resources = AiiDACp2kResources()

    assert resources.num_machines == 1
    assert resources.num_mpiprocs_per_machine == 1
    assert resources.max_wallclock_seconds == 600
    assert resources.queue_name is None
    assert resources.account is None
    assert resources.environment_variables is None
    assert resources.parser_name == "cp2k_base_parser"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"num_machines": 0},
            "num_machines must be at least 1",
        ),
        (
            {"num_mpiprocs_per_machine": 0},
            "num_mpiprocs_per_machine must be at least 1",
        ),
        (
            {"max_wallclock_seconds": 0},
            "max_wallclock_seconds must be at least 1",
        ),
    ],
)
def test_aiida_cp2k_resources_reject_invalid_values(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        AiiDACp2kResources(**kwargs)


def _fake_aiida_modules(
    *,
    process: FakeAiiDAProcess,
    code: MagicMock,
) -> tuple[
    dict[str, ModuleType],
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Build fake AiiDA modules for isolated adapter tests."""

    load_profile = MagicMock()
    submit = MagicMock(return_value=process)
    dict_factory = MagicMock(
        side_effect=lambda value: ("Dict", value)
    )
    structure_factory = MagicMock(
        side_effect=lambda **kwargs: (
            "StructureData",
            kwargs,
        )
    )
    load_code = MagicMock(return_value=code)

    aiida_module = ModuleType("aiida")
    engine_module = ModuleType("aiida.engine")
    orm_module = ModuleType("aiida.orm")

    aiida_module.load_profile = load_profile
    engine_module.submit = submit

    orm_module.Dict = dict_factory
    orm_module.StructureData = structure_factory
    orm_module.load_code = load_code

    modules = {
        "aiida": aiida_module,
        "aiida.engine": engine_module,
        "aiida.orm": orm_module,
    }

    return (
        modules,
        load_profile,
        submit,
        dict_factory,
        load_code,
    )


def test_submit_cp2k_aiida_builds_and_submits_calcjob() -> None:
    builder = MagicMock()

    computer = MagicMock()
    computer.hostname = "efp.lumi.csc.fi"

    code = MagicMock()
    code.get_builder.return_value = builder
    code.computer = computer

    process = FakeAiiDAProcess(
        pk=123,
        uuid="550e8400-e29b-41d4-a716-446655440123",
        process_state="created",
    )

    (
        modules,
        load_profile,
        submit,
        dict_factory,
        load_code,
    ) = _fake_aiida_modules(
        process=process,
        code=code,
    )

    parameters = {
        "GLOBAL": {
            "RUN_TYPE": "ENERGY",
        },
    }

    structure = object()

    resources = AiiDACp2kResources(
        num_machines=1,
        num_mpiprocs_per_machine=4,
        max_wallclock_seconds=900,
        queue_name="debug",
        account="project_465003407",
        environment_variables={
            "OMP_NUM_THREADS": "2",
        },
        parser_name="cp2k_base_parser",
    )

    with (
        patch.dict(
            sys.modules,
            modules,
        ),
        patch(
            "nsdw.execution.aiida.find_spec",
            return_value=object(),
        ),
    ):
        submission = submit_cp2k_aiida(
            calculation_id="lumi-smoke-001",
            code_label="cp2k-lumi-c@lumi-c",
            parameters=parameters,
            structure=structure,
            resources=resources,
            profile="nsdw-dev",
            label="NSDW LUMI smoke",
            description="Adapter integration test",
        )

    load_profile.assert_called_once_with("nsdw-dev")
    load_code.assert_called_once_with(
        "cp2k-lumi-c@lumi-c"
    )
    code.get_builder.assert_called_once_with()

    dict_factory.assert_called_once_with(parameters)

    assert builder.parameters == (
        "Dict",
        parameters,
    )
    assert builder.structure == (
        "StructureData",
        {
            "ase": structure,
        },
    )

    assert (
        builder.metadata.label
        == "NSDW LUMI smoke"
    )
    assert (
        builder.metadata.description
        == "Adapter integration test"
    )

    assert (
        builder.metadata.options.resources
        == {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 4,
        }
    )
    assert (
        builder.metadata.options.max_wallclock_seconds
        == 900
    )
    assert builder.metadata.options.withmpi is True
    assert (
        builder.metadata.options.queue_name
        == "debug"
    )
    assert (
        builder.metadata.options.account
        == "project_465003407"
    )
    assert (
        builder.metadata.options.environment_variables
        == {
            "OMP_NUM_THREADS": "2",
        }
    )
    assert (
        builder.metadata.options.parser_name
        == "cp2k_base_parser"
    )

    submit.assert_called_once_with(builder)

    assert submission.process is process

    assert (
        submission.result.calculation_id
        == "lumi-smoke-001"
    )
    assert (
        submission.result.state
        == ExecutionState.CREATED
    )
    assert submission.result.process_id == "123"
    assert (
        submission.result.process_uuid
        == "550e8400-e29b-41d4-a716-446655440123"
    )
    assert (
        submission.result.host
        == "efp.lumi.csc.fi"
    )


def test_submit_cp2k_aiida_uses_default_profile_and_label() -> None:
    builder = MagicMock()

    code = MagicMock()
    code.get_builder.return_value = builder
    code.computer = None

    process = FakeAiiDAProcess(
        pk=124,
        uuid="550e8400-e29b-41d4-a716-446655440124",
        process_state="waiting",
    )

    (
        modules,
        load_profile,
        submit,
        _,
        _,
    ) = _fake_aiida_modules(
        process=process,
        code=code,
    )

    with (
        patch.dict(
            sys.modules,
            modules,
        ),
        patch(
            "nsdw.execution.aiida.find_spec",
            return_value=object(),
        ),
    ):
        submission = submit_cp2k_aiida(
            calculation_id="default-label",
            code_label="cp2k-test",
            parameters={},
            structure=object(),
            resources=AiiDACp2kResources(),
        )

    load_profile.assert_called_once_with()
    submit.assert_called_once_with(builder)

    assert (
        builder.metadata.label
        == "default-label"
    )
    assert (
        submission.result.state
        == ExecutionState.SUBMITTED
    )
    assert submission.result.host is None


def test_submit_cp2k_aiida_wraps_code_loading_error() -> None:
    process = FakeAiiDAProcess(
        pk=125,
        uuid="550e8400-e29b-41d4-a716-446655440125",
        process_state="created",
    )

    code = MagicMock()

    (
        modules,
        _,
        _,
        _,
        load_code,
    ) = _fake_aiida_modules(
        process=process,
        code=code,
    )

    load_code.side_effect = RuntimeError(
        "code not found"
    )

    with (
        patch.dict(
            sys.modules,
            modules,
        ),
        patch(
            "nsdw.execution.aiida.find_spec",
            return_value=object(),
        ),
    ):
        with pytest.raises(
            AiiDAExecutionError,
            match="Could not load AiiDA code",
        ):
            submit_cp2k_aiida(
                calculation_id="bad-code",
                code_label="missing-code",
                parameters={},
                structure=object(),
                resources=AiiDACp2kResources(),
            )


def test_submit_cp2k_aiida_wraps_submission_error() -> None:
    builder = MagicMock()

    code = MagicMock()
    code.get_builder.return_value = builder
    code.computer = None

    process = FakeAiiDAProcess(
        pk=126,
        uuid="550e8400-e29b-41d4-a716-446655440126",
        process_state="created",
    )

    (
        modules,
        _,
        submit,
        _,
        _,
    ) = _fake_aiida_modules(
        process=process,
        code=code,
    )

    submit.side_effect = RuntimeError(
        "broker unavailable"
    )

    with (
        patch.dict(
            sys.modules,
            modules,
        ),
        patch(
            "nsdw.execution.aiida.find_spec",
            return_value=object(),
        ),
    ):
        with pytest.raises(
            AiiDAExecutionError,
            match="AiiDA CP2K submission failed",
        ):
            submit_cp2k_aiida(
                calculation_id="submission-error",
                code_label="cp2k-test",
                parameters={},
                structure=object(),
                resources=AiiDACp2kResources(),
            )
