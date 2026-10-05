"""Optional AiiDA execution support for NSDW."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Mapping, Protocol


from nsdw.execution.models import (
    ExecutionBackend,
    ExecutionResult,
    ExecutionState,
)


class AiiDAUnavailableError(RuntimeError):
    """Raised when an AiiDA operation is requested without AiiDA installed."""


class AiiDAExecutionError(RuntimeError):
    """Raised when NSDW cannot interpret an AiiDA execution."""


class AiiDAProcessLike(Protocol):
    """Minimal process interface required by the NSDW AiiDA adapter."""

    pk: int | None
    uuid: object
    process_state: object


@dataclass(frozen=True, slots=True)
class AiiDACp2kResources:
    """Execution resources for an AiiDA CP2K calculation."""

    num_machines: int = 1
    num_mpiprocs_per_machine: int = 1
    max_wallclock_seconds: int = 600
    queue_name: str | None = None
    account: str | None = None
    environment_variables: Mapping[str, str] | None = None
    parser_name: str | None = "cp2k_base_parser"

    def __post_init__(self) -> None:
        if self.num_machines < 1:
            raise ValueError(
                "num_machines must be at least 1"
            )

        if self.num_mpiprocs_per_machine < 1:
            raise ValueError(
                "num_mpiprocs_per_machine must be at least 1"
            )

        if self.max_wallclock_seconds < 1:
            raise ValueError(
                "max_wallclock_seconds must be at least 1"
            )


@dataclass(frozen=True, slots=True)
class AiiDASubmission:
    """Submitted AiiDA process and its NSDW execution result."""

    process: AiiDAProcessLike
    result: ExecutionResult


def aiida_available() -> bool:
    """Return whether the optional AiiDA dependency is installed."""

    return find_spec("aiida") is not None


def require_aiida() -> None:
    """Require AiiDA to be installed for the requested operation."""

    if not aiida_available():
        raise AiiDAUnavailableError(
            "AiiDA support is not installed. "
            "Install NSDW with the 'aiida' extra: "
            "pip install -e '.[aiida]'"
        )


def map_aiida_process_state(
    process_state: str,
) -> ExecutionState:
    """Map an AiiDA process state onto the NSDW lifecycle."""

    state = process_state.strip().lower()

    mapping = {
        "created": ExecutionState.CREATED,
        "waiting": ExecutionState.SUBMITTED,
        "running": ExecutionState.RUNNING,
        "finished": ExecutionState.COMPLETED,
        "excepted": ExecutionState.FAILED,
        "killed": ExecutionState.CANCELLED,
    }

    try:
        return mapping[state]
    except KeyError as exc:
        raise AiiDAExecutionError(
            "Unsupported AiiDA process state: "
            f"{process_state!r}"
        ) from exc


def build_aiida_execution_result(
    *,
    calculation_id: str,
    process: AiiDAProcessLike,
    job_id: str | None = None,
    host: str | None = None,
) -> ExecutionResult:
    """Build an NSDW execution result from an AiiDA process."""

    raw_state = process.process_state

    if raw_state is None:
        raise AiiDAExecutionError(
            "AiiDA process has no process state"
        )

    state_value = getattr(
        raw_state,
        "value",
        raw_state,
    )

    state = map_aiida_process_state(
        str(state_value)
    )

    process_id = (
        str(process.pk)
        if process.pk is not None
        else None
    )

    process_uuid = str(process.uuid)

    if not process_uuid.strip():
        raise AiiDAExecutionError(
            "AiiDA process UUID cannot be empty"
        )

    return ExecutionResult(
        calculation_id=calculation_id,
        backend=ExecutionBackend.AIIDA,
        state=state,
        job_id=job_id,
        process_id=process_id,
        process_uuid=process_uuid,
        host=host,
    )


def submit_cp2k_aiida(
    *,
    calculation_id: str,
    code_label: str,
    parameters: Mapping[str, Any],
    structure: Any,
    resources: AiiDACp2kResources,
    profile: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> AiiDASubmission:
    """Submit a CP2K calculation through AiiDA.

    Parameters are supplied as the nested CP2K dictionary expected by
    ``aiida-cp2k``. ``structure`` is expected to be an ASE ``Atoms``
    object and is converted to an AiiDA ``StructureData`` node here.

    AiiDA remains responsible for transport, scheduling, persistence,
    retrieval, parsing, and provenance. NSDW remains responsible for
    constructing the scientific CP2K input.
    """

    require_aiida()

    try:
        from aiida import load_profile
        from aiida.engine import submit
        from aiida.orm import (
            Dict,
            StructureData,
            load_code,
        )
    except ImportError as exc:
        raise AiiDAUnavailableError(
            "AiiDA CP2K execution dependencies could not be imported"
        ) from exc

    if profile is None:
        load_profile()
    else:
        load_profile(profile)

    try:
        code = load_code(code_label)
        builder = code.get_builder()
    except Exception as exc:
        raise AiiDAExecutionError(
            f"Could not load AiiDA code {code_label!r}"
        ) from exc

    builder.parameters = Dict(dict(parameters))
    builder.structure = StructureData(ase=structure)

    builder.metadata.label = (
        label
        if label is not None
        else calculation_id
    )

    if description is not None:
        builder.metadata.description = description

    builder.metadata.options.resources = {
        "num_machines": resources.num_machines,
        "num_mpiprocs_per_machine": (
            resources.num_mpiprocs_per_machine
        ),
    }

    builder.metadata.options.max_wallclock_seconds = (
        resources.max_wallclock_seconds
    )
    builder.metadata.options.withmpi = True

    if resources.queue_name is not None:
        builder.metadata.options.queue_name = (
            resources.queue_name
        )

    if resources.account is not None:
        builder.metadata.options.account = resources.account

    if resources.environment_variables is not None:
        builder.metadata.options.environment_variables = dict(
            resources.environment_variables
        )

    if resources.parser_name is not None:
        builder.metadata.options.parser_name = (
            resources.parser_name
        )

    try:
        process = submit(builder)
    except Exception as exc:
        raise AiiDAExecutionError(
            "AiiDA CP2K submission failed"
        ) from exc

    computer = getattr(code, "computer", None)

    host = (
        getattr(computer, "hostname", None)
        if computer is not None
        else None
    )

    result = build_aiida_execution_result(
        calculation_id=calculation_id,
        process=process,
        host=host,
    )

    return AiiDASubmission(
        process=process,
        result=result,
    )
