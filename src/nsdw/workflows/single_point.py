"""Single-point calculation workflow for NSDW."""

from __future__ import annotations

from pathlib import Path

from nsdw.calculators.cp2k import (
    adapt_cp2k_result,
    build_cp2k_execution_request,
    parse_cp2k_input,
    parse_cp2k_output,
)
from nsdw.execution import (
    ExecutionState,
    LocalExecutor,
)
from nsdw.models.provenance import (
    ExecutionPlatform,
    SchedulerType,
)
from nsdw.models.result import NSDWResult


class SinglePointWorkflowError(RuntimeError):
    """Raised when an NSDW single-point workflow fails."""


def run_cp2k_single_point(
    *,
    calculation_id: str,
    working_directory: str | Path,
    input_file: str | Path,
    output_file: str | Path,
    executable: str = "cp2k.psmp",
    environment: dict[str, str] | None = None,
    result_file: str | Path = "result.json",
) -> NSDWResult:
    """Run and process a local CP2K single-point calculation."""

    working_directory = (
        Path(working_directory)
        .expanduser()
        .resolve()
    )

    input_file = Path(input_file)
    output_file = Path(output_file)
    result_file = Path(result_file)

    request = build_cp2k_execution_request(
        calculation_id=calculation_id,
        working_directory=working_directory,
        input_file=input_file,
        output_file=output_file,
        executable=executable,
        environment=environment,
    )

    execution = LocalExecutor().execute(
        request
    )

    if execution.state != ExecutionState.COMPLETED:
        raise SinglePointWorkflowError(
            "CP2K execution failed with return code "
            f"{execution.return_code}"
        )

    input_path = (
        input_file
        if input_file.is_absolute()
        else working_directory / input_file
    )

    output_path = (
        output_file
        if output_file.is_absolute()
        else working_directory / output_file
    )

    parsed_input = parse_cp2k_input(
        input_path
    )

    parsed_output = parse_cp2k_output(
        output_path
    )

    result = adapt_cp2k_result(
        parsed_output,
        input_settings=parsed_input,
        input_path=input_path,
        output_path=output_path,
        calculation_id=calculation_id,
        platform=ExecutionPlatform.LOCAL,
        scheduler=SchedulerType.LOCAL,
    )

    destination = (
        result_file
        if result_file.is_absolute()
        else working_directory / result_file
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        result.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    return result
