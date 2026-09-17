"""Tests for CP2K execution-request construction."""

import pytest

from nsdw.calculators.cp2k.execution import (
    CP2KExecutionError,
    build_cp2k_execution_request,
)


def test_build_cp2k_execution_request(
    tmp_path,
):
    input_path = tmp_path / "calculation.inp"

    input_path.write_text(
        "&GLOBAL\n"
        "  PROJECT test\n"
        "  RUN_TYPE ENERGY\n"
        "&END GLOBAL\n"
    )

    request = build_cp2k_execution_request(
        calculation_id="cp2k_test",
        working_directory=tmp_path,
        input_file="calculation.inp",
        output_file="calculation.out",
    )

    assert request.calculation_id == "cp2k_test"

    assert request.command == (
        "cp2k.psmp",
        "-i",
        "calculation.inp",
        "-o",
        "calculation.out",
    )

    assert (
        request.working_directory
        == tmp_path.resolve()
    )


def test_custom_cp2k_executable(
    tmp_path,
):
    input_path = tmp_path / "test.inp"

    input_path.write_text(
        "&GLOBAL\n&END GLOBAL\n"
    )

    request = build_cp2k_execution_request(
        calculation_id="test",
        working_directory=tmp_path,
        input_file="test.inp",
        output_file="test.out",
        executable="cp2k.ssmp",
    )

    assert request.command[0] == "cp2k.ssmp"


def test_cp2k_execution_environment(
    tmp_path,
):
    input_path = tmp_path / "test.inp"

    input_path.write_text(
        "&GLOBAL\n&END GLOBAL\n"
    )

    request = build_cp2k_execution_request(
        calculation_id="test",
        working_directory=tmp_path,
        input_file="test.inp",
        output_file="test.out",
        environment={
            "OMP_NUM_THREADS": "2",
        },
    )

    assert (
        request.environment["OMP_NUM_THREADS"]
        == "2"
    )


def test_missing_cp2k_input_rejected(
    tmp_path,
):
    with pytest.raises(
        CP2KExecutionError,
        match="input file",
    ):
        build_cp2k_execution_request(
            calculation_id="test",
            working_directory=tmp_path,
            input_file="missing.inp",
            output_file="test.out",
        )


def test_missing_working_directory_rejected(
    tmp_path,
):
    with pytest.raises(
        CP2KExecutionError,
        match="Working directory",
    ):
        build_cp2k_execution_request(
            calculation_id="test",
            working_directory=(
                tmp_path / "missing"
            ),
            input_file="test.inp",
            output_file="test.out",
        )


def test_empty_cp2k_executable_rejected(
    tmp_path,
):
    input_path = tmp_path / "test.inp"

    input_path.write_text(
        "&GLOBAL\n&END GLOBAL\n"
    )

    with pytest.raises(
        CP2KExecutionError,
        match="executable",
    ):
        build_cp2k_execution_request(
            calculation_id="test",
            working_directory=tmp_path,
            input_file="test.inp",
            output_file="test.out",
            executable="   ",
        )
