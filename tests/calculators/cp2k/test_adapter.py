"""Tests for the CP2K to NSDW result adapter."""

from pathlib import Path

import pytest

from nsdw.calculators.cp2k import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KEnergy,
    ParsedCP2KResult,
    ParsedCP2KSCF,
    ParsedCP2KInput,
)
from nsdw.calculators.cp2k.adapter import (
    HARTREE_TO_EV,
    CP2KAdapterError,
    adapt_cp2k_result,
)
from nsdw.models.calculation import (
    Backend,
    CalculationStatus,
    CalculationType,
)
from nsdw.calculators.cp2k.input_parser import (
    parse_cp2k_input,
)

FIXTURES = (
    Path(__file__).parent
    / "fixtures"
)

def _completed_energy_result() -> ParsedCP2KResult:
    return ParsedCP2KResult(
        cp2k_version="2025.2",
        run_type=CP2KRunType.ENERGY,
        project_name="igzo_ordered_003_sp",
        charge=0,
        multiplicity=1,
        energy=ParsedCP2KEnergy(
            total_energy_hartree=(
                -767.248205609001843
            )
        ),
        scf=ParsedCP2KSCF(
            status=CP2KSCFStatus.CONVERGED,
            iterations=2,
        ),
        normal_termination=True,
    )


def test_adapt_completed_cp2k_energy_result():
    parsed = _completed_energy_result()

    result = adapt_cp2k_result(parsed)

    assert (
        result.calculation.id
        == "igzo_ordered_003_sp"
    )

    assert (
        result.calculation.type
        == CalculationType.SINGLE_POINT
    )

    assert (
        result.calculation.status
        == CalculationStatus.COMPLETED
    )

    assert (
        result.calculation.backend
        == Backend.CP2K
    )


def test_adapter_converts_hartree_to_ev():
    parsed = _completed_energy_result()

    result = adapt_cp2k_result(parsed)

    assert result.energy is not None
    assert result.energy.total is not None

    assert result.energy.total.unit == "eV"

    assert (
        result.energy.total.value
        == pytest.approx(
            -767.248205609001843
            * HARTREE_TO_EV
        )
    )


def test_adapter_builds_cp2k_provenance():
    parsed = _completed_energy_result()

    result = adapt_cp2k_result(parsed)

    assert result.provenance is not None

    assert (
        result.provenance.software.calculator
        == "CP2K"
    )

    assert (
        result.provenance.software.calculator_version
        == "2025.2"
    )

    assert result.provenance.cp2k is not None

    assert result.provenance.cp2k.charge == 0
    assert result.provenance.cp2k.multiplicity == 1


def test_non_converged_scf_is_failed():
    parsed = _completed_energy_result().model_copy(
        update={
            "scf": ParsedCP2KSCF(
                status=CP2KSCFStatus.NOT_CONVERGED,
            )
        }
    )

    result = adapt_cp2k_result(parsed)

    assert (
        result.calculation.status
        == CalculationStatus.FAILED
    )


def test_abnormal_termination_is_failed():
    parsed = _completed_energy_result().model_copy(
        update={
            "normal_termination": False,
        }
    )

    result = adapt_cp2k_result(parsed)

    assert (
        result.calculation.status
        == CalculationStatus.FAILED
    )


def test_missing_run_type_rejected():
    parsed = _completed_energy_result().model_copy(
        update={
            "run_type": None,
        }
    )

    with pytest.raises(
        CP2KAdapterError,
        match="run type is missing",
    ):
        adapt_cp2k_result(parsed)


def test_missing_version_rejected():
    parsed = _completed_energy_result().model_copy(
        update={
            "cp2k_version": None,
        }
    )

    with pytest.raises(
        CP2KAdapterError,
        match="version",
    ):
        adapt_cp2k_result(parsed)

def test_adapter_uses_real_cp2k_input_provenance():
    parsed = _completed_energy_result()

    input_settings = parse_cp2k_input(
        FIXTURES
        / "igzo_ordered_003_sp.inp"
    )

    result = adapt_cp2k_result(
        parsed,
        input_settings=input_settings,
    )

    assert result.provenance is not None
    assert result.provenance.cp2k is not None

    cp2k = result.provenance.cp2k

    assert cp2k.xc_functional == "PBE"

    assert cp2k.cutoff is not None
    assert cp2k.cutoff.value == pytest.approx(
        700.0
    )
    assert cp2k.cutoff.unit == "Ry"

    assert cp2k.relative_cutoff is not None
    assert (
        cp2k.relative_cutoff.value
        == pytest.approx(60.0)
    )
    assert cp2k.relative_cutoff.unit == "Ry"

    assert cp2k.eps_scf == pytest.approx(
        1.0e-7
    )

    assert cp2k.k_points == (6, 6, 1)

    assert cp2k.basis_set_file == (
        "BASIS_MOLOPT_UZH"
    )

    assert cp2k.potential_file == (
        "POTENTIAL_UZH"
    )

    assert len(cp2k.kinds) == 4

    assert cp2k.kinds[0].kind == "In"
    assert cp2k.kinds[0].element == "In"
    assert (
        cp2k.kinds[0].basis_set
        == "TZV2P-MOLOPT-PBE-GTH-q13"
    )
    assert (
        cp2k.kinds[0].potential
        == "GTH-PBE-q13"
    )

    assert cp2k.kinds[1].kind == "Ga"
    assert cp2k.kinds[1].element == "Ga"

    assert cp2k.kinds[2].kind == "Zn"
    assert cp2k.kinds[2].element == "Zn"
    assert (
        cp2k.kinds[2].potential
        == "GTH-PBE-q12"
    )

    assert cp2k.kinds[3].kind == "O"
    assert cp2k.kinds[3].element == "O"
    assert (
        cp2k.kinds[3].potential
        == "GTH-PBE-q6"
    )
def test_project_name_mismatch_rejected():
    parsed = _completed_energy_result()

    input_settings = ParsedCP2KInput(
        project_name="wrong_project",
        run_type=CP2KRunType.ENERGY,
        charge=0,
        multiplicity=1,
    )

    with pytest.raises(
        CP2KAdapterError,
        match="project-name mismatch",
    ):
        adapt_cp2k_result(
            parsed,
            input_settings=input_settings,
        )


def test_run_type_mismatch_rejected():
    parsed = _completed_energy_result()

    input_settings = ParsedCP2KInput(
        project_name=parsed.project_name,
        run_type=CP2KRunType.GEO_OPT,
        charge=0,
        multiplicity=1,
    )

    with pytest.raises(
        CP2KAdapterError,
        match="run-type mismatch",
    ):
        adapt_cp2k_result(
            parsed,
            input_settings=input_settings,
        )


def test_charge_mismatch_rejected():
    parsed = _completed_energy_result()

    input_settings = ParsedCP2KInput(
        project_name=parsed.project_name,
        run_type=CP2KRunType.ENERGY,
        charge=-1,
        multiplicity=1,
    )

    with pytest.raises(
        CP2KAdapterError,
        match="charge mismatch",
    ):
        adapt_cp2k_result(
            parsed,
            input_settings=input_settings,
        )


def test_multiplicity_mismatch_rejected():
    parsed = _completed_energy_result()

    input_settings = ParsedCP2KInput(
        project_name=parsed.project_name,
        run_type=CP2KRunType.ENERGY,
        charge=0,
        multiplicity=2,
    )

    with pytest.raises(
        CP2KAdapterError,
        match="multiplicity mismatch",
    ):
        adapt_cp2k_result(
            parsed,
            input_settings=input_settings,
        )
def test_adapter_records_input_output_files(
    tmp_path,
):
    parsed = _completed_energy_result()

    input_path = tmp_path / "test.inp"
    output_path = tmp_path / "test.out"

    input_path.write_text(
        "&GLOBAL\n&END GLOBAL\n"
    )
    output_path.write_text(
        "CP2K test output\n"
    )

    result = adapt_cp2k_result(
        parsed,
        input_path=input_path,
        output_path=output_path,
    )

    assert result.provenance is not None

    assert len(
        result.provenance.input_files
    ) == 1

    assert len(
        result.provenance.output_files
    ) == 1

    input_ref = (
        result.provenance.input_files[0]
    )
    output_ref = (
        result.provenance.output_files[0]
    )

    assert input_ref.format == "cp2k-input"
    assert output_ref.format == "cp2k-output"

    assert input_ref.path == str(
        input_path.resolve()
    )
    assert output_ref.path == str(
        output_path.resolve()
    )

    assert len(input_ref.sha256) == 64
    assert len(output_ref.sha256) == 64

    assert input_ref.size_bytes == (
        input_path.stat().st_size
    )
    assert output_ref.size_bytes == (
        output_path.stat().st_size
    )