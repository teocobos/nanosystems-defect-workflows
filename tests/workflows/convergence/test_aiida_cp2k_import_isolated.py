from nsdw.workflows.convergence.aiida_workchain import (
    CP2KConvergenceWorkChain,
)

"""Isolated CP2K import experiments."""

from pathlib import Path
import shutil

import pytest

from aiida.engine import run_get_node
from aiida.orm import Dict, InstalledCode, RemoteData
from aiida.plugins import CalculationFactory


pytest_plugins = ("aiida.tools.pytest_fixtures",)


@pytest.mark.usefixtures("aiida_profile_clean")
def test_cp2k_import_isolated(
    aiida_computer_local,
    tmp_path,
):
    """Import a completed CP2K output without scheduler submission."""

    reference = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "cp2k"
        / "recovery_reference"
        / "aiida.out"
    )

    assert reference.is_file()

    remote_directory = tmp_path / "completed_cp2k"
    remote_directory.mkdir()

    shutil.copy2(
        reference,
        remote_directory / "aiida.out",
    )

    local_computer = aiida_computer_local()

    assert local_computer.is_stored
    assert local_computer.transport_type == "core.local"

    remote_folder = RemoteData(
        computer=local_computer,
        remote_path=str(remote_directory),
    )

    calculation = CalculationFactory("cp2k")

    builder = calculation.get_builder()

    # Test-only code identity. The import pathway must not
    # execute this command.
    code = InstalledCode(
        label="cp2k-import-test",
        computer=local_computer,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
    ).store()

    builder.code = code

    builder.parameters = Dict(
        dict={
            "GLOBAL": {
                "PROJECT": "nsdw-import-test",
                "RUN_TYPE": "ENERGY",
            }
        }
    )

    builder.remote_folder = remote_folder

    builder.metadata.options.parser_name = (
        "cp2k_base_parser"
    )
    builder.metadata.options.output_filename = (
        "aiida.out"
    )

    builder.metadata.options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
    }
    builder.metadata.options.max_wallclock_seconds = 1800

    result, node = run_get_node(builder)

    assert node.is_finished_ok, (
        f"Import failed: exit status {node.exit_status}"
    )

    assert "output_parameters" in result

    parsed = result["output_parameters"].get_dict()

    assert parsed["energy_units"] == "a.u."
    assert abs(
        parsed["energy"] - (-108.61874386877)
    ) < 1e-10

    # Verify the imported CalcJob and its provenance.
    from aiida.common.links import LinkType
    from aiida.orm import CalcJobNode

    assert isinstance(node, CalcJobNode)
    assert node.is_finished_ok
    assert node.exit_status == 0

    assert node.base.attributes.get(
        CalcJobNode.IMMIGRATED_KEY
    ) is True

    assert node.computer.uuid == local_computer.uuid
    assert node.inputs.code.uuid == code.uuid

    assert node.get_option("max_wallclock_seconds") == 1800
    assert node.get_option("resources") == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
    }

    assert node.get_job_id() is None

    output = node.outputs.output_parameters

    assert output.uuid == result["output_parameters"].uuid

    incoming = output.base.links.get_incoming(
        link_type=LinkType.CREATE
    ).all()

    assert len(incoming) == 1
    assert incoming[0].node.uuid == node.uuid
    assert incoming[0].link_label == "output_parameters"


@pytest.mark.usefixtures("aiida_profile_clean")
def test_cp2k_import_parent_child_provenance(
    aiida_computer_local,
    tmp_path,
):
    """Verify genuine CALL_CALC provenance for an imported CP2K job."""
    from aiida.common.links import LinkType
    from aiida.engine import ToContext, WorkChain
    from aiida.orm import InstalledCode, WorkChainNode

    local_computer = aiida_computer_local()

    reference = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "cp2k"
        / "recovery_reference"
        / "aiida.out"
    )

    assert reference.is_file()

    remote_directory = tmp_path / "completed_cp2k"
    remote_directory.mkdir()

    shutil.copy2(
        reference,
        remote_directory / "aiida.out",
    )

    remote_folder = RemoteData(
        computer=local_computer,
        remote_path=str(remote_directory),
    )

    code = InstalledCode(
        label="cp2k-parent-child-test",
        computer=local_computer,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
    ).store()

    calculation = CalculationFactory("cp2k")

    class ImportParentProbe(WorkChain):
        @classmethod
        def define(cls, spec):
            super().define(spec)

            spec.input("code")
            spec.input("remote_folder")
            spec.input("parameters")

            spec.outline(
                cls.launch_import,
                cls.inspect_import,
            )

            spec.output("output_parameters")

            spec.exit_code(
                401,
                "ERROR_IMPORT_FAILED",
                message="Imported CP2K calculation failed.",
            )

        def launch_import(self):
            builder = calculation.get_builder()

            builder.code = self.inputs.code
            builder.remote_folder = self.inputs.remote_folder
            builder.parameters = self.inputs.parameters

            builder.metadata.options.parser_name = (
                "cp2k_base_parser"
            )
            builder.metadata.options.output_filename = (
                "aiida.out"
            )
            builder.metadata.options.resources = {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
            }
            builder.metadata.options.max_wallclock_seconds = 1800

            child = self.submit(builder)

            return ToContext(imported_cp2k=child)

        def inspect_import(self):
            child = self.ctx.imported_cp2k

            if not child.is_finished_ok:
                return self.exit_codes.ERROR_IMPORT_FAILED

            self.out(
                "output_parameters",
                child.outputs.output_parameters,
            )

    parameters = Dict(
        dict={
            "GLOBAL": {
                "PROJECT": "nsdw-import-test",
                "RUN_TYPE": "ENERGY",
            }
        }
    )

    results, parent = run_get_node(
        ImportParentProbe,
        code=code,
        remote_folder=remote_folder,
        parameters=parameters,
    )

    assert isinstance(parent, WorkChainNode)
    assert parent.is_finished_ok

    children = parent.base.links.get_outgoing(
        link_type=LinkType.CALL_CALC
    ).all()

    assert len(children) == 1

    child = children[0].node

    assert child.is_finished_ok
    assert child.base.attributes.get(
        child.IMMIGRATED_KEY
    ) is True

    assert child.get_job_id() is None

    assert (
        child.outputs.output_parameters.uuid
        == results["output_parameters"].uuid
    )

    assert abs(
        results["output_parameters"].get_dict()["energy"]
        - (-108.61874386877)
    ) < 1e-10




class CP2KImportedConvergenceTestWorkChain(
    CP2KConvergenceWorkChain
):
    """Test-only convergence WorkChain using imported CP2K outputs."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace(
            "import_folders",
            valid_type=RemoteData,
            required=True,
            dynamic=True,
        )

    def submit_candidates(self):
        candidates = self.inputs.candidates.get_dict()
        resources = self.inputs.resources.get_dict()

        from aiida.engine import ToContext
        from aiida.orm import Dict
        from aiida.plugins import CalculationFactory

        from nsdw.workflows.convergence.aiida_workchain import (
            validate_candidate_payload,
        )

        try:
            validate_candidate_payload(candidates, resources)
        except ValueError:
            return self.exit_codes.ERROR_INVALID_INPUT

        if not self.inputs.retry_only.value:
            return self.exit_codes.ERROR_INVALID_INPUT

        expected_keys = {
            f"candidate_{index:04d}"
            for index in range(len(candidates))
        }

        if set(self.inputs.import_folders) != expected_keys:
            return self.exit_codes.ERROR_INVALID_INPUT

        Cp2kCalculation = CalculationFactory("cp2k")

        running = {}
        self.ctx.candidate_names = {}

        for index, (name, parameters) in enumerate(
            candidates.items()
        ):
            internal_id = f"candidate_{index:04d}"

            builder = Cp2kCalculation.get_builder()
            builder.code = self.inputs.code
            builder.structure = self.inputs.structure
            builder.parameters = Dict(dict=parameters)
            builder.remote_folder = self.inputs.import_folders[
                internal_id
            ]

            builder.metadata.label = name
            builder.metadata.options.resources = {
                "num_machines": resources["num_machines"],
                "num_mpiprocs_per_machine": (
                    resources["num_mpiprocs_per_machine"]
                ),
            }
            builder.metadata.options.max_wallclock_seconds = (
                resources["max_wallclock_seconds"]
            )
            builder.metadata.options.withmpi = True

            for key in (
                "queue_name",
                "account",
                "environment_variables",
                "parser_name",
            ):
                if resources.get(key) is not None:
                    builder.metadata.options[key] = resources[key]

            self.ctx.candidate_names[internal_id] = name
            running[internal_id] = self.submit(builder)

        return ToContext(**running)

@pytest.mark.usefixtures("aiida_profile_clean")
def test_real_convergence_workchain_imports(
    aiida_computer_local,
    tmp_path,
):
    """Exercise the actual NSDW WorkChain using local CP2K imports."""
    from aiida.common.links import LinkType
    from aiida.orm import (
        Bool,
        InstalledCode,
        StructureData,
        WorkChainNode,
    )
    from ase.build import molecule

    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
    )
    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        validate_convergence_summary_provenance,
    )

    computer = aiida_computer_local()

    reference = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "cp2k"
        / "recovery_reference"
        / "aiida.out"
    )
    assert reference.is_file()

    folders = {}

    for label in ("40-Ry", "50-Ry"):
        directory = tmp_path / label
        directory.mkdir()
        shutil.copy2(reference, directory / "aiida.out")

        folders[label] = RemoteData(
            computer=computer,
            remote_path=str(directory),
        )

    code = InstalledCode(
        label="cp2k-nsdw-convergence-import",
        computer=computer,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
    ).store()

    structure = StructureData(
        ase=molecule("H2O"),
    )

    candidates = Dict(
        dict={
            label: {
                "GLOBAL": {
                    "PROJECT": "nsdw-import-test",
                    "RUN_TYPE": "ENERGY",
                },
                "FORCE_EVAL": {
                    "METHOD": "QS",
                    "DFT": {
                        "MGRID": {
                            "CUTOFF": cutoff,
                            "REL_CUTOFF": 40,
                        },
                    },
                },
            }
            for label, cutoff in (
                ("40-Ry", 400),
                ("50-Ry", 500),
            )
        }
    )

    resources = Dict(
        dict={
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 1800,
        }
    )

    builder = CP2KImportedConvergenceTestWorkChain.get_builder()
    builder.code = code
    builder.structure = structure
    builder.candidates = candidates
    builder.resources = resources
    builder.retry_only = Bool(True)
    builder.import_folders = {
        f"candidate_{index:04d}": folders[label]
        for index, label in enumerate(candidates.get_dict())
    }

    results, parent = run_get_node(builder)

    assert isinstance(parent, WorkChainNode)
    assert parent.is_finished_ok, (
        f"Convergence WorkChain failed: {parent.exit_status}"
    )

    assert "summary" in results
    assert "analysis" not in results

    summary = results["summary"].get_dict()
    assert list(summary) == ["40-Ry", "50-Ry"]

    assert all(
        entry["imported"] is True
        for entry in summary.values()
    )

    # The production validator deliberately requires the exact
    # production WorkChain process type. This test subclass has
    # a different process type, so validate its graph directly.
    from aiida.orm import CalcFunctionNode

    summary_creators = results["summary"].base.links.get_incoming(
        link_type=LinkType.CREATE
    ).all()

    assert len(summary_creators) == 1

    summary_creator = summary_creators[0].node
    assert isinstance(summary_creator, CalcFunctionNode)

    summary_callers = summary_creator.base.links.get_incoming(
        link_type=LinkType.CALL_CALC
    ).all()

    assert len(summary_callers) == 1
    assert summary_callers[0].node.uuid == parent.uuid

    summary_returns = parent.base.links.get_outgoing(
        link_type=LinkType.RETURN
    ).all()

    assert any(
        link.link_label == "summary"
        and link.node.uuid == results["summary"].uuid
        for link in summary_returns
    )

    with pytest.raises(
        ValueError,
        match="Unexpected convergence campaign process type",
    ):
        validate_convergence_summary_provenance(
            campaign=parent,
        )

    from aiida.orm import CalcJobNode

    called = parent.base.links.get_outgoing(
        link_type=LinkType.CALL_CALC
    ).all()

    children = [
        link.node
        for link in called
        if isinstance(link.node, CalcJobNode)
    ]

    assert len(called) == 3
    assert len(children) == 2

    for child in children:

        assert child.is_finished_ok
        assert child.base.attributes.get(
            child.IMMIGRATED_KEY
        ) is True

        from nsdw.workflows.convergence.aiida_recovery_workchain import (
            validate_retry_calculation_provenance,
        )

        with pytest.raises(
            ValueError,
            match="Imported CP2K calculations cannot validate",
        ):
            validate_retry_calculation_provenance(
                calculation=child,
                parser_output=child.outputs.output_parameters,
                expected_resources={
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "max_wallclock_seconds": 1800,
                    "queue_name": None,
                    "account": None,
                },
            )
        assert child.get_job_id() is None
        assert child.get_option(
            "max_wallclock_seconds"
        ) == 1800

        assert abs(
            child.outputs.output_parameters.get_dict()["energy"]
            - (-108.61874386877)
        ) < 1e-10


def test_recovery_import_boundary_rejects_imports():
    """Recovery validation must reject imported WorkChain inputs."""
    from types import SimpleNamespace

    from nsdw.workflows.convergence.aiida_recovery_workchain import (
        validate_recovery_child_import_boundary,
    )

    imported_child = SimpleNamespace(
        inputs={"import_folders": {"candidate_0000": object()}}
    )

    with pytest.raises(
        ValueError,
        match="Imported CP2K results are not permitted",
    ):
        validate_recovery_child_import_boundary(
            imported_child,
        )

    normal_child = SimpleNamespace(
        inputs={"retry_only": True}
    )

    assert validate_recovery_child_import_boundary(
        normal_child,
    ) is True


def test_production_convergence_rejects_import_namespace():
    """Production WorkChain must not expose test-only import inputs."""
    from nsdw.workflows.convergence.aiida_workchain import (
        CP2KConvergenceWorkChain,
    )

    production_spec = CP2KConvergenceWorkChain.spec()
    test_spec = CP2KImportedConvergenceTestWorkChain.spec()

    assert "import_folders" not in production_spec.inputs
    assert "import_folders" in test_spec.inputs

    production_builder = CP2KConvergenceWorkChain.get_builder()

    with pytest.raises(
        (AttributeError, ValueError, TypeError),
    ):
        production_builder.import_folders = {
            "candidate_0000": object(),
        }
