"""CP2K calculator integration for NSDW."""

from nsdw.calculators.cp2k.models import (
    CP2KRunType,
    CP2KSCFStatus,
    ParsedCP2KEnergy,
    ParsedCP2KResult,
    ParsedCP2KSCF,
    ParsedCP2KInput,
    ParsedCP2KKind,
    ParsedCP2KMultigrid,
    ParsedCP2KMultigridLevel,
)
from nsdw.calculators.cp2k.parser import (
    CP2KParseError,
    parse_cp2k_output,
    parse_cp2k_text,
)
from nsdw.calculators.cp2k.adapter import (
    CP2KAdapterError,
    HARTREE_TO_EV,
    adapt_cp2k_result,
)
from nsdw.calculators.cp2k.input_parser import (
    CP2KInputParseError,
    parse_cp2k_input,
    parse_cp2k_input_text,
)
from nsdw.calculators.cp2k.execution import (
    CP2KExecutionError,
    build_archer2_cp2k_job,
    build_cp2k_execution_request,
)
from nsdw.calculators.cp2k.environment import (
    CP2KEnvironmentError,
    CP2KDataDirectoryNotFoundError,
    CP2KDataFileNotFoundError,
    build_cp2k_environment,
    resolve_cp2k_data_dir,
    validate_cp2k_data_dir,
)

__all__ = [
    "ParsedCP2KMultigrid",
    "ParsedCP2KMultigridLevel",
    "CP2KRunType",
    "CP2KSCFStatus",
    "ParsedCP2KEnergy",
    "ParsedCP2KResult",
    "ParsedCP2KSCF",
    "CP2KParseError",
    "parse_cp2k_output",
    "parse_cp2k_text",
    "CP2KAdapterError",
    "HARTREE_TO_EV",
    "adapt_cp2k_result",
    "CP2KInputParseError",
    "ParsedCP2KInput",
    "ParsedCP2KKind",
    "parse_cp2k_input",
    "parse_cp2k_input_text",
    "CP2KExecutionError",
    "build_cp2k_execution_request",
    "build_archer2_cp2k_job",
    "CP2KEnvironmentError",
    "CP2KDataDirectoryNotFoundError",
    "CP2KDataFileNotFoundError",
    "build_cp2k_environment",
    "resolve_cp2k_data_dir",
    "validate_cp2k_data_dir",
]
