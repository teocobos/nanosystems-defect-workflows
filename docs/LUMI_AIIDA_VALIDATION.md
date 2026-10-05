# LUMI-C AiiDA Validation

## Summary

NSDW's AiiDA-backed CP2K production workflow has been validated end to end on LUMI-C.

The validated path is:

project.yaml
    ↓
validated CP2KProductionMethodology
    ↓
ProjectWorkspace
    ↓
CP2KProductionRecipe
    ↓
CP2KInputConfig
    ↓
build_aiida_cp2k_parameters()
    ↓
submit_cp2k_production_aiida()
    ↓
submit_cp2k_aiida()
    ↓
AiiDA
    ↓
LUMI-C
    ↓
SLURM / srun
    ↓
CP2K
    ↓
aiida-cp2k parser
    ↓
NSDW ExecutionResult

## Environment

- AiiDA profile: `nsdw-dev`
- AiiDA Computer: `lumi-c`
- Host: `efp.lumi.csc.fi`
- AiiDA CP2K code: `cp2k-lumi-c@lumi-c`
- Scheduler: SLURM
- CP2K: 2024.3
- NSDW execution backend: `aiida`

## Validation 1: Production Python API

A production calculation was submitted from an NSDW project containing persisted validated CP2K methodology.

A temporary NSDW project workspace was written to disk and reloaded before submission so that the validation exercised the persisted `project.yaml` path rather than only an in-memory configuration.

AiiDA process:

- PK: `31`
- UUID: `65ef8a70-4b26-476d-bdb6-92c4b26b3075`
- Process type: `Cp2kCalculation`
- State: `Finished [0]`
- Computer: `lumi-c`

Parsed result:

- calculation_id: `nsdw-production-lumi-001`
- backend: `aiida`
- state: `completed`
- host: `efp.lumi.csc.fi`
- finished_ok: `True`
- exit_status: `0`
- energy: `-1.1577880333216 a.u.`
- warnings: `0`

## Validation 2: Public NSDW CLI

The same production execution architecture was validated through the public NSDW command-line interface.

Command path:

`nsdw workflow production-submit`

AiiDA process:

- PK: `37`
- UUID: `26f94cac-1304-4658-aef2-3c83eb7f4507`
- Label: `nsdw-production-cli-lumi-001`
- Process type: `Cp2kCalculation`
- State: `Finished [0]`
- Computer: `lumi-c`

AiiDA inputs:

- InstalledCode: PK `5`
- parameters: PK `35`
- structure: PK `36`

AiiDA outputs:

- output_parameters: PK `40`
- remote_folder: PK `38`
- retrieved: PK `39`

AiiDA emitted:

`No restart file found in the retrieved folder.`

For this ENERGY validation calculation, no restart file was required and the calculation completed successfully with exit status 0.

## Test status

Following the production AiiDA and CLI integration work:

`545 passed, 31 warnings`

The remaining warnings are existing spglib deprecation warnings.

## Result

NSDW's first user-facing AiiDA production workflow is validated on LUMI-C.
