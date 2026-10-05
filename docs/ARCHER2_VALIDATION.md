# ARCHER2 validation

Final ARCHER2 compatibility validation completed successfully.

## Validation calculation

- Material: alpha-quartz SiO2
- Calculator: CP2K 2025.2
- Platform: ARCHER2
- Scheduler: SLURM
- NSDW version: 0.1.0
- Run type: ENERGY
- Cutoff: 600 Ry
- Relative cutoff: 60 Ry
- Basis file: BASIS_MOLOPT
- Potential file: GTH_POTENTIALS
- Nodes: 1
- Tasks per node: 128
- SLURM job ID: 15509027
- Result status: completed
- Total energy: -2951.633703865346 eV

## Validated workflow

The following production path was exercised successfully:

1. Create an NSDW material project.
2. Validate the input structure.
3. Store and promote a CP2K project methodology.
4. Generate a production CP2K package from the project methodology.
5. Submit the generated package through the ARCHER2 SLURM workflow.
6. Monitor job completion.
7. Parse the CP2K output.
8. Write the normalized NSDW `result.json`.

The generated calculation directory contained the CP2K input,
external XYZ coordinates, SLURM job script/output, CP2K output,
restart wavefunctions, and NSDW result record.

## Follow-up improvements

### Runtime CLI feedback

Long-running HPC workflows should show visible execution state rather
than leaving the terminal static while monitoring a job.

Desired information includes:

- SUBMITTED / PENDING / RUNNING / COMPLETED state
- SLURM job ID
- elapsed time
- animated spinner or progress indicator
- optional live tail of relevant CP2K output

### Provenance enrichment

The validation result exposed several fields that should be populated
more completely:

- `structure`
- `provenance.software.git_commit`
- `provenance.execution.started_at`
- `provenance.execution.completed_at`
- `provenance.execution.host`

These are follow-up provenance improvements and do not block the
successful ARCHER2 compatibility validation.

## Status

ARCHER2 compatibility checkpoint: PASSED.

Future development should avoid further ARCHER2-specific architecture
unless needed for regression compatibility. Primary HPC workflow
development now moves to AiiDA and LUMI-C.
