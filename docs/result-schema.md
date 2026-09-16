# NSDW Scientific Result Schema

## 1. Purpose

The NSDW Scientific Result defines a calculator-independent representation of
scientific calculation results.

Raw outputs from CP2K, VASP, MACE, LAMMPS, or future backends should not be
consumed directly by higher-level scientific analysis wherever practical.

Instead:

    Calculator output
          |
          v
    Calculator parser
          |
          v
    NSDW Scientific Result
          |
          v
    Scientific analysis
          |
          v
    Device-relevant quantity

This separates scientific analysis from calculator-specific file formats.


## 2. Design Principles

The result schema must be:

1. Calculator-independent
2. Explicitly versioned
3. Unit-aware
4. Machine-readable
5. Human-readable
6. Extensible
7. Reproducible
8. Suitable for validation
9. Suitable for crystalline and amorphous materials
10. Suitable for single calculations and statistical ensembles


## 3. Schema Versioning

Every result must contain:

    schema_version

Initial version:

    1.0.0

Schema changes should follow semantic versioning principles.

Major:
    incompatible schema change

Minor:
    backwards-compatible capability addition

Patch:
    backwards-compatible correction


## 4. Top-Level Result

Conceptually:

    NSDWResult
    |
    +-- schema
    +-- calculation
    +-- structure
    +-- energy
    +-- electronic
    +-- defect
    +-- migration
    +-- transport
    +-- analysis
    +-- validation
    +-- provenance

Not every section is required for every calculation.

For example, a pristine geometry optimisation may contain:

    calculation
    structure
    energy
    electronic
    validation
    provenance

while a vacancy calculation may additionally contain:

    defect

and an NEB calculation:

    migration


## 5. Calculation Metadata

Required fields should include:

    calculation.id
    calculation.type
    calculation.status
    calculation.backend

Possible calculation types include:

    geometry_optimisation
    single_point
    electronic_structure
    dos
    band_structure
    defect_relaxation
    charged_defect
    neb
    molecular_dynamics
    vibrational
    constrained_dft
    optical
    mlmd

Example:

    calculation:
        id: "igzo-vo-o001-q+2"
        type: "defect_relaxation"
        status: "completed"
        backend: "cp2k"


## 6. Units

NSDW should use a canonical internal unit system.

Recommended canonical units:

    energy              eV
    length              angstrom
    force               eV/angstrom
    stress              GPa
    time                fs
    temperature         K
    pressure            GPa
    frequency           THz
    charge              elementary charge
    diffusion           cm^2/s
    mobility            cm^2/(V s)

Raw calculator units may be retained in provenance, but scientific results
should be converted into canonical NSDW units.

Every quantity for which the unit is not obvious from the schema must carry
explicit unit information.


## 7. Structure Result

The structure section should describe the structure associated with the
scientific result.

Fields may include:

    structure.formula
    structure.n_atoms
    structure.periodic
    structure.cell
    structure.volume
    structure.density
    structure.space_group
    structure.positions
    structure.species
    structure.structure_hash

Example:

    structure:
        formula: "In3Ga3Zn3O12"
        n_atoms: 21
        periodic: true
        space_group:
            number: 160
            symbol: "R3m"


## 8. Energy Result

Potential fields:

    energy.total
    energy.free
    energy.formation
    energy.trapping
    energy.migration
    energy.reorganisation

Example:

    energy:
        total:
            value: -1234.567
            unit: "eV"

No energy value should exist without a clearly defined reference.


## 9. Electronic Result

Potential fields:

    electronic.band_gap
    electronic.vbm
    electronic.cbm
    electronic.fermi_level

    electronic.dos
    electronic.pdos

    electronic.charge
    electronic.spin

    electronic.localisation

Example:

    electronic:
        band_gap:
            value: 3.21
            unit: "eV"
            method: "PBE0-TC-LRC"

        vbm:
            value: -5.42
            unit: "eV"
            reference: "internal"

        cbm:
            value: -2.21
            unit: "eV"
            reference: "internal"


## 10. Localisation Result

Potential fields:

    electronic.localisation.ipr
    electronic.localisation.site
    electronic.localisation.species
    electronic.localisation.spin_population
    electronic.localisation.charge_population

Example:

    localisation:
        carrier: "electron"
        localised: true
        site_index: 17
        species: "In"
        ipr: 0.42

The localisation method must be recorded.


## 11. Defect Result

Potential fields:

    defect.type
    defect.name
    defect.species
    defect.site
    defect.charge
    defect.multiplicity
    defect.formation_energy
    defect.transition_levels
    defect.corrections

Example:

    defect:
        type: "vacancy"
        name: "V_O"
        species: "O"
        site_index: 12
        charge: 2


## 12. Defect Formation Energy

Formation energies must record the assumptions required to reproduce them.

Potential fields:

    defect.formation_energy.value
    defect.formation_energy.fermi_level
    defect.formation_energy.chemical_potentials
    defect.formation_energy.correction
    defect.formation_energy.temperature

This prevents formation energies calculated under different chemical
environments from being incorrectly compared.


## 13. Charge-Transition Levels

A transition level should contain:

    initial_charge
    final_charge
    energy
    reference
    correction_method

Example:

    transition_levels:
        - initial_charge: 0
          final_charge: -1
          energy:
              value: 1.35
              unit: "eV"
          reference: "VBM"


## 14. Migration Result

Potential fields:

    migration.species
    migration.initial_state
    migration.final_state
    migration.path
    migration.images
    migration.barrier
    migration.reverse_barrier
    migration.attempt_frequency

Example:

    migration:
        species: "O_i"
        mechanism: "interstitial"
        barrier:
            value: 1.70
            unit: "eV"


## 15. Diffusion Result

Potential fields:

    transport.diffusion.species
    transport.diffusion.temperature
    transport.diffusion.coefficient
    transport.diffusion.activation_energy
    transport.diffusion.method
    transport.diffusion.direction

Possible methods:

    neb_tst
    molecular_dynamics
    kinetic_network

Example:

    diffusion:
        species: "O"
        temperature:
            value: 800
            unit: "K"
        coefficient:
            value: 1.2e-10
            unit: "cm^2/s"
        method: "molecular_dynamics"


## 16. Carrier Transport Result

Band and hopping transport must remain distinguishable.

Potential transport types:

    band
    hopping
    polaron

Potential fields:

    transport.carrier
    transport.mechanism
    transport.temperature
    transport.mobility
    transport.conductivity
    transport.effective_mass

Example:

    transport:
        carrier: "electron"
        mechanism: "band"
        mobility:
            value: 18.4
            unit: "cm^2/(V s)"


## 17. Molecular Dynamics Result

Potential fields:

    md.ensemble
    md.temperature
    md.pressure
    md.timestep
    md.total_time
    md.n_steps
    md.trajectory
    md.msd

Large trajectories should not normally be embedded directly in JSON.

Instead:

    md.trajectory.path
    md.trajectory.hash
    md.trajectory.format

should reference an external artefact.


## 18. Statistical Results

Amorphous systems and ensemble calculations require distributions.

Potential fields:

    statistics.n_samples
    statistics.mean
    statistics.std
    statistics.minimum
    statistics.maximum
    statistics.median
    statistics.quantiles

Individual sample values may also be retained.

Example:

    statistics:
        quantity: "oxygen_vacancy_formation_energy"
        n_samples: 200
        mean:
            value: 4.21
            unit: "eV"
        std:
            value: 0.73
            unit: "eV"


## 19. Validation Result

Execution success and scientific validity must remain separate.

Potential fields:

    validation.execution
    validation.numerical
    validation.physical

Example:

    validation:
        execution:
            passed: true

        numerical:
            passed: true
            checks:
                scf_converged: true
                forces_converged: true

        physical:
            passed: true
            checks:
                structure_valid: true
                charge_localised: true


## 20. Provenance

Every result must retain sufficient provenance to identify its origin.

Required or strongly recommended fields:

    provenance.workflow
    provenance.workflow_version

    provenance.calculator
    provenance.calculator_version

    provenance.input_hash
    provenance.structure_hash

    provenance.command

    provenance.host
    provenance.scheduler
    provenance.job_id

    provenance.started_at
    provenance.completed_at

    provenance.git_commit

    provenance.dependencies


## 21. CP2K Provenance

For CP2K calculations, provenance should additionally retain information such
as:

    basis_set
    potential
    xc_functional
    cutoff
    relative_cutoff
    k_points
    charge
    multiplicity
    eps_scf

Hybrid calculations should retain parameters required to reproduce the
exchange treatment.


## 22. External Artefacts

Large data should be referenced rather than embedded.

Examples:

- trajectories
- wavefunctions
- restart files
- charge-density cubes
- DOS data
- band structures
- NEB images

Each artefact should record:

    path
    format
    hash
    size

Example:

    artefact:
        type: "wavefunction"
        path: "outputs/igzo-RESTART.wfn"
        format: "cp2k-wfn"
        sha256: "..."
        size_bytes: 1932735283


## 23. Missing Data

Missing quantities must not be represented using artificial numerical values.

Use:

    null

rather than values such as:

    0
    -1
    999

when a quantity was not calculated or is unavailable.

Where useful, a reason may also be recorded:

    status: "not_calculated"

or:

    status: "not_applicable"


## 24. Raw versus Derived Data

The schema should distinguish:

    raw calculator result

from:

    derived scientific result

Example:

    raw:
        total_energy

    derived:
        formation_energy
        transition_level
        diffusion_coefficient

Derived quantities should record the method and source calculations used.


## 25. Result Relationships

Scientific results often depend on other calculations.

For example:

    CTL
     |
     +-- bulk calculation
     |
     +-- q = 0 defect
     |
     +-- q = -1 defect

NSDW should therefore allow:

    parents
    children
    references

using stable calculation identifiers.


## 26. Serialisation

The primary portable serialisation format should initially be JSON.

Example:

    result.json

Human-readable YAML may be supported for configuration, but JSON should be
preferred for validated scientific result exchange.


## 27. Python Representation

The initial implementation should use typed Python models.

Recommended implementation:

    Pydantic

Conceptually:

    NSDWResult
        CalculationResult
        StructureResult
        EnergyResult
        ElectronicResult
        DefectResult
        MigrationResult
        TransportResult
        ValidationResult
        ProvenanceResult

The models should be capable of:

- validating results
- serialising results
- loading results
- generating JSON Schema
- rejecting invalid scientific records


## 28. JSON Schema

A machine-readable JSON Schema should be generated from the Python models.

Target:

    JSON Schema Draft 2020-12

The generated schema should be stored under:

    schemas/

for documentation, interoperability, and external validation.


## 29. Schema Evolution

Existing result files should remain interpretable after NSDW evolves.

Future migrations may therefore follow:

    v1 result
       |
       v
    migration
       |
       v
    v2 result

Schema migration should be explicit rather than silently modifying scientific
data.


## 30. Design Rule

Higher-level scientific analysis should consume:

    NSDWResult

rather than:

    CP2K output
    VASP OUTCAR
    arbitrary log files

Calculator-specific parsing belongs at the boundary of the system.

This gives:

    CP2K parser ----\
                     \
    VASP parser ------> NSDWResult --> scientific analysis
                     /
    MACE parser -----/

and allows the scientific analysis layer to remain independent of the
underlying calculator.
