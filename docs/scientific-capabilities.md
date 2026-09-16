# NSDW Scientific Capabilities

## 1. Purpose

Nanosystems Defect Workflows (NSDW) is a reproducible workflow framework for
atomistic semiconductor and electronic-materials modelling.

The objective is not simply to automate electronic-structure calculations.
NSDW is intended to convert atomistic simulations into physically meaningful,
device-relevant quantities while preserving reproducibility, provenance, and
portability across computational environments.

The scientific workflow is conceptually:

    material structure
          |
          v
    atomistic calculation
          |
          v
    standardised NSDW result
          |
          v
    scientific analysis
          |
          v
    materials / device quantity

NSDW should remain calculator-independent wherever possible.

Initial calculator and simulation backends include:

- CP2K
- VASP
- MACE
- LAMMPS

Supporting libraries may include:

- ASE
- pymatgen
- spglib
- doped
- Wannier90
- AMSET
- AiiDA

Existing validated scientific libraries should be reused where appropriate
rather than reimplemented unnecessarily.


## 2. Capability Levels

NSDW capabilities are divided into three development levels.

### Core

Capabilities required for the first scientifically useful NSDW release.

### Advanced

Capabilities that extend the core workflows into transport, kinetics,
disordered materials, and more complete semiconductor physics.

### Future

More computationally specialised capabilities requiring additional theory,
external packages, or substantial validation.


## 3. Capability Matrix

| Area | Capability | Level |
|---|---|---|
| Structures | Structure validation | Core |
| Structures | Symmetry analysis | Core |
| Structures | Supercell generation | Core |
| Structures | Crystalline defect enumeration | Core |
| Electronic | Total energies | Core |
| Electronic | Band structure | Core |
| Electronic | DOS / PDOS | Core |
| Electronic | VBM / CBM | Core |
| Electronic | Band gap | Core |
| Electronic | Charge / spin analysis | Core |
| Electronic | Localisation analysis / IPR | Core |
| Defects | Vacancies | Core |
| Defects | Interstitials | Core |
| Defects | Substitutions | Core |
| Defects | Antisites | Advanced |
| Defects | Charged defects | Core |
| Defects | Formation energies | Core |
| Defects | Charge-transition levels | Core |
| Defects | Chemical potentials | Core |
| Defects | Finite-size corrections | Core |
| Defects | Defect concentrations | Advanced |
| Defects | Self-consistent Fermi level | Advanced |
| Migration | NEB | Core |
| Migration | Migration barriers | Core |
| Migration | Attempt frequencies | Advanced |
| Migration | Hopping rates | Advanced |
| Migration | Diffusion coefficients | Advanced |
| Carriers | Electron / hole trapping | Core |
| Carriers | Polarons | Advanced |
| Carriers | Trapping energies | Advanced |
| Carriers | Polaron hopping | Advanced |
| Transport | Effective masses | Advanced |
| Transport | Band carrier mobility | Advanced |
| Transport | Hopping mobility | Advanced |
| Disorder | Melt-quench workflows | Advanced |
| Disorder | Amorphous ensembles | Advanced |
| Disorder | Coordination analysis | Core |
| Disorder | Structural topology | Advanced |
| Disorder | Defect-energy distributions | Advanced |
| Disorder | Structure-property correlations | Advanced |
| Interfaces | Surfaces | Advanced |
| Interfaces | Grain boundaries | Future |
| Interfaces | Heterointerfaces | Advanced |
| Interfaces | Interface trap states | Advanced |
| Interfaces | Defect segregation | Advanced |
| Interfaces | Band alignment | Advanced |
| Reliability | Trap kinetics | Advanced |
| Reliability | NBTI / PBTI descriptors | Future |
| Reliability | TDDB descriptors | Future |
| Reliability | ReRAM descriptors | Advanced |
| Reliability | Field-dependent processes | Future |
| Response | Vibrational properties | Advanced |
| Response | Dielectric response | Advanced |
| Optical | Optical transitions | Future |
| Optical | Absorption spectra | Future |
| Spectroscopy | Defect fingerprints | Future |
| ML | DFT training-data generation | Advanced |
| ML | MACE training / validation | Advanced |
| ML | ML molecular dynamics | Advanced |
| ML | DFT validation of ML structures | Advanced |


# 4. Electronic Structure


## 4.1 Ground-State Electronic Structure

### Device question

What are the fundamental electronic properties of the material?

### Required calculations

- geometry optimisation
- self-consistent electronic-structure calculation
- appropriate k-point sampling for crystalline materials
- hybrid-functional calculations where required

### Calculator outputs

- total energy
- eigenvalues
- occupations
- Fermi energy
- wavefunction information
- atomic structure
- cell information

### NSDW-derived quantities

- band gap
- VBM
- CBM
- Fermi-level position
- electronic convergence metrics

### Validation

- basis-set convergence
- cutoff convergence
- k-point convergence
- supercell convergence where appropriate
- functional validation

### Status

Core


## 4.2 Density of States and Projected Density of States

### Device question

Which atoms and orbitals contribute to the electronic states and defect levels?

### Required calculations

- converged electronic structure
- DOS / PDOS calculation

### Calculator outputs

- orbital energies
- projected states
- occupations

### NSDW-derived quantities

- DOS
- PDOS
- defect-state energies
- orbital character
- atomic contributions

### Validation

- energy-grid convergence
- smearing sensitivity
- consistency with band structure

### Status

Core


## 4.3 Band Structure

### Device question

How do electronic states disperse through the crystal?

### NSDW-derived quantities

- band gap
- direct / indirect gap
- band-edge locations
- band dispersion
- effective masses where appropriate

### Status

Core


## 4.4 Charge Localisation

### Device question

Does an injected electron or hole remain delocalised or form a localised
defect/polaron state?

### Required calculations

- neutral and charged calculations
- spin-polarised calculations where appropriate
- hybrid DFT where required

### Calculator outputs

- charge density
- spin density
- orbital populations
- wavefunctions / orbital coefficients

### NSDW-derived quantities

- localisation site
- IPR or equivalent localisation metric
- atomic charge redistribution
- spin localisation
- structural distortion

### Validation

- supercell convergence
- initial-state sensitivity
- localisation stability
- functional dependence

### Status

Core


# 5. Defect Physics


## 5.1 Defect Generation

Supported defect classes should ultimately include:

- vacancies
- interstitials
- substitutions
- antisites
- defect complexes

NSDW should identify symmetry-inequivalent sites wherever possible.

### Status

Vacancies, interstitials and substitutions: Core

Antisites and complex defects: Advanced


## 5.2 Charged Defects

### Device question

Which defect charge states are physically stable?

### Required calculations

For each defect:

    D^(q1)
    D^(q2)
    D^(q3)
    ...

### Calculator outputs

- total energy
- relaxed structure
- charge state
- electrostatic potential
- electronic structure

### NSDW-derived quantities

- charge-state stability
- structural relaxation
- localisation behaviour

### Status

Core


## 5.3 Defect Formation Energies

The standard defect formation energy is

    Delta H_f(D^q, E_F)
      = E(D^q)
      - E(bulk)
      - sum_i n_i mu_i
      + q(E_F + E_VBM)
      + E_corr

where:

- E(D^q) is the defective-cell energy
- E(bulk) is the pristine-cell energy
- n_i describes atoms added or removed
- mu_i is the corresponding chemical potential
- E_F is the Fermi level
- E_VBM is the valence-band reference
- E_corr contains finite-size / electrostatic corrections

### NSDW-derived quantities

- formation energy versus Fermi level
- stable charge states
- defect stability under different chemical environments

### Validation

- supercell convergence
- chemical-potential consistency
- potential alignment
- finite-size correction convergence

### Status

Core


## 5.4 Charge-Transition Levels

### Device question

At which Fermi-level position does a defect change charge state?

For charge states q and q':

    epsilon(q/q')

is the Fermi-level position where their formation energies are equal.

### Required calculations

- pristine bulk
- multiple defect charge states
- band-edge reference
- finite-size corrections

### NSDW-derived quantities

- thermodynamic CTLs
- stable charge states
- shallow versus deep defect behaviour
- formation-energy diagrams

### Validation

- charge-state convergence
- supercell convergence
- band-edge alignment
- correction-method validation

### Status

Core


## 5.5 Chemical Potentials

### Device question

How does the processing or growth environment affect defect stability?

Examples include:

- oxygen-rich
- oxygen-poor
- metal-rich
- metal-poor

### NSDW-derived quantities

- allowed chemical-potential region
- competing-phase constraints
- environment-dependent formation energies

### Status

Core


## 5.6 Defect and Carrier Concentrations

### Device question

Which defects dominate under a specified temperature and chemical environment?

NSDW should ultimately solve charge neutrality using contributions from:

- charged defects
- electrons
- holes
- dopants

### NSDW-derived quantities

- equilibrium defect concentrations
- electron concentration
- hole concentration
- self-consistent Fermi level
- compensation behaviour

### Status

Advanced


# 6. Carrier Trapping and Polarons


## 6.1 Carrier Trapping

### Device question

Can a defect or structural environment trap an electron or hole?

### Required calculations

- neutral reference
- electron-added state
- electron-removed state
- structural relaxation

### NSDW-derived quantities

- trapping energy
- localisation site
- structural distortion
- trap depth
- metastability

### Status

Core / Advanced


## 6.2 Polarons

### Device question

Does a carrier self-localise through coupling to the lattice?

### Required calculations

- localised initial configurations
- hybrid DFT and/or constrained DFT
- structural relaxation
- localisation analysis

### NSDW-derived quantities

- polaron formation energy
- localisation centre
- structural distortion
- electronic state
- stability

### Status

Advanced


## 6.3 Polaron Hopping

### Device question

How rapidly can a localised carrier move between sites?

### Required calculations

- initial localised state
- final localised state
- NEB and/or constrained DFT
- electronic coupling where required

### NSDW-derived quantities

- hopping barrier
- reorganisation energy
- hopping rate
- temperature dependence

### Status

Advanced


# 7. Atomic Migration and Diffusion


## 7.1 Migration Pathways

### Device question

How can a vacancy, interstitial, impurity, or ion move through the material?

### Required calculations

- initial state
- final state
- NEB pathway

### Calculator outputs

- image energies
- image forces
- structures

### NSDW-derived quantities

- migration barrier
- reaction coordinate
- transition-state structure

### Validation

- force convergence
- image convergence
- pathway convergence
- endpoint stability

### Status

Core


## 7.2 Hopping Rates

For an activated process:

    Gamma(T) = nu_0 exp[-E_m / (k_B T)]

where:

- Gamma is the hopping rate
- nu_0 is an attempt frequency
- E_m is the migration barrier

### NSDW-derived quantities

- temperature-dependent hopping rates
- competing pathway probabilities

### Status

Advanced


## 7.3 Diffusion

### Device question

How rapidly does an atomic species or defect migrate through the material?

Potential approaches include:

1. transition-state / hopping-network models
2. molecular-dynamics trajectories

For MD, NSDW may analyse the mean-square displacement:

    MSD(t) = <|r(t) - r(0)|^2>

and obtain diffusion coefficients in the appropriate diffusive regime.

### NSDW-derived quantities

- MSD
- diffusion coefficient D(T)
- activation energy
- anisotropic diffusion
- pathway distributions

### Status

Advanced


# 8. Carrier Transport


## 8.1 Effective Mass

### Device question

How strongly do the band edges favour electron or hole transport?

### Required calculations

- converged band structure
- dense sampling around band extrema

### NSDW-derived quantities

- electron effective mass
- hole effective mass
- anisotropic effective-mass tensor

### Status

Advanced


## 8.2 Band Transport

Potential workflow:

    electronic structure
          |
          v
    band interpolation
          |
          v
    scattering model
          |
          v
    Boltzmann transport
          |
          v
    mobility / conductivity

External validated transport packages should be integrated where appropriate.

### NSDW-derived quantities

- carrier mobility
- conductivity
- temperature dependence
- doping dependence

### Status

Advanced


## 8.3 Hopping Transport

For strongly localised or disordered systems:

    localised states
          |
          v
    hopping pathways
          |
          v
    hopping rates
          |
          v
    kinetic network
          |
          v
    mobility

This should remain distinct from band transport.

### Status

Advanced


# 9. Amorphous and Disordered Materials


## 9.1 Melt-Quench

Potential backends:

- CP2K AIMD
- MACE
- LAMMPS

Typical workflow:

    crystalline / random structure
              |
              v
             melt
              |
              v
          equilibrate
              |
              v
             quench
              |
              v
          relaxation
              |
              v
      amorphous structure

### Status

Advanced


## 9.2 Amorphous Ensembles

A single amorphous configuration should generally not be assumed to represent
the complete material.

NSDW should support ensembles generated across:

- random seeds
- cooling rates
- densities
- compositions
- temperatures

### NSDW-derived quantities

- coordination distributions
- bond-length distributions
- bond-angle distributions
- density
- local environments
- structural topology

### Status

Advanced


## 9.3 Defect Distributions

For disordered materials, defect properties should be represented as
distributions wherever appropriate rather than single values.

Examples:

    P(E_f)
    P(E_m)
    P[epsilon(q/q')]
    P(E_trap)

### Device relevance

These distributions can describe heterogeneous trap populations and
migration environments in amorphous electronic materials.

### Status

Advanced


## 9.4 Structure-Property Correlations

NSDW should allow correlations between local structural descriptors and
electronic/defect properties.

Potential descriptors include:

- coordination number
- bond length
- bond angle
- local density
- ring topology
- free volume
- nearest-neighbour chemistry

Potential targets include:

- trapping energy
- formation energy
- CTL
- migration barrier
- localisation strength

### Status

Advanced


# 10. Surfaces and Interfaces


## 10.1 Surfaces

Potential quantities:

- surface energy
- surface electronic states
- adsorption energy
- surface defects
- segregation energy

### Status

Advanced


## 10.2 Heterointerfaces

### Device question

How does an interface alter electronic and defect behaviour?

Potential quantities:

- interface formation energy
- band alignment
- interface dipole
- charge transfer
- interface trap states
- defect segregation

### Status

Advanced


## 10.3 Grain Boundaries

Potential quantities:

- grain-boundary energy
- defect segregation
- transport pathways
- electronic trap states

### Status

Future


# 11. Vibrational and Dielectric Properties


## 11.1 Vibrational Analysis

Potential quantities:

- normal modes
- vibrational frequencies
- local defect modes
- attempt frequencies
- thermodynamic corrections

### Device relevance

Vibrational information can support:

- defect identification
- migration-rate calculations
- carrier capture models
- experimental spectroscopy

### Status

Advanced


## 11.2 Dielectric Response

Potential quantities:

- dielectric response
- polarisation
- screening parameters

### Device relevance

Dielectric quantities are important for:

- electrostatic defect corrections
- gate dielectrics
- polarons
- charge trapping
- interfaces

### Status

Advanced


# 12. Optical and Spectroscopic Properties


## 12.1 Optical Properties

Potential calculations may include:

- TDDFT
- GW
- BSE

Potential quantities:

- excitation energies
- optical gap
- absorption spectrum
- defect-related optical transitions

### Status

Future


## 12.2 Spectroscopic Defect Identification

Goal:

    candidate defect
          |
          v
    predicted signature
          |
          v
    experimental comparison

Potential observables may include:

- vibrational signatures
- optical transitions
- X-ray absorption
- electronic states

### Status

Future


# 13. Device Reliability


## 13.1 Bias Temperature Instability

Relevant microscopic quantities may include:

- electron / hole trapping
- hydrogen migration
- proton migration
- defect creation
- defect passivation / depassivation
- activation barriers
- trap distributions

NSDW should provide microscopic quantities for reliability models rather than
attempting to replace device-scale reliability simulation.

### Status

Future


## 13.2 Time-Dependent Dielectric Breakdown

Relevant quantities may include:

- defect formation
- field-assisted migration
- defect accumulation
- structural transformation
- percolation precursors

### Status

Future


## 13.3 Resistive Switching / ReRAM

Relevant quantities may include:

- oxygen-vacancy formation
- oxygen interstitial formation
- ionic migration
- charge-state-dependent migration
- trap formation
- structural transformation

### Status

Advanced


# 14. Machine-Learned Interatomic Potentials


## 14.1 Training-Data Generation

DFT calculations should be capable of generating datasets containing:

- energies
- forces
- stresses
- structures
- metadata

### Status

Advanced


## 14.2 MACE Training and Validation

Potential workflow:

    DFT
     |
     v
    dataset
     |
     v
    MACE
     |
     v
    validation
     |
     v
    production MLMD

Validation should include:

- energy errors
- force errors
- structural properties
- unseen configurations
- relevant defect environments

### Status

Advanced


## 14.3 ML Molecular Dynamics

Applications may include:

- melt-quench
- thermal sampling
- diffusion
- large amorphous cells
- long-timescale trajectory generation

Selected configurations should be returned to DFT for validation.

### Status

Advanced


# 15. Electric-Field-Dependent Processes

Future workflows should allow investigation of how electric fields alter:

- defect stability
- migration barriers
- ionic transport
- carrier localisation
- structural transformations
- polarisation

Zero-field calculations must not automatically be interpreted as equivalent
to device operating conditions.

### Status

Future


# 16. Standard NSDW Scientific Result

Scientific analysis should not depend directly on raw calculator output.

The intended architecture is:

    CP2K ----\
              \
    VASP ------> NSDW Result --> Analysis --> Device-relevant quantity
              /
    MACE -----/

A standard result may contain categories such as:

    result.structure
    result.energy
    result.electronic
    result.defect
    result.migration
    result.transport
    result.provenance

Example conceptual fields:

    energy.total
    electronic.band_gap
    electronic.vbm
    electronic.cbm
    electronic.fermi_level

    defect.type
    defect.site
    defect.charge
    defect.formation_energy
    defect.transition_levels

    migration.initial_state
    migration.final_state
    migration.barrier

    provenance.calculator
    provenance.calculator_version
    provenance.input_hash
    provenance.structure_hash
    provenance.workflow_version

The exact schema will be defined separately.


# 17. Scientific Validation Principles

A workflow completing successfully does not demonstrate scientific validity.

NSDW should distinguish:

    execution success

from:

    numerical convergence

from:

    physical validation

Scientific workflows should therefore include validation appropriate to the
property being calculated.

Examples include:

- basis-set convergence
- cutoff convergence
- k-point convergence
- supercell convergence
- finite-size convergence
- functional validation
- charge localisation validation
- NEB convergence
- ML potential validation
- statistical convergence for amorphous ensembles

Reference systems should be maintained as regression tests where practical.


# 18. Reproducibility

Every scientific result should be traceable to:

- input structure
- input parameters
- calculator
- calculator version
- pseudopotentials
- basis sets
- workflow version
- software dependencies
- computational environment
- execution platform

NSDW should retain sufficient provenance to reproduce a calculation on a
different supported computational resource wherever practical.


# 19. Scientific Development Principle

NSDW should prioritise:

    physical question
          |
          v
    required quantity
          |
          v
    required calculation
          |
          v
    calculator execution
          |
          v
    validated analysis
          |
          v
    reproducible result

rather than:

    calculator input
          |
          v
    run calculation
          |
          v
    output file

The primary scientific objective is therefore to connect atomistic simulation
to reproducible semiconductor materials and device-relevant information.
