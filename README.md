# Nanosystems Defect Workflows (NSDW)

NSDW is a Python command-line toolkit for reproducible semiconductor
structure analysis, supercell selection, defect generation, and
calculator workflows.

The project is developed by Teofilo Cobos Freire as part of
Nanosystems Advisory Ltd.

**Current version:** 0.1.0
**Python:** 3.11 or 3.12

## Current capabilities

- Parse and validate periodic CIF structures.
- Parse standard XYZ structures with explicitly supplied lattice parameters.
- Analyse crystallographic symmetry and inequivalent atomic sites.
- Search for suitable diagonal supercells.
- Generate symmetry-inequivalent neutral vacancy structures.
- Execute existing CP2K single-point inputs locally or through the
  ARCHER2 single-point workflow.

CP2K input generation from structures and YAML configuration is planned
but is **not yet implemented**.

## Installation

Clone the repository and install NSDW in a Python 3.11 or 3.12 environment:

```bash
git clone https://github.com/teocobos/nanosystems-defect-workflows.git
cd nanosystems-defect-workflows
python -m pip install -e ".[dev]"
```

Check the installation:

```bash
nsdw --version
nsdw --help
```

## Structure input: CIF and XYZ

CIF files contain their own lattice information. No additional cell
parameters are needed.

Standard XYZ files contain atomic coordinates but do not necessarily
contain a periodic cell. For XYZ input, supply all six lattice parameters:

| Option | Meaning | Unit |
|---|---|---|
| `--a` | Lattice length a | Å |
| `--b` | Lattice length b | Å |
| `--c` | Lattice length c | Å |
| `--alpha` | Angle between b and c | degrees |
| `--beta` | Angle between a and c | degrees |
| `--gamma` | Angle between a and b | degrees |

All six options must be supplied together. NSDW does not infer or invent
the missing cell of a standard XYZ file.

The examples below use `structure.xyz` as a placeholder for your own
XYZ file and a 10 Å cubic cell as an illustrative lattice. Replace
these values with the actual cell of your structure.

### Validate a structure

CIF:

```bash
nsdw structure validate \
    examples/igzo/igzo_crystal_ingazno4_cod1521670.cif
```

XYZ:

```bash
nsdw structure validate structure.xyz \
    --a 10 --b 10 --c 10 \
    --alpha 90 --beta 90 --gamma 90
```

To obtain JSON output, append `--format json`.

### Analyse symmetry

```bash
nsdw structure symmetry structure.xyz \
    --a 10 --b 10 --c 10 \
    --alpha 90 --beta 90 --gamma 90 \
    --element O \
    --format json
```

For a CIF file, omit the six lattice options.

### Search for supercells

```bash
nsdw structure supercell structure.xyz \
    --a 10 --b 10 --c 10 \
    --alpha 90 --beta 90 --gamma 90 \
    --min-atoms 50 \
    --max-atoms 250 \
    --min-image-distance 10 \
    --format json
```

The current search evaluates diagonal supercell scaling.

### Generate neutral vacancies

```bash
nsdw defects generate-vacancies structure.xyz \
    --a 10 --b 10 --c 10 \
    --alpha 90 --beta 90 --gamma 90 \
    --species O \
    --scale 2 2 2 \
    --output workspace/example-vacancies
```

This generates symmetry-inequivalent neutral vacancies and exports
the resulting dataset. Use `--overwrite` only when you intend to replace
an existing non-empty output dataset.

For a CIF input, omit the six lattice options.

## CP2K single-point workflows

NSDW can execute an **existing CP2K input file**. These commands do
not generate the CP2K input itself.

For local execution, inspect the available arguments with:

```bash
nsdw workflow single-point --help
```

For ARCHER2 execution:

```bash
nsdw workflow single-point-archer2 --help
```

The ARCHER2 workflow requires appropriate HPC access, an account,
a supported CP2K module, and a valid calculation input.

## Testing

Run the complete regression suite:

```bash
pytest -q
```

## Documentation

Additional technical information is available in:

- `docs/architecture.md`
- `docs/methodology.md`
- `docs/result-schema.md`
- `docs/scientific-capabilities.md`
- `docs/roadmap.md`

## Development roadmap

The next milestone is reproducible CP2K single-point input generation
from CIF or XYZ structures, with YAML configuration, CLI overrides,
validated basis and pseudopotential assignments, and a calculation
manifest.

Subsequent development will extend convergence studies, defect
calculations, and HPC workflow automation.
