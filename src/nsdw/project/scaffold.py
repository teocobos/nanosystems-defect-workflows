"""Create standard NSDW material-project directory structures."""

from pathlib import Path

import yaml

from nsdw.project.models import ProjectConfig


class ProjectScaffoldError(RuntimeError):
    """Raised when an NSDW project cannot be created safely."""


CORE_DIRECTORIES = (
    "docs",
    "structures/raw",
    "structures/validated",
    "structures/supercells",
    "structures/defects",
    "calculations",
    "working/calculations",
    "workflows",
    "reports",
)

TRACKED_DIRECTORIES = (
    "docs",
    "structures/raw",
    "structures/validated",
    "structures/supercells",
    "structures/defects",
    "calculations",
    "workflows",
    "reports",
)

def create_project(
    root: Path,
    config: ProjectConfig,
) -> Path:
    """Create a new NSDW material project."""

    root = root.expanduser().resolve()

    if root.exists():
        if not root.is_dir():
            raise ProjectScaffoldError(
                f"Project path exists and is not a directory: {root}"
            )

        if any(root.iterdir()):
            raise ProjectScaffoldError(
                f"Project directory is not empty: {root}"
            )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create the standard NSDW project directory structure.
    for relative_directory in CORE_DIRECTORIES:
        (root / relative_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    # Create directories for the selected modelling components.
    for component in config.components:
        (
            root
            / "calculations"
            / component
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            root
            / "working"
            / "calculations"
            / component
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    # Preserve shareable empty directories when the project is
    # committed to Git.
    for relative_directory in TRACKED_DIRECTORIES:
        (
            root
            / relative_directory
            / ".gitkeep"
        ).touch()

    # Preserve selected component directories under calculations/.
    # The corresponding working/ directories remain local and ignored.
    for component in config.components:
        (
            root
            / "calculations"
            / component
            / ".gitkeep"
        ).touch()

    _write_project_config(
        root=root,
        config=config,
    )

    _write_readme(
        root=root,
        config=config,
    )

    _write_gitignore(
        root=root,
    )

    return root


def _write_project_config(
    root: Path,
    config: ProjectConfig,
) -> None:
    """Write project.yaml using stable field ordering."""

    destination = root / "project.yaml"

    data = config.model_dump()

    destination.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_readme(
    root: Path,
    config: ProjectConfig,
) -> None:
    """Write the initial project README."""

    if config.components:
        components = "\n".join(
            f"- {component}"
            for component in config.components
        )
    else:
        components = "- None selected"

    content = f"""# {config.name}

NSDW material-modelling project for **{config.material}**.

## Enabled components

{components}

## Project structure

- `structures/raw/` — original source structures.
- `structures/validated/` — structures checked with NSDW.
- `structures/supercells/` — generated or selected supercells.
- `structures/defects/` — generated defect structures.
- `working/calculations/` — active calculation working directories.
- `calculations/` — selected reproducible calculation records and results.
- `workflows/` — workflow configuration and supporting files.
- `reports/` — generated summaries, figures, and reports.
- `docs/` — project-specific documentation.

## Working with calculations

Active calculations should normally be performed under:

`working/calculations/`

Large temporary files, restart files, wavefunctions, and other
machine-specific calculation artefacts should not normally be committed
to Git.

Selected inputs, resolved configurations, manifests, and extracted
results intended for collaboration and reproducibility can be organised
under:

`calculations/`

## NSDW

This project was initialised with NSDW {config.nsdw_version}.
"""

    (root / "README.md").write_text(
        content,
        encoding="utf-8",
    )


def _write_gitignore(
    root: Path,
) -> None:
    """Write conservative Git defaults for research projects."""

    content = """# NSDW active calculation workspace
working/

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/

# Editors and operating systems
.vscode/
.idea/
.DS_Store
Thumbs.db

# CP2K restart and wavefunction files
*-RESTART*
*.wfn
*.wfn.bak*
*.kp

# VASP large/generated files
WAVECAR
CHGCAR
CHG
vasprun.xml

# Scheduler output
slurm-*.out
slurm-*.err

# Temporary files
*.tmp
*.bak
*~
"""

    (root / ".gitignore").write_text(
        content,
        encoding="utf-8",
    )