from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from nsdw.defects.exporter import (
    DefectExportError,
    export_vacancy_dataset,
)
from nsdw.output.builders import (
    build_structure_symmetry_output,
    build_structure_validation_output,
    build_supercell_search_output,
)
from nsdw.output.renderers import (
    render_structure_symmetry,
    render_structure_validation,
    render_supercell_search,
)
from nsdw.structures.defects import (
    DefectGenerationError,
    generate_symmetry_inequivalent_vacancies,
)
from nsdw.structures.parser import (
    StructureParseError,
    load_structure,
)
from nsdw.structures.supercell import (
    SupercellSearchError,
    search_supercells,
)
from nsdw.structures.symmetry import (
    SymmetryAnalysisError,
    analyse_symmetry,
)
from nsdw.structures.validator import (
    DEFAULT_MIN_DISTANCE,
    summarise_structure,
    validate_structure,
)
from nsdw.workflows import (
    SinglePointWorkflowError,
    run_cp2k_single_point,
)

app = typer.Typer(
    name="nsdw",
    help="Nanosystems Defect Workflows",
    no_args_is_help=True,
)

structure_app = typer.Typer(
    help="Structure parsing, validation, symmetry, and supercell commands.",
    no_args_is_help=True,
)

defects_app = typer.Typer(
    help="Generate and manage semiconductor defect structures.",
    no_args_is_help=True,
)

workflow_app = typer.Typer(
    help="Run semiconductor modelling workflows.",
    no_args_is_help=True,
)

app.add_typer(
    structure_app,
    name="structure",
)

app.add_typer(
    defects_app,
    name="defects",
)

app.add_typer(
    workflow_app,
    name="workflow",
)

console = Console()

__version__ = "0.1.0"


def version_callback(value: bool) -> None:
    """Print the NSDW version and exit."""

    if value:
        console.print(f"NSDW {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show the NSDW version and exit.",
    )
) -> None:
    """
    Nanosystems Defect Workflows.

    Automated workflows for semiconductor structure,
    convergence, and defect modelling.
    """
    pass


# ============================================================================
# Structure validation
# ============================================================================


@structure_app.command("validate")
def structure_validate(
    file: Path = typer.Argument(
        ...,
        help="Path to a CIF or XYZ structure file.",
    ),
    output_format: Literal["text", "json"] = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the result to a file.",
    ),
    min_distance: float = typer.Option(
        DEFAULT_MIN_DISTANCE,
        "--min-distance",
        help=(
            "Minimum allowed periodic interatomic "
            "distance in angstrom."
        ),
        min=0.0,
    ),
) -> None:
    """
    Parse and validate a periodic atomic structure.
    """

    try:
        structure, parser_warnings = load_structure(
            file
        )

        summary = summarise_structure(
            structure
        )

        validation = validate_structure(
            structure,
            min_distance=min_distance,
        )

        result = build_structure_validation_output(
            source_path=file,
            summary=summary,
            validation=validation,
            parser_warnings=parser_warnings,
            nsdw_version=__version__,
            minimum_distance_threshold=(
                min_distance
            ),
        )

    except (
        FileNotFoundError,
        StructureParseError,
    ) as exc:
        console.print(
            f"[bold red]ERROR:[/bold red] {exc}"
        )
        raise typer.Exit(code=1)

    if output_format == "json":
        rendered = result.model_dump_json(
            indent=2
        )

        if output is None:
            typer.echo(rendered)

        else:
            output = (
                output
                .expanduser()
                .resolve()
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output.write_text(
                rendered + "\n",
                encoding="utf-8",
            )

            console.print(
                f"[green]✓[/green] "
                f"JSON result written to: "
                f"{output}"
            )

    else:
        if output is not None:
            console.print(
                "[bold red]ERROR:[/bold red] "
                "--output currently requires "
                "--format json."
            )
            raise typer.Exit(code=2)

        render_structure_validation(
            result=result,
            console=console,
        )

    if not result.validation.valid:
        raise typer.Exit(code=1)


# ============================================================================
# Structure symmetry
# ============================================================================


@structure_app.command("symmetry")
def structure_symmetry(
    file: Path = typer.Argument(
        ...,
        help="Path to a periodic structure file.",
    ),
    element: str | None = typer.Option(
        None,
        "--element",
        "-e",
        help=(
            "Show symmetry-inequivalent sites "
            "for a selected element."
        ),
    ),
    symprec: float = typer.Option(
        1e-3,
        "--symprec",
        help="Symmetry tolerance in angstrom.",
        min=0.0,
    ),
    angle_tolerance: float = typer.Option(
        5.0,
        "--angle-tolerance",
        help=(
            "Angular symmetry tolerance "
            "in degrees."
        ),
    ),
    output_format: Literal["text", "json"] = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the result to a file.",
    ),
) -> None:
    """
    Analyse crystallographic symmetry.
    """

    try:
        structure, parser_warnings = load_structure(
            file
        )

        symmetry = analyse_symmetry(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        result = build_structure_symmetry_output(
            source_path=file,
            symmetry=symmetry,
            parser_warnings=parser_warnings,
            nsdw_version=__version__,
            selected_element=element,
        )

    except (
        FileNotFoundError,
        StructureParseError,
        SymmetryAnalysisError,
    ) as exc:
        console.print(
            f"[bold red]ERROR:[/bold red] {exc}"
        )
        raise typer.Exit(code=1)

    if output_format == "json":
        rendered = result.model_dump_json(
            indent=2
        )

        if output is None:
            typer.echo(rendered)

        else:
            output = (
                output
                .expanduser()
                .resolve()
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output.write_text(
                rendered + "\n",
                encoding="utf-8",
            )

            console.print(
                f"[green]✓[/green] "
                f"JSON result written to: "
                f"{output}"
            )

    else:
        if output is not None:
            console.print(
                "[bold red]ERROR:[/bold red] "
                "--output currently requires "
                "--format json."
            )
            raise typer.Exit(code=2)

        render_structure_symmetry(
            result=result,
            console=console,
        )


# ============================================================================
# Supercell search
# ============================================================================


@structure_app.command("supercell")
def structure_supercell(
    file: Path = typer.Argument(
        ...,
        help="Path to a periodic structure file.",
    ),
    min_atoms: int = typer.Option(
        50,
        "--min-atoms",
        help="Minimum allowed supercell atom count.",
        min=1,
    ),
    max_atoms: int = typer.Option(
        250,
        "--max-atoms",
        help="Maximum allowed supercell atom count.",
        min=1,
    ),
    min_image_distance: float = typer.Option(
        10.0,
        "--min-image-distance",
        help=(
            "Minimum periodic image-distance target "
            "in angstrom."
        ),
        min=0.0,
    ),
    max_scale: int = typer.Option(
        4,
        "--max-scale",
        help=(
            "Maximum diagonal scaling factor searched "
            "in each lattice direction."
        ),
        min=1,
    ),
    image_range: int = typer.Option(
        2,
        "--image-range",
        help=(
            "Integer lattice-image range used when "
            "determining the shortest translation."
        ),
        min=1,
    ),
    top: int = typer.Option(
        10,
        "--top",
        help=(
            "Maximum number of acceptable candidates "
            "shown in text output."
        ),
        min=1,
    ),
    output_format: Literal["text", "json"] = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the result to a file.",
    ),
) -> None:
    """
    Search and evaluate candidate diagonal supercells.
    """

    try:
        structure, parser_warnings = load_structure(
            file
        )

        search = search_supercells(
            structure,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_image_distance=min_image_distance,
            max_scale=max_scale,
            image_range=image_range,
        )

        result = build_supercell_search_output(
            source_path=file,
            search=search,
            parser_warnings=parser_warnings,
            nsdw_version=__version__,
        )

    except (
        FileNotFoundError,
        StructureParseError,
        SupercellSearchError,
    ) as exc:
        console.print(
            f"[bold red]ERROR:[/bold red] {exc}"
        )
        raise typer.Exit(code=1)

    if output_format == "json":
        rendered = result.model_dump_json(
            indent=2
        )

        if output is None:
            typer.echo(rendered)

        else:
            output = (
                output
                .expanduser()
                .resolve()
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output.write_text(
                rendered + "\n",
                encoding="utf-8",
            )

            console.print(
                f"[green]✓[/green] "
                f"JSON result written to: "
                f"{output}"
            )

    else:
        if output is not None:
            console.print(
                "[bold red]ERROR:[/bold red] "
                "--output currently requires "
                "--format json."
            )
            raise typer.Exit(code=2)

        render_supercell_search(
            result=result,
            console=console,
            top=top,
        )


# ============================================================================
# Defect generation
# ============================================================================


@defects_app.command("generate-vacancies")
def generate_vacancies_command(
    file: Path = typer.Argument(
        ...,
        help=(
            "Path to the ordered periodic parent "
            "structure."
        ),
    ),
    species: str = typer.Option(
        "O",
        "--species",
        "-s",
        help="Element to remove.",
    ),
    scale: tuple[int, int, int] = typer.Option(
        (1, 1, 1),
        "--scale",
        help=(
            "Diagonal supercell scaling as three "
            "integers, for example: --scale 4 4 1"
        ),
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help=(
            "Output directory for the generated "
            "defect dataset."
        ),
    ),
    symprec: float = typer.Option(
        1e-3,
        "--symprec",
        help="Symmetry tolerance in angstrom.",
        min=0.0,
    ),
    angle_tolerance: float = typer.Option(
        5.0,
        "--angle-tolerance",
        help=(
            "Symmetry angle tolerance in degrees."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help=(
            "Replace an existing non-empty "
            "output dataset."
        ),
    ),
) -> None:
    """
    Generate symmetry-inequivalent neutral vacancy structures.
    """

    try:
        structure, parser_warnings = (
            load_structure(
                file
            )
        )

        generation = (
            generate_symmetry_inequivalent_vacancies(
                structure,
                species=species,
                scaling=scale,
                charge_state=0,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
        )

        export = export_vacancy_dataset(
            generation=generation,
            source_path=file,
            parser_warnings=parser_warnings,
            output_directory=output,
            nsdw_version=__version__,
            overwrite=overwrite,
        )

    except (
        FileNotFoundError,
        StructureParseError,
        DefectGenerationError,
        DefectExportError,
    ) as exc:
        console.print(
            f"[bold red]ERROR:[/bold red] {exc}"
        )
        raise typer.Exit(code=1)

    console.print(
        "\n[bold green]"
        "Vacancy dataset generated"
        "[/bold green]\n"
    )

    console.print(
        f"Species:                "
        f"{generation.species}"
    )

    console.print(
        f"Charge state:           "
        f"{generation.charge_state}"
    )

    scaling_text = " × ".join(
        str(value)
        for value
        in generation.supercell_scaling
    )

    console.print(
        f"Supercell:              "
        f"{scaling_text}"
    )

    console.print(
        f"Pristine atoms:         "
        f"{generation.pristine_supercell_num_atoms}"
    )

    console.print(
        f"Inequivalent sites:     "
        f"{generation.num_inequivalent_sites}"
    )

    console.print(
        f"Defects generated:      "
        f"{len(generation.vacancies)}"
    )

    console.print(
        f"Output directory:       "
        f"{export.output_directory}"
    )

    console.print(
        f"Manifest:               "
        f"{export.manifest_file}"
    )

    if parser_warnings:
        console.print(
            "\n[bold yellow]"
            "Parser warnings"
            "[/bold yellow]"
        )

        for warning in parser_warnings:
            console.print(
                f"[yellow]⚠[/yellow] {warning}"
            )

    console.print(
        "\n[bold]Generated defects[/bold]"
    )

    for vacancy in generation.vacancies:
        console.print(
            f"  {vacancy.defect_id} "
            f"(multiplicity "
            f"{vacancy.primitive_multiplicity})"
        )

# ============================================================================
# Single-point workflow
# ============================================================================


@workflow_app.command("single-point")
def workflow_single_point(
    calculator: Literal["cp2k"] = typer.Option(
        "cp2k",
        "--calculator",
        "-c",
        help="Calculator backend.",
    ),
    workdir: Path = typer.Option(
        Path("."),
        "--workdir",
        "-w",
        help="Calculation working directory.",
    ),
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Calculator input file.",
    ),
    output_file: Path = typer.Option(
        Path("calculation.out"),
        "--output",
        "-o",
        help="Calculator output file.",
    ),
    result_file: Path = typer.Option(
        Path("result.json"),
        "--result",
        "-r",
        help="NSDW result JSON file.",
    ),
    calculation_id: str | None = typer.Option(
        None,
        "--id",
        help="Calculation identifier.",
    ),
    executable: str = typer.Option(
        "cp2k.psmp",
        "--executable",
        help="CP2K executable.",
    ),
) -> None:
    """
    Run a local single-point calculation and build an NSDW result.
    """

    workdir = (
        workdir
        .expanduser()
        .resolve()
    )

    if calculation_id is None:
        calculation_id = input_file.stem

    try:
        if calculator == "cp2k":
            result = run_cp2k_single_point(
                calculation_id=calculation_id,
                working_directory=workdir,
                input_file=input_file,
                output_file=output_file,
                executable=executable,
                result_file=result_file,
            )

    except (
        FileNotFoundError,
        SinglePointWorkflowError,
        RuntimeError,
    ) as exc:
        console.print(
            f"[bold red]ERROR:[/bold red] {exc}"
        )
        raise typer.Exit(code=1)

    console.print(
        "\n[bold green]"
        "Single-point workflow completed"
        "[/bold green]\n"
    )

    console.print(
        f"Calculation ID:  "
        f"{result.calculation.id}"
    )

    console.print(
        f"Calculator:      "
        f"{result.calculation.backend.value}"
    )

    console.print(
        f"Status:          "
        f"{result.calculation.status.value}"
    )

    if (
        result.energy is not None
        and result.energy.total is not None
    ):
        console.print(
            f"Total energy:    "
            f"{result.energy.total.value:.8f} "
            f"{result.energy.total.unit}"
        )

    destination = (
        result_file
        if result_file.is_absolute()
        else workdir / result_file
    )

    console.print(
        f"Result:          "
        f"{destination.resolve()}"
    )

if __name__ == "__main__":
    app()