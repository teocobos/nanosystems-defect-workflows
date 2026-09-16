from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nsdw.structures.parser import StructureParseError, load_structure
from nsdw.structures.validator import summarise_structure


app = typer.Typer(
    name="nsdw",
    help="Nanosystems Defect Workflows",
    no_args_is_help=True,
)

structure_app = typer.Typer(
    help="Structure parsing and validation commands.",
    no_args_is_help=True,
)

app.add_typer(structure_app, name="structure")

console = Console()

__version__ = "0.1.0"


def version_callback(value: bool) -> None:
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
    """Nanosystems Defect Workflows."""
    pass


@structure_app.command("validate")
def validate_structure(
    file: Path = typer.Argument(
        ...,
        help="Path to a CIF or XYZ structure file.",
    )
) -> None:
    """
    Parse and validate a periodic atomic structure.
    """

    try:
        structure = load_structure(file)
        summary = summarise_structure(structure)

    except (FileNotFoundError, StructureParseError) as exc:
        console.print(f"[bold red]ERROR:[/bold red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title="NSDW Structure Validation")

    table.add_column("Property")
    table.add_column("Value")

    table.add_row("Formula", summary.formula)
    table.add_row("Reduced formula", summary.reduced_formula)
    table.add_row("Atoms", str(summary.num_sites))
    table.add_row("Density", f"{summary.density:.4f} g/cm³")

    table.add_section()

    table.add_row("a", f"{summary.lattice.a:.6f} Å")
    table.add_row("b", f"{summary.lattice.b:.6f} Å")
    table.add_row("c", f"{summary.lattice.c:.6f} Å")

    table.add_row("alpha", f"{summary.lattice.alpha:.4f}°")
    table.add_row("beta", f"{summary.lattice.beta:.4f}°")
    table.add_row("gamma", f"{summary.lattice.gamma:.4f}°")

    table.add_row("Volume", f"{summary.lattice.volume:.4f} Å³")

    console.print(table)
    console.print("\n[bold green]✓ Structure parsed successfully[/bold green]")


if __name__ == "__main__":
    app()
