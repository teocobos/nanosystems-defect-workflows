from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nsdw.structures.parser import StructureParseError, load_structure
from nsdw.structures.validator import summarise_structure, validate_structure


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


@structure_app.command("validate")
def structure_validate(
    file: Path = typer.Argument(
        ...,
        help="Path to a CIF or XYZ structure file.",
    )
) -> None:
    """
    Parse and validate a periodic atomic structure.
    """

    # ------------------------------------------------------------------
    # Parse structure
    # ------------------------------------------------------------------

    try:
        structure = load_structure(file)
        summary = summarise_structure(structure)
        validation = validate_structure(structure)

    except (FileNotFoundError, StructureParseError) as exc:
        console.print(f"[bold red]ERROR:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Structure summary
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------------

    console.print("\n[bold]Validation checks[/bold]")

    for check in validation.checks:
        if check.passed:
            symbol = "[green]✓[/green]"
        else:
            symbol = "[red]✗[/red]"

        details = ""

        if check.value:
            details += f" — {check.value}"

        if check.message:
            details += f" ({check.message})"

        console.print(
            f"{symbol} {check.name}{details}"
        )

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    if validation.warnings:
        console.print("\n[bold yellow]Warnings[/bold yellow]")

        for warning in validation.warnings:
            console.print(
                f"[yellow]⚠[/yellow] {warning}"
            )

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    if validation.errors:
        console.print("\n[bold red]Errors[/bold red]")

        for error in validation.errors:
            console.print(
                f"[red]✗[/red] {error}"
            )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    if validation.valid:
        console.print(
            "\n[bold green]RESULT: STRUCTURE VALID[/bold green]"
        )

    else:
        console.print(
            "\n[bold red]RESULT: STRUCTURE INVALID[/bold red]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()