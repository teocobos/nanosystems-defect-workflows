from rich.console import Console
from rich.table import Table

from nsdw.output.models import StructureValidationOutput


def render_structure_validation(
    result: StructureValidationOutput,
    console: Console,
) -> None:
    """
    Render a structure validation result for human-readable terminal output.
    """

    table = Table(title="NSDW Structure Validation")

    table.add_column("Property")
    table.add_column("Value")

    structure = result.structure
    lattice = structure.lattice

    table.add_row("Formula", structure.formula)
    table.add_row("Reduced formula", structure.reduced_formula)
    table.add_row("Atoms", str(structure.num_sites))
    table.add_row(
        "Density",
        f"{structure.density_g_cm3:.4f} g/cm³",
    )

    table.add_section()

    table.add_row("a", f"{lattice.a_angstrom:.6f} Å")
    table.add_row("b", f"{lattice.b_angstrom:.6f} Å")
    table.add_row("c", f"{lattice.c_angstrom:.6f} Å")

    table.add_row("alpha", f"{lattice.alpha_deg:.4f}°")
    table.add_row("beta", f"{lattice.beta_deg:.4f}°")
    table.add_row("gamma", f"{lattice.gamma_deg:.4f}°")

    table.add_row(
        "Volume",
        f"{lattice.volume_angstrom3:.4f} Å³",
    )

    console.print(table)

    if result.parser.warnings:
        console.print("\n[bold yellow]Parser warnings[/bold yellow]")

        for warning in result.parser.warnings:
            console.print(f"[yellow]⚠[/yellow] {warning}")

    console.print("\n[bold]Validation checks[/bold]")

    for check in result.validation.checks:
        symbol = (
            "[green]✓[/green]"
            if check.passed
            else "[red]✗[/red]"
        )

        details = ""

        if check.value:
            details += f" — {check.value}"

        if check.message:
            details += f" ({check.message})"

        console.print(f"{symbol} {check.name}{details}")

    if result.validation.warnings:
        console.print(
            "\n[bold yellow]Validation warnings[/bold yellow]"
        )

        for warning in result.validation.warnings:
            console.print(f"[yellow]⚠[/yellow] {warning}")

    if result.validation.errors:
        console.print("\n[bold red]Errors[/bold red]")

        for error in result.validation.errors:
            console.print(f"[red]✗[/red] {error}")

    if result.validation.valid:
        console.print(
            "\n[bold green]RESULT: STRUCTURE VALID[/bold green]\n"
        )
    else:
        console.print(
            "\n[bold red]RESULT: STRUCTURE INVALID[/bold red]\n"
        )
