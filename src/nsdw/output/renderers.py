from rich.console import Console
from rich.table import Table

from nsdw.output.models import (
    StructureSymmetryOutput,
    StructureValidationOutput,
)


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
def render_structure_symmetry(
    result: StructureSymmetryOutput,
    console: Console,
) -> None:
    """
    Render symmetry analysis for human-readable terminal output.
    """

    symmetry = result.symmetry
    sites = result.site_analysis

    console.print(
        "\n[bold]NSDW Symmetry Analysis[/bold]\n"
    )

    console.print(
        f"Space group:          "
        f"[bold]{symmetry.space_group_symbol}[/bold]"
    )

    console.print(
        f"Space-group number:   "
        f"{symmetry.space_group_number}"
    )

    console.print(
        f"Crystal system:       "
        f"{symmetry.crystal_system}"
    )

    console.print(
        f"Point group:          "
        f"{symmetry.point_group}"
    )

    console.print(
        f"Symmetry operations:  "
        f"{symmetry.num_symmetry_operations}"
    )

    console.print(
        f"symprec:              "
        f"{symmetry.symprec_angstrom} Å"
    )

    console.print(
        f"angle tolerance:      "
        f"{symmetry.angle_tolerance_deg}°"
    )

    if result.parser.warnings:
        console.print(
            "\n[bold yellow]Parser warnings[/bold yellow]"
        )

        for warning in result.parser.warnings:
            console.print(
                f"[yellow]⚠[/yellow] {warning}"
            )

    if sites.selected_element:
        title = (
            f"Symmetry-inequivalent "
            f"{sites.selected_element} sites"
        )
    else:
        title = "Symmetry-inequivalent sites"

    table = Table(
        title=title,
        show_lines=False,
    )

    table.add_column("ID")
    table.add_column("Element")
    table.add_column("Index", justify="right")
    table.add_column("Atom #", justify="right")
    table.add_column(
        "Multiplicity",
        justify="right",
    )
    table.add_column("Equivalent indices")
    table.add_column("Equivalent atom #")

    for site in sites.inequivalent_sites:
        table.add_row(
            site.site_id,
            site.element,
            str(site.representative_index),
            str(site.representative_atom_number),
            str(site.multiplicity),
            str(site.equivalent_indices),
            str(site.equivalent_atom_numbers),
        )

    console.print()
    console.print(table)

    if sites.selected_element:
        console.print(
            f"\n[bold]"
            f"{sites.num_inequivalent_sites}"
            f" symmetry-inequivalent "
            f"{sites.selected_element} sites"
            f" representing "
            f"{sites.total_selected_sites}"
            f" total {sites.selected_element} sites"
            f"[/bold]\n"
        )
    else:
        console.print(
            f"\n[bold]"
            f"{sites.num_inequivalent_sites}"
            f" symmetry-inequivalent site classes"
            f" representing "
            f"{sites.total_selected_sites}"
            f" total sites"
            f"[/bold]\n"
        )