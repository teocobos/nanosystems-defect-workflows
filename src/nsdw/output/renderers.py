from rich.console import Console
from rich.table import Table

from nsdw.output.models import (
    StructureSymmetryOutput,
    StructureValidationOutput,
    SupercellSearchOutput,
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
def _format_scaling(
    scaling: list[int],
) -> str:
    return " × ".join(
        str(value)
        for value in scaling
    )


def render_supercell_search(
    result: SupercellSearchOutput,
    console: Console,
    top: int = 10,
) -> None:
    """
    Render an NSDW supercell search for terminal use.
    """

    constraints = result.constraints

    console.print(
        "\n[bold]NSDW Supercell Search[/bold]\n"
    )

    console.print(
        f"Primitive atoms:       "
        f"{result.primitive_num_atoms}"
    )

    console.print(
        f"Search method:         "
        f"{result.search_method}"
    )

    console.print(
        f"Minimum atoms:         "
        f"{constraints.min_atoms}"
    )

    console.print(
        f"Maximum atoms:         "
        f"{constraints.max_atoms}"
    )

    console.print(
        f"Minimum image target:  "
        f"{constraints.min_image_distance_angstrom:.3f} Å"
    )

    console.print(
        f"Maximum scale:         "
        f"{constraints.max_scale}"
    )

    console.print(
        f"Candidates evaluated:  "
        f"{result.num_candidates_evaluated}"
    )

    console.print(
        f"Acceptable candidates: "
        f"{result.num_acceptable_candidates}"
    )

    if result.parser.warnings:
        console.print(
            "\n[bold yellow]Parser warnings[/bold yellow]"
        )

        for warning in result.parser.warnings:
            console.print(
                f"[yellow]⚠[/yellow] {warning}"
            )

    if result.selected_candidate is not None:
        candidate = result.selected_candidate

        console.print(
            "\n[bold green]"
            "Candidate selected by ranking policy"
            "[/bold green]\n"
        )

        console.print(
            f"Scaling:               "
            f"{_format_scaling(candidate.scaling)}"
        )

        console.print(
            f"Atoms:                 "
            f"{candidate.num_atoms}"
        )

        console.print(
            f"Minimum image:         "
            f"{candidate.minimum_image_distance_angstrom:.3f} Å"
        )

        console.print(
            f"Anisotropy:            "
            f"{candidate.anisotropy_ratio:.3f}"
        )

        console.print(
            "\n[green]✓[/green] "
            "All requested hard constraints satisfied."
        )

    else:
        console.print(
            "\n[bold yellow]"
            "No candidate satisfies all requested constraints."
            "[/bold yellow]\n"
        )

        candidate = (
            result.best_separation_within_atom_limits
        )

        if candidate is not None:
            console.print(
                "[bold]"
                "Best separation within atom limits"
                "[/bold]"
            )

            console.print(
                f"Scaling:               "
                f"{_format_scaling(candidate.scaling)}"
            )

            console.print(
                f"Atoms:                 "
                f"{candidate.num_atoms}"
            )

            console.print(
                f"Minimum image:         "
                f"{candidate.minimum_image_distance_angstrom:.3f} Å"
            )

            shortfall = (
                result.diagnostics
                .image_distance_shortfall_angstrom
            )

            if shortfall is not None:
                console.print(
                    f"Distance shortfall:    "
                    f"{shortfall:.3f} Å"
                )

        candidate = (
            result.smallest_meeting_image_distance
        )

        if candidate is not None:
            console.print(
                "\n[bold]"
                "Smallest cell meeting image-distance target"
                "[/bold]"
            )

            console.print(
                f"Scaling:               "
                f"{_format_scaling(candidate.scaling)}"
            )

            console.print(
                f"Atoms:                 "
                f"{candidate.num_atoms}"
            )

            console.print(
                f"Minimum image:         "
                f"{candidate.minimum_image_distance_angstrom:.3f} Å"
            )

            atom_excess = (
                result.diagnostics.atom_excess
            )

            if atom_excess is not None:
                console.print(
                    f"Atoms above maximum:   "
                    f"{atom_excess}"
                )

    if result.acceptable_candidates:
        table = Table(
            title=(
                "Acceptable candidates "
                f"(top {top})"
            )
        )

        table.add_column("Rank", justify="right")
        table.add_column("Scaling")
        table.add_column("Atoms", justify="right")
        table.add_column(
            "Min image (Å)",
            justify="right",
        )
        table.add_column(
            "Anisotropy",
            justify="right",
        )

        for rank, candidate in enumerate(
            result.acceptable_candidates[:top],
            start=1,
        ):
            table.add_row(
                str(rank),
                _format_scaling(
                    candidate.scaling
                ),
                str(candidate.num_atoms),
                (
                    f"{candidate.minimum_image_distance_angstrom:.3f}"
                ),
                (
                    f"{candidate.anisotropy_ratio:.3f}"
                ),
            )

        console.print()
        console.print(table)

    console.print(
        "\n[dim]"
        "Selection policy: fewer atoms → lower anisotropy "
        "→ larger image separation. This is a documented "
        "computational ranking policy, not a universal "
        "physical optimum."
        "[/dim]"
    )

    console.print(
        "[dim]"
        "Current search is restricted to diagonal "
        "supercell transformations."
        "[/dim]\n"
    )