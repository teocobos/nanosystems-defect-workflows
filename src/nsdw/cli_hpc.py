"""User-facing HPC profile commands."""

from __future__ import annotations

import typer
from pydantic import ValidationError
from rich.console import Console

from nsdw.execution.profiles import (
    HPCProfile,
    HPCProfileResources,
    HPCProfileStoreError,
    default_hpc_profiles_path,
    delete_hpc_profile,
    get_hpc_profile,
    load_hpc_profiles,
    upsert_hpc_profile,
    validate_hpc_profile,
)


console = Console()

hpc_app = typer.Typer(
    help="Configure and inspect reusable HPC execution profiles.",
    no_args_is_help=True,
)


@hpc_app.command("path")
def hpc_path() -> None:
    """Show the user-level HPC profile configuration path."""

    console.print(str(default_hpc_profiles_path()))


@hpc_app.command("list")
def hpc_list() -> None:
    """List configured HPC profiles."""

    try:
        profiles = load_hpc_profiles()
    except HPCProfileStoreError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    if not profiles:
        console.print(
            "[yellow]No HPC profiles configured.[/yellow]"
        )
        return

    console.print(
        "\n[bold]Configured HPC profiles[/bold]\n"
    )

    for name in sorted(profiles):
        profile = profiles[name]

        console.print(
            f"{name}: "
            f"{profile.computer} "
            f"({profile.backend.value})"
        )


@hpc_app.command("show")
def hpc_show(
    name: str = typer.Argument(
        ...,
        help="HPC profile name.",
    ),
) -> None:
    """Show one configured HPC profile."""

    try:
        profile = get_hpc_profile(name)
    except HPCProfileStoreError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    resources = profile.resources

    console.print(
        f"\n[bold]HPC profile: {profile.name}[/bold]\n"
    )

    console.print(
        f"Backend:              {profile.backend.value}"
    )
    console.print(
        f"Scheduler:            {profile.scheduler.value}"
    )
    console.print(
        f"AiiDA profile:        "
        f"{profile.aiida_profile or 'default'}"
    )
    console.print(
        f"Computer:             {profile.computer}"
    )
    console.print(
        f"Code:                 {profile.code}"
    )
    console.print(
        f"Queue:                "
        f"{resources.queue or '-'}"
    )
    console.print(
        f"Account:              "
        f"{resources.account or '-'}"
    )
    console.print(
        f"Machines:             {resources.machines}"
    )
    console.print(
        f"MPI per machine:      "
        f"{resources.mpi_per_machine}"
    )
    console.print(
        f"OpenMP threads:       "
        f"{resources.omp_threads}"
    )
    console.print(
        f"Walltime:             "
        f"{resources.walltime_seconds} s"
    )


@hpc_app.command("configure")
def hpc_configure(
    name: str = typer.Argument(
        ...,
        help="HPC profile name.",
    ),
    computer: str = typer.Option(
        ...,
        "--computer",
        help="AiiDA Computer label.",
    ),
    code: str = typer.Option(
        ...,
        "--code",
        help="AiiDA installed-code label.",
    ),
    aiida_profile: str | None = typer.Option(
        None,
        "--aiida-profile",
        help=(
            "AiiDA profile name. "
            "Uses the AiiDA default when omitted."
        ),
    ),
    queue: str | None = typer.Option(
        None,
        "--queue",
        help="Scheduler queue or partition.",
    ),
    account: str | None = typer.Option(
        None,
        "--account",
        help="Scheduler project/account.",
    ),
    machines: int = typer.Option(
        1,
        "--machines",
        min=1,
        help="Default number of compute nodes.",
    ),
    mpi_per_machine: int = typer.Option(
        1,
        "--mpi-per-machine",
        min=1,
        help="Default MPI processes per node.",
    ),
    omp_threads: int = typer.Option(
        1,
        "--omp-threads",
        min=1,
        help="Default OpenMP threads per MPI process.",
    ),
    walltime: int = typer.Option(
        600,
        "--walltime",
        min=1,
        help="Default maximum walltime in seconds.",
    ),
) -> None:
    """Create or replace a reusable HPC profile."""

    try:
        profile = HPCProfile(
            name=name,
            aiida_profile=aiida_profile,
            computer=computer,
            code=code,
            resources=HPCProfileResources(
                machines=machines,
                mpi_per_machine=mpi_per_machine,
                omp_threads=omp_threads,
                walltime_seconds=walltime,
                queue=queue,
                account=account,
            ),
        )

        path = upsert_hpc_profile(profile)

    except (
        ValidationError,
        HPCProfileStoreError,
    ) as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold green]HPC profile configured:[/bold green] "
        f"{profile.name}"
    )
    console.print(
        f"Configuration: {path}"
    )


@hpc_app.command("delete")
def hpc_delete(
    name: str = typer.Argument(
        ...,
        help="HPC profile name.",
    ),
) -> None:
    """Delete a configured HPC profile."""

    try:
        path = delete_hpc_profile(name)
    except HPCProfileStoreError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold green]HPC profile deleted:[/bold green] {name}"
    )
    console.print(
        f"Configuration: {path}"
    )



@hpc_app.command("validate")
def hpc_validate(
    name: str = typer.Argument(
        ...,
        help="HPC profile name to validate.",
    ),
) -> None:
    """Validate an HPC profile against local AiiDA configuration."""

    try:
        profile = get_hpc_profile(name)
    except HPCProfileStoreError as exc:
        console.print(
            f"[bold red]Error:[/bold red] {exc}"
        )
        raise typer.Exit(code=1) from exc

    result = validate_hpc_profile(profile)

    console.print(
        f"\n[bold]NSDW HPC Profile Validation[/bold]\n"
    )
    console.print(f"Profile: {result.name}")
    console.print(f"Backend: {profile.backend.value}")
    console.print(f"Scheduler: {profile.scheduler.value}\n")

    for check in result.checks:
        console.print(f"[green]✓[/green] {check}")

    for error in result.errors:
        console.print(f"[red]✗[/red] {error}")

    if not result.valid:
        console.print(
            "\n[bold red]Validation failed.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print(
        "\n[bold green]Configuration validated.[/bold green]"
    )
    console.print(
        "[dim]Remote connectivity and job submission "
        "were not tested.[/dim]"
    )
