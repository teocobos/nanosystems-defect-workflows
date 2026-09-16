from collections import defaultdict
from dataclasses import dataclass

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class SymmetryAnalysisError(Exception):
    """Raised when NSDW cannot determine structure symmetry."""


@dataclass
class InequivalentSite:
    representative_index: int
    element: str
    multiplicity: int
    equivalent_indices: list[int]
    fractional_coordinates: list[float]


@dataclass
class SymmetryResult:
    space_group_symbol: str
    space_group_number: int
    hall_symbol: str
    point_group: str
    crystal_system: str

    symprec_angstrom: float
    angle_tolerance_deg: float

    num_symmetry_operations: int

    equivalent_atoms: list[int]
    inequivalent_sites: list[InequivalentSite]


def analyse_symmetry(
    structure: Structure,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
) -> SymmetryResult:
    """
    Analyse crystallographic symmetry for an ordered structure.
    """

    if not structure.is_ordered:
        raise SymmetryAnalysisError(
            "Symmetry analysis currently requires a fully ordered "
            "structure. Resolve partial occupancies before proceeding."
        )

    try:
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        dataset = analyzer.get_symmetry_dataset()

    except Exception as exc:
        raise SymmetryAnalysisError(
            f"Symmetry analysis failed: {exc}"
        ) from exc

    if dataset is None:
        raise SymmetryAnalysisError(
            "No symmetry dataset could be determined."
        )

    equivalent_atoms = [
        int(index)
        for index in dataset.equivalent_atoms
    ]

    site_groups: dict[int, list[int]] = defaultdict(list)

    for site_index, representative in enumerate(
        equivalent_atoms
    ):
        site_groups[representative].append(site_index)

    inequivalent_sites: list[InequivalentSite] = []

    for representative, members in sorted(
        site_groups.items()
    ):
        site = structure[representative]

        inequivalent_sites.append(
            InequivalentSite(
                representative_index=representative,
                element=site.specie.symbol,
                multiplicity=len(members),
                equivalent_indices=members,
                fractional_coordinates=[
                    float(value)
                    for value in site.frac_coords
                ],
            )
        )

    return SymmetryResult(
        space_group_symbol=str(dataset.international),
        space_group_number=int(dataset.number),
        hall_symbol=str(dataset.hall),
        point_group=str(
            analyzer.get_point_group_symbol()
        ),
        crystal_system=str(
            analyzer.get_crystal_system()
        ),
        symprec_angstrom=float(symprec),
        angle_tolerance_deg=float(
            angle_tolerance
        ),
        num_symmetry_operations=len(
            analyzer.get_symmetry_operations()
        ),
        equivalent_atoms=equivalent_atoms,
        inequivalent_sites=inequivalent_sites,
    )