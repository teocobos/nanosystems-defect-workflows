from dataclasses import dataclass
from itertools import product

import numpy as np
from pymatgen.core import Structure


class SupercellSearchError(Exception):
    """Raised when NSDW cannot perform a supercell search."""


@dataclass
class SupercellCandidate:
    """
    A candidate diagonal supercell evaluated by NSDW.
    """

    scaling: tuple[int, int, int]

    num_atoms: int
    volume_angstrom3: float

    a_angstrom: float
    b_angstrom: float
    c_angstrom: float

    minimum_image_distance_angstrom: float
    anisotropy_ratio: float

    meets_min_atoms: bool
    meets_max_atoms: bool
    meets_min_image_distance: bool

    acceptable: bool


@dataclass
class SupercellSearchResult:
    """
    Result of an NSDW diagonal supercell search.
    """

    primitive_num_atoms: int

    min_atoms: int
    max_atoms: int
    min_image_distance_angstrom: float

    max_scale: int
    image_range: int

    ranking_policy: str
    search_method: str

    candidates: list[SupercellCandidate]
    acceptable_candidates: list[SupercellCandidate]

    selected_candidate: SupercellCandidate | None

    best_separation_within_atom_limits: (
        SupercellCandidate | None
    )

    smallest_meeting_image_distance: (
        SupercellCandidate | None
    )


def get_minimum_lattice_translation(
    structure: Structure,
    image_range: int = 2,
) -> float:
    """
    Return the shortest non-zero periodic lattice translation.

    Integer lattice translations in the range

        [-image_range, image_range]

    are explicitly searched.

    This avoids assuming that the shortest individual lattice-vector
    length is necessarily the minimum periodic image separation for
    a non-orthogonal cell.
    """

    if image_range < 1:
        raise SupercellSearchError(
            "image_range must be at least 1."
        )

    lattice_matrix = np.asarray(
        structure.lattice.matrix,
        dtype=float,
    )

    minimum_distance = np.inf

    for i, j, k in product(
        range(-image_range, image_range + 1),
        repeat=3,
    ):
        if i == 0 and j == 0 and k == 0:
            continue

        translation = np.array(
            [i, j, k],
            dtype=float,
        )

        cartesian = (
            translation @ lattice_matrix
        )

        distance = float(
            np.linalg.norm(cartesian)
        )

        if distance < minimum_distance:
            minimum_distance = distance

    if not np.isfinite(minimum_distance):
        raise SupercellSearchError(
            "Could not determine minimum periodic "
            "lattice translation."
        )

    return float(minimum_distance)


def calculate_anisotropy_ratio(
    structure: Structure,
) -> float:
    """
    Calculate a simple lattice-length anisotropy measure.

    Ratio = longest lattice-vector length /
            shortest lattice-vector length.

    A value of 1.0 corresponds to equal lattice-vector lengths.

    This is a geometric ranking diagnostic rather than a hard
    physical acceptance criterion.
    """

    lengths = np.asarray(
        structure.lattice.abc,
        dtype=float,
    )

    shortest = float(
        np.min(lengths)
    )

    longest = float(
        np.max(lengths)
    )

    if shortest <= 0.0:
        raise SupercellSearchError(
            "Invalid lattice-vector length."
        )

    return longest / shortest


def _acceptable_ranking_key(
    candidate: SupercellCandidate,
) -> tuple:
    """
    Ranking policy for candidates satisfying all hard constraints.

    Current Phase 1 policy:

    1. fewer atoms,
    2. lower anisotropy,
    3. larger minimum image separation,
    4. deterministic scaling tuple.

    This ranking is a computational selection policy, not a claim
    that the first candidate is universally physically optimal.
    """

    return (
        candidate.num_atoms,
        candidate.anisotropy_ratio,
        -candidate.minimum_image_distance_angstrom,
        candidate.scaling,
    )


def search_supercells(
    structure: Structure,
    *,
    min_atoms: int = 50,
    max_atoms: int = 250,
    min_image_distance: float = 10.0,
    max_scale: int = 4,
    image_range: int = 2,
) -> SupercellSearchResult:
    """
    Search diagonal supercells and evaluate modelling constraints.

    Candidate supercells have the form

        diag(na, nb, nc)

    where each scaling factor ranges from 1 to ``max_scale``.

    Hard constraints
    ----------------
    A candidate is acceptable only if it satisfies:

    1. num_atoms >= min_atoms
    2. num_atoms <= max_atoms
    3. minimum periodic image distance >= min_image_distance

    Ranking objectives
    ------------------
    Candidates satisfying all hard constraints are ranked by:

    1. fewer atoms
    2. lower lattice-length anisotropy
    3. larger minimum image separation
    4. deterministic scaling tuple

    All candidates through ``max_scale`` are retained, including
    candidates exceeding ``max_atoms``. This allows NSDW to explain
    constraint trade-offs when no acceptable candidate exists.

    Notes
    -----
    This Phase 1 implementation searches diagonal supercell
    transformations only. General integer transformation matrices
    and symmetry-equivalent candidate reduction can be added later.
    """

    if len(structure) < 1:
        raise SupercellSearchError(
            "Structure contains no atomic sites."
        )

    if min_atoms < 1:
        raise SupercellSearchError(
            "min_atoms must be at least 1."
        )

    if max_atoms < min_atoms:
        raise SupercellSearchError(
            "max_atoms must be greater than or equal "
            "to min_atoms."
        )

    if min_image_distance <= 0.0:
        raise SupercellSearchError(
            "min_image_distance must be positive."
        )

    if max_scale < 1:
        raise SupercellSearchError(
            "max_scale must be at least 1."
        )

    if image_range < 1:
        raise SupercellSearchError(
            "image_range must be at least 1."
        )

    primitive_num_atoms = len(
        structure
    )

    candidates: list[
        SupercellCandidate
    ] = []

    for na, nb, nc in product(
        range(1, max_scale + 1),
        repeat=3,
    ):
        scaling = (
            na,
            nb,
            nc,
        )

        num_atoms = (
            primitive_num_atoms
            * na
            * nb
            * nc
        )

        candidate_structure = (
            structure.copy()
        )

        candidate_structure.make_supercell(
            [na, nb, nc]
        )

        minimum_image = (
            get_minimum_lattice_translation(
                candidate_structure,
                image_range=image_range,
            )
        )

        anisotropy = (
            calculate_anisotropy_ratio(
                candidate_structure
            )
        )

        meets_min_atoms = (
            num_atoms >= min_atoms
        )

        meets_max_atoms = (
            num_atoms <= max_atoms
        )

        meets_min_image_distance = (
            minimum_image
            >= min_image_distance
        )

        acceptable = (
            meets_min_atoms
            and meets_max_atoms
            and meets_min_image_distance
        )

        candidates.append(
            SupercellCandidate(
                scaling=scaling,
                num_atoms=num_atoms,
                volume_angstrom3=float(
                    candidate_structure.volume
                ),
                a_angstrom=float(
                    candidate_structure.lattice.a
                ),
                b_angstrom=float(
                    candidate_structure.lattice.b
                ),
                c_angstrom=float(
                    candidate_structure.lattice.c
                ),
                minimum_image_distance_angstrom=(
                    minimum_image
                ),
                anisotropy_ratio=anisotropy,
                meets_min_atoms=(
                    meets_min_atoms
                ),
                meets_max_atoms=(
                    meets_max_atoms
                ),
                meets_min_image_distance=(
                    meets_min_image_distance
                ),
                acceptable=acceptable,
            )
        )

    acceptable_candidates = [
        candidate
        for candidate in candidates
        if candidate.acceptable
    ]

    acceptable_candidates.sort(
        key=_acceptable_ranking_key
    )

    selected_candidate = (
        acceptable_candidates[0]
        if acceptable_candidates
        else None
    )

    within_atom_limits = [
        candidate
        for candidate in candidates
        if candidate.meets_min_atoms
        and candidate.meets_max_atoms
    ]

    within_atom_limits.sort(
        key=lambda candidate: (
            -candidate.minimum_image_distance_angstrom,
            candidate.num_atoms,
            candidate.anisotropy_ratio,
            candidate.scaling,
        )
    )

    best_separation_within_atom_limits = (
        within_atom_limits[0]
        if within_atom_limits
        else None
    )

    meeting_image_distance = [
        candidate
        for candidate in candidates
        if candidate.meets_min_atoms
        and candidate.meets_min_image_distance
    ]

    meeting_image_distance.sort(
        key=lambda candidate: (
            candidate.num_atoms,
            candidate.anisotropy_ratio,
            -candidate.minimum_image_distance_angstrom,
            candidate.scaling,
        )
    )

    smallest_meeting_image_distance = (
        meeting_image_distance[0]
        if meeting_image_distance
        else None
    )

    return SupercellSearchResult(
        primitive_num_atoms=(
            primitive_num_atoms
        ),
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        min_image_distance_angstrom=(
            min_image_distance
        ),
        max_scale=max_scale,
        image_range=image_range,
        ranking_policy=(
            "fewest_atoms_then_lowest_anisotropy_"
            "then_largest_image_distance"
        ),
        search_method="diagonal_scaling",
        candidates=candidates,
        acceptable_candidates=(
            acceptable_candidates
        ),
        selected_candidate=selected_candidate,
        best_separation_within_atom_limits=(
            best_separation_within_atom_limits
        ),
        smallest_meeting_image_distance=(
            smallest_meeting_image_distance
        ),
    )