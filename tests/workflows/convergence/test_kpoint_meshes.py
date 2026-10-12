"""Tests for automatic reciprocal-space k-point meshes."""

import math

import pytest
from pymatgen.core import Lattice, Structure

from nsdw.workflows.convergence.kpoint_meshes import (
    generate_automatic_mesh_sequence,
    generate_mesh_sequence,
    mesh_from_spacing,
)


def make_structure(a=4.0, b=4.0, c=4.0):
    return Structure(
        Lattice.orthorhombic(a, b, c),
        ["Si"],
        [[0, 0, 0]],
    )


def test_cubic_mesh_from_spacing():
    structure = make_structure()
    spacing = math.pi / 4

    assert mesh_from_spacing(structure, spacing) == (2, 2, 2)


def test_anisotropic_mesh():
    structure = make_structure(4, 4, 24)

    assert mesh_from_spacing(
        structure,
        0.5,
    ) == (4, 4, 1)


@pytest.mark.parametrize(
    "spacing",
    [0, -0.1, float("nan"), float("inf")],
)
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        mesh_from_spacing(make_structure(), spacing)


def test_generate_increasing_mesh_sequence():
    structure = make_structure()

    meshes = generate_mesh_sequence(
        structure,
        [2.0, 1.0, 0.5],
    )

    assert meshes == (
        (1, 1, 1),
        (2, 2, 2),
        (4, 4, 4),
    )


def test_duplicate_meshes_are_removed():
    structure = make_structure()

    meshes = generate_mesh_sequence(
        structure,
        [2.0, 1.8, 1.0, 0.5],
    )

    assert meshes == (
        (1, 1, 1),
        (2, 2, 2),
        (4, 4, 4),
    )


def test_rejects_insufficient_distinct_meshes():
    with pytest.raises(ValueError):
        generate_mesh_sequence(
            make_structure(),
            [2.0, 1.8],
        )


def test_rejects_decreasing_mesh_cost():
    with pytest.raises(ValueError):
        generate_mesh_sequence(
            make_structure(),
            [0.5, 1.0, 2.0],
        )


def test_automatic_meshes_are_reproducible():
    structure = make_structure()

    first = generate_automatic_mesh_sequence(structure)
    second = generate_automatic_mesh_sequence(structure)

    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4

    costs = [math.prod(mesh) for mesh in first]
    assert costs == sorted(set(costs))


def test_automatic_meshes_respect_anisotropy():
    structure = make_structure(4, 4, 24)

    meshes = generate_automatic_mesh_sequence(structure)

    assert len(meshes) == 4
    assert all(mesh[0] == mesh[1] for mesh in meshes)
    assert all(mesh[2] <= mesh[0] for mesh in meshes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"number_of_meshes": 1},
        {"initial_spacing_inv_angstrom": 0},
        {"initial_spacing_inv_angstrom": float("nan")},
        {"refinement_factor": 0},
        {"refinement_factor": 1},
        {"refinement_factor": float("nan")},
        {"max_mesh_points": 0},
    ],
)
def test_automatic_meshes_reject_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_automatic_mesh_sequence(
            make_structure(),
            **kwargs,
        )


def test_automatic_meshes_enforce_safety_limit():
    with pytest.raises(ValueError):
        generate_automatic_mesh_sequence(
            make_structure(),
            number_of_meshes=4,
            max_mesh_points=1,
        )
