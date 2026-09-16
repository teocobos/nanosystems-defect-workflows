"""Tests for NSDW structural result models."""

import pytest
from pydantic import ValidationError

from nsdw.models import SpaceGroup, StructureResult


def test_crystalline_structure_creation():
    structure = StructureResult(
        formula="In3Ga3Zn3O12",
        n_atoms=21,
        periodic=True,
        structure_hash="abc123",
        volume=246.5,
        density=6.2,
        space_group=SpaceGroup(
            number=160,
            symbol="R3m",
        ),
    )

    assert structure.n_atoms == 21
    assert structure.space_group is not None
    assert structure.space_group.number == 160


def test_amorphous_structure_can_have_no_space_group():
    structure = StructureResult(
        formula="SiO2",
        n_atoms=216,
        periodic=True,
        structure_hash="def456",
    )

    assert structure.space_group is None


def test_structure_rejects_zero_atoms():
    with pytest.raises(ValidationError):
        StructureResult(
            formula="SiO2",
            n_atoms=0,
            periodic=True,
            structure_hash="abc123",
        )


def test_space_group_rejects_invalid_number():
    with pytest.raises(ValidationError):
        SpaceGroup(
            number=231,
            symbol="invalid",
        )