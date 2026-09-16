"""Tests for unit-aware NSDW quantities."""

import math

import pytest
from pydantic import ValidationError

from nsdw.models import Quantity


def test_quantity_creation():
    quantity = Quantity(value=1.70, unit="eV")

    assert quantity.value == 1.70
    assert quantity.unit == "eV"


def test_quantity_strips_unit_whitespace():
    quantity = Quantity(value=1.70, unit="  eV  ")

    assert quantity.unit == "eV"


def test_quantity_rejects_empty_unit():
    with pytest.raises(ValidationError):
        Quantity(value=1.70, unit="   ")


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_quantity_rejects_non_finite_values(value):
    with pytest.raises(ValidationError):
        Quantity(value=value, unit="eV")


def test_quantity_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Quantity(value=1.70, unit="eV", uncertainty=0.10)


def test_quantity_is_immutable():
    quantity = Quantity(value=1.70, unit="eV")

    with pytest.raises(ValidationError):
        quantity.value = 2.00
