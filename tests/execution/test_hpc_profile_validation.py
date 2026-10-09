
"""Tests for AiiDA-backed HPC profile validation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nsdw.execution.profiles.models import HPCProfile
from nsdw.execution.profiles.validation import validate_hpc_profile


@pytest.fixture
def profile():
    return HPCProfile(
        name="lumi-c",
        aiida_profile="nsdw-dev",
        computer="lumi-c",
        code="cp2k-lumi-c@lumi-c",
    )


def install_fake_aiida(
    monkeypatch,
    *,
    profile_error=None,
    computer_error=None,
    code_error=None,
    code_computer_uuid="computer-123",
):
    """Install a minimal fake AiiDA API without contacting a database."""

    aiida = ModuleType("aiida")
    orm = ModuleType("aiida.orm")

    def load_profile(name):
        if profile_error:
            raise RuntimeError(profile_error)
        return name

    def load_computer(label):
        if computer_error:
            raise RuntimeError(computer_error)
        return SimpleNamespace(uuid="computer-123")

    def load_code(label):
        if code_error:
            raise RuntimeError(code_error)
        return SimpleNamespace(
            computer=(
                None
                if code_computer_uuid is None
                else SimpleNamespace(uuid=code_computer_uuid)
            )
        )

    aiida.load_profile = load_profile
    orm.load_computer = load_computer
    orm.load_code = load_code

    monkeypatch.setitem(sys.modules, "aiida", aiida)
    monkeypatch.setitem(sys.modules, "aiida.orm", orm)


def test_valid_profile(monkeypatch, profile):
    install_fake_aiida(monkeypatch)

    result = validate_hpc_profile(profile)

    assert result.valid
    assert result.errors == ()
    assert len(result.checks) == 4


def test_invalid_aiida_profile(monkeypatch, profile):
    install_fake_aiida(
        monkeypatch,
        profile_error="Profile not found",
    )

    result = validate_hpc_profile(profile)

    assert not result.valid
    assert any("Cannot load AiiDA profile" in e for e in result.errors)


def test_missing_computer(monkeypatch, profile):
    install_fake_aiida(
        monkeypatch,
        computer_error="Computer not found",
    )

    result = validate_hpc_profile(profile)

    assert not result.valid
    assert any("Cannot load computer" in e for e in result.errors)


def test_missing_code(monkeypatch, profile):
    install_fake_aiida(
        monkeypatch,
        code_error="Code not found",
    )

    result = validate_hpc_profile(profile)

    assert not result.valid
    assert any("Cannot load code" in e for e in result.errors)


def test_mismatched_computer(monkeypatch, profile):
    install_fake_aiida(
        monkeypatch,
        code_computer_uuid="different-computer",
    )

    result = validate_hpc_profile(profile)

    assert not result.valid
    assert any(
        "different computer" in e
        for e in result.errors
    )


def test_code_without_computer(monkeypatch, profile):
    install_fake_aiida(
        monkeypatch,
        code_computer_uuid=None,
    )

    result = validate_hpc_profile(profile)

    assert not result.valid
    assert any(
        "no associated computer" in e
        for e in result.errors
    )

