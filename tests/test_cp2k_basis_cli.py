"""Regression tests for CP2K basis convergence CLI."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nsdw.cli import app


runner = CliRunner()

STRUCTURE = (
    Path(__file__).parent
    / "data/sio2/alpha_quartz_cod1526860.cif"
)


def invoke_basis(tmp_path, *extra):
    output = tmp_path / "basis-campaign"
    result = runner.invoke(
        app,
        [
            "workflow",
            "convergence-generate",
            str(STRUCTURE),
            "--parameter", "basis",
            "--preset", "sio2-pbe-dzvp",
            "--cutoff", "400",
            "--relative-cutoff", "30",
            "--output", str(output),
            *extra,
        ],
    )
    return result, output


def test_standard_basis_sequence(tmp_path):
    result, output = invoke_basis(
        tmp_path,
        "--kpoint-mesh", "2x2x2",
    )

    assert result.exit_code == 0, result.output

    manifest = json.loads((output / "manifest.json").read_text())
    base = json.loads((output / "base-config.json").read_text())

    assert manifest["parameter"] == "basis"
    assert [c["label"] for c in manifest["candidates"]] == [
        "DZVP", "TZVP", "TZV2P"
    ]
    assert tuple(base["k_points"]) == (2, 2, 2)
    assert base["scf"]["solver"] == "DIAGONALIZATION"

    provenance = json.loads(
        (output / "basis-selection.json").read_text()
    )
    assert provenance["schema_version"] == 1
    assert provenance["strategy"] == "standard"
    assert provenance["candidates"] == [
        {"label": "DZVP", "preset": "sio2-pbe-dzvp"},
        {"label": "TZVP", "preset": "sio2-pbe-tzvp"},
        {"label": "TZV2P", "preset": "sio2-pbe-tzv2p"},
    ]

    for label in ("DZVP", "TZVP", "TZV2P"):
        input_path = next((output / label).glob("*.inp"))
        text = input_path.read_text()

        assert "CUTOFF 400" in text
        assert "REL_CUTOFF 30" in text
        assert "SCHEME MONKHORST-PACK 2 2 2" in text
        assert "&DIAGONALIZATION" in text
        assert "&OT" not in text
        assert f"{label}-MOLOPT-GTH-q4" in text
        assert f"{label}-MOLOPT-GTH-q6" in text
        assert "GTH-PBE-q4" in text
        assert "GTH-PBE-q6" in text


def test_custom_basis_sequence(tmp_path):
    result, output = invoke_basis(
        tmp_path,
        "--basis-strategy", "custom",
        "--basis-candidate", "DZVP=sio2-pbe-dzvp",
        "--basis-candidate", "TZV2P=sio2-pbe-tzv2p",
        "--kpoint-mesh", "2x2x2",
    )

    assert result.exit_code == 0, result.output

    manifest = json.loads((output / "manifest.json").read_text())

    assert [c["label"] for c in manifest["candidates"]] == [
        "DZVP", "TZV2P"
    ]

    provenance = json.loads(
        (output / "basis-selection.json").read_text()
    )
    assert provenance["strategy"] == "custom"
    assert provenance["candidates"] == [
        {"label": "DZVP", "preset": "sio2-pbe-dzvp"},
        {"label": "TZV2P", "preset": "sio2-pbe-tzv2p"},
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ("--auto",),
        ("--mesh", "2x2x2"),
        ("--values", "400"),
        ("--basis-candidate", "DZVP=sio2-pbe-dzvp"),
        ("--basis-strategy", "custom"),
        ("--basis-strategy", "custom",
         "--basis-candidate", "invalid"),
        ("--basis-strategy", "custom",
         "--basis-candidate", "A=sio2-pbe-dzvp",
         "--basis-candidate", "A=sio2-pbe-tzv2p"),
        ("--kpoint-mesh", "2x0x2"),
        ("--kpoint-mesh", "2x2"),
    ],
)
def test_invalid_basis_options_do_not_create_campaign(tmp_path, extra):
    result, output = invoke_basis(tmp_path, *extra)

    assert result.exit_code != 0
    assert not output.exists()


def test_unknown_standard_sequence_is_rejected(tmp_path):
    output = tmp_path / "basis-campaign"

    result = runner.invoke(
        app,
        [
            "workflow", "convergence-generate",
            str(STRUCTURE),
            "--parameter", "basis",
            "--preset", "igzo-uzh-tzv2p",
            "--output", str(output),
        ],
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_fixed_mesh_rejected_for_kpoint_convergence(tmp_path):
    output = tmp_path / "kpoints"

    result = runner.invoke(
        app,
        [
            "workflow", "convergence-generate",
            str(STRUCTURE),
            "--parameter", "kpoints",
            "--preset", "sio2-pbe-dzvp",
            "--mesh", "1x1x1",
            "--mesh", "2x2x2",
            "--kpoint-mesh", "2x2x2",
            "--output", str(output),
        ],
    )

    assert result.exit_code != 0
    assert not output.exists()
