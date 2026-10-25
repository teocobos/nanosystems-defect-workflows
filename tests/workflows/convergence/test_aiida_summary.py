import pytest

"""Tests for AiiDA convergence summary provenance."""

from aiida.manage.configuration import load_profile
from aiida.orm import Dict, load_node
from nsdw.workflows.convergence.aiida_workchain import (
    build_convergence_summary,
)


@pytest.mark.aiida_integration
def test_summary_preserves_energy_and_provenance():
    # Integration test against the existing AiiDA database.
    load_profile("nsdw-dev")
    calc = load_node(52)
    assert calc.is_finished_ok

    parsed = calc.outputs.output_parameters
    parameters = parsed.get_dict()

    entries = Dict(dict={
        "sio2_smoke": {
            "pk": calc.pk,
            "uuid": str(calc.uuid),
            "finished_ok": True,
            "exit_status": calc.exit_status,
            "energy_hartree": parameters["energy"],
            "output_parameters": parameters,
        }
    })

    summary = build_convergence_summary(
        entries=entries,
        candidate_0000=parsed,
    )

    assert summary.is_stored
    assert summary.get_dict()["sio2_smoke"]["energy_hartree"] == (
        -108.49635643498
    )

    incoming = summary.base.links.get_incoming().all()
    assert len(incoming) >= 1
