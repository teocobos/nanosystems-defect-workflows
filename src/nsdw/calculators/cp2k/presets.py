"""Verified CP2K basis-set and pseudopotential presets."""

from __future__ import annotations

from nsdw.calculators.cp2k.generation_models import (
    CP2KBasisPotentialConfig,
    CP2KKindConfig,
)


IGZO_UZH_TZV2P = CP2KBasisPotentialConfig(
    basis_set_file="BASIS_MOLOPT_UZH",
    potential_file="POTENTIAL_UZH",
    kinds=(
        CP2KKindConfig(
            element="In",
            basis_set="TZV2P-MOLOPT-PBE-GTH-q13",
            potential="GTH-PBE-q13",
        ),
        CP2KKindConfig(
            element="Ga",
            basis_set="TZV2P-MOLOPT-PBE-GTH-q13",
            potential="GTH-PBE-q13",
        ),
        CP2KKindConfig(
            element="Zn",
            basis_set="TZV2P-MOLOPT-PBE-GTH-q12",
            potential="GTH-PBE-q12",
        ),
        CP2KKindConfig(
            element="O",
            basis_set="TZV2P-MOLOPT-PBE-GTH-q6",
            potential="GTH-PBE-q6",
        ),
    ),
)

SIO2_PBE_DZVP = CP2KBasisPotentialConfig(
    basis_set_file="BASIS_MOLOPT",
    potential_file="GTH_POTENTIALS",
    kinds=(
        CP2KKindConfig(
            element="Si",
            basis_set="DZVP-MOLOPT-GTH-q4",
            potential="GTH-PBE-q4",
        ),
        CP2KKindConfig(
            element="O",
            basis_set="DZVP-MOLOPT-GTH-q6",
            potential="GTH-PBE-q6",
        ),
    ),
)


BASIS_POTENTIAL_PRESETS: dict[
    str,
    CP2KBasisPotentialConfig,
] = {
    "igzo-uzh-tzv2p": IGZO_UZH_TZV2P,
    "sio2-pbe-dzvp": SIO2_PBE_DZVP,
}


def get_basis_potential_preset(
    name: str,
) -> CP2KBasisPotentialConfig:
    """Return a named CP2K basis/pseudopotential preset."""

    try:
        return BASIS_POTENTIAL_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(
            sorted(BASIS_POTENTIAL_PRESETS)
        )

        raise ValueError(
            f"Unknown CP2K basis/potential preset: {name}. "
            f"Available presets: {available}"
        ) from exc
