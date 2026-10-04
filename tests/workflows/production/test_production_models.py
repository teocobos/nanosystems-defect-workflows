from pydantic import ValidationError
import pytest

from nsdw.workflows.production.models import (
    CP2KProductionRecipe,
)


def test_cp2k_production_recipe_defaults() -> None:
    recipe = CP2KProductionRecipe(
        project_name="igzo-neutral",
    )

    assert recipe.project_name == "igzo-neutral"
    assert recipe.run_type == "ENERGY_FORCE"
    assert recipe.charge == 0
    assert recipe.multiplicity == 1


def test_cp2k_production_recipe_supports_charged_state() -> None:
    recipe = CP2KProductionRecipe(
        project_name="oxygen-vacancy-plus-two",
        run_type="ENERGY_FORCE",
        charge=2,
        multiplicity=1,
    )

    assert recipe.charge == 2
    assert recipe.multiplicity == 1


def test_cp2k_production_recipe_rejects_invalid_multiplicity() -> None:
    with pytest.raises(ValidationError):
        CP2KProductionRecipe(
            project_name="invalid",
            multiplicity=0,
        )


def test_cp2k_production_recipe_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CP2KProductionRecipe(
            project_name="igzo",
            cutoff_ry=800.0,
        )
