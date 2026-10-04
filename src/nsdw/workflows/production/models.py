"""Production calculation recipe models."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class CP2KProductionRecipe(BaseModel):
    """
    Calculation-specific settings for a CP2K production calculation.

    Scientific methodology such as the functional, cutoffs, SCF settings,
    basis sets, and pseudopotentials is supplied separately from the
    project's validated production methodology.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    project_name: str = Field(
        min_length=1,
    )

    run_type: str = Field(
        default="ENERGY_FORCE",
        min_length=1,
    )

    charge: int = 0

    multiplicity: int = Field(
        default=1,
        ge=1,
    )
