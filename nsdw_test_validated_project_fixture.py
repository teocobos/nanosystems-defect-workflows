"""Test-only setup for legacy validated-project consumer tests.

This deliberately bypasses the production writer only within temporary
test projects. Never import this helper into src/nsdw.
"""
from pathlib import Path
import yaml

from nsdw.project.models import ProjectConfig
from nsdw.project.scaffold import create_project


def create_existing_validated_project(*, root, config):
    validated = ProjectConfig.model_validate(
        config.model_dump(mode="json", warnings="error")
    )
    methodology = validated.methodology.cp2k

    if methodology is None or methodology.status != "validated":
        raise ValueError("Fixture requires a validated methodology")

    initial = validated.model_copy(
        update={
            "methodology": validated.methodology.model_copy(
                update={"cp2k": None}
            )
        }
    )

    create_project(root=root, config=initial)

    metadata = Path(root) / "project.yaml"
    metadata.write_text(
        yaml.safe_dump(
            validated.model_dump(mode="json", warnings="error"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
