import json
from pathlib import Path

import pytest

from nsdw.defects.exporter import (
    DefectExportError,
    calculate_file_sha256,
    export_vacancy_dataset,
)
from nsdw.structures.defects import (
    generate_symmetry_inequivalent_vacancies,
)
from nsdw.structures.parser import (
    load_structure,
)


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

IGZO_ORDERED = (
    REPO_ROOT
    / "tests"
    / "data"
    / "igzo"
    / "igzo_crystal_ordered_003.cif"
)


def _generate_igzo():
    structure, parser_warnings = (
        load_structure(
            IGZO_ORDERED
        )
    )

    generation = (
        generate_symmetry_inequivalent_vacancies(
            structure,
            species="O",
            scaling=(4, 4, 1),
        )
    )

    return (
        generation,
        parser_warnings,
    )


def test_sha256_is_deterministic():
    first = calculate_file_sha256(
        IGZO_ORDERED
    )

    second = calculate_file_sha256(
        IGZO_ORDERED
    )

    assert first == second
    assert len(first) == 64


def test_export_creates_expected_files(
    tmp_path,
):
    generation, parser_warnings = (
        _generate_igzo()
    )

    output = (
        tmp_path
        / "igzo"
    )

    result = export_vacancy_dataset(
        generation=generation,
        source_path=IGZO_ORDERED,
        parser_warnings=parser_warnings,
        output_directory=output,
        nsdw_version="0.1.0",
    )

    assert (
        result.pristine_structure_file
        .exists()
    )

    assert (
        result.manifest_file
        .exists()
    )

    assert (
        len(
            result.defect_structure_files
        )
        == 4
    )

    assert (
        len(
            result.defect_metadata_files
        )
        == 4
    )

    for path in (
        result.defect_structure_files
        + result.defect_metadata_files
    ):
        assert path.exists()


def test_exported_defect_names(
    tmp_path,
):
    generation, parser_warnings = (
        _generate_igzo()
    )

    result = export_vacancy_dataset(
        generation=generation,
        source_path=IGZO_ORDERED,
        parser_warnings=parser_warnings,
        output_directory=(
            tmp_path
            / "igzo"
        ),
        nsdw_version="0.1.0",
    )

    filenames = [
        path.name
        for path
        in result.defect_structure_files
    ]

    assert filenames == [
        "VO_O001_4x4x1_q0.cif",
        "VO_O002_4x4x1_q0.cif",
        "VO_O003_4x4x1_q0.cif",
        "VO_O004_4x4x1_q0.cif",
    ]


def test_manifest_contains_four_defects(
    tmp_path,
):
    generation, parser_warnings = (
        _generate_igzo()
    )

    result = export_vacancy_dataset(
        generation=generation,
        source_path=IGZO_ORDERED,
        parser_warnings=parser_warnings,
        output_directory=(
            tmp_path
            / "igzo"
        ),
        nsdw_version="0.1.0",
    )

    data = json.loads(
        result.manifest_file.read_text(
            encoding="utf-8"
        )
    )

    assert (
        data["num_defects_generated"]
        == 4
    )

    assert (
        data["num_inequivalent_sites"]
        == 4
    )

    assert (
        data["supercell_scaling"]
        == [4, 4, 1]
    )

    assert (
        data[
            "pristine_supercell_num_atoms"
        ]
        == 336
    )

    assert [
        defect["defect_id"]
        for defect in data["defects"]
    ] == [
        "VO_O001",
        "VO_O002",
        "VO_O003",
        "VO_O004",
    ]


def test_metadata_contains_provenance(
    tmp_path,
):
    generation, parser_warnings = (
        _generate_igzo()
    )

    result = export_vacancy_dataset(
        generation=generation,
        source_path=IGZO_ORDERED,
        parser_warnings=parser_warnings,
        output_directory=(
            tmp_path
            / "igzo"
        ),
        nsdw_version="0.1.0",
    )

    metadata = json.loads(
        result.defect_metadata_files[
            0
        ].read_text(
            encoding="utf-8"
        )
    )

    assert (
        metadata["defect_id"]
        == "VO_O001"
    )

    assert (
        metadata["provenance"]
        ["parent_structure"]
        == IGZO_ORDERED.name
    )

    assert len(
        metadata["provenance"]
        ["parent_structure_sha256"]
    ) == 64

    assert (
        metadata["symmetry"]
        ["primitive_site_index"]
        == 9
    )

    assert (
        metadata["removed_site"]
        ["supercell_site_index"]
        == 144
    )


def test_export_refuses_to_overwrite(
    tmp_path,
):
    generation, parser_warnings = (
        _generate_igzo()
    )

    output = (
        tmp_path
        / "igzo"
    )

    export_vacancy_dataset(
        generation=generation,
        source_path=IGZO_ORDERED,
        parser_warnings=parser_warnings,
        output_directory=output,
        nsdw_version="0.1.0",
    )

    with pytest.raises(
        DefectExportError,
        match="--overwrite",
    ):
        export_vacancy_dataset(
            generation=generation,
            source_path=IGZO_ORDERED,
            parser_warnings=parser_warnings,
            output_directory=output,
            nsdw_version="0.1.0",
        )


def test_export_overwrite_replaces_dataset(
    tmp_path,
):
    generation, parser_warnings = (
        _generate_igzo()
    )

    output = (
        tmp_path
        / "igzo"
    )

    export_vacancy_dataset(
        generation=generation,
        source_path=IGZO_ORDERED,
        parser_warnings=parser_warnings,
        output_directory=output,
        nsdw_version="0.1.0",
    )

    marker = (
        output
        / "old_file.txt"
    )

    marker.write_text(
        "remove me",
        encoding="utf-8",
    )

    export_vacancy_dataset(
        generation=generation,
        source_path=IGZO_ORDERED,
        parser_warnings=parser_warnings,
        output_directory=output,
        nsdw_version="0.1.0",
        overwrite=True,
    )

    assert not marker.exists()

    assert (
        output
        / "defect_manifest.json"
    ).exists()
