import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from pymatgen.io.cif import CifWriter

from nsdw.output.builders import (
    build_defect_manifest_output,
    build_vacancy_metadata_output,
)
from nsdw.output.models import (
    DefectManifestEntryOutput,
    DefectManifestOutput,
)
from nsdw.structures.defects import (
    VacancyGenerationResult,
)


class DefectExportError(Exception):
    """Raised when NSDW cannot export a defect dataset."""


@dataclass
class DefectExportResult:
    """
    Filesystem result of a defect dataset export.
    """

    output_directory: Path

    pristine_structure_file: Path
    manifest_file: Path

    defect_structure_files: list[Path]
    defect_metadata_files: list[Path]

    manifest: DefectManifestOutput


def calculate_file_sha256(
    path: str | Path,
) -> str:
    """
    Calculate the SHA-256 digest of a source file.
    """

    source = Path(
        path
    ).expanduser().resolve()

    if not source.exists():
        raise DefectExportError(
            f"Source file does not exist: {source}"
        )

    digest = hashlib.sha256()

    with source.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _scaling_label(
    scaling: tuple[int, int, int],
) -> str:
    """
    Convert (4, 4, 1) to '4x4x1'.
    """

    return "x".join(
        str(value)
        for value in scaling
    )


def _charge_label(
    charge_state: int,
) -> str:
    """
    Return a filesystem-safe charge-state label.

    Examples
    --------
    0  -> q0
    +1 -> q+1
    -2 -> q-2
    """

    if charge_state > 0:
        return f"q+{charge_state}"

    return f"q{charge_state}"


def _relative_posix(
    path: Path,
    root: Path,
) -> str:
    """
    Return a portable manifest path relative to the dataset root.
    """

    return path.relative_to(
        root
    ).as_posix()


def _prepare_output_directory(
    output_directory: Path,
    *,
    overwrite: bool,
) -> None:
    """
    Prepare the export directory without silently destroying data.
    """

    if output_directory.exists():
        populated = any(
            output_directory.iterdir()
        )

        if populated and not overwrite:
            raise DefectExportError(
                "Output directory already exists and is not "
                f"empty: {output_directory}. "
                "Use --overwrite to replace it."
            )

        if populated and overwrite:
            shutil.rmtree(
                output_directory
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def export_vacancy_dataset(
    *,
    generation: VacancyGenerationResult,
    source_path: str | Path,
    parser_warnings: list[str],
    output_directory: str | Path,
    nsdw_version: str,
    overwrite: bool = False,
) -> DefectExportResult:
    """
    Export a complete symmetry-inequivalent vacancy dataset.

    The dataset contains:

    - pristine supercell CIF,
    - one directory per defect,
    - one CIF per defect,
    - one metadata JSON file per defect,
    - one top-level defect manifest.
    """

    source = Path(
        source_path
    ).expanduser().resolve()

    output_root = Path(
        output_directory
    ).expanduser().resolve()

    _prepare_output_directory(
        output_root,
        overwrite=overwrite,
    )

    parent_hash = (
        calculate_file_sha256(
            source
        )
    )

    scaling_label = (
        _scaling_label(
            generation.supercell_scaling
        )
    )

    pristine_directory = (
        output_root
        / "pristine"
    )

    defects_directory = (
        output_root
        / "defects"
    )

    pristine_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    defects_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pristine_filename = (
        f"pristine_{scaling_label}.cif"
    )

    pristine_path = (
        pristine_directory
        / pristine_filename
    )

    CifWriter(
        generation.pristine_supercell
    ).write_file(
        str(pristine_path)
    )

    defect_structure_files: list[
        Path
    ] = []

    defect_metadata_files: list[
        Path
    ] = []

    manifest_entries: list[
        DefectManifestEntryOutput
    ] = []

    charge_label = (
        _charge_label(
            generation.charge_state
        )
    )

    for vacancy in generation.vacancies:
        defect_directory = (
            defects_directory
            / vacancy.defect_id
        )

        defect_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        structure_filename = (
            f"{vacancy.defect_id}_"
            f"{scaling_label}_"
            f"{charge_label}.cif"
        )

        structure_path = (
            defect_directory
            / structure_filename
        )

        metadata_path = (
            defect_directory
            / "metadata.json"
        )

        CifWriter(
            vacancy.defect_structure
        ).write_file(
            str(structure_path)
        )

        structure_relative = (
            _relative_posix(
                structure_path,
                output_root,
            )
        )

        metadata_relative = (
            _relative_posix(
                metadata_path,
                output_root,
            )
        )

        metadata = (
            build_vacancy_metadata_output(
                vacancy=vacancy,
                source_path=source,
                parent_structure_sha256=(
                    parent_hash
                ),
                structure_file=(
                    structure_relative
                ),
                nsdw_version=(
                    nsdw_version
                ),
            )
        )

        metadata_path.write_text(
            metadata.model_dump_json(
                indent=2
            )
            + "\n",
            encoding="utf-8",
        )

        defect_structure_files.append(
            structure_path
        )

        defect_metadata_files.append(
            metadata_path
        )

        manifest_entries.append(
            DefectManifestEntryOutput(
                defect_id=(
                    vacancy.defect_id
                ),
                species=(
                    vacancy.species
                ),
                charge_state=(
                    vacancy.charge_state
                ),
                symmetry_site_id=(
                    vacancy.symmetry_site_id
                ),
                multiplicity=(
                    vacancy.primitive_multiplicity
                ),
                structure_file=(
                    structure_relative
                ),
                metadata_file=(
                    metadata_relative
                ),
            )
        )

    pristine_relative = (
        _relative_posix(
            pristine_path,
            output_root,
        )
    )

    manifest = (
        build_defect_manifest_output(
            source_path=source,
            parser_warnings=parser_warnings,
            parent_structure_sha256=(
                parent_hash
            ),
            species=generation.species,
            charge_state=(
                generation.charge_state
            ),
            primitive_num_atoms=(
                generation.primitive_num_atoms
            ),
            supercell_scaling=(
                generation.supercell_scaling
            ),
            pristine_supercell_num_atoms=(
                generation
                .pristine_supercell_num_atoms
            ),
            num_inequivalent_sites=(
                generation
                .num_inequivalent_sites
            ),
            pristine_structure_file=(
                pristine_relative
            ),
            defect_entries=(
                manifest_entries
            ),
            nsdw_version=(
                nsdw_version
            ),
        )
    )

    manifest_path = (
        output_root
        / "defect_manifest.json"
    )

    manifest_path.write_text(
        manifest.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    return DefectExportResult(
        output_directory=(
            output_root
        ),
        pristine_structure_file=(
            pristine_path
        ),
        manifest_file=(
            manifest_path
        ),
        defect_structure_files=(
            defect_structure_files
        ),
        defect_metadata_files=(
            defect_metadata_files
        ),
        manifest=manifest,
    )
