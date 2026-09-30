from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nsdw.models.quantity import Quantity
from nsdw.workflows.convergence.analyser import (
    ConvergenceAnalysis,
    ConvergenceReferenceAnalysis,
    ConvergenceTailAnalysis,
)
from nsdw.workflows.convergence.models import (
    ConvergenceObservation,
    ConvergenceParameter,
    ConvergenceStudyDefinition,
)


CandidateValue = Quantity | str | tuple[int, int, int] | None


@dataclass(frozen=True)
class ConvergenceReportCandidate:
    """
    Scientific reporting data for one convergence candidate.

    The three energy-difference fields deliberately represent distinct
    convergence questions:

    - adjacent: difference from the next higher-cost candidate;
    - reference: difference from the highest-cost candidate;
    - maximum higher-cost: worst difference against the complete
      higher-cost tail.

    The final candidate has no higher-cost tail and therefore has no
    tail-stability decision.
    """

    label: str
    order: int
    value: CandidateValue
    total_energy_ev: float
    energy_ev_per_atom: float
    adjacent_energy_difference_ev_per_atom: float | None
    reference_energy_difference_ev_per_atom: float
    maximum_higher_cost_energy_difference_ev_per_atom: float | None
    tail_stable: bool | None
    selected: bool


@dataclass(frozen=True)
class ConvergenceReport:
    """
    Calculator-independent scientific convergence report.

    This model contains reporting data only. Scientific convergence
    decisions remain owned by the convergence analysers.
    """

    parameter: ConvergenceParameter
    energy_tolerance_ev_per_atom: float
    structure_hash: str
    n_atoms: int
    reference_candidate_label: str
    selected_candidate_label: str | None
    energy_tolerance_satisfied: bool
    candidates: tuple[ConvergenceReportCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible representation of the report.
        """
        candidate_data: list[dict[str, Any]] = []

        for candidate in self.candidates:
            value = candidate.value

            if isinstance(value, Quantity):
                serialised_value: Any = {
                    "value": value.value,
                    "unit": value.unit,
                }
            elif isinstance(value, tuple):
                serialised_value = list(value)
            else:
                serialised_value = value

            candidate_data.append(
                {
                    "label": candidate.label,
                    "order": candidate.order,
                    "value": serialised_value,
                    "total_energy_ev": candidate.total_energy_ev,
                    "energy_ev_per_atom": candidate.energy_ev_per_atom,
                    "adjacent_energy_difference_ev_per_atom": (
                        candidate.adjacent_energy_difference_ev_per_atom
                    ),
                    "reference_energy_difference_ev_per_atom": (
                        candidate.reference_energy_difference_ev_per_atom
                    ),
                    "maximum_higher_cost_energy_difference_ev_per_atom": (
                        candidate.maximum_higher_cost_energy_difference_ev_per_atom
                    ),
                    "tail_stable": candidate.tail_stable,
                    "selected": candidate.selected,
                }
            )

        return {
            "parameter": self.parameter.value,
            "energy_tolerance_ev_per_atom": (
                self.energy_tolerance_ev_per_atom
            ),
            "structure_hash": self.structure_hash,
            "n_atoms": self.n_atoms,
            "reference_candidate_label": (
                self.reference_candidate_label
            ),
            "selected_candidate_label": (
                self.selected_candidate_label
            ),
            "energy_tolerance_satisfied": (
                self.energy_tolerance_satisfied
            ),
            "candidates": candidate_data,
        }

    def write_json(self, path: str | Path) -> None:
        """
        Write the complete structured report as formatted JSON.
        """
        output_path = Path(path)
        output_path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_csv(self, path: str | Path) -> None:
        """
        Write candidate-level convergence data as a flat CSV table.
        """
        output_path = Path(path)

        fieldnames = [
            "label",
            "order",
            "value",
            "value_unit",
            "total_energy_ev",
            "energy_ev_per_atom",
            "adjacent_energy_difference_ev_per_atom",
            "reference_energy_difference_ev_per_atom",
            "maximum_higher_cost_energy_difference_ev_per_atom",
            "tail_stable",
            "selected",
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for candidate in self.candidates:
                value = candidate.value

                if isinstance(value, Quantity):
                    csv_value: Any = value.value
                    value_unit = value.unit
                elif isinstance(value, tuple):
                    csv_value = "x".join(str(item) for item in value)
                    value_unit = ""
                else:
                    csv_value = value if value is not None else ""
                    value_unit = ""

                writer.writerow(
                    {
                        "label": candidate.label,
                        "order": candidate.order,
                        "value": csv_value,
                        "value_unit": value_unit,
                        "total_energy_ev": candidate.total_energy_ev,
                        "energy_ev_per_atom": (
                            candidate.energy_ev_per_atom
                        ),
                        "adjacent_energy_difference_ev_per_atom": (
                            candidate.adjacent_energy_difference_ev_per_atom
                            if candidate.adjacent_energy_difference_ev_per_atom
                            is not None
                            else ""
                        ),
                        "reference_energy_difference_ev_per_atom": (
                            candidate.reference_energy_difference_ev_per_atom
                        ),
                        "maximum_higher_cost_energy_difference_ev_per_atom": (
                            candidate.maximum_higher_cost_energy_difference_ev_per_atom
                            if candidate.maximum_higher_cost_energy_difference_ev_per_atom
                            is not None
                            else ""
                        ),
                        "tail_stable": (
                            candidate.tail_stable
                            if candidate.tail_stable is not None
                            else ""
                        ),
                        "selected": candidate.selected,
                    }
                )

    def write_plot(self, path: str | Path) -> None:
        """
        Write a scientific convergence plot as PNG or SVG.
    
        The plot shows two distinct energy-difference measures:
    
        - difference from the highest-cost reference candidate;
        - maximum difference against all higher-cost candidates.
    
        Values are displayed in meV/atom. The convergence tolerance and
        selected candidate are shown without recalculating any scientific
        decision.
        """
        output_path = Path(path)
        suffix = output_path.suffix.lower()
    
        if suffix not in {".png", ".svg"}:
            raise ValueError(
                "Unsupported convergence plot format "
                f"{suffix!r}; expected '.png' or '.svg'."
            )
    
        import matplotlib.pyplot as plt
    
        labels = [
            candidate.label
            for candidate in self.candidates
        ]
    
        reference_differences_mev = [
            candidate.reference_energy_difference_ev_per_atom * 1000.0
            for candidate in self.candidates
        ]
    
        tail_differences_mev = [
            (
                candidate.maximum_higher_cost_energy_difference_ev_per_atom
                * 1000.0
                if candidate.maximum_higher_cost_energy_difference_ev_per_atom
                is not None
                else float("nan")
            )
            for candidate in self.candidates
        ]
    
        tolerance_mev = (
            self.energy_tolerance_ev_per_atom * 1000.0
        )
    
        numeric_quantity_values = all(
            isinstance(candidate.value, Quantity)
            for candidate in self.candidates
        )
    
        if numeric_quantity_values:
            x_values = [
                candidate.value.value
                for candidate in self.candidates
                if isinstance(candidate.value, Quantity)
            ]
            x_tick_labels = None
    
            units = {
                candidate.value.unit
                for candidate in self.candidates
                if isinstance(candidate.value, Quantity)
            }
            unit = next(iter(units)) if len(units) == 1 else None
    
            if unit is not None:
                x_label = (
                    f"{self.parameter.value.replace('_', ' ').title()} "
                    f"({unit})"
                )
            else:
                x_label = (
                    self.parameter.value.replace("_", " ").title()
                )
        else:
            x_values = list(range(len(self.candidates)))
            x_tick_labels = labels
            x_label = (
                self.parameter.value.replace("_", " ").title()
            )
    
        fig, ax = plt.subplots(figsize=(8, 5))
    
        ax.plot(
            x_values,
            reference_differences_mev,
            marker="o",
            label="Difference from highest-cost reference",
        )
        ax.plot(
            x_values,
            tail_differences_mev,
            marker="s",
            label="Maximum difference vs higher-cost candidates",
        )
    
        ax.axhline(
            tolerance_mev,
            linestyle="--",
            label=(
                "Energy tolerance "
                f"({tolerance_mev:g} meV/atom)"
            ),
        )
    
        if self.selected_candidate_label is not None:
            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(self.candidates)
                    if candidate.label
                    == self.selected_candidate_label
                ),
                None,
            )
    
            if selected_index is not None:
                selected_x = x_values[selected_index]
    
                ax.axvline(
                    selected_x,
                    linestyle=":",
                    label=(
                        "Selected candidate "
                        f"({self.selected_candidate_label})"
                    ),
                )
    
        if x_tick_labels is not None:
            ax.set_xticks(
                x_values,
                labels=x_tick_labels,
            )
    
        ax.set_xlabel(x_label)
        ax.set_ylabel("Energy difference (meV/atom)")
        ax.set_title(
            f"{self.parameter.value.replace('_', ' ').title()} "
            "convergence"
        )
        ax.grid(
            True,
            axis="y",
            alpha=0.25,
        )
        ax.legend()
        fig.tight_layout()
    
        try:
            fig.savefig(
                output_path,
                dpi=300 if suffix == ".png" else None,
            )
        finally:
            plt.close(fig)

def build_convergence_report(
    *,
    study: ConvergenceStudyDefinition,
    observations: list[ConvergenceObservation],
    adjacent_analysis: ConvergenceAnalysis,
    reference_analysis: ConvergenceReferenceAnalysis,
    tail_analysis: ConvergenceTailAnalysis,
) -> ConvergenceReport:
    """
    Build a structured convergence report from existing analyses.

    No convergence decision is recalculated here. The report combines
    observations with the outputs of the adjacent, reference and strict
    higher-cost-tail analyses.
    """
    adjacent_by_label = {
        comparison.candidate_label: comparison
        for comparison in adjacent_analysis.comparisons
    }
    reference_by_label = {
        comparison.candidate_label: comparison
        for comparison in reference_analysis.comparisons
    }
    tail_by_label = {
        comparison.candidate_label: comparison
        for comparison in tail_analysis.comparisons
    }

    reference_label = reference_analysis.reference_candidate_label
    selected_label = tail_analysis.selected_candidate_label

    candidate_rows: list[ConvergenceReportCandidate] = []

    for observation in observations:
        label = observation.candidate.label

        adjacent_comparison = adjacent_by_label.get(label)
        reference_comparison = reference_by_label.get(label)
        tail_comparison = tail_by_label.get(label)

        if label == reference_label:
            reference_difference = 0.0
        elif reference_comparison is not None:
            reference_difference = (
                reference_comparison.energy_difference_ev_per_atom
            )
        else:
            raise ValueError(
                f"Missing reference comparison for candidate {label!r}."
            )

        candidate_rows.append(
            ConvergenceReportCandidate(
                label=label,
                order=observation.candidate.order,
                value=observation.candidate.value,
                total_energy_ev=observation.total_energy_ev,
                energy_ev_per_atom=observation.energy_ev_per_atom,
                adjacent_energy_difference_ev_per_atom=(
                    adjacent_comparison.energy_difference_ev_per_atom
                    if adjacent_comparison is not None
                    else None
                ),
                reference_energy_difference_ev_per_atom=(
                    reference_difference
                ),
                maximum_higher_cost_energy_difference_ev_per_atom=(
                    tail_comparison.maximum_energy_difference_ev_per_atom
                    if tail_comparison is not None
                    else None
                ),
                tail_stable=(
                    tail_comparison.within_energy_tolerance
                    if tail_comparison is not None
                    else None
                ),
                selected=(label == selected_label),
            )
        )

    if not observations:
        raise ValueError(
            "Cannot build a convergence report without observations."
        )

    first_observation = observations[0]

    return ConvergenceReport(
        parameter=study.parameter,
        energy_tolerance_ev_per_atom=(
            study.criterion.energy_tolerance_ev_per_atom
        ),
        structure_hash=first_observation.structure_hash,
        n_atoms=first_observation.n_atoms,
        reference_candidate_label=reference_label,
        selected_candidate_label=selected_label,
        energy_tolerance_satisfied=(
            tail_analysis.energy_tolerance_satisfied
        ),
        candidates=tuple(candidate_rows),
    )
