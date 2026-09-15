"""Matched numerical correction preserves biological programs and donor support."""

import json
from pathlib import Path

import numpy as np
import pytest

from scarf.agent.ingest import ingest
from scarf.agent.parameter_tuning.contracts import (
    ParameterCandidate,
    ParameterTuningDependencies,
)
from scarf.agent.parameter_tuning.diagnostics import augment_cluster_evaluations
from scarf.agent.parameter_tuning.execution import execute_parameter_candidate
from scarf.agent.parameter_tuning.hvg import core_hvg_evidence
from scarf.agent.parameter_tuning.selection import harmony_acceptance_gate
from scarf.agent.tools import core_artifact_reference
from scarf.datastore.datastore import DataStore
from scarf.metrics.association import coefficient_estimability
from tests.test_agent_ingest import _write_h5ad


@pytest.mark.slow
@pytest.mark.parametrize(
    ("batch_effect", "expected_benefit"), [(2.0, False), (0.5, True)]
)
def test_real_harmony_comparison_measures_batch_and_biological_preservation(
    tmp_path: Path,
    batch_effect: float,
    expected_benefit: bool,
) -> None:
    """A crossed technical effect is evaluated, with missing doublets still blocking."""
    rng = np.random.default_rng(4444)
    n_cells, rare_start = 1200, 1152
    rows = np.arange(n_cells)
    donor = rows % 6
    batch = (rows // 6) % 2
    labels = np.where(rows >= rare_start, "rare", np.where(rows < 576, "A", "B"))
    design_rows = np.unique(
        np.column_stack([donor.astype(str), batch.astype(str), labels]), axis=0
    )
    design = coefficient_estimability(
        design_rows[:, 2],
        coefficientKind="categorical",
        technicals={"batch": design_rows[:, 1]},
        technicalKinds={"batch": "categorical"},
    )
    assert design["coefficientEstimable"]
    confounded = coefficient_estimability(
        design_rows[:, 2],
        coefficientKind="categorical",
        technicals={"batch": design_rows[:, 2]},
        technicalKinds={"batch": "categorical"},
    )
    assert not confounded["coefficientEstimable"]
    counts = rng.poisson(0.2, (n_cells, 90)).astype(np.uint16)
    for mask, columns in (
        (labels == "A", slice(0, 12)),
        (labels == "B", slice(12, 24)),
        (labels == "rare", slice(24, 36)),
    ):
        counts[mask, columns] += rng.poisson(6.0, (int(mask.sum()), 12)).astype(
            np.uint16
        )
    for value, columns in ((0, slice(36, 48)), (1, slice(48, 60))):
        mask = batch == value
        counts[mask, columns] += rng.poisson(
            batch_effect, (int(mask.sum()), 12)
        ).astype(np.uint16)
    names = (
        [f"A_{i}" for i in range(12)]
        + [f"B_{i}" for i in range(12)]
        + [f"RARE_{i}" for i in range(12)]
        + [f"TECHNICAL_{i}" for i in range(24)]
        + [f"BACKGROUND_{i}" for i in range(30)]
    )
    source, target = tmp_path / "crossed.h5ad", tmp_path / "crossed.zarr"
    _write_h5ad(
        source,
        counts,
        feature_types=[b"Gene Expression"] * 90,
        feature_names=[name.encode() for name in names],
    )
    result = ingest(path=source, zarrPath=target, directions={"matrixKey": "X"})
    assert result.status == "done", result.notes
    store = DataStore(
        str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        nthreads=1,
    )
    for column, values in (
        ("donor", donor.astype(str)),
        ("batch", batch.astype(str)),
        ("population", labels),
    ):
        store.cells.insert(column, values, overwrite=True)
    cells = store.snapshot_cell_selection("I")
    genes = core_hvg_evidence(store, assay="RNA", cells=cells)["scarfDefault"]
    marker_genes = store.select_all_features(from_assay="RNA")
    normalized = store.run_normalization(cells, features=genes)
    candidates = {
        name: ParameterCandidate(
            candidateId=name,
            dimensions=21,
            neighborsK=21,
            leidenResolution=0.5,
            useHarmony=corrected,
        )
        for name, corrected in (("native", False), ("harmony", True))
    }
    deps = ParameterTuningDependencies(
        store=store,
        normalized=normalized,
        cellSelection=cells,
        fromAssay="RNA",
        candidates=candidates,
        batchColumns=("batch",),
        preservationColumns=("population",),
        columnKinds={"batch": "categorical", "population": "categorical"},
        harmonyAuthorized=True,
        maxCandidates=2,
        minClusterCells=10,
    )
    measured = [execute_parameter_candidate(deps, name) for name in candidates]
    assert all(row.status == "done" and row.eligible for row in measured), measured
    native, corrected = augment_cluster_evaluations(
        store,
        measured,
        marker_assay="RNA",
        marker_features=marker_genes,
        independent_unit_columns=("donor",),
        technical_columns=("batch",),
    )
    improvement = (
        corrected.metrics.batchMixing["batch"] - native.metrics.batchMixing["batch"]
    )
    assert (improvement > 0.05) == expected_benefit, improvement
    for metric, baseline in native.metrics.biologicalPreservation["population"].items():
        assert (
            corrected.metrics.biologicalPreservation["population"][metric]
            >= baseline - 0.05
        )
    for row in (native, corrected):
        clusters = np.asarray(
            store.load_artifact(core_artifact_reference(row.artifacts["clusters"]))[
                "values"
            ][:]
        )
        rare_labels, sizes = np.unique(clusters[rare_start:], return_counts=True)
        population = rare_labels[np.argmax(sizes)]
        members = clusters == population
        assert (members & (rows >= rare_start)).sum() / members.sum() >= 0.9
        assert len(np.unique(donor[members])) == 6
        assert (
            sum(
                name.startswith("RARE_")
                for name in row.metrics.topMarkerGenes[str(population)]
            )
            >= 3
        )
    accepted, reasons = harmony_acceptance_gate(
        native,
        corrected,
        batch_columns=("batch",),
        protected_columns=("population",),
        independent_unit_columns=("donor",),
        require_doublet_evidence=True,
    )
    assert not accepted
    assert "Matched doublet-concentration comparison is missing." in reasons
    other_checks, other_reasons = harmony_acceptance_gate(
        native,
        corrected,
        batch_columns=("batch",),
        protected_columns=("population",),
        independent_unit_columns=("donor",),
    )
    assert other_checks == expected_benefit, other_reasons
    if not expected_benefit:
        assert (
            "Harmony did not improve an approved batch metric beyond tolerance."
            in reasons
        )
    (tmp_path / "harmony_reference.json").write_text(
        json.dumps(
            {
                "cells": n_cells,
                "rareCells": n_cells - rare_start,
                "nuisanceCountRate": batch_effect,
                "batchMixingImprovement": improvement,
                "batchMixing": {
                    row.candidateId: row.metrics.batchMixing
                    for row in (native, corrected)
                },
                "preservation": {
                    row.candidateId: row.metrics.biologicalPreservation
                    for row in (native, corrected)
                },
                "crossUnitSupport": {
                    row.candidateId: row.metrics.crossUnitSupport
                    for row in (native, corrected)
                },
                "unresolvedAcceptance": reasons,
                "primaryEvaluations": len(measured),
                "limitation": "Synthetic measured correction, not model decision agreement or full workflow acceptance.",
            },
            sort_keys=True,
        )
    )
