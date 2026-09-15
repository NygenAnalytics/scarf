"""Numerical screening transfer with rare cells and repeated, unequal captures."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from scarf.agent.ingest import ingest
from scarf.agent.orchestrator.rna_tuning import (
    screening_coverage,
    screening_sizes,
    uniform_screening_selection,
)
from scarf.agent.orchestrator.models import AutomatedWorkflowConfig
from scarf.agent.parameter_tuning.hvg import core_hvg_evidence
from scarf.datastore.datastore import DataStore
from scarf.storage.selections import read_stored_selection_indices
from tests.test_agent_ingest import _write_h5ad


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_cells", "rare_cells", "automatic_policy"),
    [(1800, 24, False), (12_000, 42, True)],
    ids=["scaled-coverage-failure-regression", "bounded-default-policy-reference"],
)
def test_rare_population_transfer_requires_enlargement_beyond_metadata_coverage(
    tmp_path: Path,
    n_cells: int,
    rare_cells: int,
    automatic_policy: bool,
) -> None:
    """Measure transfer and distinguish scaled coverage checks from the real policy."""
    rng = np.random.default_rng(4444)
    rare_start = n_cells - rare_cells
    split = rare_start // 2
    counts = rng.poisson(0.2, (n_cells, 90)).astype(np.uint16)
    counts[:split, :12] += rng.poisson(6.0, (split, 12)).astype(np.uint16)
    counts[split:rare_start, 12:24] += rng.poisson(
        6.0, (rare_start - split, 12)
    ).astype(np.uint16)
    counts[rare_start:, 24:36] += rng.poisson(6.0, (rare_cells, 12)).astype(np.uint16)
    names = (
        [f"COMMON_A_{i}" for i in range(12)]
        + [f"COMMON_B_{i}" for i in range(12)]
        + [f"RARE_{i}" for i in range(12)]
        + [f"BACKGROUND_{i}" for i in range(54)]
    )
    source, target = tmp_path / "reference.h5ad", tmp_path / "reference.zarr"
    _write_h5ad(
        source,
        counts,
        feature_types=[b"Gene Expression"] * 90,
        feature_names=[name.encode() for name in names],
    )
    outcome = ingest(path=source, zarrPath=target, directions={"matrixKey": "X"})
    assert outcome.status == "done", outcome.notes
    store = DataStore(
        str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        nthreads=1,
    )
    rows = np.arange(n_cells)
    donors = rows % 6
    captures = np.asarray(
        [
            f"donor{donor}:capture{int((row // 6) % 4 != 0)}"
            for row, donor in zip(rows, donors, strict=True)
        ]
    )
    store.cells.insert("donor", donors.astype(str), overwrite=True)
    store.cells.insert("capture", captures, overwrite=True)
    store.cells.insert(
        "known_population",
        np.where(rows >= rare_start, "rare", "common"),
        overwrite=True,
    )
    full = store.snapshot_cell_selection("I")
    if automatic_policy:
        sizes = screening_sizes(n_cells, AutomatedWorkflowConfig())
        assert sizes == (10_000, 12_000)
        sample_size = sizes[0]
        assert rare_cells / n_cells < 0.01
    else:
        # This scaled regression deliberately explores unsupported populations;
        # it is not a test of the production minimum of 10,000 screening cells.
        sample_size = 1500
        initial = uniform_screening_selection(store, full, size=180, seed=4444)
        medium = uniform_screening_selection(store, full, size=700, seed=4444)
        _, initial_concerns = screening_coverage(
            store, full, initial, ["donor", "capture", "known_population"]
        )
        assert initial_concerns
        _, observed_group_concerns = screening_coverage(
            store, full, medium, ["donor", "capture"]
        )
        assert not observed_group_concerns
        _, population_concerns = screening_coverage(
            store, full, medium, ["known_population"]
        )
        assert population_concerns
    screened = uniform_screening_selection(store, full, size=sample_size, seed=4444)
    coverage, concerns = screening_coverage(
        store, full, screened, ["donor", "capture", "known_population"]
    )
    assert not concerns
    rare = next(
        row for row in coverage["groups"]["known_population"] if row["value"] == "rare"
    )
    assert rare["populationCells"] == rare_cells and rare["screeningCells"] >= 20
    genes = core_hvg_evidence(store, assay="RNA", cells=full)["scarfDefault"]
    marker_genes = store.select_all_features(from_assay="RNA")
    observed = {}
    graph_artifacts = set()
    marker_artifacts = set()
    for scope, selection in (("sample", screened), ("full", full)):
        normalized = store.run_normalization(selection, features=genes)
        pca = store.run_pca(normalized, dims=21)
        ann = store.build_ann_index(
            pca, ann_metric="l2", ann_parallel=False, rand_state=4444
        )
        neighbors = store.query_neighbors(ann, coordinates=pca, k=11)
        graph = store.build_connectivity_map(
            neighbors, local_connectivity=1.0, bandwidth=1.5
        )
        graph_artifacts.add(graph)
        clusters = store.run_leiden_clustering(
            graph,
            resolution=0.5,
            backend="igraph",
            symmetric_graph=False,
            graph_upper_only=False,
            random_seed=4444,
        )
        indices = read_stored_selection_indices(
            store.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        labels = np.asarray(store.load_artifact(clusters)["values"][:])
        labels_rare, sizes = np.unique(
            labels[indices >= rare_start], return_counts=True
        )
        rare_label = labels_rare[np.argmax(sizes)]
        rare_cluster = labels == rare_label
        recall = float(
            (rare_cluster & (indices >= rare_start)).sum()
            / (indices >= rare_start).sum()
        )
        purity = float(
            (rare_cluster & (indices >= rare_start)).sum() / rare_cluster.sum()
        )
        assert recall >= 0.9 and purity >= 0.9
        assert len(np.unique(donors[indices[rare_cluster]])) == 6
        markers = store.run_marker_search(
            clusters, features=marker_genes, from_assay="RNA"
        )
        marker_artifacts.add(markers)
        table = store.get_markers(marker=markers)
        rare_markers = set(
            table.loc[table.group_id.astype(str) == str(rare_label)]
            .sort_values("score", ascending=False)
            .head(12)["feature_name"]
        )
        assert {"RARE_0", "RARE_1", "RARE_2"}.issubset(rare_markers)
        assert len({name for name in rare_markers if name.startswith("RARE_")}) >= 10
        observed[scope] = {
            "indices": indices,
            "labels": labels,
            "recall": recall,
            "purity": purity,
            "rareMarkers": rare_markers,
        }
    agreement = adjusted_rand_score(
        observed["full"]["labels"][observed["sample"]["indices"]],
        observed["sample"]["labels"],
    )
    assert agreement > 0.9
    np.testing.assert_array_equal(
        store.cells.fetch_all("I"), np.ones(n_cells, dtype=bool)
    )
    (tmp_path / "screening_reference.json").write_text(
        json.dumps(
            {
                "fullCells": n_cells,
                "rareFullCells": rare_cells,
                "screeningCells": sample_size,
                "automaticPolicy": automatic_policy,
                "sharedCellAdjustedRandIndex": agreement,
                "rareRecall": {
                    scope: value["recall"] for scope, value in observed.items()
                },
                "rarePurity": {
                    scope: value["purity"] for scope, value in observed.items()
                },
                "primaryGraphArtifacts": len(graph_artifacts),
                "markerArtifacts": len(marker_artifacts),
                "limitation": "Synthetic numerical transfer, not LLM decision agreement or an atlas runtime benchmark.",
            },
            sort_keys=True,
        )
    )
