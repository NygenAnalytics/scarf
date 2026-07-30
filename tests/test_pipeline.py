import inspect
from threading import Event

import pytest
from scipy.sparse import tril

from scarf.storage.artifacts import ArtifactRef


def test_basic_rna_pipeline_returns_only_named_artifact_refs(
    datastore_ephemeral,
) -> None:
    artifacts = datastore_ephemeral.pipeline.run(
        pipeline_id="basic_rna_analysis",
        filtering={},
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 100,
            "hvg_key_name": "pipeline_hvgs",
        },
        pca={"dims": 5, "n_centroids": 10},
        ann_index={"ann_m": 16},
        neighbors={"k": 3},
        connectivity={},
        umap={"n_epochs": 10, "label": "pipeline_umap"},
        leiden={
            0.5: {},
            1.0: {},
        },
        paris={"n_clusters": 3, "label": "pipeline_paris"},
        doublet_scoring=False,
        markers={
            "clusters": {"leiden": 1.0},
            "gene_batch_size": 100,
        },
    )

    assert list(artifacts) == [
        "cell_selection",
        "highly_variable_features",
        "normalized",
        "pca",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "embedding_initialization",
        "umap",
        "leiden_0.5",
        "leiden_1.0",
        "paris",
        "markers",
    ]
    assert all(isinstance(ref, ArtifactRef) for ref in artifacts.values())
    assert artifacts["leiden_0.5"] != artifacts["leiden_1.0"]
    assert all(
        datastore_ephemeral.inspect_artifact(ref).complete for ref in artifacts.values()
    )
    cluster_column = "RNA_leiden_1.0"
    edited = datastore_ephemeral.cells.fetch(
        cluster_column,
        key="I",
    ).copy()
    edited[0] = int(edited.max()) + 1
    datastore_ephemeral.cells.insert(
        cluster_column,
        edited,
        key="I",
        overwrite=True,
    )
    with pytest.raises(ValueError, match="cluster labels are stale"):
        datastore_ephemeral.get_markers(
            from_assay="RNA",
            cell_key="I",
            group_key=cluster_column,
        )


def test_pipeline_rejects_unknown_id(datastore_ephemeral) -> None:
    with pytest.raises(ValueError, match="basic_rna_analysis"):
        datastore_ephemeral.pipeline.run(pipeline_id="unknown")


def test_pipeline_requires_highly_variable_features(datastore_ephemeral) -> None:
    with pytest.raises(ValueError, match="highly_variable_features"):
        datastore_ephemeral.pipeline.run(
            highly_variable_features=False,
            umap=False,
            leiden={},
            paris=False,
            doublet_scoring=False,
            markers=False,
        )


def test_pipeline_selects_clusters_by_pca_silhouette(datastore_ephemeral) -> None:
    artifacts = datastore_ephemeral.pipeline.run(
        filtering={},
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 100,
            "hvg_key_name": "pipeline_silhouette_hvgs",
        },
        pca={"dims": 5, "n_centroids": 10},
        neighbors={"k": 3},
        umap=False,
        leiden={
            0.5: {},
            1.0: {},
        },
        paris={"n_clusters": 3, "label": "pipeline_silhouette_paris"},
        clustering_concurrency=2,
        doublet_scoring=False,
        markers={
            "gene_batch_size": 100,
        },
    )

    assert "selected_clusters" in artifacts
    assert artifacts["selected_clusters"] in {
        artifacts["leiden_0.5"],
        artifacts["leiden_1.0"],
        artifacts["paris"],
    }
    assert artifacts["markers"].kind == "marker_table"


def test_pipeline_rejects_invalid_clustering_concurrency(datastore_ephemeral) -> None:
    with pytest.raises(ValueError, match="clustering_concurrency"):
        datastore_ephemeral.pipeline.run(
            clustering_concurrency=0,
            umap=False,
            leiden={},
            paris=False,
            doublet_scoring=False,
            markers=False,
        )


def test_pipeline_reuses_leiden_without_recomputing(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    options = {
        "filtering": False,
        "cell_cycle_scoring": False,
        "highly_variable_features": {
            "top_n": 100,
            "hvg_key_name": "pipeline_cached_leiden_hvgs",
        },
        "pca": {"dims": 3, "n_centroids": 5},
        "neighbors": {"k": 3},
        "umap": False,
        "leiden": {0.5: {}},
        "paris": False,
        "doublet_scoring": False,
        "markers": False,
    }
    first = datastore.pipeline.run(**options)

    def fail_compute(*_args, **_kwargs):
        pytest.fail("A reusable Leiden artifact must not be recomputed")

    monkeypatch.setattr(
        type(datastore),
        "_compute_prepared_leiden",
        staticmethod(fail_compute),
    )
    second = datastore.pipeline.run(**options)

    assert second["leiden_0.5"] == first["leiden_0.5"]


def test_pipeline_computes_leiden_with_requested_graph_options(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    original_compute = type(datastore)._compute_prepared_leiden
    observed = False

    def capture_graph(prepared, graph):
        nonlocal observed
        observed = True
        assert prepared.symmetric_graph is True
        assert prepared.graph_upper_only is True
        assert tril(graph, k=-1).nnz == 0
        return original_compute(prepared, graph)

    monkeypatch.setattr(
        type(datastore),
        "_compute_prepared_leiden",
        staticmethod(capture_graph),
    )
    datastore.pipeline.run(
        filtering=False,
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 100,
            "hvg_key_name": "pipeline_graph_options_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        neighbors={"k": 3},
        umap=False,
        leiden={
            0.5: {
                "symmetric_graph": True,
                "graph_upper_only": True,
            }
        },
        paris=False,
        doublet_scoring=False,
        markers=False,
    )

    assert observed is True


def test_leiden_public_api_does_not_accept_precomputed_membership(
    datastore_ephemeral,
) -> None:
    parameters = inspect.signature(datastore_ephemeral.run_leiden_clustering).parameters
    assert "precomputed_membership" not in parameters


def test_pipeline_overlaps_paris_compute_but_serializes_leiden_publish(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    compute_started = Event()
    paris_started = Event()
    paris_finished = Event()
    original_compute = type(datastore)._compute_prepared_leiden
    original_paris = type(datastore).run_paris_clustering
    original_publish = type(datastore)._publish_prepared_leiden

    def observed_compute(prepared, graph):
        compute_started.set()
        assert paris_started.wait(timeout=5)
        return original_compute(prepared, graph)

    def observed_paris(self, *args, **kwargs):
        paris_started.set()
        assert compute_started.wait(timeout=5)
        result = original_paris(self, *args, **kwargs)
        paris_finished.set()
        return result

    def observed_publish(self, prepared, membership):
        assert paris_finished.is_set()
        return original_publish(self, prepared, membership)

    monkeypatch.setattr(
        type(datastore),
        "_compute_prepared_leiden",
        staticmethod(observed_compute),
    )
    monkeypatch.setattr(
        type(datastore),
        "run_paris_clustering",
        observed_paris,
    )
    monkeypatch.setattr(
        type(datastore),
        "_publish_prepared_leiden",
        observed_publish,
    )

    datastore.pipeline.run(
        filtering=False,
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 100,
            "hvg_key_name": "pipeline_concurrency_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        neighbors={"k": 3},
        umap=False,
        leiden={0.5: {}},
        paris={"n_clusters": 3, "label": "pipeline_concurrency_paris"},
        clustering_concurrency=2,
        doublet_scoring=False,
        markers=False,
    )

    assert paris_finished.is_set()


def test_pipeline_rejects_unsaved_markers_before_writes(
    datastore_ephemeral,
) -> None:
    before = set(datastore_ephemeral.cells.columns)

    with pytest.raises(ValueError, match="skip_save"):
        datastore_ephemeral.pipeline.run(
            markers={"skip_save": True},
        )

    assert set(datastore_ephemeral.cells.columns) == before


def test_basic_rna_pipeline_supports_optional_harmony(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    datastore.cells.insert(
        "pipeline_batch",
        ["a" if i % 2 else "b" for i in range(datastore.cells.N)],
        overwrite=True,
    )

    artifacts = datastore.pipeline.run(
        filtering={},
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 50,
            "hvg_key_name": "pipeline_harmony_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        harmony={
            "batch_columns": ["pipeline_batch"],
            "harmony_params": {"nclust": 3},
        },
        neighbors={"k": 3},
        umap=False,
        leiden={},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )

    state = datastore.get_assay_state("RNA")
    assert "harmony" in artifacts
    assert state is not None
    assert state.batch_correction == artifacts["harmony"]


def test_basic_rna_pipeline_runs_score_steps_after_clustering(
    datastore_ephemeral,
) -> None:
    artifacts = datastore_ephemeral.pipeline.run(
        filtering={},
        cell_cycle_scoring={},
        highly_variable_features={
            "top_n": 50,
            "hvg_key_name": "pipeline_full_hvgs",
        },
        pca={"dims": 3, "n_centroids": 5},
        neighbors={"k": 3},
        umap={"n_epochs": 5, "label": "pipeline_full_umap"},
        leiden={0.5: {}},
        paris={"n_clusters": 3, "label": "pipeline_full_paris"},
        doublet_scoring={
            "clusters": "paris",
            "cluster_sample_fraction": 0.01,
            "max_cells_per_cluster": 2,
            "simulation_ratio": 0.01,
            "save_k": 3,
            "smoothing_t": 1,
            "random_seed": 9,
        },
        markers={
            "clusters": {"leiden": 0.5},
            "gene_batch_size": 100,
        },
    )

    assert artifacts["cell_cycle"].kind == "cell_cycle"
    assert artifacts["doublets"].kind == "doublet_score"
    assert artifacts["markers"].kind == "marker_table"
    assert list(artifacts).index("doublets") > list(artifacts).index("paris")
