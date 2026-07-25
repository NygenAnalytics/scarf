import pytest

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
