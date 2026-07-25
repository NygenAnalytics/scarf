import numpy as np
import pytest

from scarf.graph.state import stored_assay_graph_from_ref
from scarf.storage.artifacts import ArtifactRef, artifact_path
from tests import full_path


def _prepare_atomic_features(datastore) -> None:
    datastore.auto_filter_cells(show_qc_plots=False)
    if "I__atomic_hvgs" not in datastore.get_assay("RNA").feats.columns:
        datastore.mark_hvgs(
            from_assay="RNA",
            cell_key="I",
            top_n=100,
            hvg_key_name="atomic_hvgs",
            show_plot=False,
        )


def test_atomic_graph_methods_chain_refs_and_publish_current_results(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)

    normalized = datastore.run_normalization(
        from_assay="RNA",
        cell_key="I",
        feat_key="atomic_hvgs",
    )
    pca = datastore.run_pca(normalized, dims=5, batch_size=100)
    ann = datastore.build_ann_index(pca, batch_size=100)
    neighbors = datastore.query_neighbors(ann, k=3, batch_size=100)
    graph = datastore.build_connectivity_map(neighbors, batch_size=100)

    assert all(
        isinstance(ref, ArtifactRef) for ref in (normalized, pca, ann, neighbors, graph)
    )
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.normalized == normalized
    assert state.reduction == pca
    assert state.ann_index == ann
    assert state.neighbors == neighbors
    assert state.connectivity_map == graph
    loaded = datastore.load_graph(graph_loc=artifact_path(graph))
    assert loaded.shape[0] == int(datastore.cells.fetch_all("I").sum())
    assert np.isfinite(loaded.data).all()


def test_new_reduction_becomes_current_and_clears_downstream_refs(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
    )
    first = datastore.run_pca(normalized, dims=4)
    ann = datastore.build_ann_index(first)
    neighbors = datastore.query_neighbors(ann, k=3)

    assert datastore.run_pca(normalized, dims=4) == first
    reused_state = datastore.get_assay_state("RNA")
    assert reused_state is not None
    assert reused_state.ann_index == ann
    assert reused_state.neighbors == neighbors

    second = datastore.run_pca(normalized, dims=6)
    state = datastore.get_assay_state("RNA")

    assert state is not None
    assert state.reduction == second
    assert state.ann_index is None
    assert state.neighbors is None
    assert state.connectivity_map is None


def test_neighbor_count_changes_only_neighbor_and_connectivity_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
        update_state=False,
    )
    reduction = datastore.run_pca(
        normalized,
        dims=4,
        update_state=False,
    )
    ann = datastore.build_ann_index(
        reduction,
        update_state=False,
    )
    neighbors_three = datastore.query_neighbors(
        ann,
        k=3,
        update_state=False,
    )
    connectivity_three = datastore.build_connectivity_map(
        neighbors_three,
        update_state=False,
    )
    neighbors_four = datastore.query_neighbors(
        ann,
        k=4,
        update_state=False,
    )
    connectivity_four = datastore.build_connectivity_map(
        neighbors_four,
        update_state=False,
    )

    assert (
        datastore.run_normalization(
            from_assay="RNA",
            feat_key="atomic_hvgs",
            update_state=False,
        )
        == normalized
    )
    assert (
        datastore.run_pca(
            normalized,
            dims=4,
            update_state=False,
        )
        == reduction
    )
    assert (
        datastore.build_ann_index(
            reduction,
            update_state=False,
        )
        == ann
    )
    assert neighbors_three != neighbors_four
    assert connectivity_three != connectivity_four


def test_update_state_false_keeps_current_selection(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
    )
    current = datastore.run_pca(normalized, dims=4)
    detached = datastore.run_pca(
        normalized,
        dims=6,
        update_state=False,
    )

    assert detached != current
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction == current

    ann = datastore.build_ann_index(detached)
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction == detached
    assert state.ann_index == ann


def test_reused_normalization_publishes_requested_selection_aliases(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    source_mask = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    datastore.cells.insert(
        "normalization_source",
        source_mask,
        overwrite=True,
    )
    datastore.cells.insert(
        "normalization_alias",
        source_mask,
        overwrite=True,
    )
    feature_mask = np.asarray(
        datastore.RNA.feats.fetch_all("I__atomic_hvgs"),
        dtype=bool,
    )
    datastore.RNA.feats.insert(
        "normalization_source__atomic_hvgs",
        feature_mask,
        overwrite=True,
    )
    datastore.RNA.feats.insert(
        "normalization_alias__atomic_hvgs",
        feature_mask,
        overwrite=True,
    )
    original = datastore.run_normalization(
        from_assay="RNA",
        cell_key="normalization_source",
        feat_key="atomic_hvgs",
        update_state=False,
    )
    reused = datastore.run_normalization(
        from_assay="RNA",
        cell_key="normalization_alias",
        feat_key="atomic_hvgs",
    )

    assert reused == original
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.cell_key == "normalization_alias"
    assert state.feat_key == "atomic_hvgs"

    reduction = datastore.run_pca(reused, dims=4)
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction == reduction
    assert state.cell_key == "normalization_alias"
    assert state.feat_key == "atomic_hvgs"


def test_explicit_graph_preserves_prefixed_logical_feature_key(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    datastore.RNA.feats.insert(
        "I__I__prefixed",
        np.asarray(
            datastore.RNA.feats.fetch_all("I__atomic_hvgs"),
            dtype=bool,
        ),
        overwrite=True,
    )
    normalized = datastore.run_normalization(
        from_assay="RNA",
        cell_key="I",
        feat_key="I__prefixed",
        update_state=False,
    )
    reduction = datastore.run_pca(normalized, dims=3, update_state=False)
    ann = datastore.build_ann_index(reduction, update_state=False)
    neighbors = datastore.query_neighbors(ann, k=3, update_state=False)
    connectivity = datastore.build_connectivity_map(
        neighbors,
        update_state=False,
    )
    stored = stored_assay_graph_from_ref(datastore.zw, connectivity)

    assert stored.cell_key == "I"
    assert stored.feat_key == "I__prefixed"


def test_historical_neighbors_use_normalized_execution_keys(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    mask = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    datastore.cells.insert("selection_a", mask, overwrite=True)
    datastore.cells.insert("selection_b", mask, overwrite=True)
    datastore._ensure_cell_selection("selection_a")
    feature_mask = np.asarray(
        datastore.RNA.feats.fetch_all("I__atomic_hvgs"),
        dtype=bool,
    )
    datastore.RNA.feats.insert(
        "selection_b__atomic_hvgs",
        feature_mask,
        overwrite=True,
    )

    normalized = datastore.run_normalization(
        from_assay="RNA",
        cell_key="selection_b",
        feat_key="atomic_hvgs",
    )
    reduction = datastore.run_pca(normalized, dims=4)
    ann = datastore.build_ann_index(reduction)
    neighbors = datastore.query_neighbors(ann, k=3)
    datastore.run_normalization(
        from_assay="RNA",
        cell_key="I",
        feat_key="atomic_hvgs",
    )

    stale = mask.copy()
    selected = np.flatnonzero(stale)
    excluded = np.flatnonzero(~stale)
    assert len(selected) > 0 and len(excluded) > 0
    stale[selected[0]] = False
    stale[excluded[0]] = True
    datastore.cells.insert("selection_a", stale, overwrite=True, force=True)

    assert datastore._keys_from_knn_path(
        "RNA",
        artifact_path(neighbors),
    ) == ("selection_b", "atomic_hvgs")


def test_reduction_and_harmony_reject_changed_normalized_selection(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
    )
    reduction = datastore.run_pca(normalized, dims=4)
    datastore.cells.insert(
        "atomic_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    mask = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    selected = np.flatnonzero(mask)
    excluded = np.flatnonzero(~mask)
    assert len(selected) > 0 and len(excluded) > 0
    mask[selected[0]] = False
    mask[excluded[0]] = True
    datastore.cells.insert("I", mask, overwrite=True, force=True)

    with pytest.raises(ValueError, match="no longer matches"):
        datastore.run_pca(
            normalized,
            dims=5,
            invalidate_cache=True,
        )
    with pytest.raises(ValueError, match="no longer matches"):
        datastore.run_harmony(
            ["atomic_batch"],
            reduction,
            harmony_params={"nclust": 5},
            invalidate_cache=True,
        )


def test_datastore_inspects_and_loads_artifact_read_only(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    ref = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
    )

    status = datastore.inspect_artifact(ref)
    group = datastore.load_artifact(ref)

    assert status.complete
    assert status.operation == "run_normalization"
    assert status.parameters is not None
    assert "data" in group
    with pytest.raises(ValueError):
        group.attrs["invalid"] = True


def test_atomic_harmony_becomes_default_ann_coordinates(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    batches = np.where(np.arange(datastore.cells.N) % 2, "a", "b")
    datastore.cells.insert("atomic_batch", batches, overwrite=True)
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
    )
    pca = datastore.run_pca(normalized, dims=5)

    corrected = datastore.run_harmony(
        ["atomic_batch"],
        pca,
        harmony_params={"nclust": 5},
    )
    ann = datastore.build_ann_index(batch_size=100)

    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.batch_correction == corrected
    assert state.ann_index == ann
    ann_inputs = datastore.inspect_artifact(ann).inputs
    assert ann_inputs is not None
    assert ann_inputs["coordinates"] == corrected.to_dict()


def test_lsi_and_custom_reduction_have_distinct_public_methods(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
    )
    lsi = datastore.run_lsi(normalized, dims=3)
    n_features = datastore.load_artifact(normalized)["data"].shape[1]
    loadings = np.eye(n_features, 2, dtype=np.float64)
    custom = datastore.run_custom_reduction(
        loadings,
        normalized,
        update_state=False,
    )

    assert datastore.inspect_artifact(lsi).operation == "run_lsi"
    assert datastore.inspect_artifact(custom).operation == "run_custom_reduction"
    ann = datastore.build_ann_index(custom, update_state=False)
    neighbors = datastore.query_neighbors(ann, k=3, update_state=False)
    connectivity = datastore.build_connectivity_map(
        neighbors,
        update_state=False,
    )
    stored = stored_assay_graph_from_ref(datastore.zw, connectivity)
    assert stored.reduction_method == "custom"
    assert stored.dims == 2
    assert stored.pca_cell_key is None
    assert stored.feat_scaling is False


def test_atomic_chain_matches_released_knn_golden(
    make_graph,
    datastore_ephemeral,
) -> None:
    del make_graph
    datastore = datastore_ephemeral
    if "I__hvgs" not in datastore.get_assay("RNA").feats.columns:
        datastore.auto_filter_cells(show_qc_plots=False)
        datastore.mark_hvgs(
            from_assay="RNA",
            cell_key="I",
            top_n=100,
            hvg_key_name="hvgs",
            show_plot=False,
            bin_strategy="fixed",
        )
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="hvgs",
        update_state=False,
        invalidate_cache=True,
    )
    reduction = datastore.run_pca(
        normalized,
        dims=11,
        update_state=False,
        invalidate_cache=True,
    )
    ann = datastore.build_ann_index(
        reduction,
        update_state=False,
        invalidate_cache=True,
    )
    neighbors = datastore.query_neighbors(
        ann,
        coordinates=reduction,
        k=11,
        update_state=False,
        invalidate_cache=True,
    )
    group = datastore.load_artifact(neighbors)

    np.testing.assert_array_equal(
        group["indices"][:],
        np.load(full_path("knn_indices.npy")),
    )
    np.testing.assert_allclose(
        group["distances"][:],
        np.load(full_path("knn_distances.npy")),
        rtol=0,
        atol=1e-3,
    )


def test_corrupt_ann_bytes_are_not_reused(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_atomic_features(datastore)
    normalized = datastore.run_normalization(
        from_assay="RNA",
        feat_key="atomic_hvgs",
    )
    reduction = datastore.run_pca(normalized, dims=3)
    first = datastore.build_ann_index(reduction)
    ann_group = datastore.zw[artifact_path(first)]
    ann_group["ann_idx_bytes"][:] = 0

    repaired = datastore.build_ann_index(reduction)
    reused = datastore.build_ann_index(reduction)

    assert repaired != first
    assert reused == repaired
    assert datastore.inspect_artifact(repaired).complete
