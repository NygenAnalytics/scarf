import numpy as np
import pytest

from scarf import ArtifactRef
from scarf.storage.artifacts import parse_artifact_path
from scarf.utils import configure_output
from tests.fixtures_datastore import _has_graph, build_neighbourhood_graph

pytestmark = pytest.mark.slow


def _active_cell_count(datastore) -> int:
    return len(datastore.cells.active_index("I"))


def _ensure_graph(datastore):
    assert _has_graph(datastore), "analyzed datastore fixture has no RNA/I/hvgs graph"


def _clear_umap_columns(datastore):
    for column in ("RNA_UMAP1", "RNA_UMAP2"):
        if column in datastore.cells.columns:
            datastore.cells.drop(column)


def _clear_projection(datastore, name: str):
    artifact_path = datastore._projection_artifact_path("RNA", name)
    if artifact_path is not None:
        del datastore.zw[artifact_path]
        projections = datastore.z["RNA"].get("projections")
        if projections is not None:
            artifacts = dict(projections.attrs.get("artifacts", {}))
            artifacts.pop(name, None)
            projections.attrs["artifacts"] = artifacts
        return
    projections = datastore.z["RNA"].get("projections")
    if projections is not None and name in projections:
        del projections[name]


def _projection_group(datastore, name: str):
    return datastore._load_complete_projection(name, "RNA", "I")


def test_run_umap_recomputes_coordinates(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    _clear_umap_columns(ds)

    ds.run_umap(n_epochs=10, parallel=False)

    umap1 = ds.cells.fetch("RNA_UMAP1")
    umap2 = ds.cells.fetch("RNA_UMAP2")
    assert len(umap1) == _active_cell_count(ds)
    assert len(umap2) == _active_cell_count(ds)


def test_run_umap_uses_explicit_progress_for_verbose_output(
    analyzed_datastore_ephemeral,
    monkeypatch,
):
    import scarf.embeddings.umap as umap_module

    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    observed: list[bool] = []

    def fake_fit_transform(*, graph, verbose, **_kwargs):
        observed.append(verbose)
        return np.zeros((graph.shape[0], 2)), None, None

    monkeypatch.setattr(umap_module, "fit_transform", fake_fit_transform)
    try:
        configure_output(progress=False)
        ds.run_umap(
            n_epochs=10,
            label="progress_off_umap",
            invalidate_cache=True,
        )
        configure_output(progress=True)
        ds.run_umap(
            n_epochs=10,
            label="progress_on_umap",
            invalidate_cache=True,
        )
    finally:
        configure_output(progress=False)

    assert observed == [False, True]


def test_run_leiden_writes_cluster_labels(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)

    ds.run_leiden_clustering(label="ephemeral_leiden")

    groups = ds.cells.fetch("RNA_ephemeral_leiden", key="I")
    assert len(groups) == _active_cell_count(ds)
    assert np.unique(groups).size >= 1


def test_run_mapping_writes_projection(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "freshmap")

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="freshmap",
        target_feat_key="hvgs_freshmap",
        save_k=3,
    )

    assert _projection_group(ds, "freshmap")["indices"].shape[0] == _active_cell_count(
        ds
    )
    loaded = ds.get_mapping_result("freshmap", load_arrays=True)
    assert loaded.indices is not None
    assert loaded.distances is not None
    assert loaded.corrected_latent is None
    assert loaded.uncorrected_latent is None


def test_mapping_projection_reuses_and_explicitly_invalidates(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "projection_reuse")
    kwargs = {
        "target_assay": ds.RNA,
        "target_name": "projection_reuse",
        "target_feat_key": "hvgs_projection_reuse",
        "save_k": 3,
    }

    ds.run_mapping(**kwargs)
    first = ds._projection_artifact_path("RNA", "projection_reuse")
    ds.run_mapping(**kwargs)
    assert ds._projection_artifact_path("RNA", "projection_reuse") == first

    ds.run_mapping(**kwargs, invalidate_cache=True)
    assert ds._projection_artifact_path("RNA", "projection_reuse") != first


def test_run_mapping_supports_all_features_key(datastore_ephemeral):
    ds = datastore_ephemeral
    ds.auto_filter_cells(show_qc_plots=False)
    build_neighbourhood_graph(ds, feat_key="I", dims=5, k=3, n_centroids=10)

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="all_features_map",
        target_feat_key="all_features_map_target",
        save_k=3,
    )

    assert _projection_group(ds, "all_features_map").attrs["complete"]


def test_run_mapping_with_coral_writes_corrected_data(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "freshmap_coral")

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="freshmap_coral",
        target_feat_key="hvgs_fresh_coral",
        save_k=3,
        run_coral=True,
    )

    normed_loc = "normed__I__hvgs_fresh_coral"
    assert "data_coral" in ds.RNA.z[normed_loc]
    assert _projection_group(ds, "freshmap_coral").attrs["complete"]


def test_coral_mapping_cache_and_rebuild_are_equivalent(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="coral_cached",
        target_feat_key="hvgs_coral_cached",
        save_k=3,
        run_coral=True,
    )
    cached = _projection_group(ds, "coral_cached")
    cached_indices = cached["indices"][:]
    cached_distances = cached["distances"][:]

    state = ds.get_assay_state("RNA")
    assert state is not None and state.ann_index is not None
    ann = ds.zw[ds.inspect_artifact(state.ann_index).path]
    del ann["ann_idx_bytes"]
    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="coral_rebuilt",
        target_feat_key="hvgs_coral_rebuilt",
        save_k=3,
        run_coral=True,
    )
    rebuilt = _projection_group(ds, "coral_rebuilt")

    assert np.array_equal(cached_indices, rebuilt["indices"][:])
    np.testing.assert_allclose(cached_distances, rebuilt["distances"][:])


def test_run_unified_umap_after_mapping(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "freshmap_unified_src")
    _clear_projection(ds, "unified_UMAP")

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="freshmap_unified_src",
        target_feat_key="hvgs_unified_src",
        save_k=3,
    )
    ds.run_unified_umap(target_names=["freshmap_unified_src"], n_epochs=10)

    x, y, _reference_count, _target_counts, _target_names = (
        ds._load_unified_layout_data("unified_UMAP", "RNA")
    )
    coords = np.column_stack((x, y))
    assert coords.shape[1] == 2
    assert coords.shape[0] >= _active_cell_count(ds)


def test_build_and_reload_symphony_mapping_reference(
    analyzed_datastore_ephemeral, tmp_path
):
    import numpy as np
    import pandas as pd

    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    active_batches = ds.cells.fetch("mapping_batch", key="I").copy()

    reference = ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
        harmony_params={"nclust": 5},
    )
    state = ds.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    assert state.batch_correction is not None
    assert state.ann_index is not None
    reduction = ds.zw[ds.inspect_artifact(state.reduction).path]
    correction = ds.zw[ds.inspect_artifact(state.batch_correction).path]
    reference_coordinates = correction["data"][:].copy()
    reference_loadings = reduction["loadings"][:].copy()
    reference_ann = ds.zw[ds.inspect_artifact(state.ann_index).path]
    reference_ann_metadata = dict(reference_ann.attrs)
    correction_parameters = ds.inspect_artifact(state.batch_correction).parameters
    assert correction_parameters is not None
    assert correction_parameters["harmony_parameters"]["nclust"] == 5
    assert ds.get_assay_state("RNA") == state
    loaded = ds.get_mapping_reference(feat_key="hvgs")
    with pytest.raises(ValueError, match="matches the reference path"):
        loaded.map_query(
            ds.RNA,
            "unsafe_symphony_self",
            "hvgs",
            save_k=3,
        )
    result = loaded.map_query(
        ds.RNA,
        "symphony_self",
        "hvgs_symphony_self",
        save_k=3,
        query_batches=pd.DataFrame({"mapping_batch": active_batches}),
    )
    del ds.RNA.z["normed__I__hvgs_symphony_self"]
    reused = loaded.map_query(
        ds.RNA,
        "symphony_self",
        "hvgs_symphony_self",
        save_k=3,
        query_batches=pd.DataFrame({"mapping_batch": active_batches}),
    )
    assert reused.projection_path == result.projection_path

    assert reference.model.n_dims == loaded.model.n_dims
    assert reference.metadata["harmony_parameters"]["nclust"] == 5
    assert result.n_cells == _active_cell_count(ds)
    projection = _projection_group(ds, "symphony_self")
    assert projection.attrs["complete"]
    assert projection["corrected_latent"].shape[1] == loaded.model.n_dims
    assert projection["indices"].chunks[1] == 3
    assert projection["distances"].chunks[1] == 3
    assert projection.attrs["query_batch_columns"] == ["mapping_batch"]
    assert isinstance(projection.attrs["query_batch_fingerprint"], str)
    original_indices = projection["indices"][:].copy()
    original_corrected = projection["corrected_latent"][:].copy()

    meta = ds.get_mapping_result("symphony_self")
    assert meta.n_cells == result.n_cells
    assert meta.correction_method == "symphony"
    assert meta.projection_path == ds._projection_artifact_path(
        "RNA",
        "symphony_self",
    )
    assert meta.indices is None
    assert meta.corrected_latent is None
    assert "featureCoverage" in meta.diagnostics

    loaded_result = ds.get_mapping_result("symphony_self", load_arrays=True)
    np.testing.assert_array_equal(loaded_result.indices, original_indices)
    np.testing.assert_allclose(loaded_result.corrected_latent, original_corrected)
    assert loaded_result.uncorrected_latent is not None
    assert loaded_result.uncorrected_latent.shape == original_corrected.shape
    np.testing.assert_array_equal(correction["data"][:], reference_coordinates)
    np.testing.assert_array_equal(reduction["loadings"][:], reference_loadings)
    assert dict(reference_ann.attrs) == reference_ann_metadata
    artifact = ds.z[loaded.artifact_path]
    artifact.attrs["complete"] = False
    try:
        with pytest.raises(ValueError, match="incomplete"):
            next(ds.get_mapping_score("symphony_self"))
    finally:
        artifact.attrs["complete"] = True

    from scarf.datastore.datastore import DataStore

    read_only = DataStore(ds.zarr_loc, default_assay="RNA", zarr_mode="r")
    in_memory = read_only.get_mapping_reference(feat_key="hvgs").map_query(
        ds.RNA,
        "symphony_read_only",
        "hvgs_symphony_read_only",
        save_k=3,
        query_batches=pd.DataFrame({"mapping_batch": active_batches}),
    )
    assert in_memory.projection_path == ""
    assert in_memory.indices is not None
    assert in_memory.corrected_latent is not None
    persisted_indices = projection["indices"][:]
    overlap = np.mean(
        [
            len(set(expected) & set(observed)) / len(expected)
            for expected, observed in zip(persisted_indices, in_memory.indices)
        ]
    )
    assert overlap >= 0.99
    np.testing.assert_allclose(
        projection["corrected_latent"][:],
        in_memory.corrected_latent,
        rtol=1e-10,
        atol=1e-12,
    )

    read_only_single_thread = DataStore(
        ds.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
        nthreads=1,
    )
    single_thread = read_only_single_thread.get_mapping_reference(
        feat_key="hvgs"
    ).map_query(
        ds.RNA,
        "symphony_single_thread",
        "hvgs_symphony_single_thread",
        save_k=3,
        query_batches=pd.DataFrame({"mapping_batch": active_batches}),
    )
    assert single_thread.corrected_latent is not None
    np.testing.assert_allclose(
        in_memory.corrected_latent,
        single_thread.corrected_latent,
        rtol=1e-10,
        atol=1e-12,
    )

    import zarr

    result_store = zarr.open_group(
        str(tmp_path / "mapping_result.zarr"), mode="w"
    ).create_group("projection")
    streamed = read_only.get_mapping_reference(feat_key="hvgs").map_query(
        ds.RNA,
        "symphony_streamed",
        "hvgs_symphony_streamed",
        save_k=3,
        query_batches=pd.DataFrame({"mapping_batch": active_batches}),
        result_store=result_store,
    )
    assert streamed.indices is None
    assert result_store["indices"].shape[0] == _active_cell_count(ds)

    ds.cells.insert(
        "mapping_batch",
        np.repeat("changed", ds.cells.N),
        overwrite=True,
    )
    assert ds.get_mapping_reference(feat_key="hvgs").artifact_path == (
        reference.artifact_path
    )

    artifact_count = len(ds.list_artifacts(kind="mapping_reference", from_assay="RNA"))
    replacement_batches = np.where(np.arange(ds.cells.N) % 2 == 0, "c", "d")
    ds.cells.insert(
        "mapping_batch",
        replacement_batches,
        overwrite=True,
    )
    rebuilt_reference = ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
        harmony_params={"nclust": 5},
    )
    assert rebuilt_reference.artifact_path != reference.artifact_path
    assert rebuilt_reference.ann_path != reference.ann_path
    assert (
        len(ds.list_artifacts(kind="mapping_reference", from_assay="RNA"))
        == artifact_count + 1
    )
    old_reference_result = reference.map_query(
        ds.RNA,
        "old_reference_replay",
        "hvgs_old_reference_replay",
        save_k=3,
        query_batches=pd.DataFrame({"mapping_batch": active_batches}),
    )
    old_projection = _projection_group(ds, "old_reference_replay")
    np.testing.assert_array_equal(old_projection["indices"][:], original_indices)
    np.testing.assert_allclose(
        old_projection["corrected_latent"][:],
        original_corrected,
        rtol=1e-10,
        atol=1e-12,
    )
    assert old_reference_result.projection_path


def test_mapping_rejects_reference_normalization_overwrite(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    state = ds.get_assay_state("RNA")
    assert state is not None
    assert state.normalized is not None
    assert state.reduction is not None
    assert state.ann_index is not None
    normed = ds.zw[ds.inspect_artifact(state.normalized).path]
    reduction = ds.zw[ds.inspect_artifact(state.reduction).path]
    ann = ds.zw[ds.inspect_artifact(state.ann_index).path]
    reference_data = normed["data"][:].copy()
    reference_loadings = reduction["loadings"][:].copy()
    ann_metadata = dict(ann.attrs)

    with pytest.raises(ValueError, match="matches the reference path"):
        ds.run_mapping(
            target_assay=ds.RNA,
            target_name="unsafe_self_map",
            target_feat_key="hvgs",
            save_k=3,
        )

    from scarf.datastore.datastore import DataStore

    separately_opened = DataStore(ds.zarr_loc, default_assay="RNA")
    with pytest.raises(ValueError, match="matches the reference path"):
        ds.run_mapping(
            target_assay=separately_opened.RNA,
            target_name="unsafe_separate_self_map",
            target_feat_key="hvgs",
            save_k=3,
        )

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="safe_self_map",
        target_feat_key="hvgs_safe_self_map",
        save_k=3,
    )

    np.testing.assert_array_equal(normed["data"][:], reference_data)
    np.testing.assert_array_equal(reduction["loadings"][:], reference_loadings)
    assert dict(ann.attrs) == ann_metadata


def test_cached_harmony_can_rebuild_missing_mapping_artifact(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    state = ds.get_assay_state("RNA")
    assert state is not None
    first_harmony = state.batch_correction
    first_ref = state.named_results["mapping_reference"]
    artifact = ds.zw[ds.inspect_artifact(first_ref).path]
    for name in (
        "reference_distance_quantiles",
        "reference_distance_values",
    ):
        values = artifact[name][:]
        del artifact[name]
        with pytest.raises(ValueError, match="missing required arrays"):
            ds.get_mapping_reference(feat_key="hvgs")
        artifact.create_array(name, data=values)
    del artifact["sigma"]
    with pytest.raises(ValueError, match="missing required arrays"):
        ds.get_mapping_reference(feat_key="hvgs")

    rebuilt = ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
    )

    assert rebuilt.metadata["complete"]
    rebuilt_state = ds.get_assay_state("RNA")
    assert rebuilt_state is not None
    assert rebuilt_state.batch_correction == first_harmony
    assert rebuilt_state.named_results["mapping_reference"] != first_ref


def test_mapping_reference_update_keys_false_returns_detached_reference(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("detached_mapping_batch", batches, overwrite=True)
    initial_state = ds.get_assay_state("RNA")

    reference = ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["detached_mapping_batch"],
        k=3,
        n_centroids=10,
        update_keys=False,
    )

    assert reference.metadata["complete"]
    assert ds.get_assay_state("RNA") == initial_state


def test_mapping_reference_rebuilds_after_missing_artifact(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    first_state = ds.get_assay_state("RNA")
    assert first_state is not None
    first_ref = first_state.named_results["mapping_reference"]
    del ds.zw[ds.inspect_artifact(first_ref).path]

    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    rebuilt_state = ds.get_assay_state("RNA")
    assert rebuilt_state is not None
    assert rebuilt_state.named_results["mapping_reference"] != first_ref
    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="upgraded_harmonized_map",
        target_feat_key="hvgs_upgraded_harmonized",
        save_k=3,
    )

    assert _projection_group(ds, "upgraded_harmonized_map").attrs["complete"]


def test_mapping_reference_tracks_changed_batch_values(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    first = ds.get_assay_state("RNA")
    assert first is not None
    first_ref = first.named_results["mapping_reference"]

    changed = np.where(np.arange(ds.cells.N) % 3 == 0, "a", "b")
    ds.cells.insert("mapping_batch_changed", changed, overwrite=True)
    rebuilt = ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch_changed"],
        k=3,
        n_centroids=10,
    )
    second = ds.get_assay_state("RNA")
    assert second is not None
    assert second.named_results["mapping_reference"] != first_ref
    assert rebuilt.metadata["batch_columns"] == ["mapping_batch_changed"]


def test_deprecated_reference_scaling_flags_do_not_break_mapping(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)

    with pytest.warns(DeprecationWarning, match="ignored"):
        ds.run_mapping(
            target_assay=ds.RNA,
            target_name="deprecated_scaling_flags",
            target_feat_key="hvgs_deprecated_scaling",
            ref_mu=False,
            ref_sigma=False,
            save_k=3,
        )

    assert _projection_group(ds, "deprecated_scaling_flags").attrs["complete"]


def test_mapping_feature_scaling_uses_isolated_ann_cache(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    original_state = ds.get_assay_state("RNA")
    assert original_state is not None and original_state.ann_index is not None
    original_ann_path = ds.inspect_artifact(original_state.ann_index).path

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="unscaled_mapping",
        target_feat_key="hvgs_unscaled_mapping",
        feat_scaling=False,
        save_k=3,
    )

    projection = _projection_group(ds, "unscaled_mapping")
    mapping_ann_path = projection.attrs["ann_path"]
    assert mapping_ann_path != original_ann_path
    mapping_ann_ref = parse_artifact_path(mapping_ann_path)
    mapping_ann_inputs = ds.inspect_artifact(mapping_ann_ref).inputs
    assert mapping_ann_inputs is not None
    reduction_ref = ArtifactRef.from_dict(mapping_ann_inputs["coordinates"])
    reduction_parameters = ds.inspect_artifact(reduction_ref).parameters
    assert reduction_parameters is not None
    assert not reduction_parameters["feat_scaling"]
    assert ds.get_assay_state("RNA") == original_state


def test_mapping_reference_rebuilds_unscaled_state_with_pca_scaling(
    analyzed_datastore_ephemeral,
) -> None:
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    build_neighbourhood_graph(
        ds,
        feat_key="hvgs",
        feat_scaling=False,
        dims=5,
        k=3,
        n_centroids=10,
    )
    ds.cells.insert(
        "mapping_reference_batch",
        np.where(np.arange(ds.cells.N) % 2, "a", "b"),
        overwrite=True,
    )

    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_reference_batch"],
        dims=5,
        k=3,
        n_centroids=10,
    )

    state = ds.get_assay_state("RNA")
    assert state is not None and state.reduction is not None
    parameters = ds.inspect_artifact(state.reduction).parameters
    assert parameters is not None
    assert parameters["feat_scaling"] is True
    assert ds.inspect_artifact(state.reduction).operation == "run_pca"


def test_mapping_reference_identical_call_reuses_complete_chain(
    analyzed_datastore_ephemeral,
) -> None:
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    ds.cells.insert(
        "mapping_reuse_batch",
        np.where(np.arange(ds.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    kwargs = {
        "feat_key": "hvgs",
        "batch_columns": ["mapping_reuse_batch"],
        "dims": 5,
        "k": 3,
        "n_centroids": 10,
    }

    first_reference = ds.build_mapping_reference(**kwargs)
    first_state = ds.get_assay_state("RNA")
    second_reference = ds.build_mapping_reference(**kwargs)
    second_state = ds.get_assay_state("RNA")

    assert first_reference.artifact_path == second_reference.artifact_path
    assert second_state == first_state


def test_projection_uses_stored_feature_key_and_rejects_stale_provenance(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="provenance_map",
        target_feat_key="hvgs_provenance_map",
        save_k=3,
    )
    projection = _projection_group(ds, "provenance_map")
    build_neighbourhood_graph(
        ds,
        feat_key="I",
        dims=5,
        k=3,
        n_centroids=10,
    )
    score = next(ds.get_mapping_score("provenance_map"))[1]
    assert np.all(np.isfinite(score))

    original_fingerprint = projection.attrs["reduction_fingerprint"]
    projection.attrs["reduction_fingerprint"] = "stale"
    try:
        with pytest.raises(ValueError, match="changed reduction"):
            next(ds.get_mapping_score("provenance_map"))
    finally:
        projection.attrs["reduction_fingerprint"] = original_fingerprint


def test_mapping_missing_feature_policies_and_legacy_intersection(
    analyzed_datastore_ephemeral,
):
    from scarf.datastore.datastore import DataStore

    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    shared_ids = ds.RNA.feats.fetch("ids", key="I__hvgs")[: ds.assay2.feats.N]
    ds.assay2.feats.insert(
        "ids",
        shared_ids,
        overwrite=True,
        force=True,
    )
    mapped = DataStore(
        ds.zarr_loc,
        assay_types={"RNA": "RNA", "assay2": "RNA"},
        default_assay="RNA",
        min_features_per_cell=0,
        min_cells_per_feature=0,
    )

    with pytest.raises(ValueError, match="missing"):
        mapped.run_mapping(
            target_assay=mapped.assay2,
            target_name="missing_error",
            target_feat_key="missing_error_target",
            missing_feature_policy="error",
            save_k=3,
        )

    mapped.run_mapping(
        target_assay=mapped.assay2,
        target_name="missing_zero",
        target_feat_key="missing_zero_target",
        missing_feature_policy="zero",
        save_k=3,
    )
    missing_zero = _projection_group(mapped, "missing_zero")
    assert missing_zero.attrs["complete"]
    assert missing_zero.attrs["feature_coverage"] == pytest.approx(
        len(shared_ids) / len(mapped.RNA.feats.fetch("ids", key="I__hvgs"))
    )
    zero_aligned = mapped.assay2.z["normed__I__missing_zero_target/data"][:]
    np.testing.assert_array_equal(zero_aligned[:, len(shared_ids) :], 0.0)

    with pytest.warns(DeprecationWarning, match="exclude_missing"):
        mapped.run_mapping(
            target_assay=mapped.assay2,
            target_name="missing_intersection",
            target_feat_key="missing_intersection_target",
            exclude_missing=True,
            save_k=3,
        )
    assert "I__hvgs_common_missing_intersection" in mapped.RNA.feats.columns
    intersection_projection = _projection_group(mapped, "missing_intersection")
    intersection_ann_path = intersection_projection.attrs["ann_path"]
    intersection_ref = parse_artifact_path(intersection_ann_path)
    assert intersection_ref.kind == "intersection_ann_index"
    assert intersection_ann_path in mapped.z
    intersection_inputs = mapped.inspect_artifact(intersection_ref).inputs
    assert intersection_inputs is not None
    assert (
        intersection_inputs["selected_feature_fingerprint"]
        == intersection_projection.attrs["selected_feature_fingerprint"]
    )
    assert np.all(
        np.isfinite(next(mapped.get_mapping_score("missing_intersection"))[1])
    )
    intersection_evidence = mapped.get_target_label_evidence(
        "missing_intersection",
        reference_class_group="ids",
    )
    assert np.all(np.isfinite(intersection_evidence["referenceDistancePercentile"]))
    mapped.run_mapping(
        target_assay=mapped.assay2,
        target_name="missing_intersection_unscaled",
        target_feat_key="missing_intersection_unscaled_target",
        missing_feature_policy="intersection",
        feat_scaling=False,
        save_k=3,
    )
    unscaled_intersection = _projection_group(
        mapped,
        "missing_intersection_unscaled",
    )
    assert not unscaled_intersection.attrs["ann_feature_scaling"]
    unscaled_ref = parse_artifact_path(unscaled_intersection.attrs["ann_path"])
    unscaled_inputs = mapped.inspect_artifact(unscaled_ref).inputs
    assert unscaled_inputs is not None
    source_ann_ref = ArtifactRef.from_dict(unscaled_inputs["source_ann_index"])
    source_ann_inputs = mapped.inspect_artifact(source_ann_ref).inputs
    assert source_ann_inputs is not None
    source_reduction_ref = ArtifactRef.from_dict(source_ann_inputs["coordinates"])
    source_reduction_parameters = mapped.inspect_artifact(
        source_reduction_ref
    ).parameters
    assert source_reduction_parameters is not None
    assert not source_reduction_parameters["feat_scaling"]
    assert np.all(
        np.isfinite(next(mapped.get_mapping_score("missing_intersection_unscaled"))[1])
    )

    batches = np.where(np.arange(mapped.cells.N) % 2 == 0, "a", "b")
    mapped.cells.insert("mapping_batch", batches, overwrite=True)
    reference = mapped.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    result = reference.map_query(
        mapped.assay2,
        "missing_reference_mean",
        "missing_reference_mean_target",
        save_k=3,
        missing_feature_policy="reference_mean",
    )
    assert result.n_cells == mapped.cells.active_index("I").shape[0]
    assert result.diagnostics["featureCoverage"] == pytest.approx(
        len(shared_ids) / len(reference.feature_ids)
    )
    mean_loaded = mapped.get_mapping_result(
        "missing_reference_mean",
        load_arrays=True,
    )
    zero_result = reference.map_query(
        mapped.assay2,
        "missing_reference_zero",
        "missing_reference_zero_target",
        save_k=3,
        missing_feature_policy="zero",
    )
    assert zero_result.n_cells == mapped.cells.active_index("I").shape[0]
    zero_loaded = mapped.get_mapping_result(
        "missing_reference_zero",
        load_arrays=True,
    )
    assert mean_loaded.uncorrected_latent is not None
    assert zero_loaded.uncorrected_latent is not None
    missing = slice(len(shared_ids), len(reference.feature_ids))
    expected_shift = (
        -reference.model.feature_means[missing]
        / reference.model.feature_scales[missing]
    ) @ reference.model.loadings[missing]
    np.testing.assert_allclose(
        zero_loaded.uncorrected_latent - mean_loaded.uncorrected_latent,
        np.broadcast_to(
            expected_shift,
            mean_loaded.uncorrected_latent.shape,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
