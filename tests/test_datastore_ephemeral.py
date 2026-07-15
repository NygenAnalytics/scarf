import numpy as np
import pytest

from tests.fixtures_datastore import _has_graph


def _active_cell_count(datastore) -> int:
    return len(datastore.cells.active_index("I"))


def _ensure_graph(datastore):
    if not _has_graph(datastore):
        datastore.auto_filter_cells(show_qc_plots=False)
        datastore.mark_hvgs(top_n=100, show_plot=False)
        datastore.make_graph(feat_key="hvgs")


def _clear_umap_columns(datastore):
    for column in ("RNA_UMAP1", "RNA_UMAP2"):
        if column in datastore.cells.columns:
            datastore.cells.drop(column)


def _clear_projection(datastore, name: str):
    projections = datastore.z["RNA"].get("projections")
    if projections is not None and name in projections:
        del projections[name]


def test_run_umap_recomputes_coordinates(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    _clear_umap_columns(ds)

    ds.run_umap(n_epochs=10, parallel=False)

    umap1 = ds.cells.fetch("RNA_UMAP1")
    umap2 = ds.cells.fetch("RNA_UMAP2")
    assert len(umap1) == _active_cell_count(ds)
    assert len(umap2) == _active_cell_count(ds)


def test_run_leiden_writes_cluster_labels(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)

    ds.run_leiden_clustering(label="ephemeral_leiden")

    groups = ds.cells.fetch("RNA_ephemeral_leiden", key="I")
    assert len(groups) == _active_cell_count(ds)
    assert np.unique(groups).size >= 1


def test_run_mapping_writes_projection(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "freshmap")

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="freshmap",
        target_feat_key="hvgs_freshmap",
        save_k=3,
    )

    assert "freshmap" in ds.z["RNA"]["projections"]
    assert ds.z["RNA"]["projections"]["freshmap"]["indices"].shape[
        0
    ] == _active_cell_count(ds)


def test_run_mapping_supports_all_features_key(datastore_ephemeral):
    ds = datastore_ephemeral
    ds.auto_filter_cells(show_qc_plots=False)
    ds.make_graph(feat_key="I", dims=5, k=3, n_centroids=10)

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="all_features_map",
        target_feat_key="all_features_map_target",
        save_k=3,
    )

    assert ds.z["RNA"]["projections"]["all_features_map"].attrs["complete"]


def test_run_mapping_with_coral_writes_corrected_data(datastore_ephemeral):
    ds = datastore_ephemeral
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
    assert "freshmap_coral" in ds.z["RNA"]["projections"]


def test_coral_mapping_cache_and_rebuild_are_equivalent(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="coral_cached",
        target_feat_key="hvgs_coral_cached",
        save_k=3,
        run_coral=True,
    )
    cached = ds.z["RNA"]["projections"]["coral_cached"]
    cached_indices = cached["indices"][:]
    cached_distances = cached["distances"][:]

    normed = ds.z["RNA"]["normed__I__hvgs"]
    reduction = ds.z[normed.attrs["latest_reduction"]]
    ann = ds.z[reduction.attrs["latest_ann"]]
    del ann["ann_idx_bytes"]
    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="coral_rebuilt",
        target_feat_key="hvgs_coral_rebuilt",
        save_k=3,
        run_coral=True,
    )
    rebuilt = ds.z["RNA"]["projections"]["coral_rebuilt"]

    assert np.array_equal(cached_indices, rebuilt["indices"][:])
    np.testing.assert_allclose(cached_distances, rebuilt["distances"][:])


def test_run_unified_umap_after_mapping(datastore_ephemeral):
    ds = datastore_ephemeral
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

    coords = ds.z["RNA"]["projections"]["unified_UMAP"][:]
    assert coords.shape[1] == 2
    assert coords.shape[0] >= _active_cell_count(ds)


def test_build_and_reload_symphony_mapping_reference(datastore_ephemeral, tmp_path):
    import numpy as np
    import pandas as pd

    ds = datastore_ephemeral
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
    normed = ds.z["RNA"]["normed__I__hvgs"]
    reduction = ds.z[normed.attrs["latest_reduction"]]
    reference_coordinates = reduction["harmonizedData"][:].copy()
    reference_loadings = reduction["reduction"][:].copy()
    reference_ann = ds.z[reduction.attrs["latest_ann"]]
    reference_ann_metadata = dict(reference_ann.attrs)
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

    assert reference.model.n_dims == loaded.model.n_dims
    assert reference.metadata["harmonyParameters"]["nclust"] == 5
    assert result.n_cells == _active_cell_count(ds)
    projection = ds.z["RNA"]["projections"]["symphony_self"]
    assert projection.attrs["complete"]
    assert projection["correctedLatent"].shape[1] == loaded.model.n_dims
    assert projection["indices"].chunks[1] == 3
    assert projection["distances"].chunks[1] == 3
    assert projection.attrs["queryBatchColumns"] == ["mapping_batch"]
    assert isinstance(projection.attrs["queryBatchHash"], str)
    original_indices = projection["indices"][:].copy()
    original_corrected = projection["correctedLatent"][:].copy()
    np.testing.assert_array_equal(reduction["harmonizedData"][:], reference_coordinates)
    np.testing.assert_array_equal(reduction["reduction"][:], reference_loadings)
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
        projection["correctedLatent"][:],
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
    with pytest.raises(ValueError, match="stale"):
        ds.get_mapping_reference(feat_key="hvgs")

    artifact_count = len(reduction["mappingReferences"])
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
    assert len(reduction["mappingReferences"]) == artifact_count + 1
    old_reference_result = reference.map_query(
        ds.RNA,
        "old_reference_replay",
        "hvgs_old_reference_replay",
        save_k=3,
        query_batches=pd.DataFrame({"mapping_batch": active_batches}),
    )
    old_projection = ds.z["RNA"]["projections"]["old_reference_replay"]
    np.testing.assert_array_equal(old_projection["indices"][:], original_indices)
    np.testing.assert_allclose(
        old_projection["correctedLatent"][:],
        original_corrected,
        rtol=1e-10,
        atol=1e-12,
    )
    assert old_reference_result.projection_path


def test_mapping_rejects_reference_normalization_overwrite(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    normed = ds.z["RNA"]["normed__I__hvgs"]
    reduction = ds.z[normed.attrs["latest_reduction"]]
    ann = ds.z[reduction.attrs["latest_ann"]]
    reference_data = normed["data"][:].copy()
    reference_loadings = reduction["reduction"][:].copy()
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

    np.testing.assert_array_equal(
        ds.z["RNA"]["normed__I__hvgs"]["data"][:], reference_data
    )
    np.testing.assert_array_equal(reduction["reduction"][:], reference_loadings)
    assert dict(ann.attrs) == ann_metadata


def test_cached_harmony_can_rebuild_missing_mapping_artifact(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    normed = ds.z["RNA"]["normed__I__hvgs"]
    reduction = ds.z[normed.attrs["latest_reduction"]]
    assert "harmonizedData" in reduction
    artifact_hash = reduction.attrs["latestMappingReference"]
    artifact = reduction["mappingReferences"][artifact_hash]
    legacy = reduction.create_group("mappingReference")
    for key, value in dict(artifact.attrs).items():
        if key not in {
            "algorithmVariant",
            "annContract",
            "artifactHash",
            "correctedCoordinatesHash",
        }:
            legacy.attrs[key] = value
    legacy.attrs["schemaVersion"] = 1
    for name in (
        "featureIds",
        "featureMeans",
        "featureScales",
        "loadings",
        "centroids",
        "rawCentroids",
        "correctedCentroids",
        "clusterMass",
        "sigma",
    ):
        legacy.create_array(name, data=np.asarray(artifact[name][:]))
    del reduction["mappingReferences"]
    del reduction.attrs["latestMappingReference"]

    with pytest.warns(DeprecationWarning, match="legacy"):
        legacy_reference = ds.get_mapping_reference(feat_key="hvgs")
    assert legacy_reference.metadata["schemaVersion"] == 1
    del reduction["mappingReference"]

    from scarf.datastore.datastore import DataStore

    read_only = DataStore(ds.zarr_loc, default_assay="RNA", zarr_mode="r")
    with pytest.raises(ValueError, match="zarr_mode='r\\+'"):
        read_only.run_mapping(
            target_assay=ds.RNA,
            target_name="read_only_legacy_harmony",
            target_feat_key="hvgs_read_only_legacy_harmony",
            save_k=3,
        )

    cached_ann = ds.z[reduction.attrs["latest_ann"]]
    cached_knn = ds.z[cached_ann.attrs["latest_knn"]]
    cached_knn["distances"][0, 0] = 123456.0
    rebuilt = ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
    )

    assert rebuilt.metadata["complete"]
    assert "mappingReferences" in reduction
    rebuilt_ann = ds.z[rebuilt.ann_path]
    rebuilt_knn = ds.z[rebuilt_ann.attrs["latest_knn"]]
    assert np.isfinite(rebuilt_knn["distances"][0, 0])
    assert rebuilt_knn["distances"][0, 0] != 123456.0


def test_run_mapping_upgrades_legacy_harmonized_graph(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    normed = ds.z["RNA"]["normed__I__hvgs"]
    reduction = ds.z[normed.attrs["latest_reduction"]]
    del reduction["mappingReferences"]
    del reduction.attrs["latestMappingReference"]

    with pytest.warns(DeprecationWarning, match="rebuilt once"):
        ds.run_mapping(
            target_assay=ds.RNA,
            target_name="upgraded_harmonized_map",
            target_feat_key="hvgs_upgraded_harmonized",
            save_k=3,
        )

    assert ds.z["RNA"]["projections"]["upgraded_harmonized_map"].attrs["complete"]


def test_legacy_harmony_upgrade_requires_recorded_batch_columns(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    ds.build_mapping_reference(
        feat_key="hvgs",
        batch_columns=["mapping_batch"],
        k=3,
        n_centroids=10,
    )
    normed = ds.z["RNA"]["normed__I__hvgs"]
    reduction = ds.z[normed.attrs["latest_reduction"]]
    del reduction["mappingReferences"]
    del reduction.attrs["latestMappingReference"]
    del reduction["harmonizedData"].attrs["batches"]

    with pytest.raises(ValueError, match="does not record its batch columns"):
        ds.run_mapping(
            target_assay=ds.RNA,
            target_name="legacy_harmony_without_batches",
            target_feat_key="hvgs_legacy_harmony_without_batches",
            save_k=3,
        )


def test_deprecated_reference_scaling_flags_do_not_break_mapping(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
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

    assert ds.z["RNA"]["projections"]["deprecated_scaling_flags"].attrs["complete"]


def test_mapping_feature_scaling_uses_isolated_ann_cache(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    normed = ds.z["RNA"]["normed__I__hvgs"]
    reduction = ds.z[normed.attrs["latest_reduction"]]
    original_ann_path = reduction.attrs["latest_ann"]

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="unscaled_mapping",
        target_feat_key="hvgs_unscaled_mapping",
        feat_scaling=False,
        save_k=3,
    )

    projection = ds.z["RNA"]["projections"]["unscaled_mapping"]
    mapping_ann_path = projection.attrs["annPath"]
    assert mapping_ann_path != original_ann_path
    assert mapping_ann_path.endswith("__unscaled")
    assert not ds.z[mapping_ann_path].attrs["featureScaling"]
    assert reduction.attrs["latest_ann"] == original_ann_path


def test_projection_uses_stored_feature_key_and_rejects_stale_provenance(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="provenance_map",
        target_feat_key="hvgs_provenance_map",
        save_k=3,
    )
    projection = ds.z["RNA"]["projections"]["provenance_map"]
    original_latest = ds.RNA.z.attrs["latest_feat_key"]
    ds.RNA.z.attrs["latest_feat_key"] = "I"
    try:
        score = next(ds.get_mapping_score("provenance_map"))[1]
    finally:
        ds.RNA.z.attrs["latest_feat_key"] = original_latest
    assert np.all(np.isfinite(score))

    original_hash = projection.attrs["reductionHash"]
    projection.attrs["reductionHash"] = "stale"
    try:
        with pytest.raises(ValueError, match="changed reduction"):
            next(ds.get_mapping_score("provenance_map"))
    finally:
        projection.attrs["reductionHash"] = original_hash


def test_mapping_missing_feature_policies_and_legacy_intersection(
    datastore_ephemeral,
):
    from scarf.datastore.datastore import DataStore

    ds = datastore_ephemeral
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
    missing_zero = mapped.z["RNA"]["projections"]["missing_zero"]
    assert missing_zero.attrs["complete"]
    assert missing_zero.attrs["featureCoverage"] == pytest.approx(
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
    intersection_projection = mapped.z["RNA"]["projections"]["missing_intersection"]
    intersection_ann_path = intersection_projection.attrs["annPath"]
    assert "__intersection_" in intersection_ann_path
    assert intersection_ann_path in mapped.z
    assert (
        mapped.z[intersection_ann_path].attrs["selectedFeatureHash"]
        == intersection_projection.attrs["selectedFeatureHash"]
    )
    assert (
        mapped.z[intersection_ann_path].attrs["sourceAnnPath"]
        == intersection_projection.attrs["annSourcePath"]
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
    unscaled_intersection = mapped.z["RNA"]["projections"][
        "missing_intersection_unscaled"
    ]
    assert not unscaled_intersection.attrs["annFeatureScaling"]
    assert "__unscaled" in unscaled_intersection.attrs["annSourcePath"]
    assert not mapped.z[unscaled_intersection.attrs["annPath"]].attrs["featureScaling"]
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
    mean_aligned = mapped.assay2.z["normed__I__missing_reference_mean_target/data"][:]
    np.testing.assert_allclose(
        mean_aligned[:, len(shared_ids) :],
        np.broadcast_to(
            reference.model.feature_means[np.newaxis, len(shared_ids) :],
            mean_aligned[:, len(shared_ids) :].shape,
        ),
        rtol=0,
        atol=0,
    )
