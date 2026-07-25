import numpy as np
import pytest

from scarf.writers import create_zarr_dataset


def _ensure_graph(datastore) -> None:
    try:
        datastore.get_latest_graph_loc("RNA", "I", "hvgs")
    except KeyError:
        datastore.auto_filter_cells(show_qc_plots=False)
        datastore.mark_hvgs(top_n=100, show_plot=False, bin_strategy="fixed")
        datastore.make_graph(feat_key="hvgs")


def _projection_store(datastore, name: str, indices: np.ndarray, distances: np.ndarray):
    source = datastore.RNA
    _ensure_graph(datastore)
    if "projections" not in source.z:
        source.z.create_group("projections")
    store = source.z["projections"].create_group(name, overwrite=True)
    zi = create_zarr_dataset(store, "indices", (len(indices),), "u8", indices.shape)
    zd = create_zarr_dataset(store, "distances", (len(indices),), "f8", distances.shape)
    zi[:] = indices
    zd[:] = distances
    store.attrs["complete"] = True
    store.attrs["assay"] = "RNA"
    return store


def test_label_transfer_uses_every_neighbor_and_abstains_on_ties(datastore_ephemeral):
    datastore = datastore_ephemeral
    reference_ids = datastore.cells.fetch("ids", key="I")
    _projection_store(
        datastore,
        "manual_labels",
        np.array([[0, 1], [0, 1]], dtype=np.uint64),
        np.array([[0.0, 1.0], [1.0, 1.0]]),
    )

    labels = datastore.get_target_classes(
        "manual_labels",
        reference_class_group="ids",
        threshold_fraction=0.5,
    )

    assert labels.tolist() == [reference_ids[0], "NA"]


def test_mapping_score_uses_all_saved_neighbors(datastore_ephemeral):
    datastore = datastore_ephemeral
    _projection_store(
        datastore,
        "manual_scores",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[1.0, 1.0]]),
    )

    score = next(
        datastore.get_mapping_score(
            "manual_scores", log_transform=False, multiplier=1.0
        )
    )[1]

    np.testing.assert_allclose(score[:2], [0.25, 0.25])


def test_mapping_evidence_and_fixed_layout_projection(datastore_ephemeral):
    datastore = datastore_ephemeral
    _projection_store(
        datastore,
        "manual_evidence",
        np.array([[0, 1], [0, 1]], dtype=np.uint64),
        np.array([[0.0, 1.0], [1.0, 1.0]]),
    )
    layout1 = np.arange(datastore.cells.N, dtype=float)
    layout2 = layout1 * 2
    datastore.cells.insert("fixed_layout1", layout1, overwrite=True)
    datastore.cells.insert("fixed_layout2", layout2, overwrite=True)

    evidence = datastore.get_target_label_evidence(
        "manual_evidence",
        reference_class_group="ids",
        threshold_fraction=0.75,
    )
    layout_path = datastore.project_mapping_layout("manual_evidence", "fixed_layout")
    projected = datastore.z[layout_path]["data"][:]

    assert evidence["isUnknown"].tolist() == [False, True]
    assert evidence["label"].tolist()[1] == "NA"
    assert set(
        [
            "voteFraction",
            "voteEntropy",
            "topTwoMargin",
            "featureCoverage",
            "referenceDistancePercentile",
        ]
    ).issubset(evidence.columns)
    np.testing.assert_allclose(projected[0], [0.0, 0.0])
    np.testing.assert_allclose(projected[1], [0.5, 1.0])


def test_label_threshold_boundary_and_subset_index(datastore_ephemeral):
    datastore = datastore_ephemeral
    labels = np.repeat("other", datastore.cells.N).astype(object)
    labels[:2] = ["winner", "runner_up"]
    datastore.cells.insert("boundary_labels", labels, overwrite=True)
    _projection_store(
        datastore,
        "boundary_labels",
        np.array([[0, 1], [0, 1]], dtype=np.uint64),
        np.array([[1.0, 9.0], [9.0, 1.0]]),
    )

    predictions = datastore.get_target_classes(
        "boundary_labels",
        reference_class_group="boundary_labels",
        threshold_fraction=0.75,
        target_subset=[1],
        na_val="unknown",
    )

    assert predictions.index.tolist() == [1]
    assert predictions.tolist() == ["runner_up"]
    evidence = datastore.get_target_label_evidence(
        "boundary_labels",
        reference_class_group="boundary_labels",
        threshold_fraction=0.75,
        na_val="unknown",
    )
    assert evidence["label"].tolist() == ["winner", "runner_up"]
    np.testing.assert_allclose(evidence["voteFraction"], [0.75, 0.75])


def test_master_projection_without_provenance_warns_on_read(datastore_ephemeral):
    datastore = datastore_ephemeral
    store = _projection_store(
        datastore,
        "master_projection",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[1.0, 1.0]]),
    )
    del store.attrs["complete"]

    with pytest.warns(DeprecationWarning, match="predates"):
        result = datastore.get_target_classes(
            "master_projection",
            reference_class_group="ids",
        )

    assert len(result) == 1
    store.attrs["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        datastore.get_target_classes(
            "master_projection",
            reference_class_group="ids",
        )


def test_incomplete_projection_is_rejected(datastore_ephemeral):
    datastore = datastore_ephemeral
    store = _projection_store(
        datastore,
        "incomplete_projection",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[1.0, 1.0]]),
    )
    store.attrs["complete"] = False

    with pytest.raises(ValueError, match="incomplete"):
        datastore.get_target_classes(
            "incomplete_projection",
            reference_class_group="ids",
        )


def test_read_only_fixed_layout_returns_array(datastore_ephemeral):
    from scarf.datastore.datastore import DataStore

    datastore = datastore_ephemeral
    _projection_store(
        datastore,
        "read_only_layout",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[1.0, 1.0]]),
    )
    layout1 = np.arange(datastore.cells.N, dtype=float)
    layout2 = layout1 * 2
    datastore.cells.insert("readonly_layout1", layout1, overwrite=True)
    datastore.cells.insert("readonly_layout2", layout2, overwrite=True)
    read_only = DataStore(
        datastore.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )

    projected = read_only.project_mapping_layout("read_only_layout", "readonly_layout")

    assert isinstance(projected, np.ndarray)
    np.testing.assert_allclose(projected[0], [0.5, 1.0])


def test_conformal_prediction_sets_are_exposed_in_label_evidence(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _projection_store(
        datastore,
        "conformal_evidence",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[0.0, 1.0]]),
    )

    evidence = datastore.get_target_label_evidence(
        "conformal_evidence",
        reference_class_group="ids",
        calibration_nonconformity=np.array([0.1, 0.2, 0.3]),
        conformal_alpha=0.2,
    )

    assert "predictionSet" in evidence
    assert datastore.cells.fetch("ids", key="I")[0] in evidence.loc[0, "predictionSet"]


def test_uninformative_projection_rows_are_forced_unknown(datastore_ephemeral):
    datastore = datastore_ephemeral
    store = _projection_store(
        datastore,
        "uninformative_evidence",
        np.array([[0, 1], [0, 1]], dtype=np.uint64),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    uninformative = create_zarr_dataset(
        store,
        "uninformative",
        (2,),
        "bool",
        (2,),
    )
    uninformative[:] = [True, False]

    evidence = datastore.get_target_label_evidence(
        "uninformative_evidence",
        reference_class_group="ids",
        threshold_fraction=0.0,
    )

    assert evidence["isUnknown"].tolist() == [True, False]
    assert evidence.loc[0, "label"] == "NA"
    classes = datastore.get_target_classes(
        "uninformative_evidence",
        reference_class_group="ids",
        threshold_fraction=0.0,
        na_val="unknown",
    )
    assert classes.iloc[0] == "unknown"

    decision = datastore._label_vote_decision(
        datastore.cells.fetch("ids", key="I"),
        np.array([0, 1]),
        np.array([0.0, 0.0]),
        0.0,
        "unknown",
    )
    assert decision[0] == "unknown"
    assert decision[4]


def test_reference_distance_percentile_is_query_composition_invariant(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _projection_store(
        datastore,
        "distance_single",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[1.0, 4.0]]),
    )
    _projection_store(
        datastore,
        "distance_composed",
        np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint64),
        np.array([[1.0, 4.0], [0.1, 0.2], [10.0, 20.0]]),
    )

    single = datastore.get_target_label_evidence(
        "distance_single",
        reference_class_group="ids",
    )
    composed = datastore.get_target_label_evidence(
        "distance_composed",
        reference_class_group="ids",
    )

    assert (
        single.loc[0, "referenceDistancePercentile"]
        == composed.loc[0, "referenceDistancePercentile"]
    )


def test_legacy_projection_without_provenance_marker_warns(datastore_ephemeral):
    datastore = datastore_ephemeral
    _projection_store(
        datastore,
        "legacy_projection",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[0.0, 1.0]]),
    )

    with pytest.warns(DeprecationWarning, match="predates projection provenance"):
        datastore.get_target_classes(
            "legacy_projection",
            reference_class_group="ids",
            threshold_fraction=0.5,
        )


def test_partial_provenance_projection_is_rejected_not_downgraded(datastore_ephemeral):
    # A projection that carries the provenance marker but is missing the rest of
    # the provenance metadata is a corrupt current write, not a legacy store, so
    # it must raise rather than silently fall back to the legacy read path.
    datastore = datastore_ephemeral
    store = _projection_store(
        datastore,
        "partial_provenance",
        np.array([[0, 1]], dtype=np.uint64),
        np.array([[0.0, 1.0]]),
    )
    marker = create_zarr_dataset(store, "reference_feature_indices", (1,), "u8", (1,))
    marker[:] = [0]

    with pytest.raises(ValueError, match="incomplete provenance metadata"):
        datastore.get_target_classes(
            "partial_provenance",
            reference_class_group="ids",
            threshold_fraction=0.5,
        )
