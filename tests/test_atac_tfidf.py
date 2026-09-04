from collections.abc import Callable

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.datastore.datastore import DataStore
from scarf.matrix import ChunkedArray
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.plotting.heatmaps import _prepare_marker_heatmap
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    callable_identity,
    fingerprint_stored_arrays,
    inspect_artifact,
)
from scarf.utils import controlled_compute
from scarf.writers import SparseToZarr


def _build_store(store_path, counts):
    writer = SparseToZarr(
        csr_matrix(counts),
        zarr_loc=str(store_path),
        cell_ids=[f"cell_{i}" for i in range(counts.shape[0])],
        feature_ids=[f"peak_{i}" for i in range(counts.shape[1])],
        assay_name="ATAC",
        nthreads=1,
    )
    writer.dump(batch_size=2)
    return DataStore(
        str(store_path),
        default_assay="ATAC",
        min_features_per_cell=0,
        nthreads=1,
    )


@pytest.fixture
def atac_tfidf_store(tmp_path):
    counts = np.array(
        [
            [2, 0, 1, 0],
            [1, 3, 0, 0],
            [0, 1, 4, 1],
            [0, 2, 0, 5],
        ],
        dtype=np.uint32,
    )
    store = _build_store(tmp_path / "atac_tfidf.zarr", counts)
    store.cells.insert(
        "subset",
        np.array([True, True, False, False]),
        overwrite=True,
    )
    return store, counts


def _reference_tfidf(
    counts: np.ndarray,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
    *,
    renormalize_subset: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    selected = counts[np.ix_(cell_idx, feat_idx)].astype(np.float64)
    terms_per_document = (
        selected.sum(axis=1, dtype=np.float64)
        if renormalize_subset
        else counts.sum(axis=1, dtype=np.float64)[cell_idx]
    )
    terms_per_document[terms_per_document == 0] = 1
    document_frequency = np.count_nonzero(selected, axis=0)
    term_frequency = selected / terms_per_document.reshape(-1, 1)
    idf = np.log2(1 + len(cell_idx) / (document_frequency + 1))
    return term_frequency * idf, document_frequency


def _feature_summary_ref(store: DataStore, selection: ArtifactRef) -> ArtifactRef:
    status = inspect_artifact(store.zw, selection)
    raw = (status.inputs or {})["feature_summary"]
    assert isinstance(raw, dict)
    return ArtifactRef.from_dict(raw)


def test_subset_tfidf_and_feature_summary_match_reference(atac_tfidf_store):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    cell_idx = store.cells.active_index("subset")
    feat_idx = np.arange(assay.feats.N, dtype=np.int64)
    expected, expected_df = _reference_tfidf(counts, cell_idx, feat_idx)

    selection = store.select_prevalent_peaks(
        store.snapshot_cell_selection("subset"),
        from_assay="ATAC",
        top_n=2,
    )
    summary_ref = _feature_summary_ref(store, selection)
    summary_status = inspect_artifact(store.zw, summary_ref)
    assert summary_status.operation == "summarize_atac_features"
    assert summary_status.parameters == {
        "normalization_method": callable_identity(assay.normMethod),
    }
    assert set(summary_status.inputs or {}) == {"cell_selection"}
    summary = store.load_artifact(summary_ref)
    assert set(summary.array_keys()) == {"prevalence", "document_frequency"}
    assert isinstance(summary.attrs["ordered_feature_ids_fingerprint"], str)
    assert summary.attrs["payload_fingerprint"] == fingerprint_stored_arrays(
        store.zw[artifact_path(summary_ref)],
        ("prevalence", "document_frequency"),
    )
    np.testing.assert_array_equal(summary["document_frequency"][:], expected_df)
    np.testing.assert_allclose(
        summary["prevalence"][:],
        expected.sum(axis=0),
        rtol=1e-12,
        atol=1e-12,
    )
    observed = controlled_compute(assay.normed(cell_idx, feat_idx), assay.nthreads)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert not any(name.startswith("summary_stats_") for name in assay.z)


def test_subset_renormalization_uses_only_selected_peak_counts(atac_tfidf_store):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    cell_idx = np.array([0, 1, 2], dtype=np.int64)
    feat_idx = np.array([0, 1], dtype=np.int64)
    expected_full, _ = _reference_tfidf(counts, cell_idx, feat_idx)
    expected_subset, _ = _reference_tfidf(
        counts,
        cell_idx,
        feat_idx,
        renormalize_subset=True,
    )

    observed_full = controlled_compute(
        assay.normed(cell_idx, feat_idx, renormalize_subset=False),
        assay.nthreads,
    )
    observed_subset = controlled_compute(
        assay.normed(cell_idx, feat_idx, renormalize_subset=True),
        assay.nthreads,
    )

    np.testing.assert_allclose(observed_full, expected_full, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        observed_subset,
        expected_subset,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.allclose(observed_full, observed_subset)


def test_atac_rejects_log_transform(atac_tfidf_store):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    cell_idx = store.cells.active_index("subset")
    feat_idx = np.arange(assay.feats.N, dtype=np.int64)
    feature_ref = store.set_feature_selection(
        from_assay="ATAC",
        mask=np.ones(assay.feats.N, dtype=bool),
    )
    cell_selection = store.snapshot_cell_selection("subset")

    with pytest.raises(ValueError, match="does not support log_transform"):
        assay.normed(cell_idx, feat_idx, log_transform=True)
    with pytest.raises(ValueError, match="does not support log_transform"):
        store.run_normalization(
            cell_selection,
            feature_ref,
            log_transform=True,
        )


def test_atac_marker_heatmap_defaults_to_unlogged_tfidf(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    feature_ref = store.set_feature_selection(
        from_assay="ATAC",
        mask=np.ones(assay.feats.N, dtype=bool),
    )
    cell_selection = store.snapshot_cell_selection("I")
    cluster_plan = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="ATAC",
        kind="cluster_labels",
        operation="test_atac_clusters",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": ((assay.cells.N,), "i")},
    )
    write_cell_data_artifact(
        store.zw,
        cluster_plan,
        {"values": np.array([0, 0, 1, 1])},
    )
    clusters = cluster_plan.ref
    marker = store.run_marker_search(
        clusters,
        from_assay="ATAC",
        features=feature_ref,
    )
    observed_log_transform: list[bool] = []
    original = assay.normed

    def recording_normed(cell_idx=None, feat_idx=None, **kwargs):
        observed_log_transform.append(kwargs["log_transform"])
        return original(cell_idx, feat_idx, **kwargs)

    monkeypatch.setattr(assay, "normed", recording_normed)
    prepared = _prepare_marker_heatmap(
        store,
        marker=marker,
        topn=1,
        log_transform=None,
        vmin=-1,
        vmax=2,
    )

    assert observed_log_transform == [False]
    assert not prepared["matrix"].empty


def test_run_normalization_uses_explicit_feature_artifact(atac_tfidf_store):
    store, counts = atac_tfidf_store
    selected_features = np.array([True, True, False, False])
    feature_ref = store.set_feature_selection(
        from_assay="ATAC",
        mask=selected_features,
    )
    cell_idx = store.cells.active_index("subset")
    feat_idx = np.flatnonzero(selected_features)
    expected, _ = _reference_tfidf(
        counts,
        cell_idx,
        feat_idx,
        renormalize_subset=True,
    )
    cell_selection = store.snapshot_cell_selection("subset")

    normalized = store.run_normalization(
        cell_selection,
        feature_ref,
        renormalize_subset=True,
    )

    status = inspect_artifact(store.zw, normalized)
    observed = np.asarray(store.zw[status.path]["data"][:])
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
    assert status.parameters["log_transform"] is False
    assert status.parameters["renormalize_subset"] is True
    assert (
        ArtifactRef.from_dict((status.inputs or {})["feature_selection"]) == feature_ref
    )


def test_document_frequency_is_feature_subset_invariant(atac_tfidf_store):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    cell_idx = np.array([0, 2], dtype=np.int64)
    all_features = np.arange(counts.shape[1], dtype=np.int64)
    selected_features = np.array([3, 1], dtype=np.int64)

    assay.normed(cell_idx, all_features)
    all_df = np.asarray(assay.n_docs_per_term).copy()
    assay.normed(cell_idx, selected_features)

    np.testing.assert_array_equal(assay.n_docs_per_term, all_df[selected_features])


def _count_document_frequency_passes(monkeypatch) -> Callable[[], int]:
    original = ChunkedArray.count_nonzero
    calls = 0

    def counting_count_nonzero(self, axis=None):
        nonlocal calls
        calls += 1
        return original(self, axis)

    monkeypatch.setattr(ChunkedArray, "count_nonzero", counting_count_nonzero)
    return lambda: calls


def test_normed_recomputes_document_frequency_without_mounted_cache(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    cell_idx = store.cells.active_index("subset")
    feat_idx = np.arange(assay.feats.N, dtype=np.int64)
    passes = _count_document_frequency_passes(monkeypatch)

    assay.normed(cell_idx, feat_idx)
    assay.normed(cell_idx, feat_idx)

    assert passes() == 2
    assert not any(name.startswith("summary_stats_") for name in assay.z)


def test_normed_respects_arbitrary_cell_order(atac_tfidf_store):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    cell_idx = np.array([3, 0, 2], dtype=np.int64)
    feat_idx = np.arange(assay.feats.N, dtype=np.int64)
    expected, expected_df = _reference_tfidf(counts, cell_idx, feat_idx)

    observed = controlled_compute(assay.normed(cell_idx, feat_idx), assay.nthreads)

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(assay.n_docs_per_term, expected_df)


def test_fused_stats_accumulate_across_stream_blocks(atac_tfidf_store):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    assay.rawData = ChunkedArray(
        assay.rawData._backing,
        nthreads=assay.nthreads,
        resources=assay.resources,
        block_size=1,
    )
    cell_idx = np.arange(counts.shape[0], dtype=np.int64)
    feat_idx = np.arange(assay.feats.N, dtype=np.int64)
    expected, expected_df = _reference_tfidf(counts, cell_idx, feat_idx)

    document_frequency, prevalence = assay._streaming_tfidf_feature_stats(
        cell_idx,
        feat_idx,
    )

    np.testing.assert_array_equal(document_frequency, expected_df)
    np.testing.assert_allclose(
        prevalence,
        expected.sum(axis=0),
        rtol=1e-12,
        atol=1e-12,
    )


def test_custom_normalizer_feature_summary_is_an_artifact(
    atac_tfidf_store,
    monkeypatch,
):
    store, counts = atac_tfidf_store
    assay = store.ATAC

    def identity_normalizer(_assay, selected):
        return selected

    identity_normalizer.artifact_identity = "test.identity_normalizer"
    assay.normMethod = identity_normalizer

    def fail_fused(*_args, **_kwargs):
        pytest.fail("Custom normalizers must not use the TF-IDF fused path")

    monkeypatch.setattr(assay, "_streaming_tfidf_feature_stats", fail_fused)
    selection = store.select_prevalent_peaks(
        store.snapshot_cell_selection("subset"),
        from_assay="ATAC",
        top_n=2,
    )
    summary = store.load_artifact(_feature_summary_ref(store, selection))
    expected = counts[store.cells.active_index("subset")].sum(axis=0)
    np.testing.assert_array_equal(summary["prevalence"][:], expected)


def test_custom_normalizer_skips_document_frequency_pass(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    assay.normMethod = lambda _assay, selected: selected

    def fail_count_nonzero(*_args, **_kwargs):
        pytest.fail("Custom normalizers must not trigger document frequency")

    monkeypatch.setattr(ChunkedArray, "count_nonzero", fail_count_nonzero)
    feat_idx = np.arange(assay.feats.N, dtype=np.int64)
    assay.normed(store.cells.active_index("subset"), feat_idx)

    np.testing.assert_array_equal(
        assay.n_docs_per_term,
        assay.feats.fetch_all("nCells")[feat_idx],
    )


def test_cells_without_accessible_peaks_keep_summary_finite(tmp_path):
    counts = np.array(
        [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1]],
        dtype=np.uint32,
    )
    store = _build_store(tmp_path / "atac_empty_cell.zarr", counts)
    store.cells.insert("all_cells", np.ones(4, dtype=bool), overwrite=True)
    selection = store.select_prevalent_peaks(
        store.snapshot_cell_selection("all_cells"),
        from_assay="ATAC",
        top_n=2,
    )
    summary = store.load_artifact(_feature_summary_ref(store, selection))

    assert np.isfinite(np.asarray(summary["prevalence"][:])).all()
    assert np.isfinite(np.asarray(summary["document_frequency"][:])).all()


def test_empty_cell_selection_is_shape_safe(atac_tfidf_store):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    feat_idx = np.array([0, 1], dtype=np.int64)

    values = controlled_compute(
        assay.normed(np.array([], dtype=np.int64), feat_idx),
        assay.nthreads,
    )

    assert values.shape == (0, len(feat_idx))
    np.testing.assert_array_equal(
        assay.n_docs_per_term,
        np.zeros(len(feat_idx), dtype=np.int64),
    )


def test_atac_normed_validates_boolean_options(atac_tfidf_store):
    store, _ = atac_tfidf_store
    assay = store.ATAC

    with pytest.raises(TypeError, match="log_transform must be a boolean"):
        assay.normed(log_transform=1)
    with pytest.raises(TypeError, match="renormalize_subset must be a boolean"):
        assay.normed(renormalize_subset="yes")


def test_subset_term_counts_require_data_and_handle_empty_corpora(
    atac_tfidf_store,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    empty_cells = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="Selected counts are required"):
        assay._terms_per_document(
            empty_cells,
            counts=None,
            renormalize_subset=True,
        )

    terms = assay._terms_per_document(
        empty_cells,
        counts=ChunkedArray.from_numpy(np.empty((0, 2))),
        renormalize_subset=True,
    )
    assert terms.shape == (0,)


def test_prevalent_peak_provenance_has_no_version_token(atac_tfidf_store):
    store, _ = atac_tfidf_store
    columns_before = set(store.ATAC.feats.columns)
    ref = store.select_prevalent_peaks(
        store.snapshot_cell_selection("subset"),
        from_assay="ATAC",
        top_n=2,
    )
    status = inspect_artifact(store.zw, ref)

    assert status.parameters == {"top_n": 2}
    assert set(status.inputs or {}) == {"feature_summary"}
    assert "algorithm_version" not in status.parameters
    assert store.resolve_features("ATAC", ref) == ref
    assert set(store.ATAC.feats.columns) == columns_before
