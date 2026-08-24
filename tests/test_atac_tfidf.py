from collections.abc import Callable

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf.assay import norm_tf_idf
from scarf.datastore.datastore import DataStore
from scarf.matrix import ChunkedArray
from scarf.plotting.heatmaps import _prepare_marker_heatmap
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
        min_cells_per_feature=0,
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


def test_subset_tfidf_and_fused_prevalence_match_reference(atac_tfidf_store):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    cell_idx = store.cells.active_index("subset")
    feat_idx = assay.feats.active_index("I")
    expected, expected_df = _reference_tfidf(counts, cell_idx, feat_idx)

    assay.set_feature_stats("subset")

    stats = assay.z["summary_stats_subset"]
    np.testing.assert_array_equal(
        np.asarray(stats["document_frequency"][:])[feat_idx],
        expected_df,
    )
    np.testing.assert_allclose(
        np.asarray(stats["prevalence"][:])[feat_idx],
        expected.sum(axis=0),
        rtol=1e-12,
        atol=1e-12,
    )
    observed = controlled_compute(assay.normed(cell_idx, feat_idx), assay.nthreads)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


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
        assay.normed(
            cell_idx,
            feat_idx,
            renormalize_subset=False,
        ),
        assay.nthreads,
    )
    observed_subset = controlled_compute(
        assay.normed(
            cell_idx,
            feat_idx,
            renormalize_subset=True,
        ),
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
    feat_idx = assay.feats.active_index("I")

    with pytest.raises(
        ValueError,
        match="ATAC TF-IDF does not support log_transform",
    ):
        assay.normed(cell_idx, feat_idx, log_transform=True)
    with pytest.raises(
        ValueError,
        match="ATAC TF-IDF does not support log_transform",
    ):
        store.run_normalization(
            from_assay="ATAC",
            cell_key="subset",
            feat_key="I",
            log_transform=True,
        )


def test_atac_marker_heatmap_defaults_to_unlogged_tfidf(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    store.cells.insert(
        "atac_cluster",
        np.array([0, 0, 1, 1]),
        overwrite=True,
    )
    store.run_marker_search(
        from_assay="ATAC",
        group_key="atac_cluster",
        cell_key="I",
        feat_key="I",
    )
    observed_log_transform: list[bool] = []
    original = assay.normed

    def recording_normed(cell_idx=None, feat_idx=None, **kwargs):
        observed_log_transform.append(kwargs["log_transform"])
        return original(cell_idx, feat_idx, **kwargs)

    monkeypatch.setattr(assay, "normed", recording_normed)
    prepared = _prepare_marker_heatmap(
        store,
        from_assay="ATAC",
        group_key="atac_cluster",
        cell_key="I",
        topn=1,
        log_transform=None,
        vmin=-1,
        vmax=2,
    )

    assert observed_log_transform == [False]
    assert not prepared["matrix"].empty
    with pytest.raises(
        ValueError,
        match="ATAC TF-IDF does not support log_transform",
    ):
        _prepare_marker_heatmap(
            store,
            from_assay="ATAC",
            group_key="atac_cluster",
            cell_key="I",
            topn=1,
            log_transform=True,
            vmin=-1,
            vmax=2,
        )


def test_atac_mapping_requires_rna_query(
    atac_tfidf_store,
    analyzed_datastore_ephemeral,
):
    store, _ = atac_tfidf_store
    state = analyzed_datastore_ephemeral.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors is not None
    reference = analyzed_datastore_ephemeral.build_mapping_reference(state.neighbors)

    with pytest.raises(TypeError, match="RNA query assays"):
        store.run_mapping(
            reference,
            "unsupported_atac_mapping",
            query_assay="ATAC",
        )


def test_run_normalization_applies_and_records_atac_subset_renormalization(
    atac_tfidf_store,
):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    selected_features = np.array([True, True, False, False])
    assay.feats.insert(
        "subset__two_peaks",
        selected_features,
        overwrite=True,
    )
    cell_idx = store.cells.active_index("subset")
    feat_idx = np.flatnonzero(selected_features)
    expected, _ = _reference_tfidf(
        counts,
        cell_idx,
        feat_idx,
        renormalize_subset=True,
    )

    normalized = store.run_normalization(
        from_assay="ATAC",
        cell_key="subset",
        feat_key="two_peaks",
        renormalize_subset=True,
    )

    status = store.inspect_artifact(normalized)
    observed = np.asarray(store.zw[status.path]["data"][:])
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
    assert status.parameters["log_transform"] is False
    assert status.parameters["renormalize_subset"] is True
    assert (
        store.run_normalization(
            from_assay="ATAC",
            cell_key="subset",
            feat_key="two_peaks",
        )
        == normalized
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

    np.testing.assert_array_equal(
        assay.n_docs_per_term,
        all_df[selected_features],
    )


def _count_document_frequency_passes(monkeypatch) -> Callable[[], int]:
    original = ChunkedArray.count_nonzero
    calls = 0

    def counting_count_nonzero(self, axis=None):
        nonlocal calls
        calls += 1
        return original(self, axis)

    monkeypatch.setattr(ChunkedArray, "count_nonzero", counting_count_nonzero)
    return lambda: calls


def test_normed_reuses_matching_df_and_recomputes_for_new_cells(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    cell_idx = store.cells.active_index("subset")
    feat_idx = assay.feats.active_index("I")
    assay.set_feature_stats("subset")

    passes = _count_document_frequency_passes(monkeypatch)
    assay.normed(cell_idx, feat_idx)
    assert passes() == 0

    assay.normed(np.array([0, 2], dtype=np.int64), feat_idx)
    assert passes() == 1


def test_normed_respects_arbitrary_cell_order(atac_tfidf_store):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    cell_idx = np.array([3, 0, 2], dtype=np.int64)
    feat_idx = assay.feats.active_index("I")
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
    feat_idx = assay.feats.active_index("I")
    assert assay.rawData[:, feat_idx][cell_idx, :].numblocks[0] == len(cell_idx)
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


def test_fused_stats_reserve_accumulators_and_float_block(
    atac_tfidf_store,
    monkeypatch,
):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    assay.rawData = ChunkedArray(
        assay.rawData._backing,
        nthreads=assay.nthreads,
        resources=assay.resources,
        block_size=2,
    )
    captured_resident_bytes: list[int] = []
    original = ChunkedArray._stream_blocks

    def record_stream(
        self,
        *,
        nthreads,
        msg,
        prefetch,
        row_mask,
        resident_bytes=0,
    ):
        captured_resident_bytes.append(resident_bytes)
        return original(
            self,
            nthreads=nthreads,
            msg=msg,
            prefetch=prefetch,
            row_mask=row_mask,
            resident_bytes=resident_bytes,
        )

    monkeypatch.setattr(ChunkedArray, "_stream_blocks", record_stream)
    cell_idx = np.arange(counts.shape[0], dtype=np.int64)
    feat_idx = assay.feats.active_index("I")

    assay._streaming_tfidf_feature_stats(cell_idx, feat_idx)

    float_itemsize = np.dtype(np.float64).itemsize
    expected = (
        len(cell_idx) * float_itemsize
        + 4 * len(feat_idx) * float_itemsize
        + 2 * len(feat_idx) * float_itemsize
    )
    assert captured_resident_bytes == [expected]


def test_normed_recomputes_df_when_stored_stats_predate_current_normalizer(
    atac_tfidf_store,
    monkeypatch,
):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    cell_idx = store.cells.active_index("subset")
    feat_idx = assay.feats.active_index("I")
    assay.set_feature_stats("subset")
    assay.z["summary_stats_subset"].attrs["normalization_identity"] = "older-tfidf"
    expected, _ = _reference_tfidf(counts, cell_idx, feat_idx)

    passes = _count_document_frequency_passes(monkeypatch)
    observed = controlled_compute(assay.normed(cell_idx, feat_idx), assay.nthreads)

    assert passes() == 1
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_normed_recomputes_df_for_features_absent_from_stored_stats(
    atac_tfidf_store,
    monkeypatch,
):
    store, counts = atac_tfidf_store
    assay = store.ATAC
    assay.feats.update_key(np.array([True, True, True, False]), "I")
    assay.set_feature_stats("subset")
    stored = np.asarray(assay.z["summary_stats_subset"]["document_frequency"][:])
    assert np.isnan(stored[3])

    assay.feats.reset_key("I")
    cell_idx = store.cells.active_index("subset")
    feat_idx = assay.feats.active_index("I")
    expected, _ = _reference_tfidf(counts, cell_idx, feat_idx)

    passes = _count_document_frequency_passes(monkeypatch)
    observed = controlled_compute(assay.normed(cell_idx, feat_idx), assay.nthreads)

    assert passes() == 1
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_filtered_prevalence_to_normalization_path_reuses_subset_df(
    atac_tfidf_store,
    monkeypatch,
):
    store, counts = atac_tfidf_store
    store.mark_prevalent_peaks(
        from_assay="ATAC",
        cell_key="subset",
        top_n=2,
    )

    def fail_count_nonzero(*_args, **_kwargs):
        pytest.fail("Normalization should reuse DF from peak prevalence")

    monkeypatch.setattr(ChunkedArray, "count_nonzero", fail_count_nonzero)
    normalized = store.run_normalization(
        from_assay="ATAC",
        cell_key="subset",
        feat_key="prevalent_peaks",
    )

    cell_idx = store.cells.active_index("subset")
    feat_idx = store.ATAC.feats.active_index("subset__prevalent_peaks")
    expected, _ = _reference_tfidf(counts, cell_idx, feat_idx)
    status = store.inspect_artifact(normalized)
    observed = np.asarray(store.zw[status.path]["data"][:])
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
    assert status.parameters["log_transform"] is False
    assert status.parameters["renormalize_subset"] is False
    assert status.parameters["normalization_method"] == {
        "external_hook": True,
        "identity": "scarf.assay.norm_tf_idf:selected-cell-df:total-count-tf",
    }


@pytest.mark.parametrize(
    "damage",
    [
        "missing_df",
        "missing_prevalence",
        "wrong_identity",
        "wrong_cell_digest",
        "malformed_df",
        "nan_df",
        "oversized_df",
    ],
)
def test_tfidf_stats_reject_stale_or_malformed_cache(
    atac_tfidf_store,
    monkeypatch,
    damage,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    assay.set_feature_stats("subset")
    stats = assay.z["summary_stats_subset"]

    if damage == "missing_df":
        del stats["document_frequency"]
    elif damage == "missing_prevalence":
        del stats["prevalence"]
    elif damage == "wrong_identity":
        stats.attrs["normalization_identity"] = "old-tfidf"
    elif damage == "wrong_cell_digest":
        stats.attrs["cell_index_digest"] = "digest-from-another-corpus"
    elif damage == "oversized_df":
        stats["document_frequency"][0] = store.cells.N + 1
    elif damage == "malformed_df":
        del stats["document_frequency"]
        stats.create_array(
            "document_frequency",
            data=np.array([1.0]),
        )
    elif damage == "nan_df":
        stats["document_frequency"][0] = np.nan

    original: Callable = assay._streaming_tfidf_feature_stats
    calls = 0

    def counting_stats(cell_idx, feat_idx):
        nonlocal calls
        calls += 1
        return original(cell_idx, feat_idx)

    monkeypatch.setattr(assay, "_streaming_tfidf_feature_stats", counting_stats)
    assay.set_feature_stats("subset")

    assert calls == 1
    repaired = assay.z["summary_stats_subset"]
    assert "document_frequency" in repaired
    assert "prevalence" in repaired
    assert repaired.attrs["normalization_identity"] == getattr(
        norm_tf_idf,
        "artifact_identity",
    )


def test_custom_normalizer_keeps_generic_prevalence_path(
    atac_tfidf_store,
    monkeypatch,
):
    store, counts = atac_tfidf_store
    assay = store.ATAC

    def identity_normalizer(_assay, selected):
        return selected

    assay.normMethod = identity_normalizer

    def fail_fused(*_args, **_kwargs):
        pytest.fail("Custom normalizers must not use the TF-IDF fused path")

    monkeypatch.setattr(assay, "_streaming_tfidf_feature_stats", fail_fused)
    assay.set_feature_stats("subset")

    feat_idx = assay.feats.active_index("I")
    expected = counts[store.cells.active_index("subset")].sum(axis=0)
    stats = assay.z["summary_stats_subset"]
    np.testing.assert_array_equal(
        np.asarray(stats["prevalence"][:])[feat_idx],
        expected[feat_idx],
    )
    assert "document_frequency" not in stats


def test_custom_normalizer_skips_document_frequency_pass(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    assay.normMethod = lambda _assay, selected: selected

    def fail_count_nonzero(*_args, **_kwargs):
        pytest.fail("Custom normalizers must not trigger a document frequency pass")

    monkeypatch.setattr(ChunkedArray, "count_nonzero", fail_count_nonzero)
    feat_idx = assay.feats.active_index("I")
    assay.normed(store.cells.active_index("subset"), feat_idx)

    np.testing.assert_array_equal(
        assay.n_docs_per_term,
        assay.feats.fetch_all("nCells")[feat_idx],
    )


def test_cells_without_accessible_peaks_keep_stats_finite_and_cacheable(
    tmp_path,
    monkeypatch,
):
    counts = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.uint32,
    )
    store = _build_store(tmp_path / "atac_empty_cell.zarr", counts)
    assay = store.ATAC
    store.cells.insert(
        "all_cells", np.ones(counts.shape[0], dtype=bool), overwrite=True
    )
    cell_idx = store.cells.active_index("all_cells")
    feat_idx = assay.feats.active_index("I")
    expected, expected_df = _reference_tfidf(counts, cell_idx, feat_idx)

    assay.set_feature_stats("all_cells")
    stats = assay.z["summary_stats_all_cells"]
    np.testing.assert_array_equal(
        np.asarray(stats["document_frequency"][:])[feat_idx],
        expected_df,
    )
    np.testing.assert_allclose(
        np.asarray(stats["prevalence"][:])[feat_idx],
        expected.sum(axis=0),
        rtol=1e-12,
        atol=1e-12,
    )

    original: Callable = assay._streaming_tfidf_feature_stats
    calls = 0

    def counting_stats(inner_cell_idx, inner_feat_idx):
        nonlocal calls
        calls += 1
        return original(inner_cell_idx, inner_feat_idx)

    monkeypatch.setattr(assay, "_streaming_tfidf_feature_stats", counting_stats)
    assay.set_feature_stats("all_cells")
    assert calls == 0

    observed = controlled_compute(assay.normed(cell_idx, feat_idx), assay.nthreads)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


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


def test_peak_prevalence_rejects_empty_cell_corpus(atac_tfidf_store):
    store, _ = atac_tfidf_store
    store.cells.insert(
        "empty",
        np.zeros(store.cells.N, dtype=bool),
        overwrite=True,
    )

    with pytest.raises(
        ValueError,
        match="Peak prevalence requires selected cells and features",
    ):
        store.ATAC.set_feature_stats("empty")


def test_atac_normed_validates_boolean_options_and_default_indices(
    atac_tfidf_store,
):
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


def test_tfidf_identity_requires_the_normalizer_contract(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    monkeypatch.delattr(norm_tf_idf, "artifact_identity")

    with pytest.raises(RuntimeError, match="must define artifact_identity"):
        store.ATAC._normalization_identity()


def test_cached_document_frequency_skips_malformed_candidates(
    atac_tfidf_store,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    cell_idx = np.array([1, 0], dtype=np.int64)
    feat_idx = np.array([assay.feats.N], dtype=np.int64)
    attrs = {
        "cell_index_digest": assay._cell_index_digest(cell_idx),
        "normalization_identity": assay._normalization_identity(),
    }

    assay.z.create_array(
        "summary_stats_coverage_array",
        data=np.array([1.0]),
        overwrite=True,
    )
    group_node = assay.z.create_group(
        "summary_stats_coverage_group_node",
        overwrite=True,
    )
    group_node.attrs.update(attrs)
    group_node.create_group("document_frequency")

    wrong_shape = assay.z.create_group(
        "summary_stats_coverage_wrong_shape",
        overwrite=True,
    )
    wrong_shape.attrs.update(attrs)
    wrong_shape.create_array("document_frequency", data=np.array([1.0]))

    invalid_index = assay.z.create_group(
        "summary_stats_coverage_invalid_index",
        overwrite=True,
    )
    invalid_index.attrs.update(attrs)
    invalid_index.create_array(
        "document_frequency",
        data=np.ones(assay.feats.N),
    )

    assert assay._cached_document_frequency(cell_idx, feat_idx) is None


def test_tfidf_cache_validation_rejects_unreadable_arrays(
    atac_tfidf_store,
    monkeypatch,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC
    cell_idx = store.cells.active_index("subset")
    feat_idx = assay.feats.active_index("I")
    attrs = {
        "cell_index_digest": assay._cell_index_digest(cell_idx),
        "normalization_identity": assay._normalization_identity(),
    }
    monkeypatch.setattr(assay, "_validate_stats_loc", lambda *_args, **_kwargs: True)

    assert not assay._valid_tfidf_stats("missing_stats", cell_idx, feat_idx)

    group_node = assay.z.create_group("coverage_stats_group_node", overwrite=True)
    group_node.attrs.update(attrs)
    group_node.create_group("prevalence")
    group_node.create_array(
        "document_frequency",
        data=np.ones(assay.feats.N),
    )
    assert not assay._valid_tfidf_stats(
        "coverage_stats_group_node",
        cell_idx,
        feat_idx,
    )

    invalid_index = assay.z.create_group("coverage_stats_invalid_index", overwrite=True)
    invalid_index.attrs.update(attrs)
    invalid_index.create_array("prevalence", data=np.ones(assay.feats.N))
    invalid_index.create_array(
        "document_frequency",
        data=np.ones(assay.feats.N),
    )
    assert not assay._valid_tfidf_stats(
        "coverage_stats_invalid_index",
        cell_idx,
        np.array([assay.feats.N], dtype=np.int64),
    )


def test_legacy_prevalent_peak_metadata_and_argument_validation(
    atac_tfidf_store,
):
    store, _ = atac_tfidf_store
    assay = store.ATAC

    with pytest.raises(ValueError, match="less than total number"):
        assay._prevalent_peak_mask("subset", assay.feats.N)
    with pytest.raises(TypeError, match="positive integer"):
        assay._prevalent_peak_mask("subset", 0)

    with pytest.warns(DeprecationWarning):
        assay.mark_prevalent_peaks("subset", 2, "top_peaks")

    values = assay.feats.fetch_all("subset__top_peaks")
    assert values.dtype == bool
    assert int(values.sum()) == 2
