import numpy as np
import pandas as pd

from scarf.markers import _batch_stats, mannwhitneyu_from_ranks, sort_marker_results
from scarf.utils import controlled_compute


def _reference_calc(
    vdf: pd.DataFrame, groups: np.ndarray, group_set: np.ndarray
) -> np.ndarray:
    """Original pandas implementation used as the parity reference."""
    ranked_vdf = vdf.rank(method="dense")
    ranked_vdf_average = vdf.rank(method="average")
    r = ranked_vdf.groupby(groups).mean().reindex(group_set)
    r = r / r.sum()
    g = np.array([pd.Series(groups).value_counts().reindex(group_set).values]).T
    g_o = len(groups) - g
    s = vdf.groupby(groups).sum().reindex(group_set)
    m = s / g
    m_o = (s.sum() - s) / g_o
    s2 = (vdf > 0).groupby(groups).sum().reindex(group_set)
    e = s2 / g
    e_o = (s2.sum() - s2) / g_o
    fc = (m / m_o).fillna(0)
    pvals = mannwhitneyu_from_ranks(ranked_vdf_average, groups, group_set)
    return np.array(
        [r.values, m.values, m_o.values, e.values, e_o.values, fc.values, pvals.values]
    ).T


def test_batch_stats_matches_pandas_reference():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 250, 16
    # Zero-inflated counts exercise the tie correction path.
    data = rng.poisson(0.6, size=(n_cells, n_genes)).astype(np.float64)
    groups = rng.integers(1, 4, size=n_cells)
    group_set = np.array(sorted(set(groups)))
    idx_map = {v: i for i, v in enumerate(group_set)}
    int_indices = np.array([idx_map[x] for x in groups])
    group_counts = pd.Series(groups).value_counts().reindex(group_set).values

    ref = _reference_calc(pd.DataFrame(data), groups, group_set)
    got = _batch_stats(data, int_indices, group_counts, n_cells)

    # score, mean, mean_rest, frac_exp, frac_exp_rest
    assert np.allclose(got[:, :, :5], ref[:, :, :5], atol=1e-6)
    # fold_change agrees where the reference is finite
    finite = np.isfinite(ref[:, :, 5])
    assert np.allclose(got[:, :, 5][finite], ref[:, :, 5][finite], atol=1e-6)
    # two-sided p-values
    assert np.allclose(got[:, :, 6], ref[:, :, 6], atol=1e-6)


def test_iter_raw_feature_columns_matches_normed(datastore):
    assay = datastore.RNA
    cell_idx = assay.cells.active_index("I")
    feat_idx = assay.feats.active_index("I")
    scalar = assay.cells.fetch_all(assay.name + "_nCounts")[cell_idx]

    streamed = controlled_compute(
        assay.normed(cell_idx=cell_idx, feat_idx=feat_idx), assay.nthreads
    )

    cols = []
    mats = []
    for mat, batch_cols in assay.iter_raw_feature_columns(
        cell_idx=cell_idx,
        feat_idx=feat_idx,
        batch_size=37,
        scalar=scalar,
        sf=float(assay.sf),
        prefetch_depth=3,
    ):
        mats.append(mat)
        cols.append(batch_cols)

    fast = np.hstack(mats)
    assert np.array_equal(np.concatenate(cols), feat_idx)
    assert fast.shape == streamed.shape
    assert np.allclose(fast, streamed, rtol=1e-4, atol=1e-4)


def test_read_prenormed_batches_yields_expected_chunks():
    import zarr
    from zarr.storage import MemoryStore

    from scarf.markers import read_prenormed_batches

    root = zarr.open_group(store=MemoryStore(), mode="w")
    n_cells = 12
    for batch_id in range(5):
        arr = root.create_array(str(batch_id), shape=(n_cells,), dtype="f8")
        arr[:] = float(batch_id)

    cell_idx = np.arange(n_cells)
    batches = list(
        read_prenormed_batches(root, cell_idx, batch_size=2, desc="test batches")
    )

    assert len(batches) == 3
    assert batches[0].shape == (n_cells, 2)
    assert batches[1].shape == (n_cells, 2)
    assert batches[2].shape == (n_cells, 1)
    first_chunk = batches[0].iloc[:, 0]
    last_chunk = batches[2].iloc[:, 0]
    assert first_chunk.nunique() == 1
    assert last_chunk.nunique() == 1
    assert set(first_chunk.unique()).issubset({0.0, 1.0, 2.0, 3.0, 4.0})


def test_compact_marker_save_roundtrip():
    import zarr
    from zarr.storage import MemoryStore

    from scarf.datastore.datastore import DataStore, _load_marker_cluster_frame

    index = np.array([10, 5, 7], dtype=np.int32)
    source = pd.DataFrame(
        {
            "score": [0.9, 0.4, 0.2],
            "mean": [1.0, 2.0, 3.0],
            "mean_rest": [0.5, 0.5, 0.5],
            "frac_exp": [0.8, 0.2, 0.1],
            "frac_exp_rest": [0.3, 0.3, 0.3],
            "fold_change": [2.0, 4.0, 6.0],
            "p_value": [0.01, 0.02, 0.03],
        },
        index=index,
    )
    source["feature_index"] = source.index
    source = sort_marker_results(source)
    markers = {1: source}
    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    DataStore._write_marker_slot(slot, markers)
    feature_names = np.array([f"g{i}" for i in range(11)])
    loaded = _load_marker_cluster_frame(
        slot,
        slot["1"],
        feature_names,
        group_id=1,
    )
    assert slot.attrs["layout"] == "compact_v2"
    assert list(slot.group_keys()) == ["1", "feature_index"] or "feature_index" in slot
    assert len(loaded) == 3
    assert loaded.iloc[0]["score"] == 0.9
    assert loaded.iloc[0]["feature_name"] == "g10"
