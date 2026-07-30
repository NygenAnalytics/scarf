import numpy as np
import pandas as pd
import pytest

import scarf.features.markers.search as marker_search_module
from scarf.assay import norm_lib_size
from scipy.stats import linregress
from scipy.stats import mannwhitneyu

from scarf.features.markers import (
    find_markers_by_rank,
    find_markers_by_regression,
    mannwhitneyu_from_ranks,
    sort_marker_results,
)
from scarf.features.markers.rank import (
    _batch_stats,
    _batch_stats_gene_major,
    _marker_stats_batch,
    _marker_stats_gene_major,
)
from scarf.features.markers.regression import (
    _regression_batch_results,
    _regression_r_batch,
)
from scarf.utils import controlled_compute


def _reference_calc(
    vdf: pd.DataFrame, groups: np.ndarray, group_set: np.ndarray
) -> np.ndarray:
    """Independent pandas and SciPy implementation used as the parity reference."""
    ranked_vdf = vdf.rank(method="dense")
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
    pvals = pd.DataFrame(
        np.vstack(
            [
                mannwhitneyu(
                    vdf.loc[groups == group].to_numpy(),
                    vdf.loc[groups != group].to_numpy(),
                    axis=0,
                    alternative="two-sided",
                    method="asymptotic",
                    use_continuity=True,
                ).pvalue
                for group in group_set
            ]
        ),
        index=group_set,
        columns=vdf.columns,
    )
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


def test_rank_paths_match_scipy_continuity_correction():
    data = np.array(
        [
            [8, 0, 0],
            [7, 1, 1],
            [6, 1, 2],
            [5, 2, 3],
            [4, 3, 0],
            [3, 3, 1],
            [2, 4, 2],
            [1, 5, 3],
        ],
        dtype=np.uint32,
    )
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    group_set = np.array([0, 1])
    group_counts = np.bincount(groups)
    expected = np.stack(
        [
            mannwhitneyu(
                data[groups == group],
                data[groups != group],
                axis=0,
                alternative="two-sided",
                method="asymptotic",
                use_continuity=True,
            ).pvalue
            for group in group_set
        ],
        axis=1,
    )

    ranked = pd.DataFrame(data).rank(method="average")
    from_ranks = mannwhitneyu_from_ranks(ranked, groups, group_set).to_numpy().T
    cell_major = _batch_stats(data, groups, group_counts, len(groups))[:, :, 6]
    gene_major = _batch_stats_gene_major(
        data.T,
        np.ones(len(groups), dtype=np.float32),
        1.0,
        False,
        groups,
        group_counts,
        len(groups),
    )[:, :, 6]

    np.testing.assert_allclose(from_ranks, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cell_major, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gene_major, expected, rtol=1e-12, atol=1e-12)


def test_tie_correction_survives_more_than_two_million_tied_values():
    # A cubed int64 tie group wraps negative beyond roughly 2.08 million ties, which
    # inflates the variance instead of collapsing it.
    n_zeros = 2_200_000
    values = np.concatenate([np.zeros(n_zeros), np.array([10.0, 20.0, 30.0, 40.0])])
    groups = np.concatenate(
        [np.ones(n_zeros, dtype=np.int64), np.zeros(4, dtype=np.int64)]
    )
    group_set = np.array([0, 1])
    group_counts = np.bincount(groups)
    n_total = values.size

    ranked = pd.DataFrame({"feature": values}).rank(method="average")
    from_ranks = mannwhitneyu_from_ranks(ranked, groups, group_set)["feature"]
    cell_major = _batch_stats(values[:, None], groups, group_counts, n_total)[0, :, 6]
    gene_major = _batch_stats_gene_major(
        values.astype(np.uint32)[None, :],
        np.ones(n_total, dtype=np.float32),
        1.0,
        False,
        groups,
        group_counts,
        n_total,
    )[0, :, 6]

    # The feature is confined to one group's four cells, so separation is maximal.
    assert (from_ranks.to_numpy() < 1e-100).all()
    assert (cell_major < 1e-100).all()
    assert (gene_major < 1e-100).all()


def test_mannwhitneyu_from_ranks_returns_one_for_zero_variance():
    ranked = pd.DataFrame({"constant": np.ones(4)}).rank(method="average")
    groups = np.array([0, 0, 1, 1])

    with np.errstate(divide="raise", invalid="raise"):
        p_values = mannwhitneyu_from_ranks(ranked, groups, np.array([0, 1]))

    np.testing.assert_array_equal(p_values["constant"], [1.0, 1.0])


def test_batch_stats_distinguishes_zero_fold_change_from_zero_rest_sentinel():
    data = np.array(
        [
            [0.0, 2.0, 1.0],
            [0.0, 4.0, 3.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ]
    )
    stats = _batch_stats(
        data,
        int_indices=np.array([0, 0, 1, 1]),
        group_counts=np.array([2, 2]),
        n_total=4,
    )

    assert np.array_equal(stats[0, :, 5], [0.0, 0.0])
    assert stats[1, 0, 5] == pytest.approx(100.1)
    assert stats[1, 1, 5] == pytest.approx(0.0)
    assert np.array_equal(stats[2, :, 5], [1.0, 1.0])
    assert np.array_equal(stats[0, :, 6], [1.0, 1.0])


def test_marker_stats_python_kernel_matches_compiled_kernel():
    data = np.array(
        [
            [0.0, 5.0, 0.0, 1.0],
            [0.0, 4.0, 1.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 3.0, 2.0],
            [0.0, 0.0, 4.0, 3.0],
            [0.0, 0.0, 5.0, 3.0],
        ],
        dtype=np.float32,
    )
    int_indices = np.array([0, 0, 1, 1, 2, 2])
    group_counts = np.array([2, 2, 2], dtype=np.float32)
    n_total = np.float32(len(data))

    python_stats = _marker_stats_batch.py_func(
        data,
        int_indices,
        group_counts,
        n_total,
    )
    compiled_stats = _marker_stats_batch(
        data,
        int_indices,
        group_counts,
        n_total,
    )

    np.testing.assert_allclose(python_stats, compiled_stats)
    assert python_stats[1, 0, 5] == pytest.approx(100.1)
    assert np.array_equal(python_stats[0, :, 5], [0.0, 0.0, 0.0])


def test_marker_stats_python_kernel_handles_single_cell_population():
    stats = _marker_stats_batch.py_func(
        np.array([[0.0, 2.0]], dtype=np.float32),
        np.array([0]),
        np.array([1], dtype=np.float32),
        np.float32(1),
    )

    np.testing.assert_allclose(
        stats[:, 0, :],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 1.0, 0.0, 100.1, 0.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "raw",
    [
        np.array(
            [
                [1, 2, 3],
                [3, 2, 1],
                [2, 1, 4],
                [4, 3, 2],
            ],
            dtype=np.uint32,
        ),
        np.zeros((4, 3), dtype=np.uint32),
        np.full((4, 3), 2, dtype=np.uint32),
        np.array(
            [
                [0, 0, 0],
                [0, 0, 5],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint32,
        ),
        np.array(
            [
                [0, 2, 1],
                [0, 2, 1],
                [3, 2, 0],
                [3, 2, 0],
            ],
            dtype=np.uint32,
        ),
        np.array([[0, 2, 2]], dtype=np.uint32),
    ],
    ids=[
        "no-zeros",
        "all-zero",
        "constant",
        "single-nonzero",
        "heavy-ties",
        "single-cell",
    ],
)
@pytest.mark.parametrize("log_transform", [False, True])
def test_gene_major_zero_aware_kernel_is_bit_identical(
    raw: np.ndarray,
    log_transform: bool,
) -> None:
    n_cells = raw.shape[0]
    groups = np.arange(n_cells, dtype=np.int64) % max(1, min(3, n_cells))
    group_counts = np.bincount(
        groups,
        minlength=int(groups.max()) + 1,
    )
    scalar = raw.sum(axis=1).astype(np.float32)
    scalar[scalar == 0] = 1
    normalized = (float(1000.0) * raw.astype(np.float32)) / scalar[:, None]
    if log_transform:
        normalized = np.log1p(normalized)

    expected = _batch_stats(
        normalized,
        groups,
        group_counts,
        n_cells,
    )
    observed = _batch_stats_gene_major(
        raw.T,
        scalar,
        1000.0,
        log_transform,
        groups,
        group_counts,
        n_cells,
    )

    np.testing.assert_array_equal(observed, expected)


def test_gene_major_python_kernel_matches_compiled_kernel() -> None:
    raw = np.array(
        [
            [0, 2, 0, 4],
            [1, 2, 0, 0],
            [1, 0, 3, 4],
            [0, 0, 3, 0],
        ],
        dtype=np.uint32,
    ).T
    scalar = np.array([2, 4, 6, 8], dtype=np.float32)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    group_counts = np.array([2, 2], dtype=np.float32)
    destinations = np.arange(raw.shape[0], dtype=np.int64)
    compiled = np.zeros((raw.shape[0], 2, 7), dtype=np.float64)
    python = np.zeros_like(compiled)
    args = (
        raw,
        scalar,
        np.float32(1000),
        False,
        groups,
        group_counts,
        np.float32(4),
        destinations,
    )

    _marker_stats_gene_major(*args, compiled)
    _marker_stats_gene_major.py_func(*args, python)

    np.testing.assert_allclose(compiled, python, rtol=1e-7, atol=1e-12)


def test_sort_marker_results_adds_deterministic_tie_breakers():
    named = pd.DataFrame(
        {
            "score": [0.8, 0.8, 0.8],
            "p_value": [0.02, 0.01, 0.01],
            "feature_name": ["zeta", "beta", "alpha"],
        },
        index=[7, 8, 9],
    )

    sorted_named = sort_marker_results(named)
    unnamed = named.drop(columns="feature_name").iloc[[0, 2]].copy()
    unnamed["p_value"] = 0.01
    sorted_unnamed = sort_marker_results(unnamed)

    assert "feature_index" not in named
    assert sorted_named["feature_name"].tolist() == ["alpha", "beta", "zeta"]
    assert sorted_named["feature_index"].tolist() == [9, 8, 7]
    assert sorted_unnamed["feature_index"].tolist() == [7, 9]


def test_find_markers_by_regression_handles_expression_threshold():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield pd.DataFrame(
                {
                    "correlated": [0.0, 1.0, 2.0, 3.0],
                    "at_threshold": [0.0, 1.0, 2.0, 0.0],
                    "too_sparse": [0.0, 0.0, 1.0, 0.0],
                    "constant": [1.0, 1.0, 1.0, 1.0],
                }
            )

    result = find_markers_by_regression(
        Assay(),
        cell_key="I",
        feat_key="I",
        regressor=np.arange(4),
        min_cells=2,
    )

    assert result.loc["correlated", "r_value"] == pytest.approx(1.0)
    assert result.loc["correlated", "p_value"] < 1e-10
    assert result.loc["at_threshold", "r_value"] != 0.0
    assert np.array_equal(result.loc["too_sparse"].to_numpy(), [0.0, 1.0])
    assert np.array_equal(result.loc["constant"].to_numpy(), [0.0, 1.0])


def test_regression_r_batch_matches_py_func():
    rng = np.random.default_rng(0)
    data = rng.poisson(0.8, size=(40, 8)).astype(np.float64)
    regressor = np.linspace(0.0, 1.0, 40)
    x_centered = regressor - regressor.mean()
    ssxm = float(np.dot(x_centered, x_centered) / regressor.size)
    kwargs = (
        np.ascontiguousarray(data),
        np.ascontiguousarray(x_centered),
        ssxm,
        2,
        float(np.finfo(float).eps),
    )
    compiled = _regression_r_batch(*kwargs)
    python = _regression_r_batch.py_func(*kwargs)
    np.testing.assert_allclose(compiled[0], python[0])
    np.testing.assert_array_equal(compiled[1], python[1])


def test_regression_batch_matches_linregress():
    rng = np.random.default_rng(1)
    n_cells = 50
    regressor = np.linspace(-1.0, 2.0, n_cells)
    data = np.column_stack(
        [
            2.0 * regressor + 0.1,
            -1.5 * regressor,
            rng.poisson(0.5, size=n_cells).astype(float),
            np.zeros(n_cells),
            np.where(np.arange(n_cells) < 3, 1.0, 0.0),
        ]
    )
    x_centered = regressor - regressor.mean()
    ssxm = float(np.dot(x_centered, x_centered) / n_cells)
    labels = np.array(["pos", "neg", "sparseish", "constant", "too_sparse"])
    r_vals, p_vals = _regression_batch_results(
        np.ascontiguousarray(data),
        np.ascontiguousarray(x_centered),
        ssxm,
        regressor,
        min_cells=5,
        feature_labels=labels,
    )
    for i, label in enumerate(labels):
        v = data[:, i]
        if (v > 0).sum() >= 5 and np.ptp(v) > np.finfo(float).eps:
            ref = linregress(regressor, v)
            assert r_vals[i] == pytest.approx(ref.rvalue, rel=1e-10, abs=1e-12)
            assert p_vals[i] == pytest.approx(ref.pvalue, rel=1e-8, abs=1e-12)
        else:
            assert r_vals[i] == 0.0
            assert p_vals[i] == 1.0


def test_find_markers_by_regression_two_cell_fallback():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 1.0]})

    result = find_markers_by_regression(
        Assay(),
        cell_key="I",
        feat_key="I",
        regressor=np.array([0.0, 1.0]),
        min_cells=1,
    )
    assert result.loc["a", "r_value"] == pytest.approx(1.0)
    assert np.array_equal(result.loc["b"].to_numpy(), [0.0, 1.0])


def test_find_markers_by_regression_rejects_non_dataframe_batches():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield np.ones((3, 1))

    with pytest.raises(TypeError, match="DataFrames"):
        find_markers_by_regression(
            Assay(),
            cell_key="I",
            feat_key="I",
            regressor=np.arange(3),
            min_cells=1,
        )


def test_find_markers_by_regression_identifies_nonfinite_feature():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield pd.DataFrame({"bad_feature": [0.0, np.nan, 1.0]})

    with pytest.raises(ValueError, match="bad_feature"):
        find_markers_by_regression(
            Assay(),
            cell_key="I",
            feat_key="I",
            regressor=np.arange(3),
            min_cells=1,
        )


def test_find_markers_by_rank_rejects_fast_path_for_non_rna_assay():
    import numba

    class Cells:
        @staticmethod
        def fetch(_group_key, _cell_key):
            return np.array([0, 1])

    class Assay:
        def __init__(self):
            self.cells = Cells()
            self.normMethod = norm_lib_size
            self.sf = 1.0

    previous_threads = numba.get_num_threads()
    with pytest.raises(TypeError, match="requires an RNAassay"):
        find_markers_by_rank(
            Assay(),
            group_key="cluster",
            cell_key="I",
            feat_key="I",
            batch_size=2,
            n_threads=1,
        )
    assert numba.get_num_threads() == previous_threads


def test_find_markers_by_rank_slow_path_returns_groupwise_statistics():
    data = np.array(
        [
            [2.0, 0.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 5.0, 1.0],
            [0.0, 0.0, 5.0, 1.0],
        ]
    )

    class Cells:
        @staticmethod
        def fetch(_group_key, _cell_key):
            return np.array(["a", "a", "b", "b"])

    class Feats:
        @staticmethod
        def active_index(_feat_key):
            return np.array([10, 11, 12, 13])

    class Assay:
        cells = Cells()
        feats = Feats()
        normMethod = None
        sf = None

        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield pd.DataFrame(data[:, :2])
            yield pd.DataFrame(data[:, 2:])

    results = find_markers_by_rank(
        Assay(),
        group_key="cluster",
        cell_key="I",
        feat_key="I",
        batch_size=2,
        n_threads=1,
    )
    group_a = results["a"].set_index("feature_index")
    group_b = results["b"].set_index("feature_index")

    assert group_a.loc[10, "fold_change"] == pytest.approx(100.1)
    assert group_a.loc[11, "fold_change"] == pytest.approx(0.0)
    assert group_a.loc[13, "fold_change"] == pytest.approx(1.0)
    assert group_b.loc[12, "fold_change"] == pytest.approx(100.1)
    assert group_b.loc[10, "fold_change"] == pytest.approx(0.0)
    assert np.isfinite(group_a["p_value"]).all()
    assert np.isfinite(group_b["p_value"]).all()


def test_find_markers_fast_raw_path_computes_groupwise_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zarr
    from zarr.storage import MemoryStore

    from scarf.storage.budget import ResourceBudget

    data = np.array(
        [
            [4.0, 0.0, 1.0, 0.0],
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 5.0, 1.0, 2.0],
            [0.0, 6.0, 1.0, 2.0],
        ]
    )

    class Cells:
        @staticmethod
        def fetch(_group_key, _cell_key):
            return np.array(["a", "a", "b", "b"])

        @staticmethod
        def active_index(_cell_key):
            return np.arange(4)

        @staticmethod
        def fetch_all(_key):
            return data.sum(axis=1)

    class Feats:
        @staticmethod
        def active_index(_feat_key):
            return np.arange(4)

    class FakeRNA:
        def __init__(self):
            self.cells = Cells()
            self.feats = Feats()
            self.normMethod = norm_lib_size
            self.sf = 1_000.0
            self.name = "RNA"
            self.resources = ResourceBudget(1024**3, 2)
            root = zarr.open_group(store=MemoryStore(), mode="w")
            self.raw = root.create_array(
                "counts",
                data=data.astype(np.uint32),
                chunks=(2, 2),
            )

        def _raw_feature_stream_source(self):
            return self.raw, 1, 0

        @staticmethod
        def iter_raw_feature_major_blocks(cell_idx, plan, **_kwargs):
            for block in plan.blocks:
                yield (
                    block,
                    np.ascontiguousarray(
                        data[np.asarray(cell_idx)][:, block.indices].T
                    ),
                    0.01,
                    "memory",
                )

    monkeypatch.setattr(marker_search_module, "RNAassay", FakeRNA)
    results = find_markers_by_rank(
        FakeRNA(),
        group_key="cluster",
        cell_key="I",
        feat_key="I",
        batch_size=2,
        n_threads=1,
    )

    assert set(results) == {"a", "b"}
    for frame in results.values():
        assert len(frame) == data.shape[1]
        assert np.isfinite(frame["p_value"]).all()


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
    ):
        mats.append(mat)
        cols.append(batch_cols)

    fast = np.hstack(mats)
    assert np.array_equal(np.concatenate(cols), feat_idx)
    assert fast.shape == streamed.shape
    assert np.allclose(fast, streamed, rtol=1e-4, atol=1e-4)


def test_compact_marker_save_roundtrip():
    import zarr
    from zarr.storage import MemoryStore

    from scarf.datastore._operations.features import _load_marker_cluster_frame
    from scarf.datastore.datastore import DataStore

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
    assert "layout" not in slot.attrs
    assert "feature_index" in slot
    assert "stats" in slot["1"]
    assert len(loaded) == 3
    assert loaded.iloc[0]["score"] == 0.9
    assert loaded.iloc[0]["feature_name"] == "g10"


def test_legacy_marker_names_and_scores_are_readable():
    import zarr
    from zarr.storage import MemoryStore

    from scarf.datastore._operations.features import _load_marker_cluster_frame

    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    cluster = slot.create_group("1")
    cluster.create_array(
        "names",
        data=np.array(["id2", "id0"]),
    )
    cluster.create_array(
        "scores",
        data=np.array([0.9, 0.8]),
    )

    loaded = _load_marker_cluster_frame(
        slot,
        cluster,
        np.array(["gene0", "gene1", "gene2"]),
        group_id=1,
        feature_ids=np.array(["id0", "id1", "id2"]),
    )

    assert loaded["feature_index"].tolist() == [2, 0]
    assert loaded["feature_name"].tolist() == ["gene2", "gene0"]
    assert loaded["score"].tolist() == [0.9, 0.8]


def test_explicit_marker_gene_batch_size_reaches_search(
    datastore_ephemeral,
    monkeypatch,
):
    groups = np.arange(datastore_ephemeral.cells.N) % 2
    datastore_ephemeral.cells.insert("batch_contract_groups", groups, overwrite=True)
    captured = {}
    expected = {"group": pd.DataFrame()}

    def capture_marker_search(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "scarf.features.markers.find_markers_by_rank",
        capture_marker_search,
    )
    monkeypatch.setattr(
        datastore_ephemeral,
        "_get_latest_keys",
        lambda from_assay, cell_key, feat_key: ("RNA", "I", "I"),
    )

    result = datastore_ephemeral.run_marker_search(
        group_key="batch_contract_groups",
        gene_batch_size=100,
        skip_save=True,
    )

    assert result is expected
    assert captured["batch_size"] == 100
