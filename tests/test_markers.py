import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest

import scarf.features.markers.search as marker_search_module
from scarf.assay import norm_lib_size
from scarf.datastore.datastore import DataStore
from scipy.stats import linregress
from scipy.stats import mannwhitneyu

from scarf.features.markers import (
    RankMarkerResult,
    find_markers_by_rank,
    find_markers_by_regression,
    mannwhitneyu_from_ranks,
    sort_marker_results,
)
from scarf.features.markers.rank import (
    _batch_stats,
    _gene_major_feature,
    _marker_stats_batch,
    _marker_stats_gene_major,
)
from scarf.features.markers.regression import (
    _REG_SENTINEL,
    _regression_batch_results,
    _regression_r_batch,
)
from scarf.storage.artifacts import ArtifactRef
from scarf.features.statistical import adjust_pvalues


def _gene_major_stats(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: int,
    *,
    threads: int = 2,
) -> np.ndarray:
    """Run the feature-major kernel over every row and return its z statistics."""
    out = np.zeros((raw.shape[0], len(group_counts), 8), dtype=np.float64)
    invalid = _marker_stats_gene_major(
        np.ascontiguousarray(raw),
        np.asarray(totals, dtype=np.float64),
        float(size_factor),
        bool(log_transform),
        np.asarray(int_indices, dtype=np.int64),
        np.asarray(group_counts, dtype=np.float64),
        float(n_total),
        np.arange(raw.shape[0], dtype=np.int64),
        threads,
        out,
    )
    assert invalid == -1
    return out


def _p_values(statistics: np.ndarray) -> np.ndarray:
    """Return the stored two-sided p-values of kernel statistics of every group."""
    n_features, n_groups, _ = statistics.shape
    result = RankMarkerResult(
        group_ids=np.arange(n_groups),
        group_sizes=np.full(n_groups, 2),
        feature_index=np.arange(n_features),
        statistics=statistics,
    )
    return np.column_stack(
        [result.stored_statistics(group)[:, 6] for group in range(n_groups)]
    )


def test_marker_public_contract_requires_explicit_artifacts() -> None:
    producer = inspect.signature(DataStore.run_marker_search)
    assert list(producer.parameters)[:2] == ["self", "clusters"]
    assert "features" in producer.parameters
    assert "group_key" not in producer.parameters
    assert "cell_key" not in producer.parameters
    assert "skip_save" not in producer.parameters
    loader = inspect.signature(DataStore.get_markers)
    assert loader.parameters["marker"].default is inspect.Parameter.empty


def _cluster_labels(store, values: np.ndarray) -> ArtifactRef:
    """Freeze one label per active cell as a cluster-label artifact."""
    store.cells.insert("test_clusters", np.asarray(values), overwrite=True)
    return store.snapshot_cluster_labels(
        "test_clusters", cell_selection=store.snapshot_cell_selection()
    )


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
    assert np.allclose(_p_values(got), ref[:, :, 6], atol=1e-6)


def test_batch_stats_preserves_float64_near_ties_against_scipy():
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    values = 1.0 + 1e-9 * np.array([5, 6, 7, 8, 1, 2, 3, 4])
    data = values[:, None]
    assert np.unique(data.astype(np.float32)).size == 1

    expected = mannwhitneyu(
        values[groups == 0],
        values[groups == 1],
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    got = _batch_stats(data, groups, np.bincount(groups), len(groups))

    assert _p_values(got)[0, 0] == pytest.approx(expected.pvalue, rel=1e-12, abs=1e-15)
    assert got[0, 0, 7] == pytest.approx(
        expected.statistic / 16.0,
        rel=1e-12,
        abs=1e-15,
    )


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
    cell_major = _p_values(_batch_stats(data, groups, group_counts, len(groups)))
    gene_major = _p_values(
        _gene_major_stats(
            data.T,
            np.ones(len(groups)),
            1.0,
            False,
            groups,
            group_counts,
            len(groups),
        )
    )

    np.testing.assert_allclose(from_ranks, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cell_major, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gene_major, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method_name", ["norm_lib_size", "norm_dummy"])
def test_complementary_groups_get_the_exact_mann_whitney_p_value(
    tmp_path, method_name
) -> None:
    import scarf.assay.normalization as normalization

    # The product of these group sizes exceeds 2**24, which float32 rounds.
    labels = np.repeat(np.array(["a", "b"]), [4_097, 4_099])
    counts = np.random.default_rng(7).poisson(1.5, size=(labels.size, 3))
    values = counts.astype(np.float64)
    store = _count_store(tmp_path, values, "uint16")
    store.RNA.normMethod = getattr(normalization, method_name)
    result = find_markers_by_rank(
        store.RNA, labels, np.arange(labels.size), np.arange(3)
    )

    if method_name == "norm_lib_size":
        # The zero-aware kernel ranks the float32 rounding of each value;
        # SciPy computes in the dtype of its input.
        totals = values.sum(axis=1)
        totals[totals == 0] = 1
        values = 1000.0 * values / totals[:, None]
        values = values.astype(np.float32).astype(np.float64)
    expected = mannwhitneyu(
        values[labels == "a"],
        values[labels == "b"],
        axis=0,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    ).pvalue
    first, second = (result.stored_statistics(group)[:, 6] for group in "ab")
    # Both one-versus-rest tests of two groups are the same test.
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, expected, rtol=1e-12)


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
    cell_major = _p_values(
        _batch_stats(values[:, None], groups, group_counts, n_total)
    )[0]
    gene_major = _p_values(
        _gene_major_stats(
            values.astype(np.uint32)[None, :],
            np.ones(n_total),
            1.0,
            False,
            groups,
            group_counts,
            n_total,
        )
    )[0]

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
    assert np.array_equal(_p_values(stats)[0], [1.0, 1.0])


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
    group_counts = np.array([2, 2, 2], dtype=np.float64)
    n_total = float(len(data))

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
        np.array([1], dtype=np.float64),
        1.0,
    )

    np.testing.assert_allclose(
        stats[:, 0, :7],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 1.0, 0.0, 100.1, 0.0],
            ]
        ),
    )
    assert np.isnan(stats[:, 0, 7]).all()


@pytest.mark.parametrize(
    ("sums", "detected", "ranks", "dense_ranks", "sizes", "n_total", "ties"),
    [
        ([3, 0, 0], [2, 0, 0], [9, 4, 8], [5, 2, 4], [2, 2, 2], 6, 6),
        # An empty group and an empty complement, with no ranks or ties.
        ([0, 0], [0, 0], [0, 0], [0, 0], [0, 1], 1, 0),
    ],
    ids=["groups", "empty"],
)
def test_group_statistics_python_kernel_matches_compiled_kernel(
    sums, detected, ranks, dense_ranks, sizes, n_total, ties
) -> None:
    from scarf.features.markers.rank import _write_group_statistics

    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (sums, detected, ranks, dense_ranks, sizes)
    ]
    outcomes = []
    for kernel in (_write_group_statistics, _write_group_statistics.py_func):
        out = np.zeros((len(sizes), 8))
        kernel(out, *arrays, float(n_total), float(ties))
        outcomes.append(out)

    np.testing.assert_array_equal(outcomes[0], outcomes[1])
    assert np.isfinite(outcomes[0][:, :7]).all()


def test_radix_argsort_python_kernel_sorts_like_a_stable_sort() -> None:
    from scarf.features.markers.rank import _argsort_positive

    values = np.array([3.0, 0.5, 3.0, 1e-30, 7.25, 0.5, 1e30], dtype=np.float32)
    for kernel in (_argsort_positive, _argsort_positive.py_func):
        # The kernel orders the first ``n`` values of longer scratch arrays.
        order = np.full(values.size + 2, -1, dtype=np.int64)
        buckets = np.empty(2048, dtype=np.int64)
        kernel(values, values.size, order, np.empty_like(order), buckets)

        np.testing.assert_array_equal(
            order[: values.size], np.argsort(values, kind="stable")
        )


def test_batch_stats_rejects_malformed_values() -> None:
    codes = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="two-dimensional"):
        _batch_stats(np.zeros(4), codes, np.array([4]), 4)
    with pytest.raises(ValueError, match="labels must match"):
        _batch_stats(np.zeros((4, 2)), codes, np.array([4]), 4, np.arange(3))


def test_marker_groups_use_natural_order_and_string_ids(
    datastore_ephemeral, tmp_path
) -> None:
    store = datastore_ephemeral
    n_cells = len(store.cells.active_index("I"))
    clusters = _cluster_labels(store, np.arange(n_cells, dtype=np.int64) % 12)
    marker = store.run_marker_search(
        clusters,
        from_assay="RNA",
        features=store.set_feature_selection(feature_indexes=np.arange(20)),
        nthreads=1,
    )
    natural = [str(value) for value in range(12)]

    table = store.get_markers(marker=marker, min_score=-1, min_frac_exp=-1)
    assert list(dict.fromkeys(table["group_id"])) == natural
    assert all(isinstance(value, str) for value in table["group_id"])
    one = store.get_markers(marker=marker, group_id=10, min_score=-1, min_frac_exp=-1)
    assert set(one["group_id"]) == {"10"}
    pd.testing.assert_frame_equal(
        one,
        table[table["group_id"] == "10"].reset_index(drop=True),
    )
    with pytest.raises(ValueError, match="no group '12'"):
        store.get_markers(marker=marker, group_id=12)

    out_file = tmp_path / "markers.csv"
    store.export_markers_to_csv(marker, str(out_file), min_score=-1, min_frac_exp=-1)
    assert list(pd.read_csv(out_file).columns) == natural


def test_marker_group_order_matches_plot_category_order() -> None:
    from scarf.utils.arrays import sort_categories

    assert sort_categories(["10", "2", "-1", "1.5"]) == ["-1", "1.5", "2", "10"]
    # Mixed labels put numbers first by value, then natural text.
    assert sort_categories(["b", "10", "a10", "2", "a2"]) == [
        "2",
        "10",
        "a2",
        "a10",
        "b",
    ]
    # A "nan" label is text, so its position does not depend on input order.
    assert sort_categories(["nan", "2", "10"]) == ["2", "10", "nan"]
    assert sort_categories(["10", "nan", "2"]) == ["2", "10", "nan"]
    # Digit runs precede text where labels differ in kind, and only decimal
    # notation is numeric, so "1_10" is not the number 110.
    assert sort_categories(["B cell", "2_T", "2_1", "1_10", "1_2", "²"]) == [
        "1_2",
        "1_10",
        "2_1",
        "2_T",
        "B cell",
        "²",
    ]
    assert sort_categories(["b", "B"]) == sort_categories(["B", "b"]) == ["B", "b"]


def test_marker_readers_accept_mixed_digit_and_text_labels(
    datastore_ephemeral, tmp_path
) -> None:
    store = datastore_ephemeral
    n_cells = len(store.cells.active_index("I"))
    labels = np.array(["B cell", "2_T", "1_10"])[np.arange(n_cells) % 3]
    marker = store.run_marker_search(
        _cluster_labels(store, labels),
        from_assay="RNA",
        features=store.set_feature_selection(feature_indexes=np.arange(20)),
        nthreads=1,
    )
    expected = ["1_10", "2_T", "B cell"]

    table = store.get_markers(marker=marker, min_score=-1, min_frac_exp=-1)
    assert list(dict.fromkeys(table["group_id"])) == expected
    one = store.get_markers(
        marker=marker, group_id="2_T", min_score=-1, min_frac_exp=-1
    )
    assert set(one["group_id"]) == {"2_T"}
    out_file = tmp_path / "markers.csv"
    store.export_markers_to_csv(marker, str(out_file), min_score=-1, min_frac_exp=-1)
    assert list(pd.read_csv(out_file).columns) == expected


def _refuse_marker_search(*_args, **_kwargs):
    raise AssertionError("the rank marker search must not run")


@pytest.mark.parametrize("label", ["NK/T", "NK\\T", ".", ".."])
def test_marker_search_rejects_labels_that_cannot_name_a_group_before_search(
    datastore_ephemeral, monkeypatch, label
) -> None:
    import scarf.features.markers as markers_package

    store = datastore_ephemeral
    n_cells = len(store.cells.active_index("I"))
    labels = np.where(np.arange(n_cells) % 2 == 0, label, "B")
    clusters = _cluster_labels(store, labels)
    features = store.set_feature_selection(feature_indexes=np.arange(20))
    monkeypatch.setattr(markers_package, "find_markers_by_rank", _refuse_marker_search)

    with pytest.raises(ValueError, match="cannot name a stored marker group"):
        store.run_marker_search(clusters, from_assay="RNA", features=features)
    assert store.list_artifacts(kind="marker_table", from_assay="RNA") == []


@pytest.mark.parametrize("label", ["", "  "])
def test_blank_labels_cannot_name_a_marker_group(label) -> None:
    # Snapshots reject blank labels, but imported label artifacts can hold them.
    from scarf.datastore._operations.features import _validate_marker_group_name

    with pytest.raises(ValueError, match="cannot name a stored marker group"):
        _validate_marker_group_name(label)


def test_marker_search_on_a_read_only_store_reuses_but_never_searches(
    datastore_ephemeral, monkeypatch
) -> None:
    import scarf.features.markers as markers_package

    store = datastore_ephemeral
    n_cells = len(store.cells.active_index("I"))
    features = store.set_feature_selection(feature_indexes=np.arange(20))
    saved = _cluster_labels(store, np.arange(n_cells) % 2)
    fresh = _cluster_labels(store, np.arange(n_cells) % 3)
    marker = store.run_marker_search(saved, from_assay="RNA", features=features)
    read_only = DataStore(store.zarr_loc, zarr_mode="r", nthreads=1)
    monkeypatch.setattr(markers_package, "find_markers_by_rank", _refuse_marker_search)

    assert (
        read_only.run_marker_search(saved, from_assay="RNA", features=features)
        == marker
    )
    with pytest.raises(PermissionError, match="run_marker_search requires"):
        read_only.run_marker_search(fresh, from_assay="RNA", features=features)


def test_saved_marker_refs_keep_feature_specific_results_addressable(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    assay = store.RNA
    assert assay.feats.N >= 16
    clusters = _cluster_labels(
        store,
        np.arange(len(store.cells.active_index("I")), dtype=np.int64) % 2,
    )
    first_indices = np.arange(8, dtype=np.int64)
    second_indices = np.arange(8, 16, dtype=np.int64)
    first_selection = store.set_feature_selection(
        feature_indexes=first_indices,
    )
    second_selection = store.set_feature_selection(
        feature_indexes=second_indices,
    )
    cell_columns = set(store.cells.columns)
    feature_columns = set(assay.feats.columns)
    assay_keys = set(assay.z.keys())

    first = store.run_marker_search(
        clusters,
        from_assay="RNA",
        features=first_selection,
        nthreads=1,
    )
    second = store.run_marker_search(
        clusters,
        from_assay="RNA",
        features=second_selection,
        nthreads=1,
    )

    assert isinstance(first, ArtifactRef)
    assert isinstance(second, ArtifactRef)
    assert first != second
    first_table = store.get_markers(
        marker=first,
        min_score=-1,
        min_frac_exp=-1,
    )
    second_table = store.get_markers(
        marker=second,
        min_score=-1,
        min_frac_exp=-1,
    )
    assert set(first_table["feature_index"]) == set(first_indices)
    assert set(second_table["feature_index"]) == set(second_indices)
    assert set(store.cells.columns) == cell_columns
    assert set(assay.feats.columns) == feature_columns
    assert set(assay.z.keys()) == assay_keys
    for marker in (first, second):
        group = store.load_artifact(marker)
        assert "feature_names" in group
        assert "feature_ids" in group
    with pytest.raises(TypeError, match="marker"):
        store.get_markers()  # type: ignore[call-arg]


_COUNT_DTYPES = ("uint8", "uint16", "uint32", "int32", "int64", "float32", "float64")


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
@pytest.mark.parametrize("dtype", _COUNT_DTYPES)
@pytest.mark.parametrize("log_transform", [False, True])
def test_gene_major_zero_aware_kernel_is_bit_identical(
    raw: np.ndarray,
    dtype: str,
    log_transform: bool,
) -> None:
    from scarf.assay.normalization import library_size_values

    counts = raw.astype(dtype)
    n_cells = counts.shape[0]
    groups = np.arange(n_cells, dtype=np.int64) % max(1, min(3, n_cells))
    group_counts = np.bincount(
        groups,
        minlength=int(groups.max()) + 1,
    )
    totals = counts.sum(axis=1, dtype=np.float64)
    totals[totals == 0] = 1
    # The dense kernel ranks the float32 rounding of the float64 values.
    normalized = library_size_values(
        counts, totals, 1000.0, dtype=np.float64, log_transform=log_transform
    ).astype(np.float32)

    expected = _batch_stats(
        normalized,
        groups,
        group_counts,
        n_cells,
    )
    observed = _gene_major_stats(
        counts.T,
        totals,
        1000.0,
        log_transform,
        groups,
        group_counts,
        n_cells,
    )

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("log_transform", [False, True])
def test_gene_major_kernel_ranks_fractional_counts_like_the_dense_kernel(
    dtype: str, log_transform: bool
) -> None:
    from scarf.assay.normalization import library_size_values

    rng = np.random.default_rng(5)
    # Values on a 0.1 grid with many zeros and ties, as corrected counts hold.
    counts = np.round(rng.random((60, 7)) * 3, 1).astype(dtype)
    counts[rng.random(counts.shape) < 0.5] = 0
    counts[:, 0] = 0
    groups = np.arange(60, dtype=np.int64) % 4
    group_counts = np.bincount(groups)
    totals = counts.sum(axis=1, dtype=np.float64)
    totals[totals == 0] = 1
    normalized = library_size_values(
        counts, totals, 1000.0, dtype=np.float64, log_transform=log_transform
    ).astype(np.float32)

    np.testing.assert_array_equal(
        _gene_major_stats(
            counts.T, totals, 1000.0, log_transform, groups, group_counts, 60
        ),
        _batch_stats(normalized, groups, group_counts, 60),
    )


def _gene_major_inputs(raw: np.ndarray) -> tuple[np.ndarray, ...]:
    n_cells = raw.shape[1]
    groups = np.arange(n_cells, dtype=np.int64) % 2
    return (
        np.ascontiguousarray(raw),
        np.full(n_cells, 10.0),
        1000.0,
        False,
        groups,
        np.bincount(groups).astype(np.float64),
        float(n_cells),
    )


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
    args = (*_gene_major_inputs(raw), np.arange(raw.shape[0], dtype=np.int64), 3)
    compiled = np.zeros((raw.shape[0], 2, 8), dtype=np.float64)
    python = np.zeros_like(compiled)

    assert _marker_stats_gene_major(*args, compiled) == -1
    assert _marker_stats_gene_major.py_func(*args, python) == -1

    np.testing.assert_array_equal(compiled, python)
    assert compiled.any()


@pytest.mark.parametrize(
    "counts",
    [
        np.array([0.0, 2.0, 1.0, 3.0, 0.0, 4.0]),
        # Tied nonzero values beside a single zero.
        np.array([0.0, 2.0, 2.0, 3.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 1.0, 3.0, 5.0, 4.0]),
        # A positive count whose value rounds to a float32 zero.
        np.array([0.0, 2.0, 1e-49, 3.0, 0.0, 4.0]),
        np.array([0.0, 2.0, -1.0, 3.0, 0.0, 4.0]),
        np.array([0.0, 2.0, np.nan, 3.0, 0.0, 4.0]),
        np.array([0.0, 2.0, np.inf, 3.0, 0.0, 4.0]),
    ],
    ids=["valid", "ties", "no-zeros", "rounds-to-zero", "negative", "nan", "inf"],
)
@pytest.mark.parametrize("log_transform", [False, True])
def test_gene_major_feature_python_kernel_matches_compiled_kernel(
    counts: np.ndarray, log_transform: bool
) -> None:
    raw, totals, size_factor, _, groups, group_counts, n_total = _gene_major_inputs(
        counts[None, :]
    )
    outcomes = []
    for kernel in (_gene_major_feature, _gene_major_feature.py_func):
        out = np.zeros((2, 8))
        scratch = [np.empty(counts.size, dtype=np.float32)] + [
            np.empty(counts.size, dtype=np.int64) for _ in range(3)
        ]
        # NumPy scalars warn on the log of an invalid value; compiled code does not.
        with np.errstate(invalid="ignore"):
            valid = kernel(
                raw[0],
                totals,
                size_factor,
                log_transform,
                groups,
                group_counts,
                n_total,
                out,
                *scratch,
                np.empty(2048, dtype=np.int64),
                np.zeros(2),
            )
        outcomes.append((valid, out))

    (compiled_valid, compiled), (python_valid, python) = outcomes
    assert (
        compiled_valid
        == python_valid
        == bool(np.isfinite(counts).all() and (counts >= 0).all())
    )
    np.testing.assert_array_equal(compiled, python)
    assert compiled.any() == compiled_valid


@pytest.mark.parametrize("threads", [1, 2, 3, 8])
def test_gene_major_kernel_results_do_not_depend_on_threads(threads: int) -> None:
    rng = np.random.default_rng(2)
    raw = rng.poisson(0.8, size=(9, 50)).astype(np.uint16)
    destinations = np.array([3, -1, 0, 5, -1, 1, 2, 4, -1], dtype=np.int64)
    args = (*_gene_major_inputs(raw), destinations)
    expected = np.zeros((6, 2, 8), dtype=np.float64)
    assert _marker_stats_gene_major(*args, 1, expected) == -1
    observed = np.zeros_like(expected)

    assert _marker_stats_gene_major(*args, threads, observed) == -1

    np.testing.assert_array_equal(observed, expected)


def test_gene_major_kernel_skips_unselected_source_rows() -> None:
    raw = np.array(
        [
            [0, 2, 0, 4],
            [1, 2, 0, 0],
            [1, 0, 3, 4],
            [0, 0, 3, 0],
        ],
        dtype=np.uint32,
    ).T
    inputs = _gene_major_inputs(raw)
    observed = np.zeros((2, 2, 8), dtype=np.float64)
    destinations = np.array([1, -1, 0, -1], dtype=np.int64)

    assert _marker_stats_gene_major(*inputs, destinations, 2, observed) == -1
    expected = np.zeros_like(observed)
    reordered = (raw[[2, 0]], *inputs[1:])
    assert (
        _marker_stats_gene_major(*reordered, np.arange(2, dtype=np.int64), 2, expected)
        == -1
    )

    np.testing.assert_array_equal(observed, expected)


_INVALID_COUNTS = {"negative": -2.0, "nan": np.nan, "inf": np.inf, "overflow": 1e37}
# Signed counts hold negative values, and floating-point counts every invalid one.
_INVALID_CASES = [
    (dtype, bad)
    for dtype in _COUNT_DTYPES
    for bad in _INVALID_COUNTS
    if np.dtype(dtype).kind == "f"
    or (np.dtype(dtype).kind == "i" and bad == "negative")
]


@pytest.mark.parametrize(("dtype", "bad"), _INVALID_CASES)
def test_gene_major_kernel_reports_invalid_normalized_values(
    dtype: str, bad: str
) -> None:
    value = _INVALID_COUNTS[bad]
    raw = np.array([[1, 0, 2, 3], [0, 4, 1, 1], [2, 2, 0, 1], [5, 0, 0, 1]])
    raw = raw.astype(dtype)
    # Rows 1 and 3 are invalid; the first invalid row is reported.
    raw[1, 3] = value
    raw[3, 0] = value
    # One count per cell makes 1e37 overflow float32 after normalization.
    totals = np.ones(4)
    destinations = np.arange(4, dtype=np.int64)
    for kernel in (_marker_stats_gene_major, _marker_stats_gene_major.py_func):
        out = np.zeros((4, 2, 8))
        invalid = kernel(
            raw,
            totals,
            1000.0,
            False,
            np.array([0, 0, 1, 1], dtype=np.int64),
            np.array([2, 2], dtype=np.float64),
            4.0,
            destinations,
            2,
            out,
        )
        assert invalid == 1
        assert not out[1].any()


def test_gene_major_kernel_treats_values_that_round_to_zero_as_zeros() -> None:
    # 1e-45 normalizes to a float32 zero, so it ranks with the zeros.
    raw = np.array([[1.5, 0.0, 1e-45, 3.0]], dtype=np.float32)
    zeroed = np.array([[1.5, 0.0, 0.0, 3.0]], dtype=np.float32)
    totals = np.full(4, 1e30)
    stats = [
        _gene_major_stats(
            values, totals, 1000.0, True, np.array([0, 0, 1, 1]), [2, 2], 4
        )
        for values in (raw, zeroed)
    ]

    np.testing.assert_array_equal(stats[0], stats[1])
    assert stats[0][0, :, 3].tolist() == [0.5, 0.5]


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


def _feature_batch(columns: dict[str, list[float]]):
    """Return a feature-major batch as ``iter_normed_feature_wise`` yields it."""
    return np.array(list(columns.values())), np.array(list(columns))


def test_find_markers_by_regression_handles_expression_threshold():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield _feature_batch(
                {
                    "correlated": [0.0, 1.0, 2.0, 3.0],
                    "at_threshold": [0.0, 1.0, 2.0, 0.0],
                    "too_sparse": [0.0, 0.0, 1.0, 0.0],
                    "constant": [1.0, 1.0, 1.0, 1.0],
                }
            )

    result = find_markers_by_regression(
        Assay(),
        cell_idx=np.arange(4),
        feat_idx=np.arange(4),
        regressor=np.arange(4),
        min_cells=2,
    )

    assert result.loc["correlated", "r_value"] == pytest.approx(1.0)
    assert result.loc["correlated", "p_value"] < 1e-10
    assert result.loc["at_threshold", "r_value"] != 0.0
    assert result.loc["too_sparse", "r_value"] == 0.0
    assert np.isnan(result.loc["too_sparse", "p_value"])
    assert np.isnan(result.loc["too_sparse", "p_value_adjusted"])
    assert result.loc["constant", "r_value"] == 0.0
    assert np.isnan(result.loc["constant", "p_value"])
    assert np.isnan(result.loc["constant", "p_value_adjusted"])


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


def test_regression_r_does_not_depend_on_the_value_scale():
    regressor = np.linspace(0.0, 1.0, 40)
    values = np.random.default_rng(4).poisson(0.8, size=40) + 3.0 * regressor
    # Power-of-two scales change no rounding, so r must not change at all.
    data = np.column_stack([values, values * 2.0**600, values * 2.0**-40])
    x_centered = regressor - regressor.mean()
    ssxm = float(np.dot(x_centered, x_centered) / regressor.size)
    for kernel in (_regression_r_batch, _regression_r_batch.py_func):
        r_vals, status = kernel(data, x_centered, ssxm, 1, float(np.finfo(float).eps))

        np.testing.assert_array_equal(status, [0, 0, 0])
        np.testing.assert_array_equal(r_vals, np.full(3, r_vals[0]))
        assert r_vals[0] == pytest.approx(linregress(regressor, values).rvalue)


@pytest.mark.parametrize(
    "scale", [1.0, 2.0**-600, 2.0**600, 4e307], ids=["unit", "tiny", "huge", "limit"]
)
def test_find_markers_by_regression_does_not_depend_on_the_regressor_scale(scale):
    values = np.array([0.0, 1.0, 3.0, 2.0, 5.0])
    regressor = np.arange(5.0)

    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield _feature_batch({"values": values, "scaled": values * 2.0**600})

    result = find_markers_by_regression(
        Assay(),
        cell_idx=np.arange(5),
        feat_idx=np.arange(2),
        regressor=regressor * scale,
        min_cells=1,
    )

    expected = linregress(regressor, values)
    assert result.loc["values", "r_value"] == pytest.approx(expected.rvalue, rel=1e-12)
    assert result.loc["values", "p_value"] == pytest.approx(expected.pvalue, rel=1e-8)
    np.testing.assert_array_equal(result.loc["scaled"], result.loc["values"])


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
    r_vals, p_vals, status = _regression_batch_results(
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
            assert status[i] == 0
        else:
            assert r_vals[i] == 0.0
            assert np.isnan(p_vals[i])
            assert status[i] == 1


@pytest.mark.parametrize(
    ("regressor", "values"),
    [
        ([0.0, 1.0], [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        ([0.0, 1e300], [[0.0, 1e300, 1e300], [1e300, 0.0, 1e300]]),
        # Least squares rounds the correlation of these points below one.
        ([0.03, 0.12], [[0.65, 0.67, 1.0], [0.67, 0.65, 1.0]]),
    ],
    ids=["unit", "huge", "rounded"],
)
def test_two_cell_regression_preserves_r_but_marks_inference_untested(
    regressor, values
):
    regressor = np.array(regressor)
    data = np.array(values)
    x_centered = regressor - regressor.mean()
    r_vals, p_vals, status = _regression_batch_results(
        data,
        x_centered,
        1.0,
        regressor,
        min_cells=1,
        feature_labels=np.array(["increasing", "decreasing", "constant"]),
    )

    # Two distinct points correlate exactly along their slope.
    np.testing.assert_array_equal(r_vals, [1.0, -1.0, 0.0])
    assert np.isnan(p_vals).all()
    np.testing.assert_array_equal(
        status,
        np.full(data.shape[1], _REG_SENTINEL, dtype=np.int8),
    )


def test_find_markers_by_regression_two_cell_batches_are_unadjusted():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield _feature_batch({"a": [0.0, 1.0], "b": [1.0, 1.0]})
            yield _feature_batch({"c": [1.0, 0.0]})

    result = find_markers_by_regression(
        Assay(),
        cell_idx=np.arange(2),
        feat_idx=np.arange(3),
        regressor=np.array([0.0, 1.0]),
        min_cells=1,
    )
    assert result.loc["a", "r_value"] == pytest.approx(1.0)
    assert result.loc["c", "r_value"] == pytest.approx(-1.0)
    assert "p_value_adjusted" in result.columns
    assert result.loc["b", "r_value"] == 0.0
    assert result["p_value"].isna().all()
    assert result["p_value_adjusted"].isna().all()


def test_find_markers_by_regression_identifies_nonfinite_feature():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield _feature_batch({"bad_feature": [0.0, np.nan, 1.0]})

    with pytest.raises(ValueError, match="bad_feature"):
        find_markers_by_regression(
            Assay(),
            cell_idx=np.arange(3),
            feat_idx=np.arange(1),
            regressor=np.arange(3),
            min_cells=1,
        )


@pytest.mark.parametrize(
    ("cells", "features", "regressor", "min_cells", "message"),
    [
        (np.zeros((2, 2)), np.arange(1), np.arange(2.0), 1, "feat_idx must be one-"),
        (np.array([], dtype=int), np.arange(1), np.array([]), 1, "non-empty indices"),
        (np.array([0, 0, 1]), np.arange(1), np.arange(3.0), 1, "unique non-negative"),
        (np.arange(3), np.array([-1]), np.arange(3.0), 1, "unique non-negative"),
        (np.arange(3), np.arange(1), np.zeros((3, 1)), 1, "regressor must be one-"),
        (np.arange(3), np.arange(1), np.arange(2.0), 1, "align with cell_idx"),
        (np.arange(3), np.arange(1), np.array([0.0, np.inf, 1.0]), 1, "only finite"),
        (np.arange(3), np.arange(1), np.ones(3), 1, "two distinct values"),
        (np.arange(3), np.arange(1), np.arange(3.0), 0, "at least 1"),
    ],
)
def test_find_markers_by_regression_rejects_invalid_inputs_before_reading(
    cells, features, regressor, min_cells, message
):
    with pytest.raises(ValueError, match=message):
        find_markers_by_regression(object(), cells, features, regressor, min_cells)


def test_find_markers_by_regression_checks_the_batches_it_reads():
    class Assay:
        def __init__(self, batches):
            self.batches = batches

        def iter_normed_feature_wise(self, **_kwargs):
            yield from self.batches

    def search(batches):
        return find_markers_by_regression(
            Assay(batches), np.arange(3), np.arange(1), np.arange(3.0), 1
        )

    empty = search([])
    assert empty.empty
    assert list(empty.columns) == ["r_value", "p_value", "p_value_adjusted"]
    untested = search([_feature_batch({"constant": [1.0, 1.0, 1.0]})])
    assert untested.loc["constant", "r_value"] == 0.0
    assert np.isnan(untested.loc["constant", ["p_value", "p_value_adjusted"]]).all()
    with pytest.raises(ValueError, match="number of selected cells"):
        search([_feature_batch({"short": [0.0, 1.0]})])


def test_find_markers_by_rank_rejects_fast_path_for_non_rna_assay():
    import numba

    class Cells:
        @staticmethod
        def fetch(_group_key, _cell_key):
            return np.array([0, 0, 1, 1])

    class Assay:
        def __init__(self):
            self.cells = Cells()
            self.normMethod = norm_lib_size
            self.sf = 1.0

    previous_threads = numba.get_num_threads()
    with pytest.raises(TypeError, match="requires an RNAassay"):
        find_markers_by_rank(
            Assay(),
            groups=np.array([0, 0, 1, 1]),
            cell_idx=np.arange(4),
            feat_idx=np.arange(1),
            nthreads=1,
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

    class Assay:
        cells = Cells()
        normMethod = None
        sf = None

        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield pd.DataFrame(data[:, :2], columns=[10, 11])
            yield pd.DataFrame(data[:, 2:], columns=[12, 13])

    results = find_markers_by_rank(
        Assay(),
        groups=np.array(["a", "a", "b", "b"]),
        cell_idx=np.arange(4),
        feat_idx=np.array([13, 10, 12, 11]),
        nthreads=1,
    )
    names = np.array([f"g{index}" for index in range(14)])
    group_a = results.table("a", names).set_index("feature_index")
    group_b = results.table("b", names).set_index("feature_index")

    np.testing.assert_array_equal(results.group_ids, ["a", "b"])
    np.testing.assert_array_equal(results.group_sizes, [2, 2])
    np.testing.assert_array_equal(results.feature_index, [10, 11, 12, 13])

    assert group_a.loc[10, "fold_change"] == pytest.approx(100.1)
    assert group_a.loc[11, "fold_change"] == pytest.approx(0.0)
    assert group_a.loc[13, "fold_change"] == pytest.approx(1.0)
    assert group_b.loc[12, "fold_change"] == pytest.approx(100.1)
    assert group_b.loc[10, "fold_change"] == pytest.approx(0.0)
    assert np.isfinite(group_a["p_value"]).all()
    assert np.isfinite(group_b["p_value"]).all()
    assert "auc" in group_a.columns
    assert "p_value_adjusted" in group_a.columns
    assert np.isfinite(group_a["auc"]).all()
    assert np.isfinite(group_a["p_value_adjusted"]).all()


@pytest.mark.parametrize(
    ("batches", "message"),
    [
        ([[11, 10], [12, 13]], "follow the requested features"),
        ([[10, 11]], "cover every feature"),
    ],
    ids=["reordered", "incomplete"],
)
def test_find_markers_by_rank_places_normalized_batches_by_their_features(
    batches: list[list[int]], message: str
) -> None:
    class Assay:
        normMethod = None
        sf = None

        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            for labels in batches:
                yield pd.DataFrame(np.ones((4, len(labels))), columns=labels)

    with pytest.raises(RuntimeError, match=message):
        find_markers_by_rank(
            Assay(),
            groups=np.array(["a", "a", "b", "b"]),
            cell_idx=np.arange(4),
            feat_idx=np.array([10, 11, 12, 13]),
            nthreads=1,
        )


@pytest.mark.parametrize("batch_format", ["dataframe", "tuple"])
@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf],
    ids=["nan", "inf"],
)
def test_find_markers_by_rank_rejects_nonfinite_slow_batches(
    batch_format: str,
    bad_value: float,
) -> None:
    data = np.array(
        [
            [2.0, 0.0, 1.0, 3.0],
            [4.0, 0.0, 2.0, bad_value],
            [0.0, 1.0, 3.0, 5.0],
            [0.0, 2.0, 4.0, 6.0],
        ]
    )

    class Cells:
        @staticmethod
        def fetch(_group_key: str, _cell_key: str) -> np.ndarray:
            return np.array(["a", "a", "b", "b"])

    class Assay:
        cells = Cells()
        normMethod = None
        sf = None

        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            for start in (0, 2):
                values = data[:, start : start + 2]
                labels = np.array([10 + start, 11 + start])
                if batch_format == "dataframe":
                    yield pd.DataFrame(values, columns=labels)
                else:
                    yield values.T, labels

    with pytest.raises(
        ValueError,
        match=r"Feature .*13.* contains non-finite normalized values",
    ):
        find_markers_by_rank(
            Assay(),
            groups=np.array(["a", "a", "b", "b"]),
            cell_idx=np.arange(4),
            feat_idx=np.array([10, 11, 12, 13]),
            nthreads=1,
        )


def test_find_markers_fast_raw_path_computes_groupwise_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zarr
    from zarr.storage import MemoryStore

    from scarf.storage import feature_stream
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

    class FakeRNA:
        def __init__(self):
            from scarf.storage.sharding import write_counts_t

            self.cells = Cells()
            self.normMethod = norm_lib_size
            self.sf = 1_000.0
            self.name = "RNA"
            self.resources = ResourceBudget(1024**3, 2)
            from scarf.storage.count_matrix import (
                persist_count_matrix_plan,
                plan_count_matrix_pair,
            )

            root = zarr.open_group(store=MemoryStore(), mode="w")
            values = data.astype(np.uint32)
            plan = plan_count_matrix_pair(
                values.shape[0], values.shape[1], values.dtype
            )
            self.raw = root.create_array(
                "counts",
                shape=plan.counts.shape,
                chunks=plan.counts.chunks,
                shards=plan.counts.shards,
                dtype=values.dtype,
                overwrite=True,
            )
            self.raw[:] = values
            persist_count_matrix_plan(root, plan)
            persist_count_matrix_plan(self.raw, plan)
            from tests.storage_helpers import finalize_test_counts

            finalize_test_counts(self.raw)
            counts_t = write_counts_t(self.raw, root)
            assert counts_t is not None
            self.rawDataT = counts_t

    monkeypatch.setattr(marker_search_module, "RNAassay", FakeRNA)

    def reject_selected_copy(_values, _keep):
        raise AssertionError("fast marker search copied selected feature rows")

    monkeypatch.setattr(
        feature_stream,
        "selected_feature_values",
        reject_selected_copy,
    )
    results = find_markers_by_rank(
        FakeRNA(),
        groups=np.array(["a", "a", "b", "b"]),
        cell_idx=np.arange(4),
        feat_idx=np.array([3, 0, 2]),
        nthreads=1,
    )

    np.testing.assert_array_equal(results.group_ids, ["a", "b"])
    np.testing.assert_array_equal(results.feature_index, [0, 2, 3])
    assert results.statistics.shape == (3, 2, 8)
    for group_id in results.group_ids:
        stored = results.stored_statistics(group_id)
        assert stored.shape == (3, 9)
        assert np.isfinite(stored).all()
    # Feature 0 is expressed only in group a, feature 3 only in group b.
    assert results.statistics[0, :, 3].tolist() == [1.0, 0.0]
    assert results.statistics[2, :, 3].tolist() == [0.0, 1.0]


def _rank_result(
    group_ids=(1, 4),
    group_sizes=(10, 40),
    feature_index=(5, 7, 10),
) -> RankMarkerResult:
    rng = np.random.default_rng(3)
    statistics = rng.random((len(feature_index), len(group_ids), 8))
    # Column 6 holds Mann-Whitney z statistics.
    statistics[:, :, 6] = rng.normal(scale=3.0, size=statistics.shape[:2])
    return RankMarkerResult(
        group_ids=np.asarray(group_ids),
        group_sizes=np.asarray(group_sizes),
        feature_index=np.asarray(feature_index),
        statistics=statistics,
    )


def test_rank_marker_result_finishes_one_stored_table_per_group():
    from scipy.special import ndtr

    from scarf.features.markers.table import MARKER_STAT_COLUMNS

    result = _rank_result()
    for position, group_id in enumerate(result.group_ids):
        rank = np.asarray(result.statistics[:, position])
        stored = result.stored_statistics(group_id)
        p_values = 2.0 * ndtr(-np.abs(rank[:, 6]))

        assert stored.shape == (3, len(MARKER_STAT_COLUMNS))
        np.testing.assert_array_equal(stored[:, 6], p_values)
        rounded = [index for index in range(8) if index != 6]
        np.testing.assert_array_equal(stored[:, rounded], np.round(rank[:, rounded], 5))
        np.testing.assert_array_equal(stored[:, 8], adjust_pvalues(p_values, "fdr_bh"))
    with pytest.raises(ValueError, match="no group 2"):
        result.stored_statistics(2)
    statistics = np.array(result.statistics)
    statistics[1, 0, 1] = np.inf
    poisoned = RankMarkerResult(
        group_ids=result.group_ids,
        group_sizes=result.group_sizes,
        feature_index=result.feature_index,
        statistics=statistics,
    )
    with pytest.raises(ValueError, match="must all be finite"):
        poisoned.stored_statistics(1)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"group_ids": np.array([1, 1])}, "unique labels"),
        ({"group_sizes": np.array([10])}, "integer sizes"),
        ({"group_sizes": np.array([10.0, 40.0])}, "integer sizes"),
        # A stored group and its complement need at least two cells each.
        (
            {
                "group_ids": np.array([1]),
                "group_sizes": np.array([50]),
                "statistics": np.zeros((3, 1, 8)),
            },
            "at least two populated groups",
        ),
        ({"group_sizes": np.array([1, 40])}, "at least two cells in every group"),
        ({"group_sizes": np.array([-3, 40])}, "at least two cells in every group"),
        ({"feature_index": np.array([7, 5, 10])}, "ascending unique"),
        # Differences of unsigned indices wrap instead of turning negative.
        ({"feature_index": np.array([7, 5, 10], dtype=np.uint64)}, "ascending unique"),
        ({"feature_index": np.array([7, 5, 10], dtype=np.uint8)}, "ascending unique"),
        ({"feature_index": np.array([5, 5, 10])}, "ascending unique"),
        ({"feature_index": np.array([-1, 5, 10])}, "non-negative"),
        ({"feature_index": np.array([5.0, 7.0, 10.0])}, "ascending unique"),
        (
            {
                "feature_index": np.array([], dtype=np.int64),
                "statistics": np.zeros((0, 2, 8)),
            },
            "one or more ascending unique",
        ),
        ({"statistics": np.zeros((3, 2, 8), dtype=np.float32)}, "float64"),
        ({"statistics": np.zeros((3, 2, 9))}, "8 statistics per group"),
        ({"statistics": np.zeros((2, 2, 8))}, "one row per feature"),
    ],
)
def test_rank_marker_result_rejects_inconsistent_arrays(changes, message):
    result = _rank_result()
    arrays = {
        "group_ids": result.group_ids,
        "group_sizes": result.group_sizes,
        "feature_index": result.feature_index,
        "statistics": np.array(result.statistics),
        **changes,
    }

    with pytest.raises(ValueError, match=message):
        RankMarkerResult(**arrays)


def test_compact_marker_save_roundtrip():
    import zarr
    from zarr.storage import MemoryStore

    from scarf.datastore.datastore import DataStore
    from scarf.features.markers.table import MARKER_STAT_COLUMNS, load_marker_table

    result = _rank_result()
    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    feature_names = np.array([f"g{i}" for i in range(11)])
    feature_ids = np.array([f"id{i}" for i in range(11)])
    DataStore._write_marker_slot(
        slot,
        result,
        workers=2,
        feature_names=feature_names,
        feature_ids=feature_ids,
    )

    assert "schema_version" not in slot.attrs
    assert list(slot.attrs["stat_columns"]) == list(MARKER_STAT_COLUMNS)
    assert slot["feature_index"].dtype == np.int32
    np.testing.assert_array_equal(slot["feature_index"][:], [5, 7, 10])
    np.testing.assert_array_equal(slot["feature_names"][:], feature_names)
    np.testing.assert_array_equal(slot["feature_ids"][:], feature_ids)
    assert sorted(slot.group_keys()) == ["1", "4"]
    for group_id, n_group in ((1, 10), (4, 40)):
        cluster = slot[str(group_id)]
        assert cluster.attrs["n_group"] == n_group
        assert cluster.attrs["n_reference"] == 50 - n_group
        np.testing.assert_array_equal(
            cluster["stats"][:], result.stored_statistics(group_id)
        )
        # ``get_markers`` reads each group by its stored string name.
        table = result.table(group_id, feature_names)
        assert table["group_id"].tolist() == [str(group_id)] * 3
        pd.testing.assert_frame_equal(
            load_marker_table(slot, cluster, feature_names, group_id=str(group_id)),
            table,
        )


@pytest.mark.parametrize(
    ("feature_index", "message"),
    [
        ((0, 1, int(np.iinfo(np.int32).max) + 1), "fit non-negative int32"),
        ((0, 1, 2), "must index feature_names"),
    ],
    ids=["int32", "names"],
)
def test_marker_writer_rejects_feature_indices_it_cannot_store_before_writing(
    feature_index, message
):
    import zarr
    from zarr.storage import MemoryStore

    from scarf.datastore.datastore import DataStore

    result = _rank_result(feature_index=feature_index)
    slot = zarr.open_group(store=MemoryStore(), mode="w").create_group("slot")

    with pytest.raises(ValueError, match=message):
        DataStore._write_marker_slot(
            slot,
            result,
            feature_names=np.array(["g0", "g1"]),
            feature_ids=np.array(["id0", "id1"]),
        )

    assert dict(slot.attrs) == {}
    assert list(slot.array_keys()) == []
    assert list(slot.group_keys()) == []


def test_marker_artifact_write_preserves_legacy_marker_subtree():
    import zarr
    from zarr.storage import MemoryStore

    from scarf.datastore.datastore import DataStore
    from scarf.features.markers.table import load_marker_table
    from scarf.storage.artifact_writer import (
        ArrayRequirement,
        AttributeRequirement,
        finish_artifact,
        plan_artifact,
        start_artifact,
    )

    root = zarr.open_group(store=MemoryStore(), mode="w")
    legacy_slot = root.create_group("RNA/markers/I__legacy")
    legacy_cluster = legacy_slot.create_group("1")
    legacy_cluster.create_array(
        "feature_index",
        data=np.array([1], dtype=np.int32),
    )
    legacy_cluster.create_array("score", data=np.array([0.5]))
    legacy_values = {
        name: np.asarray(legacy_cluster[name][:]).copy()
        for name in legacy_cluster.array_keys()
    }
    legacy_attrs = dict(legacy_slot.attrs)

    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="marker_table",
        operation="run_marker_search",
        parameters={},
        inputs={},
        execution_options={},
        required_arrays=(ArrayRequirement("feature_index", dtype_kind="i"),),
        required_attributes=(
            AttributeRequirement(
                "stat_columns",
                expected_types=(list, tuple),
            ),
        ),
    )
    artifact = start_artifact(root, planned)
    statistics = np.zeros((2, 2, 8))
    # Feature 1 scores higher than feature 0 in group 1.
    statistics[:, 0, 0] = [0.4, 0.8]
    DataStore._write_marker_slot(
        artifact,
        RankMarkerResult(
            group_ids=np.array([1, 2]),
            group_sizes=np.array([10, 20]),
            feature_index=np.array([0, 1]),
            statistics=statistics,
        ),
        feature_names=np.array(["g0", "g1"]),
        feature_ids=np.array(["id0", "id1"]),
    )
    finish_artifact(artifact, planned)

    assert dict(legacy_slot.attrs) == legacy_attrs
    assert set(legacy_cluster.array_keys()) == set(legacy_values)
    for name, expected in legacy_values.items():
        np.testing.assert_array_equal(legacy_cluster[name][:], expected)
    loaded = load_marker_table(
        artifact,
        artifact["1"],
        np.array(["g0", "g1"]),
        group_id=1,
    )
    assert loaded["feature_name"].tolist() == ["g1", "g0"]
    assert "schema_version" not in artifact.attrs


def test_load_marker_table_ignores_stale_schema_version_attribute():
    from scarf.features.markers.table import load_marker_table

    slot, cluster = _make_canonical_marker_slot()
    slot.attrs["schema_version"] = 99
    loaded = load_marker_table(
        slot,
        cluster,
        np.array(["g0", "g1"]),
        group_id=1,
    )

    assert loaded["feature_name"].tolist() == ["g1", "g0"]


def _make_canonical_marker_slot(columns=None):
    import zarr
    from zarr.storage import MemoryStore

    from scarf.features.markers.table import MARKER_STAT_COLUMNS

    if columns is None:
        columns = MARKER_STAT_COLUMNS
    values = {
        "score": np.array([0.4, 0.8]),
        "mean": np.array([0.5, 1.0]),
        "mean_rest": np.array([0.5, 0.5]),
        "frac_exp": np.array([0.3, 0.9]),
        "frac_exp_rest": np.array([0.2, 0.2]),
        "fold_change": np.array([1.0, 2.0]),
        "p_value": np.array([0.04, 0.01]),
        "auc": np.array([0.6, 0.9]),
        "p_value_adjusted": np.array([0.04, 0.02]),
    }
    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    slot.attrs.update(
        {
            "stat_columns": list(columns),
            "method": "mannwhitneyu",
            "alternative": "two-sided",
            "tie_correction": True,
            "continuity_correction": True,
            "adjustment_method": "fdr_bh",
            "adjustment_scope": "within_group_all_tested_features",
        }
    )
    slot.create_array("feature_index", data=np.array([0, 1], dtype=np.int32))
    cluster = slot.create_group("1")
    cluster.attrs.update({"n_group": 10, "n_reference": 20})
    cluster.create_array(
        "stats",
        data=np.column_stack([values[column] for column in columns]),
    )
    return slot, cluster


def test_canonical_marker_reader_accepts_reordered_named_columns():
    from scarf.features.markers.table import (
        MARKER_STAT_COLUMNS,
        load_marker_table,
    )

    columns = tuple(reversed(MARKER_STAT_COLUMNS))
    slot, cluster = _make_canonical_marker_slot(columns)
    loaded = load_marker_table(
        slot,
        cluster,
        np.array(["g0", "g1"]),
        group_id=1,
    )

    assert loaded["feature_name"].tolist() == ["g1", "g0"]
    assert loaded["score"].tolist() == pytest.approx([0.8, 0.4])
    assert loaded["auc"].tolist() == pytest.approx([0.9, 0.6])
    assert loaded["p_value_adjusted"].tolist() == pytest.approx([0.02, 0.04])


def test_canonical_marker_reader_rejects_missing_stats_array():
    from scarf.features.markers.table import load_marker_table

    slot, cluster = _make_canonical_marker_slot()
    del cluster["stats"]

    with pytest.raises(ValueError, match="require feature_index and stats"):
        load_marker_table(
            slot,
            cluster,
            np.array(["g0", "g1"]),
            group_id=1,
        )


def test_canonical_marker_reader_rejects_non_finite_statistics():
    from scarf.features.markers.table import (
        MARKER_STAT_COLUMNS,
        load_marker_table,
    )

    slot, cluster = _make_canonical_marker_slot()
    stats = np.asarray(cluster["stats"][:])
    stats[0, MARKER_STAT_COLUMNS.index("fold_change")] = np.inf
    cluster["stats"][:] = stats

    with pytest.raises(ValueError, match="statistics must all be finite"):
        load_marker_table(
            slot,
            cluster,
            np.array(["g0", "g1"]),
            group_id=1,
        )


def test_canonical_marker_reader_rejects_all_nan_adjusted_values():
    from scarf.features.markers.table import (
        MARKER_STAT_COLUMNS,
        load_marker_table,
    )

    slot, cluster = _make_canonical_marker_slot()
    stats = np.asarray(cluster["stats"][:])
    stats[:, MARKER_STAT_COLUMNS.index("p_value_adjusted")] = np.nan
    cluster["stats"][:] = stats

    with pytest.raises(ValueError, match="p_value_adjusted.*finite"):
        load_marker_table(
            slot,
            cluster,
            np.array(["g0", "g1"]),
            group_id=1,
        )


def _replace_node(group, name, data=None) -> None:
    """Replace a member of ``group`` by an array of ``data``, or by a group."""
    del group[name]
    if data is None:
        group.create_group(name)
    else:
        group.create_array(name, data=np.asarray(data))


_MARKER_PAYLOAD_CORRUPTIONS = {
    "columns-text": (
        lambda slot, _: slot.attrs.update({"stat_columns": "score"}),
        "sequence of names",
    ),
    "columns-number": (
        lambda slot, _: slot.attrs.update({"stat_columns": ["score", 3]}),
        "only strings",
    ),
    "columns-unknown": (
        lambda slot, _: slot.attrs.update(
            {"stat_columns": [*slot.attrs["stat_columns"], "bogus"]}
        ),
        "unknown columns: bogus",
    ),
    "columns-incomplete": (
        lambda slot, _: slot.attrs.update(
            {"stat_columns": slot.attrs["stat_columns"][:-1]}
        ),
        "complete named stat_columns",
    ),
    "index-missing": (
        lambda slot, _: slot.__delitem__("feature_index"),
        "require feature_index and stats",
    ),
    "index-group": (
        lambda slot, _: _replace_node(slot, "feature_index"),
        "must be an array",
    ),
    "index-2d": (
        lambda slot, _: _replace_node(slot, "feature_index", [[0, 1]]),
        "one-dimensional",
    ),
    "index-float": (
        lambda slot, _: _replace_node(slot, "feature_index", [0.0, 1.0]),
        "integer dtype",
    ),
    "index-repeated": (
        lambda slot, _: _replace_node(slot, "feature_index", [1, 1]),
        "unique values",
    ),
    "index-empty": (
        lambda slot, _: _replace_node(slot, "feature_index", np.array([], int)),
        "must contain marker rows",
    ),
    "stats-rows": (
        lambda _, cluster: _replace_node(cluster, "stats", np.zeros((3, 9))),
        "do not align with feature_index",
    ),
    "stats-integer": (
        lambda _, cluster: _replace_node(cluster, "stats", np.zeros((2, 9), int)),
        "floating dtype",
    ),
}


@pytest.mark.parametrize("corruption", sorted(_MARKER_PAYLOAD_CORRUPTIONS))
def test_canonical_marker_reader_rejects_malformed_payloads(corruption):
    from scarf.features.markers.table import load_marker_table

    slot, cluster = _make_canonical_marker_slot()
    corrupt, message = _MARKER_PAYLOAD_CORRUPTIONS[corruption]
    corrupt(slot, cluster)

    with pytest.raises((TypeError, ValueError), match=message):
        load_marker_table(slot, cluster, np.array(["g0", "g1"]), group_id="1")


def test_canonical_marker_reader_requires_one_dimensional_feature_names():
    from scarf.features.markers.table import load_marker_table

    slot, cluster = _make_canonical_marker_slot()
    with pytest.raises(ValueError, match="names must be one-dimensional"):
        load_marker_table(slot, cluster, np.array([["g0"], ["g1"]]), group_id="1")


def test_marker_slot_validation_checks_every_group_against_its_counts():
    from scarf.features.markers.table import _validate_marker_slot

    slot, _ = _make_canonical_marker_slot()
    names = np.array(["g0", "g1"])
    # Several workers read the groups of a remote store concurrently.
    for workers in (1, 2):
        _validate_marker_slot(
            slot, names, expected_group_cell_counts={"1": (10, 20)}, workers=workers
        )
    _validate_marker_slot(slot, names)
    with pytest.raises(ValueError, match="stale cell counts"):
        _validate_marker_slot(slot, names, expected_group_cell_counts={"1": (11, 19)})
    with pytest.raises(ValueError, match="do not match the requested groups"):
        _validate_marker_slot(slot, names, expected_group_cell_counts={"2": (10, 20)})
    del slot["1"]
    with pytest.raises(ValueError, match="must contain populated groups"):
        _validate_marker_slot(slot, names)


@pytest.mark.parametrize(
    "corruption",
    [
        "incomplete_provenance",
        "feature_identity",
        "slot_metadata",
        "group_metadata",
        "stats_shape",
        "missing_stats",
        "stat_values",
        "adjusted_values",
    ],
)
def test_marker_cache_reuse_revalidates_canonical_payload(
    datastore_ephemeral,
    monkeypatch,
    corruption,
):
    import scarf.features.markers as marker_algorithms
    from scarf.features.markers.table import MARKER_STAT_COLUMNS
    from scarf.storage.artifacts import artifact_path

    assay = datastore_ephemeral.RNA
    clusters = _cluster_labels(
        datastore_ephemeral,
        np.arange(len(datastore_ephemeral.cells.active_index("I"))) % 2,
    )
    feature_mask = np.zeros(assay.feats.N, dtype=bool)
    feature_mask[:8] = True
    feature_selection = datastore_ephemeral.set_feature_selection(
        from_assay="RNA",
        mask=feature_mask,
    )
    arguments = {
        "clusters": clusters,
        "from_assay": "RNA",
        "features": feature_selection,
        "nthreads": 1,
    }
    old_ref = datastore_ephemeral.run_marker_search(**arguments)
    old_artifact = datastore_ephemeral.zw[artifact_path(old_ref)]
    first_group_name = sorted(old_artifact.group_keys())[0]
    first_group = old_artifact[first_group_name]
    if corruption == "incomplete_provenance":
        provenance = dict(old_artifact.attrs["provenance"])
        parameters = dict(provenance["parameters"])
        for field_name in (
            "method",
            "alternative",
            "tie_correction",
            "continuity_correction",
            "adjustment_method",
            "adjustment_scope",
        ):
            parameters.pop(field_name)
        provenance["parameters"] = parameters
        old_artifact.attrs["provenance"] = provenance
    elif corruption == "feature_identity":
        stored_indices = np.asarray(old_artifact["feature_index"][:])
        stored_indices[0] = int(stored_indices.max()) + 1
        old_artifact["feature_index"][:] = stored_indices
    elif corruption == "slot_metadata":
        old_artifact.attrs["method"] = "ttest"
    elif corruption == "group_metadata":
        first_group.attrs["n_group"] = 1
    elif corruption == "stats_shape":
        stats = np.asarray(first_group["stats"][:, :-1])
        del first_group["stats"]
        first_group.create_array("stats", data=stats)
    elif corruption == "missing_stats":
        del first_group["stats"]
    elif corruption == "stat_values":
        stats = np.asarray(first_group["stats"][:])
        stats[0, MARKER_STAT_COLUMNS.index("fold_change")] = np.inf
        first_group["stats"][:] = stats
    else:
        stats = np.asarray(first_group["stats"][:])
        stats[:, MARKER_STAT_COLUMNS.index("p_value_adjusted")] = np.nan
        first_group["stats"][:] = stats

    original = marker_algorithms.find_markers_by_rank
    calls = 0

    def tracked_marker_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        marker_algorithms,
        "find_markers_by_rank",
        tracked_marker_search,
    )
    new_ref = datastore_ephemeral.run_marker_search(**arguments)
    status = datastore_ephemeral.inspect_artifact(new_ref)
    assert calls == 1
    assert new_ref != old_ref
    assert status.parameters["method"] == "mannwhitneyu"
    assert status.parameters["alternative"] == "two-sided"
    assert status.parameters["tie_correction"] is True
    assert status.parameters["continuity_correction"] is True
    assert status.parameters["adjustment_method"] == "fdr_bh"
    assert status.parameters["adjustment_scope"] == "within_group_all_tested_features"
    assert "schema_version" not in status.parameters
    assert "algorithm_version" not in status.parameters
    assert "correction_method" not in status.parameters


@pytest.mark.parametrize(
    ("owner", "metadata_name"),
    [
        ("slot", "method"),
        ("slot", "alternative"),
        ("slot", "tie_correction"),
        ("slot", "continuity_correction"),
        ("slot", "adjustment_method"),
        ("slot", "adjustment_scope"),
        ("slot", "stat_columns"),
        ("cluster", "n_group"),
        ("cluster", "n_reference"),
    ],
)
def test_canonical_marker_reader_rejects_incomplete_metadata(
    owner,
    metadata_name,
):
    from scarf.features.markers.table import load_marker_table

    slot, cluster = _make_canonical_marker_slot()
    target = slot if owner == "slot" else cluster
    del target.attrs[metadata_name]

    with pytest.raises(ValueError, match=metadata_name):
        load_marker_table(
            slot,
            cluster,
            np.array(["g0", "g1"]),
            group_id=1,
        )


@pytest.mark.parametrize(
    ("owner", "metadata_name", "value"),
    [
        ("slot", "method", "ttest"),
        ("slot", "alternative", "greater"),
        ("slot", "tie_correction", False),
        ("slot", "continuity_correction", False),
        ("slot", "adjustment_method", "bonferroni"),
        ("slot", "adjustment_scope", "all_groups"),
        ("cluster", "n_group", 1),
        ("cluster", "n_reference", 1),
    ],
)
def test_canonical_marker_reader_rejects_invalid_metadata(
    owner,
    metadata_name,
    value,
):
    from scarf.features.markers.table import load_marker_table

    slot, cluster = _make_canonical_marker_slot()
    target = slot if owner == "slot" else cluster
    target.attrs[metadata_name] = value

    with pytest.raises(ValueError, match=metadata_name):
        load_marker_table(
            slot,
            cluster,
            np.array(["g0", "g1"]),
            group_id=1,
        )


def test_canonical_marker_reader_rejects_malformed_stat_columns():
    from scarf.features.markers.table import (
        MARKER_STAT_COLUMNS,
        load_marker_table,
    )

    slot, cluster = _make_canonical_marker_slot()
    slot.attrs["stat_columns"] = [
        *MARKER_STAT_COLUMNS[:-1],
        MARKER_STAT_COLUMNS[0],
    ]

    with pytest.raises(ValueError, match="duplicate"):
        load_marker_table(
            slot,
            cluster,
            np.array(["g0", "g1"]),
            group_id=1,
        )


def test_canonical_marker_reader_rejects_negative_feature_index():
    from scarf.features.markers.table import load_marker_table

    slot, cluster = _make_canonical_marker_slot()
    slot["feature_index"][:] = np.array([-1, 0], dtype=np.int32)

    with pytest.raises(ValueError, match="out-of-range"):
        load_marker_table(
            slot,
            cluster,
            np.array(["g0", "g1"]),
            group_id=1,
        )


def test_bh_adjusted_pvalues_match_statsmodels_and_preserve_order():
    from statsmodels.stats.multitest import multipletests

    p_values = np.array([0.04, 0.01, 0.2, np.nan, 0.03])
    adjusted = adjust_pvalues(p_values, "fdr_bh")
    mask = np.isfinite(p_values)
    _, expected, _, _ = multipletests(p_values[mask], method="fdr_bh")
    assert adjusted[mask].tolist() == pytest.approx(list(expected))
    assert np.isnan(adjusted[3])
    reordered = p_values[[1, 0, 4, 3, 2]]
    adjusted_reordered = adjust_pvalues(reordered, "fdr_bh")
    restore = np.empty_like(adjusted_reordered)
    restore[[1, 0, 4, 3, 2]] = adjusted_reordered
    np.testing.assert_allclose(restore, adjusted, equal_nan=True)


def test_marker_auc_matches_scipy_mannwhitneyu():
    data = np.array(
        [
            [8.0, 0.0],
            [7.0, 1.0],
            [6.0, 1.0],
            [5.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [1.0, 5.0],
            [2.0, 6.0],
        ]
    )
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    group_counts = np.bincount(groups)
    got = _batch_stats(data, groups, group_counts, len(groups))
    for gene in range(data.shape[1]):
        for group in (0, 1):
            sample = data[groups == group, gene]
            rest = data[groups != group, gene]
            u = mannwhitneyu(
                sample,
                rest,
                alternative="two-sided",
                method="asymptotic",
                use_continuity=True,
            ).statistic
            expected_auc = u / (len(sample) * len(rest))
            assert got[gene, group, 7] == pytest.approx(expected_auc, abs=1e-6)


def test_find_markers_by_rank_rejects_invalid_group_sizes():
    class Cells:
        def __init__(self, groups):
            self._groups = groups

        def fetch(self, _group_key, _cell_key):
            return self._groups

    class Assay:
        def __init__(self, groups):
            self.cells = Cells(groups)
            self.normMethod = None
            self.sf = None

        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield pd.DataFrame([[1.0], [2.0], [3.0]])

    with pytest.raises(ValueError, match="at least two populated groups"):
        find_markers_by_rank(
            Assay(np.array([0, 0, 0])),
            groups=np.array([0, 0, 0]),
            cell_idx=np.arange(3),
            feat_idx=np.array([0]),
            nthreads=1,
        )
    with pytest.raises(ValueError, match="at least two cells in every group"):
        find_markers_by_rank(
            Assay(np.array([0, 1, 1])),
            groups=np.array([0, 1, 1]),
            cell_idx=np.arange(3),
            feat_idx=np.array([0]),
            nthreads=1,
        )
    with pytest.raises(ValueError, match="writers must be at least 1"):
        find_markers_by_rank(
            Assay(np.array([0, 0, 1, 1])),
            groups=np.array([0, 0, 1, 1]),
            cell_idx=np.arange(4),
            feat_idx=np.array([0]),
            writers=0,
        )


_GROUPS = np.array([0, 0, 1, 1])


@pytest.mark.parametrize(
    ("groups", "cells", "features", "message"),
    [
        (np.zeros((2, 2)), np.arange(2), np.arange(1), "must be one-dimensional"),
        (_GROUPS[:3], np.arange(4), np.arange(1), "align with cell_idx"),
        (_GROUPS[:0], np.arange(0), np.arange(1), "non-empty"),
        (_GROUPS, np.arange(4), np.arange(0), "non-empty"),
        (_GROUPS, np.array([0, 1, 2, -3]), np.arange(1), "unique non-negative"),
        (_GROUPS, np.array([0, 1, 2, 2]), np.arange(1), "unique non-negative"),
        (_GROUPS, np.arange(4), np.array([-1]), "unique non-negative"),
        (_GROUPS, np.arange(4), np.array([1, 1]), "unique non-negative"),
    ],
)
def test_find_markers_by_rank_rejects_invalid_indices_before_reading(
    groups, cells, features, message
):
    with pytest.raises(ValueError, match=message):
        find_markers_by_rank(object(), groups, cells, features)


def test_pseudotime_bh_excludes_untested_features():
    class Assay:
        @staticmethod
        def iter_normed_feature_wise(**_kwargs):
            yield _feature_batch(
                {
                    "tested": [0.0, 1.0, 2.0, 3.0],
                    "untested": [0.0, 0.0, 0.0, 0.0],
                }
            )

    result = find_markers_by_regression(
        Assay(),
        cell_idx=np.arange(4),
        feat_idx=np.arange(2),
        regressor=np.array([0.0, 1.0, 2.0, 3.0]),
        min_cells=2,
    )
    assert np.isfinite(result.loc["tested", "p_value"])
    assert np.isnan(result.loc["untested", "p_value"])
    assert result.loc["tested", "p_value_adjusted"] == pytest.approx(
        float(adjust_pvalues(np.array([result.loc["tested", "p_value"]]), "fdr_bh")[0])
    )
    assert np.isnan(result.loc["untested", "p_value_adjusted"])


def test_marker_search_does_not_accept_gene_batch_size(
    datastore_ephemeral,
):
    clusters = _cluster_labels(
        datastore_ephemeral,
        np.arange(len(datastore_ephemeral.cells.active_index("I"))) % 2,
    )
    feature_ref = datastore_ephemeral.set_feature_selection(
        from_assay="RNA",
        mask=np.ones(datastore_ephemeral.RNA.feats.N, dtype=bool),
    )
    with pytest.raises(TypeError, match="gene_batch_size"):
        datastore_ephemeral.run_marker_search(
            clusters,
            features=feature_ref,
            gene_batch_size=100,
        )


def test_marker_search_does_not_accept_n_threads(
    datastore_ephemeral,
):
    clusters = _cluster_labels(
        datastore_ephemeral,
        np.arange(len(datastore_ephemeral.cells.active_index("I"))) % 2,
    )
    feature_ref = datastore_ephemeral.set_feature_selection(
        from_assay="RNA",
        mask=np.ones(datastore_ephemeral.RNA.feats.N, dtype=bool),
    )
    with pytest.raises(TypeError, match="n_threads"):
        datastore_ephemeral.run_marker_search(
            clusters,
            features=feature_ref,
            n_threads=4,
        )


@pytest.mark.parametrize("method_name", ["norm_clr", "norm_dummy", "norm_tf_idf"])
def test_dense_adapters_rank_countst_batches_like_the_dense_kernel(
    monkeypatch, method_name
) -> None:
    import scarf.assay.normalization as normalization
    from scarf.storage.budget import ResourceBudget
    from tests.test_feature_stream import _counts_t_with_plan

    # A selection of some features of each read group copies their rows.
    values = (np.arange(8 * 12, dtype=np.uint32).reshape(8, 12) * 7) % 11
    counts_t = _counts_t_with_plan(values)
    method = getattr(normalization, method_name)
    feature_index = np.array([1, 2, 10, 11])
    term_totals = values.sum(axis=1).astype(np.float64)
    document_frequency = (values[:, feature_index] > 0).sum(axis=0)

    class FakeAssay:
        def __init__(self) -> None:
            self.normMethod = method
            self.sf = 1000.0
            self.name = "RNA"
            self.resources = ResourceBudget(8 * 1024 * 1024, 2)
            self.rawDataT = counts_t

        def _fit_tf_idf(self, cell_idx, feat_idx, **_kwargs):
            np.testing.assert_array_equal(feat_idx, feature_index)
            return None, (term_totals, len(cell_idx), document_frequency)

    monkeypatch.setattr(marker_search_module, "ATACassay", FakeAssay)
    groups = np.array(["a", "a", "a", "b", "b", "b", "b", "a"])
    result = find_markers_by_rank(
        FakeAssay(),
        groups=groups,
        cell_idx=np.arange(8),
        feat_idx=feature_index[::-1],
        nthreads=1,
    )

    raw = values[:, feature_index]
    if method_name == "norm_clr":
        reference = normalization.clr_values(raw)
    elif method_name == "norm_tf_idf":
        reference = normalization.tfidf_values(
            raw,
            term_totals,
            normalization.inverse_document_frequency(8, document_frequency),
        )
    else:
        reference = raw
    codes = (groups == "b").astype(np.int64)
    np.testing.assert_array_equal(
        result.statistics, _batch_stats(reference, codes, np.bincount(codes), 8)
    )


def test_tf_idf_marker_search_requires_an_atac_assay() -> None:
    import scarf.assay.normalization as normalization
    from tests.test_feature_stream import _counts_t_with_plan

    class FakeAssay:
        def __init__(self) -> None:
            self.normMethod = normalization.norm_tf_idf
            self.rawDataT = _counts_t_with_plan(np.ones((4, 3), dtype=np.uint32))

    with pytest.raises(TypeError, match="requires an ATACassay"):
        find_markers_by_rank(FakeAssay(), _GROUPS, np.arange(4), np.arange(3))


def _dtype_counts(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return counts with edge-case genes and the group label of each cell."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes = 300, 24
    groups = np.repeat(np.arange(3), n_cells // 3)
    means = rng.gamma(0.6, 1.5, size=(3, n_genes))
    means[:, 0] = 0.0  # never expressed
    means[1:, 1] = 0.0  # expressed in one group only
    counts = rng.poisson(means[groups]).astype(np.float64)
    counts[:, 2] = 2.0  # equal counts in every cell
    return counts, np.array([f"g{group}" for group in groups])


def _count_store(tmp_path, counts: np.ndarray, dtype: str) -> DataStore:
    from tests.storage_helpers import write_count_store

    path = str(tmp_path / f"{dtype}.zarr")
    write_count_store(path, {"RNA": counts}, dtype)
    return DataStore(
        path,
        default_assay="RNA",
        min_features_per_cell=0,
        mito_pattern="",
        ribo_pattern="",
        nthreads=2,
    )


def _stored_marker_stats(store: DataStore, ref: ArtifactRef) -> dict[str, np.ndarray]:
    group = store.load_artifact(ref)
    return {name: np.asarray(group[name]["stats"][:]) for name in group.group_keys()}


def _refuse_dense_kernel(*_args, **_kwargs):
    raise AssertionError("library-size markers used the dense kernel")


@pytest.mark.parametrize("log_transform", [False, True])
def test_library_size_markers_match_across_count_dtypes(
    tmp_path, monkeypatch, log_transform
) -> None:
    monkeypatch.setattr(marker_search_module, "_batch_stats", _refuse_dense_kernel)
    counts, labels = _dtype_counts()
    tables = {}
    for dtype in _COUNT_DTYPES:
        store = _count_store(tmp_path, counts, dtype)
        assert store.RNA.rawDataT.dtype == np.dtype(dtype)
        ref = store.run_marker_search(
            _cluster_labels(store, labels),
            features=store.select_all_features(from_assay="RNA"),
            log_transform=log_transform,
        )
        tables[dtype] = _stored_marker_stats(store, ref)

    expected = tables["uint32"]
    assert sorted(expected) == ["g0", "g1", "g2"]
    for dtype, observed in tables.items():
        assert observed.keys() == expected.keys()
        for name in expected:
            np.testing.assert_array_equal(observed[name], expected[name], err_msg=dtype)


@pytest.mark.parametrize(
    ("cells", "feature", "value", "message"),
    [
        ([3], 5, -2.0, r"Feature 5 of RNA has a negative or non-finite"),
        ([3], 23, -50.0, r"RNA_nCounts holds negative or non-finite totals"),
        ([3, 3], 22, 1e308, r"RNA_nCounts holds negative or non-finite totals"),
    ],
    ids=["negative-count", "negative-total", "infinite-total"],
)
def test_library_size_markers_reject_invalid_counts_and_totals(
    tmp_path, cells, feature, value, message
) -> None:
    counts, labels = _dtype_counts()
    # Invalid totals come from features that are not tested.
    for offset, cell in enumerate(cells):
        counts[cell, feature + offset] = value
    store = _count_store(tmp_path, counts, "float64")
    clusters = _cluster_labels(store, labels)
    tested = np.arange(22 if feature >= 22 else 24)
    features = store.set_feature_selection(from_assay="RNA", feature_indexes=tested)

    with pytest.raises(ValueError, match=message):
        store.run_marker_search(clusters, features=features)
    assert store.list_artifacts(kind="marker_table", from_assay="RNA") == []


@pytest.mark.parametrize(
    ("features", "value", "message"),
    [
        ([5], -2.0, r"Feature 5 of RNA has a negative or non-finite"),
        # A negative total makes every value of the cell positive.
        (list(range(22)), -1.0, r"tested features of RNA hold negative or non"),
        pytest.param(
            [20, 21],
            1e308,
            r"tested features of RNA hold negative or non",
            # The subset total overflows to infinity.
            marks=pytest.mark.filterwarnings("ignore:overflow encountered"),
        ),
    ],
    ids=["negative-count", "negative-total", "infinite-total"],
)
def test_subset_renormalized_markers_reject_invalid_counts_and_totals(
    tmp_path, features, value, message
) -> None:
    counts, labels = _dtype_counts()
    counts[3, features] = value
    # Untested counts keep the library total positive, so the cell stays active.
    counts[3, 22:] = 100.0
    store = _count_store(tmp_path, counts, "float64")
    clusters = _cluster_labels(store, labels)
    tested = store.set_feature_selection(
        from_assay="RNA", feature_indexes=np.arange(22)
    )

    with pytest.raises(ValueError, match=message):
        store.run_marker_search(clusters, features=tested, renormalize_subset=True)
    assert store.list_artifacts(kind="marker_table", from_assay="RNA") == []


@pytest.mark.parametrize("method_name", ["norm_lib_size", "norm_clr", "norm_dummy"])
@pytest.mark.parametrize("axis", ["cell", "feat"])
def test_rank_markers_refuse_indices_past_the_end_of_counts_t(
    tmp_path, method_name, axis
) -> None:
    import scarf.assay.normalization as normalization

    counts, labels = _dtype_counts()
    store = _count_store(tmp_path, counts, "uint16")
    store.RNA.normMethod = getattr(normalization, method_name)
    indices = {"cell": np.arange(counts.shape[0]), "feat": np.arange(counts.shape[1])}
    # The last index is one past the end of its axis.
    indices[axis][-1] += 1

    with pytest.raises(IndexError, match=f"{axis}_idx contains an out-of-range"):
        find_markers_by_rank(store.RNA, labels, indices["cell"], indices["feat"])


def _saved_marker_table(
    tmp_path,
) -> tuple[DataStore, ArtifactRef, ArtifactRef, ArtifactRef]:
    """Return a small store, its clusters and features, and their marker table."""
    counts, labels = _dtype_counts()
    store = _count_store(tmp_path, counts, "uint16")
    clusters = _cluster_labels(store, labels)
    features = store.select_all_features(from_assay="RNA")
    return (
        store,
        clusters,
        features,
        store.run_marker_search(clusters, features=features),
    )


def test_marker_search_and_reader_refuse_references_of_other_kinds(tmp_path) -> None:
    store, clusters, features, _ = _saved_marker_table(tmp_path)

    with pytest.raises(TypeError, match="clusters must be an ArtifactRef"):
        store.run_marker_search("clusters", features=features)
    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        store.run_marker_search(clusters, features="RNA")
    with pytest.raises(ValueError, match="must be a complete clustering artifact"):
        store.run_marker_search(features, features=features)
    with pytest.raises(TypeError, match="marker must be an ArtifactRef"):
        store.get_markers("markers")
    with pytest.raises(ValueError, match="identify an assay marker_table artifact"):
        store.get_markers(clusters)
    missing = ArtifactRef(
        scope="assay", kind="marker_table", artifact_id="0" * 64, assay="RNA"
    )
    with pytest.raises(ValueError, match="does not exist"):
        store.get_markers(missing)


def test_marker_search_refuses_clusters_of_an_empty_cell_selection(tmp_path) -> None:
    counts, labels = _dtype_counts()
    store = _count_store(tmp_path, counts, "uint16")
    store.cells.insert("none", np.zeros(store.cells.N, dtype=bool), overwrite=True)
    store.cells.insert("labels", labels, overwrite=True)
    clusters = store.snapshot_cluster_labels(
        "labels", cell_selection=store.snapshot_cell_selection("none")
    )

    with pytest.raises(ValueError, match="no active cells"):
        store.run_marker_search(
            clusters, features=store.select_all_features(from_assay="RNA")
        )
    assert store.list_artifacts(kind="marker_table", from_assay="RNA") == []


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("incomplete", "is incomplete"),
        ("cell_selection-missing", "cell selection is missing"),
        ("cell_selection-invalid", "cell selection is invalid"),
        ("feature_selection-missing", "feature selection is missing"),
        ("feature_selection-invalid", "feature selection is invalid"),
        ("clusters-missing", "cluster input is missing"),
        ("clusters-invalid", "cluster input is invalid"),
        # A reference of another kind in place of the clusters.
        ("clusters-features", "cluster input is invalid"),
        ("feature_names", "missing frozen feature identities"),
        ("groups", "contains no groups"),
    ],
)
def test_get_markers_rejects_a_marker_table_it_cannot_trace(
    tmp_path, corruption, message
) -> None:
    from scarf.storage.artifacts import artifact_path

    store, _, features, ref = _saved_marker_table(tmp_path)
    group = store.zw[artifact_path(ref)]
    if corruption == "incomplete":
        group.attrs["complete"] = False
    elif corruption == "feature_names":
        del group["feature_names"]
    elif corruption == "groups":
        for name in list(group.group_keys()):
            del group[name]
    else:
        name, _, mode = corruption.partition("-")
        provenance = dict(group.attrs["provenance"])
        inputs = dict(provenance["inputs"])
        if mode == "missing":
            del inputs[name]
        else:
            inputs[name] = {"kind": name} if mode == "invalid" else features.to_dict()
        group.attrs["provenance"] = {**provenance, "inputs": inputs}

    with pytest.raises(ValueError, match=message):
        store.get_markers(ref)


def test_marker_search_refuses_labels_and_names_that_do_not_align(tmp_path) -> None:
    from scarf.metadata.selection import resolve_complete_labels

    store, clusters, features, _ = _saved_marker_table(tmp_path)
    labels = resolve_complete_labels(store.zw, clusters, name="clusters")
    inputs = {
        "assay": store.RNA,
        "cell_selection": labels.source_cell_selection,
        "clusters": clusters,
        "feature_selection": store.resolve_features("RNA", features),
    }

    with pytest.raises(ValueError, match="one label per selected cell"):
        store._run_marker_search_artifact(cluster_values=labels.values[1:], **inputs)
    with pytest.raises(ValueError, match="align with the assay feature axis"):
        store._run_marker_search_artifact(
            cluster_values=labels.values, feature_names=np.array(["g0"]), **inputs
        )


def test_marker_search_recomputes_a_table_whose_feature_names_changed(
    tmp_path,
) -> None:
    from scarf.storage.artifacts import artifact_path

    store, clusters, features, ref = _saved_marker_table(tmp_path)
    names = store.zw[artifact_path(ref)]["feature_names"]
    names[0] = "renamed"

    assert store.run_marker_search(clusters, features=features) != ref


@pytest.mark.parametrize("renormalize_subset", [False, True])
@pytest.mark.parametrize("log_transform", [False, True])
def test_library_size_marker_means_are_float32_rounded_float64_values(
    tmp_path, monkeypatch, log_transform, renormalize_subset
) -> None:
    monkeypatch.setattr(marker_search_module, "_batch_stats", _refuse_dense_kernel)
    counts, labels = _dtype_counts()
    store = _count_store(tmp_path, counts, "uint32")
    tested = np.arange(2, 20)
    result = find_markers_by_rank(
        store.RNA,
        labels,
        np.arange(len(labels)),
        tested,
        log_transform=log_transform,
        renormalize_subset=renormalize_subset,
    )
    # Subset renormalization divides by the total over the tested features.
    totals = counts[:, tested if renormalize_subset else slice(None)].sum(axis=1)
    totals[totals == 0] = 1
    values = 1000.0 * counts[:, tested] / totals[:, None]
    if log_transform:
        values = np.log1p(values)
    rounded = values.astype(np.float32).astype(np.float64)

    for position, group_id in enumerate(result.group_ids):
        members = rounded[labels == group_id]
        # The kernel adds the values of each group in cell order.
        expected = np.cumsum(members, axis=0)[-1] / len(members)
        np.testing.assert_array_equal(result.statistics[:, position, 1], expected)


def _reference_stored_stats(statistics: np.ndarray) -> np.ndarray:
    """Finish one group's kernel statistics the way marker tables always have."""
    from scipy.special import ndtr

    stored = np.empty((statistics.shape[0], 9))
    for column in range(8):
        stored[:, column] = np.round(statistics[:, column], 5)
    stored[:, 6] = 2.0 * ndtr(-np.abs(statistics[:, 6]))
    stored[:, 8] = adjust_pvalues(stored[:, 6], "fdr_bh")
    return stored


@pytest.mark.parametrize("method_name", ["norm_lib_size", "norm_clr", "norm_dummy"])
def test_stored_marker_tables_are_the_reference_statistics(
    tmp_path, method_name
) -> None:
    import scarf.assay.normalization as normalization

    counts, labels = _dtype_counts()
    store = _count_store(tmp_path, counts, "uint16")
    store.RNA.normMethod = getattr(normalization, method_name)
    ref = store.run_marker_search(
        _cluster_labels(store, labels),
        features=store.select_all_features(from_assay="RNA"),
    )

    if method_name == "norm_lib_size":
        totals = counts.sum(axis=1)
        totals[totals == 0] = 1
        values = (1000.0 * counts / totals[:, None]).astype(np.float32)
    elif method_name == "norm_clr":
        values = normalization.clr_values(counts.astype(np.uint16))
    else:
        values = counts
    group_ids, codes = np.unique(labels, return_inverse=True)
    reference = _batch_stats(values, codes, np.bincount(codes), len(labels))
    stored = _stored_marker_stats(store, ref)
    for position, group_id in enumerate(group_ids):
        np.testing.assert_array_equal(
            stored[group_id], _reference_stored_stats(reference[:, position])
        )


def _rebuild_interned_strings() -> None:
    """Make CPython rebuild its interned-string table before a trace starts.

    Pathlib interns every path part it parses, and those strings soon die;
    their slots fill the table until an insertion rebuilds it. A rebuild
    inside a trace adds a block the size of the table, because the table it
    replaces predates the trace. A rebuilt table holds far more insertions
    than a marker search and its writes make.
    """
    import sys
    import tracemalloc

    chunk = 1 << 12
    for start in range(0, 1 << 21, chunk):
        # Strings made before the trace leave a rebuilt table as the only
        # sizeable block that the trace sees.
        names = [
            f"scarf-test-interned-{index}" for index in range(start, start + chunk)
        ]
        tracemalloc.start()
        try:
            for name in names:
                sys.intern(name)
            current, _ = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if current > 1 << 16:
            return


def _layout_store(tmp_path, values: np.ndarray, **datastore_options) -> DataStore:
    """Write uint16 RNA counts in small read groups and open them."""
    import zarr

    from scarf.storage.count_matrix import CountMatrixPolicy
    from scarf.storage.schema import create_cell_data, create_zarr_count_assay
    from scarf.writers.counts_t import finalize_writer_counts_t
    from tests.storage_helpers import finalize_test_counts

    path = str(tmp_path / "layout.zarr")
    root = zarr.open_group(path, mode="w")
    n_cells, n_features = values.shape
    cell_ids = np.array([f"c{index}" for index in range(n_cells)])
    feature_ids = np.array([f"f{index}" for index in range(n_features)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    # Read groups of 32 features over sharded bands of 2,500 cells.
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        n_cells,
        feature_ids,
        feature_ids,
        dtype="uint16",
        policy=CountMatrixPolicy(unitBytes=n_cells * 32 * 2, chunkBytes=32 * 2_500 * 2),
    )
    counts[:] = values
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "RNA", None)
    return DataStore(
        path,
        default_assay="RNA",
        min_features_per_cell=0,
        mito_pattern="",
        ribo_pattern="",
        **datastore_options,
    )


@pytest.mark.parametrize("method_name", ["norm_lib_size", "norm_dummy"])
@pytest.mark.parametrize("n_groups", [20, 400])
def test_marker_search_and_write_fit_the_bytes_the_search_reserves(
    tmp_path, method_name, n_groups
) -> None:
    import tracemalloc

    import zarr

    import scarf.assay.normalization as normalization
    from scarf.storage.execution import execution_report_scope

    rng = np.random.default_rng(0)
    values = rng.poisson(0.15, size=(20_000, 256)).astype(np.uint16)
    store = _layout_store(tmp_path, values, nthreads=4, mem_budget="24M")
    store.RNA.normMethod = getattr(normalization, method_name)
    labels = rng.integers(0, n_groups, size=len(values))
    cells = np.arange(len(values))
    features = np.arange(values.shape[1])
    names = np.asarray(store.RNA.feats.fetch_all("names"))
    # Load the kernels, codecs, and pools, and import what finishes a table,
    # so that the trace holds only what the search and its writes allocate.
    DataStore._write_marker_slot(
        zarr.open_group(str(tmp_path / "warm.zarr"), mode="w"),
        find_markers_by_rank(store.RNA, labels, cells, features[:4], writers=2),
        workers=2,
        feature_names=names,
        feature_ids=names,
    )
    slot = zarr.open_group(str(tmp_path / "markers.zarr"), mode="w")
    _rebuild_interned_strings()

    with execution_report_scope() as reports:
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            result = find_markers_by_rank(store.RNA, labels, cells, features, writers=2)
            DataStore._write_marker_slot(
                slot, result, workers=2, feature_names=names, feature_ids=names
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    (plan,) = [
        report.as_metrics()
        for report in reports
        if report.unitKind == "countsTReadGroup"
    ]
    # NumPy, Zarr, and Numba buffers are traced. The search and the writes
    # that follow it hold no more than the search reserved.
    assert peak - base <= int(plan["reservedBytes"]) <= 24 * 1024**2
    assert sorted(slot.group_keys()) == sorted(str(group) for group in result.group_ids)


_SCRATCH_CHILD = textwrap.dedent(
    """
    import json
    import sys
    import threading
    from pathlib import Path

    import numba
    import numpy as np

    from scarf.features.markers import search
    from scarf.features.markers.rank import gene_major_rank_scratch_bytes
    from scarf.storage import feature_stream
    from tests.test_markers import _layout_store

    directory = Path(sys.argv[1])
    values = np.load(directory / "values.npy")
    n_cells, n_features = values.shape
    store = _layout_store(directory, values, nthreads=8)
    lock = threading.Lock()
    active = 0
    calls = []
    charged = []
    original_kernel = search._marker_stats_gene_major
    original_map = feature_stream.map_feature_read_groups

    def kernel(*args):
        global active
        with lock:
            active += 1
            calls.append((args[8], numba.get_num_threads(), active))
        try:
            return original_kernel(*args)
        finally:
            with lock:
                active -= 1

    def stream(*args, **kwargs):
        charged.append(kwargs["scratchBytes"])
        return original_map(*args, **kwargs)

    search._marker_stats_gene_major = kernel
    feature_stream.map_feature_read_groups = stream
    result = search.find_markers_by_rank(
        store.RNA, np.arange(n_cells) % 5, np.arange(n_cells), np.arange(n_features)
    )
    threads = max(call[0] for call in calls)
    concurrent = max(call[2] for call in calls)
    print("SCRATCH:" + json.dumps({
        "workers": store.RNA.resources.workers,
        "threads": sorted({call[0] for call in calls}),
        "numbaThreads": sorted({call[1] for call in calls}),
        "concurrentCalls": concurrent,
        "charged": charged[0] - result.statistics.nbytes,
        "kernelScratch": gene_major_rank_scratch_bytes(
            n_cells=n_cells,
            n_groups=5,
            n_features=32,
            nthreads=concurrent * threads,
        ),
    }))
    """
)


def test_marker_kernel_scratch_covers_the_kernel_threads_that_run(tmp_path) -> None:
    import json
    import os
    import subprocess
    import sys

    values = np.random.default_rng(1).poisson(0.3, size=(12_000, 128))
    np.save(tmp_path / "values.npy", values.astype(np.uint16))
    env = {**os.environ, "NUMBA_NUM_THREADS": "2", "SCARF_WORKERS": "8"}

    completed = subprocess.run(
        [sys.executable, "-c", _SCRATCH_CHILD, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    (line,) = [
        line for line in completed.stdout.splitlines() if line.startswith("SCRATCH:")
    ]
    observed = json.loads(line.removeprefix("SCRATCH:"))
    # Eight workers plan one ordered kernel call at a time, whose two slots
    # are the two Numba threads that run it.
    assert observed["workers"] == 8
    assert observed["concurrentCalls"] == 1
    assert observed["threads"] == observed["numbaThreads"] == [2]
    assert observed["charged"] >= observed["kernelScratch"]
