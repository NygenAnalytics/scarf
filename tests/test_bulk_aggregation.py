import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scarf import DataStore
from scarf.assay import norm_lib_size_log
from scarf.features.aggregation import _accumulate_group_counts, aggregate_rna_groups
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.writers import SparseToZarr

from .doublet_fixtures import write_doublet_target_zarr
from .test_feature_stream import _counts_t_with_plan


def _bulk_store(tmp_path, counts, *, workers=1):
    path = str(tmp_path / "counts.zarr")
    ids = np.array([f"g{i}" for i in range(counts.shape[1])])
    write_doublet_target_zarr(
        path,
        "RNA",
        csr_matrix(counts),
        ids,
        ids,
        dtype=str(counts.dtype),
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=4096, chunkBytes=128),
    )
    store = DataStore(
        path, default_assay="RNA", min_features_per_cell=0, nthreads=workers
    )
    store.cells.insert("all_cells", np.ones(len(counts), dtype=bool))
    return store, store.snapshot_cell_selection("all_cells")


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("return_fraction", [False, True])
def test_bulk_kernel_matches_grouped_counts(normalize, return_fraction):
    raw = np.array([[0, 90, 3, 7], [8, 80, 0, 2], [1, 70, 4, 0]], dtype=np.uint16)
    selected = np.array([0, 2, 3])
    codes = np.array([1, 0, 1])
    scalars = np.array([9.0, 7.0, 9.0]) if normalize else None
    expected = np.zeros((3, 3))
    expected_fractions = np.zeros((3, 3))
    for group in range(3):
        counts = raw[:, selected[codes == group]].astype(float)
        expected_fractions[:, group] = (counts > 0).sum(axis=1)
        if normalize:
            counts *= 100 / scalars[codes == group]
        expected[:, group] = counts.sum(axis=1)
    for kernel in (_accumulate_group_counts.py_func, _accumulate_group_counts):
        values = np.zeros((3, 3)).T
        fractions = np.zeros((3, 3)).T if return_fraction else None
        kernel(raw, selected, codes, scalars, 100.0, values, fractions)
        np.testing.assert_allclose(values, expected)
        if return_fraction:
            np.testing.assert_array_equal(fractions, expected_fractions)


@pytest.mark.parametrize(
    ("codes", "scalars", "message"),
    [
        ([0], None, "group codes"),
        ([-1, 0], None, "group codes"),
        ([0, 2], None, "group codes"),
        ([0, 1], [1.0], "normalization scalars"),
    ],
)
def test_bulk_rejects_misaligned_groups_and_scalars(codes, scalars, message):
    counts_t = _counts_t_with_plan(np.ones((3, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match=message):
        aggregate_rna_groups(
            counts_t,
            np.arange(2),
            np.array(codes),
            2,
            scalars=None if scalars is None else np.array(scalars),
            size_factor=100,
            return_fraction=False,
            resources=ResourceBudget(1_000_000, 1),
        )


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@pytest.mark.parametrize("replicates", [1, 3])
@pytest.mark.parametrize("workers", [1, 4])
def test_bulk_counts_t_matches_selected_groups_and_empty_replicates(
    tmp_path,
    aggregation,
    replicates,
    workers,
):
    counts = np.random.default_rng(71).integers(0, 200, (7, 24), dtype=np.uint16)
    counts[:, -1] = 0
    store, _ = _bulk_store(tmp_path, counts, workers=workers)
    selected = np.arange(7) != 5
    store.cells.insert("selected", selected)
    cells = store.snapshot_cell_selection("selected")
    store.cells.insert("group", np.array(["a", "a", "b", "skip", "a", "b", "b"]))
    store.cells.insert("secondary", np.array(["x", "y", "x", "x", "x", "y", "y"]))
    actual, fractions = store.make_bulk(
        "group",
        cell_selection=cells,
        secondary_groups="secondary",
        null_vals=["skip"],
        aggr_type=aggregation,
        pseudo_reps=replicates,
        return_fraction=True,
        feature_label="id",
        random_seed=61,
    )
    expected_columns = []
    for name, rows in {"a_x": [0, 4], "a_y": [1], "b_x": [2], "b_y": [6]}.items():
        shuffled = np.random.RandomState(61).choice(rows, len(rows), replace=False)
        for i, part in enumerate(np.array_split(shuffled, replicates)):
            column = name if replicates == 1 else f"{name}_Rep{i + 1}"
            expected_columns.append(column)
            raw = counts[sorted(part)]
            if len(part) == 0:
                expected = np.zeros(24)
                expected_fraction = expected
            else:
                expected_fraction = (raw > 0).mean(axis=0)
                expected = (
                    raw.sum(axis=0)
                    if aggregation == "sum"
                    else (
                        raw.astype(float) * store.RNA.sf / raw.sum(axis=1)[:, None]
                    ).mean(axis=0)
                )
            np.testing.assert_allclose(actual[column], expected[:-1])
            np.testing.assert_array_equal(fractions[column], expected_fraction[:-1])
    assert list(actual.columns) == list(fractions.columns) == expected_columns
    assert list(actual.index) == [f"g{i}" for i in range(23)]


@pytest.mark.parametrize("workers", [1, 4])
def test_bulk_sum_keeps_large_integers_and_ignores_custom_normalizer(tmp_path, workers):
    counts = np.array([[2**53 + 3, 1], [2**53 + 7, 2]], dtype=np.uint64)
    store, cells = _bulk_store(tmp_path, counts, workers=workers)
    store.cells.insert("group", np.array(["a", "a"]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw bulk sums must not call the normalizer")

    store.RNA.normMethod = forbidden
    actual = store.make_bulk("group", cell_selection=cells, aggr_type="sum")
    np.testing.assert_array_equal(actual["a"], counts.sum(axis=0))
    assert actual["a"].dtype == np.uint64


@pytest.mark.parametrize("workers", [1, 4])
def test_bulk_mean_uses_stored_totals_and_preserves_zero_total_behavior(
    tmp_path, workers
):
    counts = np.array([[100, 10], [200, 20], [0, 0]], dtype=np.uint16)
    store, cells = _bulk_store(tmp_path, counts, workers=workers)
    store.cells.insert("group", np.array(["a", "b", "b"]))
    store.zw["cellData"].create_array(
        "RNA_nCounts", data=np.array([220, 220, 0]), overwrite=True
    )
    actual, fractions = store.make_bulk(
        "group",
        cell_selection=cells,
        return_fraction=True,
        remove_empty_features=False,
    )
    np.testing.assert_allclose(
        actual["a"], counts[0].astype(float) * store.RNA.sf / 220
    )
    np.testing.assert_array_equal(actual["b"], 0)
    np.testing.assert_array_equal(fractions["b"], 0.5)


@pytest.mark.parametrize("normalizer", [norm_lib_size_log, lambda assay, raw: raw + 7])
@pytest.mark.parametrize("replicates", [1, 3])
def test_bulk_mean_retains_other_normalizers(tmp_path, normalizer, replicates):
    counts = np.array([[100, 10], [200, 20]], dtype=np.uint32)
    store, cells = _bulk_store(tmp_path, counts)
    store.cells.insert("group", np.array(["a", "a"]))
    store.RNA.normMethod = normalizer
    normalized = store.RNA.normed(
        cell_idx=np.arange(2), feat_idx=np.arange(2)
    ).compute()
    actual, fractions = store.make_bulk(
        "group",
        cell_selection=cells,
        pseudo_reps=replicates,
        return_fraction=True,
        random_seed=61,
    )
    shuffled = np.random.RandomState(61).choice(2, 2, replace=False)
    for i, rows in enumerate(np.array_split(shuffled, replicates)):
        column = "a" if replicates == 1 else f"a_Rep{i + 1}"
        expected = normalized[rows].mean(axis=0) if len(rows) else np.zeros(2)
        expected_fraction = (
            (counts[rows] > 0).mean(axis=0) if len(rows) else np.zeros(2)
        )
        np.testing.assert_allclose(actual[column], expected)
        np.testing.assert_array_equal(fractions[column], expected_fraction)


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_bulk_excludes_masked_labels_like_null_values(tmp_path, aggregation):
    from .test_pipeline import _insert_nullable_cell_column

    counts = np.random.default_rng(13).integers(1, 50, (6, 8), dtype=np.uint16)
    store, cells = _bulk_store(tmp_path, counts)
    # Placeholders equal real labels: 0 is a group and "" would be a sub-group.
    _insert_nullable_cell_column(
        store,
        "group",
        np.array([0, 1, 0, 0, 1, 0]),
        np.array([False, False, True, False, False, True]),
    )
    _insert_nullable_cell_column(
        store,
        "secondary",
        np.array(["x", "x", "y", "", "y", "x"]),
        np.array([False, False, False, True, False, False]),
    )
    store.cells.insert("group_nulls", np.array(["0", "1", "-", "0", "1", "-"]))
    store.cells.insert("secondary_nulls", np.array(["x", "x", "y", "-", "y", "x"]))
    options = {
        "cell_selection": cells,
        "aggr_type": aggregation,
        "feature_label": "id",
        "remove_empty_features": False,
    }

    masked = store.make_bulk("group", secondary_groups="secondary", **options)
    nulled = store.make_bulk(
        "group_nulls",
        secondary_groups="secondary_nulls",
        null_vals=["-"],
        secondary_null_vals=["-"],
        **options,
    )

    assert list(masked.columns) == ["0_x", "0_y", "1_x", "1_y"]
    pd.testing.assert_frame_equal(masked, nulled)
    if aggregation == "sum":
        np.testing.assert_array_equal(masked["0_x"], counts[0])
        np.testing.assert_array_equal(masked["0_y"], 0)


def test_bulk_leaves_nan_none_and_blank_labels_out_of_every_group(tmp_path):
    counts = np.arange(1, 25, dtype=np.uint16).reshape(6, 4)
    store, cells = _bulk_store(tmp_path, counts)
    store.cells.insert("float_group", np.array([1.0, 1.0, np.nan, 2.0, np.nan, 2.0]))
    store.cells.insert(
        "text_group",
        np.array(["a", "", "a", None, "b", " "], dtype=object),
    )
    options = {
        "cell_selection": cells,
        "aggr_type": "sum",
        "remove_empty_features": False,
    }

    by_float = store.make_bulk("float_group", **options)
    by_text = store.make_bulk("text_group", **options)

    assert list(by_float.columns) == ["1.0", "2.0"]
    np.testing.assert_array_equal(by_float["1.0"], counts[[0, 1]].sum(axis=0))
    np.testing.assert_array_equal(by_float["2.0"], counts[[3, 5]].sum(axis=0))
    assert list(by_text.columns) == ["a", "b"]
    np.testing.assert_array_equal(by_text["a"], counts[[0, 2]].sum(axis=0))


def test_bulk_rejects_colliding_column_names(tmp_path):
    store, cells = _bulk_store(tmp_path, np.ones((4, 3), dtype=np.uint16))
    store.cells.insert("group", np.array(["a_b", "a_b", "a", "a"]))
    store.cells.insert("secondary", np.array(["c", "c", "b_c", "b_c"]))

    with pytest.raises(ValueError, match="'a_b_c' is produced by more than one"):
        store.make_bulk("group", cell_selection=cells, secondary_groups="secondary")


def test_bulk_mean_fits_non_rna_normalization_once(tmp_path):
    counts = np.random.default_rng(5).integers(0, 30, (12, 5)).astype(np.uint32)
    counts[:, 0] += 1
    path = str(tmp_path / "adt.zarr")
    SparseToZarr(
        csr_matrix(counts),
        zarr_loc=path,
        cell_ids=[f"c{i}" for i in range(12)],
        feature_ids=[f"p{i}" for i in range(5)],
        assay_name="ADT",
        nthreads=1,
    ).dump(batch_size=4)
    store = DataStore(path, default_assay="ADT", min_features_per_cell=0)
    store.cells.insert("all_cells", np.ones(12, dtype=bool))
    cells = store.snapshot_cell_selection("all_cells")
    store.cells.insert("group", np.repeat(["a", "b", "c"], 4))
    options = {"cell_selection": cells, "remove_empty_features": False}

    every_group = store.make_bulk("group", **options)
    without_b = store.make_bulk("group", null_vals=["b"], **options)

    # CLR geometric means come from every selected cell, not from each group.
    normalized = np.log1p(counts / np.exp(np.log1p(counts).mean(axis=0)))
    for group, rows in (("a", slice(0, 4)), ("b", slice(4, 8)), ("c", slice(8, 12))):
        np.testing.assert_allclose(every_group[group], normalized[rows].mean(axis=0))
    assert list(without_b.columns) == ["a", "c"]
    np.testing.assert_allclose(without_b["a"], every_group["a"])


def test_bulk_rejects_unknown_aggregation(tmp_path):
    store, cells = _bulk_store(tmp_path, np.ones((2, 3), dtype=np.uint16))
    store.cells.insert("group", np.array(["a", "a"]))
    with pytest.raises(ValueError, match="aggr_type"):
        store.make_bulk("group", cell_selection=cells, aggr_type="median")


def test_bulk_sum_preserves_non_rna_assays(datastore_ephemeral):
    store = datastore_ephemeral
    selected = np.arange(store.cells.N) < 2
    store.cells.insert("bulk_cells", selected)
    store.cells.insert("bulk_group", np.repeat("a", store.cells.N))
    cells = store.snapshot_cell_selection("bulk_cells")
    counts = store.assay2.rawData[np.flatnonzero(selected)].compute()
    values, fractions = store.make_bulk(
        "bulk_group",
        cell_selection=cells,
        from_assay="assay2",
        aggr_type="sum",
        return_fraction=True,
        remove_empty_features=False,
        feature_label="id",
    )
    np.testing.assert_array_equal(values["a"], counts.sum(axis=0))
    np.testing.assert_array_equal(fractions["a"], (counts > 0).mean(axis=0))


def test_bulk_output_is_admitted_before_streaming(tmp_path):
    counts = np.ones((4, 24), dtype=np.uint16)
    store, _ = _bulk_store(tmp_path, counts)
    with pytest.raises(MemoryError, match="operation limit"):
        aggregate_rna_groups(
            store.RNA.rawDataT,
            np.arange(4),
            np.arange(4),
            1_000_000,
            scalars=None,
            size_factor=1_000,
            return_fraction=True,
            resources=ResourceBudget(1_000_000, 1),
        )


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_bulk_aggregates_floating_point_counts_in_float64(aggregation):
    counts = np.array(
        [[0.25, 1.5, 2.0], [200.0, 400.0, 1.0], [1.0, 0.0, 0.25], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    selected = np.array([0, 1, 3])
    codes = np.array([0, 0, 2])
    totals = counts[selected].sum(axis=1, dtype=np.float64)
    actual, fractions = aggregate_rna_groups(
        _counts_t_with_plan(counts),
        selected,
        codes,
        4,
        scalars=totals if aggregation == "mean" else None,
        size_factor=1000,
        return_fraction=True,
        resources=ResourceBudget(16 * 1024**2, 4),
    )
    assert actual.dtype == np.float64
    assert fractions is not None
    for code in range(4):
        members = codes == code
        raw = counts[selected[members]]
        if aggregation == "mean":
            values = raw.astype(np.float64) * 1000 / totals[members, None]
            expected = values.sum(axis=0) / max(1, members.sum())
        else:
            expected = raw.sum(axis=0)
        np.testing.assert_allclose(actual[:, code], expected)
        np.testing.assert_array_equal(
            fractions[:, code], (raw > 0).sum(axis=0) / max(1, members.sum())
        )


@pytest.mark.parametrize("codes", [[0, 1], [0, 1, 2], [-2, 0, 1]])
def test_normalized_aggregation_rejects_codes_off_its_rows(codes):
    from scarf.features.aggregation import aggregate_normalized_groups
    from scarf.matrix import ChunkedArray

    with pytest.raises(ValueError, match="align with normalized rows"):
        aggregate_normalized_groups(
            ChunkedArray.from_numpy(np.ones((3, 2))), np.array(codes), 2, nthreads=1
        )


def test_normalized_aggregation_reserves_its_group_sums(tmp_path):
    import tracemalloc

    import zarr
    from zarr.storage import LocalStore

    from scarf.features.aggregation import aggregate_normalized_groups
    from scarf.matrix import ChunkedArray
    from scarf.storage.execution import execution_report_scope

    rng = np.random.default_rng(0)
    values = rng.poisson(0.3, size=(8_000, 64)).astype(np.uint16)
    root = zarr.open_group(store=LocalStore(str(tmp_path)), mode="w")
    counts = root.create_array(
        "counts",
        shape=values.shape,
        chunks=(1_000, 64),
        shards=(1_000, 64),
        dtype=np.uint16,
        fill_value=0,
    )
    counts[:] = values
    # The sums of 6,000 groups outweigh the two row blocks read at a time.
    codes = rng.integers(-1, 6_000, size=len(values))
    budget = 6_000_000

    def aggregate() -> np.ndarray:
        normalized = ChunkedArray(
            counts, nthreads=2, resources=ResourceBudget(budget, 2)
        )
        return aggregate_normalized_groups(normalized * 0.5, codes, 6_000, nthreads=2)

    aggregate()
    with execution_report_scope() as reports:
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            means = aggregate()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    (plan,) = [report.plan for report in reports]
    assert plan.residentBytes >= 6_000 * 64 * 8
    assert peak - base <= plan.reservedBytes <= budget
    grouped = codes >= 0
    expected = np.zeros((6_000, 64))
    np.add.at(expected, codes[grouped], values[grouped] * 0.5)
    expected /= np.maximum(np.bincount(codes[grouped], minlength=6_000), 1)[:, None]
    np.testing.assert_allclose(means, expected.T)


def test_bulk_sums_of_float_counts_are_exact_above_float32_precision():
    # Beyond 2**24 float32 holds only even integers, so a sum returned in the
    # storage dtype rounded an exact odd total.
    counts = np.array([[2.0**24], [1.0], [1.0], [1.0]], dtype=np.float32)
    selected = np.arange(4)

    actual, _ = aggregate_rna_groups(
        _counts_t_with_plan(counts),
        selected,
        np.zeros(4, dtype=np.int64),
        1,
        scalars=None,
        size_factor=1000,
        return_fraction=False,
        resources=ResourceBudget(16 * 1024**2, 1),
    )

    assert actual.dtype == np.float64
    assert actual[0, 0] == 2**24 + 3
