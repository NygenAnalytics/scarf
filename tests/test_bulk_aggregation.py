import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf import DataStore
from scarf.assay import norm_lib_size_log
from scarf.features.aggregation import _accumulate_group_counts, aggregate_rna_groups
from scarf.quality_control.doublets import write_doublet_target_zarr
from scarf.storage.budget import ResourceBudget
from scarf.storage.count_matrix import CountMatrixPolicy

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
    store.cells.insert("RNA_nCounts", np.array([220, 220, 0]), overwrite=True)
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
def test_bulk_preserves_half_precision_counts_and_sum_dtype(aggregation):
    counts = np.array(
        [[0.25, 1.5, 2.0], [200.0, 400.0, 1.0], [1.0, 0.0, 0.25], [0.0, 1.0, 0.0]],
        dtype=np.float16,
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
    assert actual.dtype == (np.float64 if aggregation == "mean" else np.float16)
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
