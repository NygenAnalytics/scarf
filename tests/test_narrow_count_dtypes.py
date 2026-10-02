"""Normalization arithmetic does not depend on the count storage dtype.

Every count dtype normalizes in float64 from the counts and float64 totals,
and a persisted value is that result rounded once to float32, so the same
counts in the same count layout give bit-identical results in every storage
dtype. Sums over cells follow the stored row blocks and countsT cell bands,
so these stores keep one chunk in every dtype.
"""

import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.assay.feature_summary import ensure_feature_summary
from scarf.assay.normalization import (
    _normalize_count_block,
    inverse_document_frequency,
    norm_clr,
    norm_lib_size,
    norm_lib_size_log,
    tfidf_values,
)
from scarf.datastore.datastore import DataStore
from scarf.mapping.features import _normalization_parameters
from scarf.matrix import ChunkedArray
from scarf.metadata.selection import CellField
from scarf.storage.artifact_writer import artifact_transaction, plan_artifact
from scarf.storage.artifacts import artifact_group, callable_identity
from scarf.trajectory.artifacts import (
    MARKER_INPUTS,
    validate_aggregation_parameters,
    validate_marker_parameters,
)
from tests.storage_helpers import write_count_store

SIZE_FACTOR = 1000
# Every dtype that holds the counts of ``_multimodal_counts`` exactly.
COUNT_DTYPES = ["uint8", "uint16", "uint32", "int32", "int64", "float32", "float64"]
KERNEL_DTYPES = ["bool", "uint64", "int16", *COUNT_DTYPES]
RNA_SUBSET = np.array([0, 2, 3, 5, 8, 13, 21, 34, 55, 89], dtype=np.int64)
ATAC_SUBSET = np.arange(0, 40, 3, dtype=np.int64)


def _lib_size_reference(
    values: np.ndarray,
    *,
    feat_idx: np.ndarray | None = None,
    renormalize_subset: bool = False,
    log_transform: bool = False,
) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float64)
    selected = counts if feat_idx is None else counts[:, feat_idx]
    totals = (selected if renormalize_subset else counts).sum(axis=1)
    totals[totals == 0] = 1
    reference = SIZE_FACTOR * selected / totals[:, None]
    return np.log1p(reference) if log_transform else reference


def _clr_reference(values: np.ndarray) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float64)
    scale = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    return np.log1p(counts / scale[None, :])


def _tfidf_reference(
    values: np.ndarray,
    *,
    feat_idx: np.ndarray | None = None,
    renormalize_subset: bool = False,
) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float64)
    selected = counts if feat_idx is None else counts[:, feat_idx]
    totals = (selected if renormalize_subset else counts).sum(axis=1)
    totals[totals == 0] = 1
    idf = inverse_document_frequency(len(counts), np.count_nonzero(selected, axis=0))
    return selected / totals[:, None] * idf[None, :]


def _multimodal_counts() -> dict[str, np.ndarray]:
    """RNA, ADT, and ATAC counts that every dtype in COUNT_DTYPES holds."""
    rng = np.random.default_rng(17)
    n_cells, n_genes = 150, 100
    rna = np.minimum(
        rng.poisson(rng.gamma(0.6, 3.0, size=n_genes), size=(n_cells, n_genes)), 255
    )
    rna[:, 0] = rng.integers(128, 256, size=n_cells)
    adt = rng.poisson(rng.gamma(2.0, 20.0, size=6), size=(n_cells, 6))
    atac = rng.poisson(0.3, size=(n_cells, 40))
    return {"RNA": rna, "ADT": np.minimum(adt, 254) + 1, "ATAC": atac}


MULTIMODAL_COUNTS = _multimodal_counts()


def _store(path: Any, counts: dict[str, np.ndarray], dtype: Any) -> DataStore:
    # These stores are small, so every dtype gets the same one-chunk layout.
    zarr_loc = str(path / f"{np.dtype(dtype).name}.zarr")
    write_count_store(zarr_loc, counts, dtype)
    return DataStore(
        zarr_loc,
        default_assay=next(iter(counts)),
        min_features_per_cell=0,
        nthreads=1,
    )


def _feature_mask(store: DataStore, assay: str, feat_idx: np.ndarray) -> Any:
    mask = np.zeros(store.get_assay(assay).feats.N, dtype=bool)
    mask[feat_idx] = True
    return store.set_feature_selection(from_assay=assay, mask=mask)


def _every_cell(store: DataStore) -> Any:
    """Select every cell, including cells without counts that ``I`` excludes."""
    store.cells.insert("everyone", np.ones(store.cells.N, dtype=bool), overwrite=True)
    return store.snapshot_cell_selection("everyone")


def _arrays(store: DataStore, ref: Any, prefix: str, results: dict[str, Any]) -> None:
    group = artifact_group(store.zw, ref)
    for name in sorted(group.array_keys()):
        results[f"{prefix}/{name}"] = np.asarray(group[name][:])


def _feature_batches(
    store: DataStore,
    feat_idx: np.ndarray,
    **norm_params: Any,
) -> np.ndarray:
    """Collect ``iter_normed_feature_wise`` batches as a cells-by-features array."""
    columns = {int(feature): column for column, feature in enumerate(feat_idx)}
    values = np.full((store.cells.N, len(feat_idx)), np.nan)
    for batch, labels in store.RNA.iter_normed_feature_wise(
        np.arange(store.cells.N, dtype=np.int64),
        feat_idx,
        batch_size=None,
        msg=None,
        as_dataframe=False,
        **norm_params,
    ):
        for row, label in zip(batch, labels, strict=True):
            values[:, columns[int(label)]] = row
    return values


def _count_results(store: DataStore) -> dict[str, Any]:
    """Compute the normalized results that the invariance tests compare."""
    cells = store.snapshot_cell_selection()
    rna = _feature_mask(store, "RNA", RNA_SUBSET)
    cell_idx = np.arange(store.cells.N, dtype=np.int64)
    results: dict[str, Any] = {}
    default = store.run_normalization(cells, rna)
    for name, ref in {
        "rna_default": default,
        "rna_library_log": store.run_normalization(
            cells, rna, renormalize_subset=False, log_transform=True
        ),
        "adt_clr": store.run_normalization(
            cells, store.select_all_features(from_assay="ADT")
        ),
        "atac_tfidf": store.run_normalization(
            cells, store.select_all_features(from_assay="ATAC")
        ),
        "atac_tfidf_subset": store.run_normalization(
            cells,
            _feature_mask(store, "ATAC", ATAC_SUBSET),
            renormalize_subset=True,
        ),
        "summary": ensure_feature_summary(store.zw, store.RNA, cells),
        "summary_log": ensure_feature_summary(
            store.zw, store.RNA, cells, log_transform=True
        ),
        "summary_atac": ensure_feature_summary(store.zw, store.ATAC, cells),
        "hvgs": store.select_hvgs(
            cells,
            top_n=30,
            min_cells=5,
            max_cells=np.inf,
            n_bins=10,
            bin_strategy="fixed",
            show_plot=False,
        ),
        "pca": store.run_pca(default, dims=5, local_cache=False),
        "cell_cycle": store.run_cell_cycle_scoring(
            cells,
            s_genes=[f"RNA{index}" for index in range(10)],
            g2m_genes=[f"RNA{index}" for index in range(10, 20)],
            ctrl_size=5,
            n_bins=5,
        ),
        "percentage": store.run_feature_percentage(cells, rna),
    }.items():
        _arrays(store, ref, name, results)
    results["normed/library"] = store.RNA.normed(cell_idx, RNA_SUBSET).compute()
    results["normed/subset_log"] = store.RNA.normed(
        cell_idx, RNA_SUBSET, renormalize_subset=True, log_transform=True
    ).compute()
    results["normed/adt"] = store.ADT.normed(cell_idx).compute()
    results["normed/atac"] = store.ATAC.normed(cell_idx).compute()
    results["feature_batches/library_log"] = _feature_batches(
        store, RNA_SUBSET, log_transform=True
    )
    results["feature_batches/subset"] = _feature_batches(
        store, RNA_SUBSET, renormalize_subset=True
    )
    store.cells.insert("group", np.where(cell_idx % 3 == 0, "a", "b"), overwrite=True)
    tests = store.run_statistical_testing(
        [f"RNA{index}" for index in range(3)], CellField("group")
    )
    results["tests"] = np.asarray(tests.value_fingerprints)
    for key, table in tests.tables.items():
        results[f"tests/{key}"] = table
    for aggr_type in ("sum", "mean"):
        results[f"bulk/{aggr_type}"] = store.make_bulk(
            "group", aggr_type=aggr_type, remove_empty_features=False
        )
    return results


def _sum_dtype(dtype: Any) -> np.dtype:
    """Return the dtype of exact raw sums: NumPy's integer sum dtype or float64."""
    if np.dtype(dtype).kind == "f":
        return np.dtype(np.float64)
    return np.empty(0, dtype=dtype).sum().dtype


@pytest.fixture(scope="module")
def count_results(tmp_path_factory: Any) -> Any:
    """Return the results of the multimodal counts stored in a given dtype."""
    cache: dict[str, dict[str, Any]] = {}

    def results(dtype: str) -> dict[str, Any]:
        if dtype not in cache:
            store = _store(tmp_path_factory.mktemp(dtype), MULTIMODAL_COUNTS, dtype)
            cache[dtype] = _count_results(store)
        return cache[dtype]

    return results


@pytest.mark.parametrize(
    "dtype", [dtype for dtype in COUNT_DTYPES if dtype != "float64"]
)
@pytest.mark.filterwarnings("ignore:Cell-level statistical testing:UserWarning")
def test_normalized_results_do_not_depend_on_the_count_dtype(count_results, dtype):
    expected = count_results("float64")
    actual = count_results(dtype)

    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        if isinstance(values, pd.DataFrame):
            # Raw sums keep an integer dtype on integer stores.
            pd.testing.assert_frame_equal(
                actual[name],
                values,
                check_dtype=name != "bulk/sum",
                check_exact=True,
            )
        else:
            np.testing.assert_array_equal(actual[name], values, err_msg=name)
    assert set(actual["bulk/sum"].dtypes) == {_sum_dtype(dtype)}


@pytest.mark.filterwarnings("ignore:Cell-level statistical testing:UserWarning")
def test_normalized_results_are_the_float64_reference(count_results):
    results = count_results("float64")
    rna, adt, atac = (MULTIMODAL_COUNTS[name] for name in ("RNA", "ADT", "ATAC"))

    payloads = {
        "rna_default": _lib_size_reference(
            rna, feat_idx=RNA_SUBSET, renormalize_subset=True, log_transform=True
        ),
        "rna_library_log": _lib_size_reference(
            rna, feat_idx=RNA_SUBSET, log_transform=True
        ),
        "adt_clr": _clr_reference(adt),
        "atac_tfidf": _tfidf_reference(atac),
        "atac_tfidf_subset": _tfidf_reference(
            atac, feat_idx=ATAC_SUBSET, renormalize_subset=True
        ),
    }
    for name, expected in payloads.items():
        data = results[f"{name}/data"]
        assert data.dtype == np.float32
        # The payload is the float64 value rounded once to float32.
        np.testing.assert_array_equal(data, expected.astype(np.float32), err_msg=name)
        widened = data.astype(np.float64)
        np.testing.assert_allclose(
            results[f"{name}/feature_sum"], widened.sum(axis=0), rtol=1e-12
        )
        np.testing.assert_allclose(
            results[f"{name}/feature_squared_sum"],
            np.square(widened).sum(axis=0),
            rtol=1e-12,
        )
    np.testing.assert_array_equal(
        results["normed/library"], _lib_size_reference(rna, feat_idx=RNA_SUBSET)
    )
    np.testing.assert_array_equal(
        results["normed/subset_log"],
        _lib_size_reference(
            rna, feat_idx=RNA_SUBSET, renormalize_subset=True, log_transform=True
        ),
    )
    np.testing.assert_array_equal(results["normed/adt"], _clr_reference(adt))
    np.testing.assert_array_equal(results["normed/atac"], _tfidf_reference(atac))
    # Batches streamed from countsT hold the values that normed computes.
    np.testing.assert_array_equal(
        results["feature_batches/library_log"],
        _lib_size_reference(rna, feat_idx=RNA_SUBSET, log_transform=True),
    )
    np.testing.assert_array_equal(
        results["feature_batches/subset"],
        _lib_size_reference(rna, feat_idx=RNA_SUBSET, renormalize_subset=True),
    )

    for name, log_transform in (("summary", False), ("summary_log", True)):
        normalized = _lib_size_reference(rna, log_transform=log_transform)
        mean = normalized.mean(axis=0)
        np.testing.assert_allclose(
            results[f"{name}/normed_tot"], normalized.sum(axis=0), rtol=1e-12
        )
        np.testing.assert_array_equal(
            results[f"{name}/normed_n"], (normalized > 0).sum(axis=0)
        )
        np.testing.assert_allclose(
            results[f"{name}/sigmas"],
            np.square(normalized).mean(axis=0) - np.square(mean),
            rtol=1e-9,
            atol=1e-9,
        )
    np.testing.assert_allclose(
        results["summary_atac/prevalence"],
        _tfidf_reference(atac).sum(axis=0),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        results["percentage/values"],
        100 * rna[:, RNA_SUBSET].sum(axis=1) / rna.sum(axis=1),
    )
    groups = np.where(np.arange(len(rna)) % 3 == 0, "a", "b")
    for group in ("a", "b"):
        members = groups == group
        np.testing.assert_array_equal(
            results["bulk/sum"][group].to_numpy(), rna[members].sum(axis=0)
        )
        np.testing.assert_allclose(
            results["bulk/mean"][group].to_numpy(),
            _lib_size_reference(rna)[members].mean(axis=0),
            rtol=1e-12,
        )


WIDE_COUNTS = np.array(
    [
        [9_000_001, 9_000_002, 4, 0, 6],
        [8_388_609, 1, 0, 3, 0],
        [16_777_215, 16_777_213, 1, 2, 9],
        [0, 0, 0, 0, 0],
        [5, 7, 11, 13, 17],
        [1, 0, 2, 0, 3],
    ]
)
WIDE_SUBSET = np.array([0, 1, 2], dtype=np.int64)


@pytest.mark.parametrize("dtype", ["uint32", "int32", "int64", "float32", "float64"])
def test_totals_beyond_float32_precision_are_exact(tmp_path, dtype):
    # The library total 18,000,013, subset totals such as 18,000,007, and sums
    # such as 34,165,831 are odd integers that float32 cannot hold, although
    # it holds every count.
    store = _store(tmp_path, {"RNA": WIDE_COUNTS, "ATAC": WIDE_COUNTS}, dtype)
    cells = _every_cell(store)
    features = _feature_mask(store, "RNA", WIDE_SUBSET)
    cell_idx = np.arange(len(WIDE_COUNTS), dtype=np.int64)

    for renormalize_subset in (False, True):
        for log_transform in (False, True):
            flags = {
                "renormalize_subset": renormalize_subset,
                "log_transform": log_transform,
            }
            expected = _lib_size_reference(WIDE_COUNTS, feat_idx=WIDE_SUBSET, **flags)
            normed = store.RNA.normed(cell_idx, WIDE_SUBSET, **flags).compute()
            np.testing.assert_array_equal(normed, expected)
            # The cell without counts normalizes to zero.
            np.testing.assert_array_equal(normed[3], 0)
            # Batches streamed from countsT divide by the same totals.
            np.testing.assert_array_equal(
                _feature_batches(store, WIDE_SUBSET, **flags), expected
            )
            payload = store.run_normalization(cells, features, **flags)
            np.testing.assert_array_equal(
                artifact_group(store.zw, payload)["data"][:],
                expected.astype(np.float32),
            )

    tfidf = store.run_normalization(
        cells, _feature_mask(store, "ATAC", WIDE_SUBSET), renormalize_subset=True
    )
    np.testing.assert_array_equal(
        artifact_group(store.zw, tfidf)["data"][:],
        _tfidf_reference(
            WIDE_COUNTS, feat_idx=WIDE_SUBSET, renormalize_subset=True
        ).astype(np.float32),
    )

    part = WIDE_COUNTS[:, WIDE_SUBSET].sum(axis=1)
    totals = WIDE_COUNTS.sum(axis=1)
    percentages = np.full(len(totals), np.nan)
    np.divide(100.0 * part, totals, out=percentages, where=totals != 0)
    percentage = store.run_feature_percentage(cells, features)
    np.testing.assert_array_equal(
        artifact_group(store.zw, percentage)["values"][:], percentages
    )

    store.cells.insert("bulk", np.full(len(cell_idx), "all"), overwrite=True)
    # RNA sums stream from countsT, and other assays sum their counts.
    for assay in ("RNA", "ATAC"):
        sums = store.make_bulk(
            "bulk",
            from_assay=assay,
            cell_selection=cells,
            aggr_type="sum",
            remove_empty_features=False,
        )
        assert sums["all"].dtype == _sum_dtype(dtype), assay
        np.testing.assert_array_equal(
            sums["all"].to_numpy(), WIDE_COUNTS.sum(axis=0), err_msg=assay
        )


@pytest.mark.parametrize("dtype", ["bool", "uint8", "float32"])
def test_binary_counts_normalize_alike_in_every_dtype(tmp_path, dtype):
    rng = np.random.default_rng(29)
    rna = rng.random((40, 12)) < 0.4
    rna[:, 0] = True
    atac = rng.random((40, 20)) < 0.3
    store = _store(tmp_path, {"RNA": rna, "ATAC": atac}, dtype)
    assert store.RNA.rawData.dtype == np.dtype(dtype)
    cells = store.snapshot_cell_selection()
    feat_idx = np.arange(1, 12)
    features = _feature_mask(store, "RNA", feat_idx)

    # The default normalization runs the subset kernel over countsT bands.
    for renormalize_subset, log_transform in ((True, True), (False, False)):
        flags = {
            "renormalize_subset": renormalize_subset,
            "log_transform": log_transform,
        }
        payload = store.run_normalization(cells, features, **flags)
        np.testing.assert_array_equal(
            artifact_group(store.zw, payload)["data"][:],
            _lib_size_reference(rna, feat_idx=feat_idx, **flags).astype(np.float32),
        )
    tfidf = store.run_normalization(cells, store.select_all_features(from_assay="ATAC"))
    np.testing.assert_array_equal(
        artifact_group(store.zw, tfidf)["data"][:],
        _tfidf_reference(atac).astype(np.float32),
    )
    np.testing.assert_array_equal(
        store.RNA.normed(np.arange(40), np.arange(12)).compute(),
        _lib_size_reference(rna),
    )


def _kernel_counts(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(5)
    raw = np.minimum(rng.poisson(rng.gamma(0.8, 20.0, size=17), size=(41, 17)), 255)
    raw[:, 0] += 1
    return raw > 0 if dtype == "bool" else raw.astype(dtype)


@pytest.mark.parametrize("block_size", [1, 3, 64])
@pytest.mark.parametrize("dtype", KERNEL_DTYPES)
def test_normalizers_compute_float64_values_for_every_dtype(dtype, block_size):
    values = _kernel_counts(dtype)
    counts = ChunkedArray.from_numpy(values, block_size=block_size)
    widened = values.astype(np.float64)
    totals = widened.sum(axis=1)
    assay = SimpleNamespace(sf=SIZE_FACTOR, scalar=totals)

    for method, log_transform in ((norm_lib_size, False), (norm_lib_size_log, True)):
        actual = method(assay, counts).compute()
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(
            actual, _lib_size_reference(widened, log_transform=log_transform)
        )
    # CLR sums log1p over the row blocks, so the float64 counts in the same
    # blocks give its exact value.
    clr = norm_clr(assay, counts).compute()
    exact = norm_clr(assay, ChunkedArray.from_numpy(widened, block_size=block_size))
    np.testing.assert_array_equal(clr, exact.compute())
    np.testing.assert_allclose(clr, _clr_reference(widened), rtol=1e-14)
    idf = inverse_document_frequency(len(values), np.count_nonzero(values, axis=0))
    np.testing.assert_array_equal(
        tfidf_values(counts, totals, idf).compute(), tfidf_values(widened, totals, idf)
    )
    np.testing.assert_array_equal(
        _normalize_count_block(values, scaleFactor=SIZE_FACTOR, logTransform=True),
        _lib_size_reference(
            widened, renormalize_subset=True, log_transform=True
        ).astype(np.float32),
    )


def test_float32_counts_normalize_as_their_float64_values():
    rng = np.random.default_rng(9)
    fractional = (rng.poisson(5.0, size=(30, 8)) + rng.random((30, 8))).astype(
        np.float32
    )
    widened = fractional.astype(np.float64)
    assay = SimpleNamespace(sf=SIZE_FACTOR, scalar=widened.sum(axis=1))

    for method in (norm_lib_size, norm_lib_size_log, norm_clr):
        np.testing.assert_array_equal(
            method(assay, ChunkedArray.from_numpy(fractional)).compute(),
            method(assay, ChunkedArray.from_numpy(widened)).compute(),
        )
    for log_transform in (False, True):
        np.testing.assert_array_equal(
            _normalize_count_block(
                fractional, scaleFactor=SIZE_FACTOR, logTransform=log_transform
            ),
            _normalize_count_block(
                widened, scaleFactor=SIZE_FACTOR, logTransform=log_transform
            ),
        )


@pytest.mark.parametrize("dtype", ["uint32", "int32", "uint64", "int64"])
@pytest.mark.parametrize("log_transform", [False, True])
@pytest.mark.parametrize("block_size", [1, 3])
def test_library_size_normalization_does_not_overflow_wide_integers(
    dtype, log_transform, block_size
):
    largest_safe = int(np.iinfo(dtype).max) // SIZE_FACTOR
    values = np.array([[largest_safe, 1], [largest_safe + 1, 1]], dtype=dtype)
    counts = ChunkedArray.from_numpy(values, block_size=block_size)
    assay = SimpleNamespace(
        sf=SIZE_FACTOR,
        scalar=values.sum(axis=1).astype(np.float64),
    )
    method = norm_lib_size_log if log_transform else norm_lib_size

    actual = method(assay, counts).compute()

    assert actual.dtype == np.float64
    np.testing.assert_allclose(
        actual,
        _lib_size_reference(values, log_transform=log_transform),
        rtol=1e-15,
    )


def test_clr_on_uint8_counts_does_not_accumulate_in_float16(tmp_path):
    # log1p of uint8 counts is float16 by default, and its per-feature sum
    # over these cells passes the float16 maximum of 65504.
    n_cells = 15_000
    rng = np.random.default_rng(3)
    values = rng.integers(150, 256, size=(n_cells, 3))
    store = _store(tmp_path, {"ADT": values}, "uint8")
    assert store.ADT.rawData.dtype == np.uint8
    assert np.all(np.log1p(values).sum(axis=0) > np.finfo(np.float16).max)

    actual = store.ADT.normed(cell_idx=np.arange(n_cells)).compute()

    expected = _clr_reference(values)
    assert np.all(expected > 0)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_chunked_ufuncs_honor_a_requested_dtype():
    values = np.arange(24, dtype=np.uint16).reshape(6, 4) * 300
    counts = ChunkedArray.from_numpy(values, block_size=4)

    wrapped = np.multiply(counts, SIZE_FACTOR)
    assert wrapped.dtype == np.uint16
    widened = np.multiply(counts, SIZE_FACTOR, dtype=np.float64)
    assert widened.dtype == np.float64
    logged = np.log1p(counts, dtype=np.float64)
    assert logged.dtype == np.float64

    selected = widened[[1, 4, 5], :][:, [0, 3]]
    expected = SIZE_FACTOR * values[[1, 4, 5]][:, [0, 3]].astype(np.float64)
    np.testing.assert_array_equal(selected.compute(), expected)
    np.testing.assert_array_equal(logged.compute(), np.log1p(values.astype(np.float64)))
    restored = pickle.loads(pickle.dumps(selected))
    np.testing.assert_array_equal(restored.compute(), expected)


REMOVED_FIELDS = [
    pytest.param("count_arithmetic", "float64", id="count_arithmetic"),
    pytest.param("zero_total_divisor", "one", id="zero_total_divisor"),
]
_MARKER_RECORD = {
    "normalization": {"log_transform": False, "renormalize_subset": False},
    "normalization_method": callable_identity(norm_lib_size),
    "size_factor": 1000.0,
    "association_method": "pearson",
    "p_value_method": "student_t",
    "adjustment_method": "fdr_bh",
    "adjustment_scope": "tested_features",
    "min_cells": 10,
}
_AGGREGATION_RECORD = {
    "normalization": _MARKER_RECORD["normalization"],
    "normalization_method": _MARKER_RECORD["normalization_method"],
    "size_factor": 1000.0,
    "min_exp": 1e-3,
    "window_size": 20,
    "chunk_size": 10,
    "smoothen": True,
    "z_scale": True,
    "n_neighbours": 2,
    "n_clusters": 3,
    "ann_params": {},
    "nan_cluster_value": -1,
}
_REFERENCE_RECORD = {
    "normalization_method": {"external_hook": True, **callable_identity(norm_lib_size)},
    "size_factor": 1000.0,
    "log_transform": True,
    "renormalize_subset": False,
}
REFERENCE_REMEDY = (
    "Recompute the normalization with run_normalization, then PCA and its descendants"
)


@pytest.mark.parametrize(("name", "value"), REMOVED_FIELDS)
def test_strict_loaders_reject_records_with_removed_fields(tmp_path, name, value):
    for validate, record in (
        (validate_marker_parameters, _MARKER_RECORD),
        (validate_aggregation_parameters, _AGGREGATION_RECORD),
        (_normalization_parameters, _REFERENCE_RECORD),
    ):
        validate(record)
        with pytest.raises(ValueError):
            validate({**record, name: value})
    with pytest.raises(ValueError, match=f"parameters: {name}. {REFERENCE_REMEDY}$"):
        _normalization_parameters({**_REFERENCE_RECORD, name: value})

    # A pseudotime-marker record of an earlier release candidate fails to load
    # with a request to recompute it.
    store = _store(tmp_path, {"RNA": WIDE_COUNTS}, "uint32")
    planned = plan_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="pseudotime_markers",
        operation="run_pseudotime_marker_search",
        parameters={**_MARKER_RECORD, name: value},
        inputs=dict.fromkeys(MARKER_INPUTS, "recorded"),
        execution_options={},
    )
    with artifact_transaction(store.zw, planned):
        pass
    with pytest.raises(
        ValueError,
        match="come from an earlier release; rerun run_pseudotime_marker_search",
    ):
        store.load_pseudotime_markers(planned.ref)


def test_normalizations_recorded_with_removed_fields_are_recomputed(tmp_path):
    store = _store(tmp_path, {"RNA": WIDE_COUNTS}, "uint32")
    cells = _every_cell(store)
    features = store.select_all_features(from_assay="RNA")
    flags = {"renormalize_subset": False, "log_transform": True}
    current = store.run_normalization(cells, features, **flags)
    assert store.run_normalization(cells, features, **flags) == current

    # Records that earlier release candidates salted are never reused.
    group = artifact_group(store.zw, current)
    provenance = dict(group.attrs["provenance"])
    provenance["parameters"] = {
        **provenance["parameters"],
        "count_arithmetic": "float64",
        "zero_total_divisor": "one",
    }
    group.attrs["provenance"] = provenance

    recomputed = store.run_normalization(cells, features, **flags)
    assert recomputed != current
    assert set(store.inspect_artifact(recomputed).parameters) == {
        "normalization_method",
        "size_factor",
        "log_transform",
        "renormalize_subset",
    }
    np.testing.assert_array_equal(
        artifact_group(store.zw, recomputed)["data"][:],
        _lib_size_reference(WIDE_COUNTS, log_transform=True).astype(np.float32),
    )


def test_mapping_reference_lineages_with_removed_fields_rebuild_as_advised(tmp_path):
    counts = np.random.default_rng(43).integers(0, 200, size=(30, 24))
    store = _store(tmp_path, {"RNA": counts}, "uint32")
    cells = store.snapshot_cell_selection()
    features = store.select_all_features(from_assay="RNA")
    flags = {"renormalize_subset": False, "log_transform": True}

    def neighbors_of(normalized: Any) -> Any:
        pca = store.run_pca(normalized, dims=3, local_cache=False)
        return store.query_neighbors(store.build_ann_index(pca), coordinates=pca, k=4)

    normalized = store.run_normalization(cells, features, **flags)
    neighbors = neighbors_of(normalized)
    # The normalization of the lineage records a field of an earlier release
    # candidate.
    group = artifact_group(store.zw, normalized)
    provenance = dict(group.attrs["provenance"])
    provenance["parameters"] = {
        **provenance["parameters"],
        "count_arithmetic": "float64",
    }
    group.attrs["provenance"] = provenance
    message = f"parameters: count_arithmetic. {REFERENCE_REMEDY}$"
    with pytest.raises(ValueError, match=message):
        store.build_mapping_reference(neighbors)

    # A new normalization alone leaves the neighbors on the earlier lineage.
    recomputed = store.run_normalization(cells, features, **flags)
    assert recomputed != normalized
    with pytest.raises(ValueError, match=message):
        store.build_mapping_reference(neighbors)
    # PCA and its descendants recomputed from it build the reference.
    rebuilt = neighbors_of(recomputed)
    reference = store.get_mapping_reference(store.build_mapping_reference(rebuilt))
    assert reference.neighbors == rebuilt
    assert set(reference.metadata["normalization_parameters"]) == set(_REFERENCE_RECORD)
