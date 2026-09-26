"""Count arithmetic on narrow integer stores and bit-identity on wide stores.

H5AD imports store integral counts in the smallest lossless unsigned dtype,
so most imported scRNA-seq stores hold uint8 or uint16 counts. Normalization
must promote them before scaling or taking logarithms. Stores with 32- or
64-bit integer or floating-point counts must keep their previous values
bit for bit.
"""

import itertools
import pickle
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scarf.assay.feature_summary import ensure_feature_summary
from scarf.assay.normalization import norm_clr, norm_lib_size, norm_lib_size_log
from scarf.datastore.datastore import DataStore
from scarf.mapping.features import _normalization_parameters
from scarf.matrix import ChunkedArray
from scarf.metadata.selection import CellField
from scarf.storage.artifacts import artifact_group, callable_identity
from scarf.trajectory.artifacts import (
    validate_aggregation_parameters,
    validate_marker_parameters,
)

SIZE_FACTOR = 1000


def _counts(
    n_cells: int,
    n_features: int,
    max_count: int,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Integral counts whose first feature is near ``max_count`` in every cell."""
    rng = np.random.default_rng(seed)
    values = rng.poisson(
        rng.gamma(0.6, 2.0, size=n_features), size=(n_cells, n_features)
    )
    values = values.astype(np.float64)
    values[:, 1] = np.maximum(values[:, 1], 1)
    values[:, 0] = rng.integers(max_count // 2, max_count + 1, size=n_cells)
    return values


def _h5ad_store(
    tmp_path: Any,
    values: np.ndarray,
    *,
    source_dtype: Any = np.float32,
    feature_type: str | list[str] | None = None,
) -> DataStore:
    """Write ``values`` as an H5AD file and import it through the H5AD reader.

    ``feature_type`` splits the features into assays, one type per feature or
    one type for all of them.
    """
    import anndata as ad

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    n_cells, n_features = values.shape
    var = pd.DataFrame(
        {"gene_short_name": [f"G{index}" for index in range(n_features)]},
        index=[f"g{index}" for index in range(n_features)],
    )
    if feature_type is not None:
        var["feature_types"] = feature_type
    ad.AnnData(
        X=csr_matrix(values.astype(source_dtype)),
        obs=pd.DataFrame(index=[f"cell{index}" for index in range(n_cells)]),
        var=var,
    ).write_h5ad(tmp_path / "counts.h5ad")
    reader = H5adReader(str(tmp_path / "counts.h5ad"))
    try:
        H5adToZarr(
            reader,
            zarr_loc=str(tmp_path / "counts.zarr"),
            assay_split_key=None if feature_type is None else "feature_types",
            nthreads=1,
        ).dump()
    finally:
        reader.close()
    has_rna = feature_type is None or "Gene Expression" in np.atleast_1d(feature_type)
    return DataStore(
        str(tmp_path / "counts.zarr"),
        default_assay="RNA" if has_rna else "ADT",
        min_features_per_cell=0,
        nthreads=1,
    )


def _multimodal_counts(max_count: int, *, n_cells: int = 240) -> np.ndarray:
    """RNA counts along one continuous trajectory, then six ADT features."""
    rng = np.random.default_rng(17)
    n_genes = 120
    position = np.linspace(0.0, 1.0, n_cells)[:, None]
    peaks = rng.random(n_genes)[None, :]
    means = rng.gamma(0.6, 2.0, size=n_genes) * (
        1 + 8 * np.exp(-np.square(position - peaks) / 0.02)
    )
    rna = rng.poisson(means)
    rna[:, 0] = rng.integers(max_count // 2, max_count + 1, size=n_cells)
    adt = rng.poisson(rng.gamma(2.0, 20.0, size=6), size=(n_cells, 6)) + 1
    return np.hstack([rna, adt]).astype(np.float64)


MULTIMODAL_TYPES = ["Gene Expression"] * 120 + ["Antibody Capture"] * 6


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


def _feature_mask(store: DataStore, assay: str, feat_idx: np.ndarray) -> Any:
    mask = np.zeros(getattr(store, assay).feats.N, dtype=bool)
    mask[feat_idx] = True
    return store.set_feature_selection(from_assay=assay, mask=mask)


# Old expressions, kept verbatim to prove that wide and floating-point counts
# keep their previous values bit for bit.
def _old_lib_size(assay: Any, counts: ChunkedArray) -> ChunkedArray:
    return assay.sf * counts / assay.scalar.reshape(-1, 1)


def _old_lib_size_log(assay: Any, counts: ChunkedArray) -> ChunkedArray:
    return np.log1p(assay.sf * counts / assay.scalar.reshape(-1, 1))


NARROW = [
    pytest.param(200, np.uint8, id="uint8"),
    pytest.param(3000, np.uint16, id="uint16"),
]


@pytest.mark.parametrize(("max_count", "stored"), NARROW)
def test_h5ad_import_stores_integral_counts_in_narrow_dtype(
    tmp_path, max_count, stored
):
    values = _counts(30, 12, max_count)
    store = _h5ad_store(tmp_path, values)

    assert store.RNA.rawData.dtype == np.dtype(stored)
    np.testing.assert_array_equal(store.RNA.rawData.compute(), values)


@pytest.mark.parametrize("log_transform", [False, True])
@pytest.mark.parametrize(("max_count", "stored"), NARROW)
def test_rna_normed_matches_float64_reference(
    tmp_path, max_count, stored, log_transform
):
    values = _counts(80, 30, max_count)
    store = _h5ad_store(tmp_path, values)
    cell_idx = np.arange(80, dtype=np.int64)

    normed = store.RNA.normed(cell_idx=cell_idx, log_transform=log_transform)
    actual = normed.compute()

    assert actual.dtype == np.float64
    np.testing.assert_allclose(
        actual,
        _lib_size_reference(values, log_transform=log_transform),
        rtol=1e-12,
        atol=0,
    )

    subset = np.array([0, 2, 5, 7], dtype=np.int64)
    renormalized = store.RNA.normed(
        cell_idx=cell_idx,
        feat_idx=subset,
        renormalize_subset=True,
        log_transform=log_transform,
    ).compute()
    np.testing.assert_allclose(
        renormalized,
        _lib_size_reference(
            values,
            feat_idx=subset,
            renormalize_subset=True,
            log_transform=log_transform,
        ),
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize(("max_count", "stored"), NARROW)
def test_get_cell_vals_matches_float64_reference(tmp_path, max_count, stored):
    values = _counts(60, 20, max_count)
    store = _h5ad_store(tmp_path, values)

    actual = store.get_cell_vals(from_assay="RNA", cell_key="I", k="G0")

    np.testing.assert_allclose(
        actual, _lib_size_reference(values)[:, 0], rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    ("renormalize_subset", "log_transform"),
    [(False, False), (False, True), (True, True)],
)
@pytest.mark.parametrize(("max_count", "stored"), NARROW)
def test_run_normalization_payload_matches_float64_reference(
    tmp_path, max_count, stored, renormalize_subset, log_transform
):
    values = _counts(70, 24, max_count)
    store = _h5ad_store(tmp_path, values)
    feat_idx = np.array([0, 1, 3, 4, 9, 15], dtype=np.int64)

    ref = store.run_normalization(
        store.snapshot_cell_selection(),
        _feature_mask(store, "RNA", feat_idx),
        renormalize_subset=renormalize_subset,
        log_transform=log_transform,
    )
    group = artifact_group(store.zw, ref)
    data = np.asarray(group["data"][:])

    expected = _lib_size_reference(
        values,
        feat_idx=feat_idx,
        renormalize_subset=renormalize_subset,
        log_transform=log_transform,
    )
    assert data.dtype == np.float32
    np.testing.assert_allclose(data, expected.astype(np.float32), rtol=1e-6, atol=0)
    widened = data.astype(np.float64)
    np.testing.assert_allclose(group["feature_sum"][:], widened.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(
        group["feature_squared_sum"][:], np.square(widened).sum(axis=0), rtol=1e-12
    )


@pytest.mark.parametrize("log_transform", [False, True])
@pytest.mark.parametrize(("max_count", "stored"), NARROW)
def test_rna_feature_summary_matches_float64_reference(
    tmp_path, max_count, stored, log_transform
):
    values = _counts(50, 16, max_count)
    store = _h5ad_store(tmp_path, values)

    ref = ensure_feature_summary(
        store.zw,
        store.RNA,
        store.snapshot_cell_selection(),
        log_transform=log_transform,
    )
    group = artifact_group(store.zw, ref)

    normalized = _lib_size_reference(values, log_transform=log_transform)
    mean = normalized.mean(axis=0)
    np.testing.assert_allclose(
        group["normed_tot"][:], normalized.sum(axis=0), rtol=1e-12
    )
    np.testing.assert_array_equal(group["normed_n"][:], (normalized > 0).sum(axis=0))
    np.testing.assert_allclose(
        group["sigmas"][:],
        np.square(normalized).mean(axis=0) - np.square(mean),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("renormalize_subset", [False, True])
@pytest.mark.parametrize(("max_count", "stored"), NARROW)
def test_rna_feature_batches_match_float64_reference(
    tmp_path, max_count, stored, renormalize_subset
):
    values = _counts(40, 10, max_count)
    store = _h5ad_store(tmp_path, values)
    cell_idx = np.arange(40, dtype=np.int64)
    feat_idx = np.array([0, 3, 6], dtype=np.int64)

    batches = list(
        store.RNA.iter_normed_feature_wise(
            cell_idx,
            feat_idx,
            batch_size=None,
            msg=None,
            as_dataframe=True,
            renormalize_subset=renormalize_subset,
        )
    )
    actual = pd.concat(batches, axis=1)[feat_idx].to_numpy(dtype=np.float64)

    expected = _lib_size_reference(
        values, feat_idx=feat_idx, renormalize_subset=renormalize_subset
    )
    # The default path normalizes countsT batches in float32.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_clr_on_uint8_counts_does_not_accumulate_in_float16(tmp_path):
    # log1p of uint8 counts is float16, and its per-feature sum over these
    # cells passes the float16 maximum of 65504. The old expression summed
    # in float16, where the total stalls or overflows, so the geometric mean
    # and every CLR value were wrong.
    n_cells = 15_000
    rng = np.random.default_rng(3)
    values = rng.integers(150, 256, size=(n_cells, 3)).astype(np.float64)
    store = _h5ad_store(tmp_path, values, feature_type="Antibody Capture")
    assert store.ADT.rawData.dtype == np.uint8
    assert np.all(np.log1p(values).sum(axis=0) > np.finfo(np.float16).max)

    cell_idx = np.arange(n_cells, dtype=np.int64)
    actual = store.ADT.normed(cell_idx=cell_idx).compute()

    expected = _clr_reference(values)
    assert np.all(expected > 0)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(("max_count", "stored"), NARROW)
def test_clr_normalization_payload_matches_float64_reference(
    tmp_path, max_count, stored
):
    values = _counts(90, 6, max_count, seed=11)
    store = _h5ad_store(tmp_path, values, feature_type="Antibody Capture")
    assert store.ADT.rawData.dtype == np.dtype(stored)

    ref = store.run_normalization(
        store.snapshot_cell_selection(),
        store.select_all_features(from_assay="ADT"),
    )
    data = np.asarray(artifact_group(store.zw, ref)["data"][:])

    # CLR is now computed in float64 before the float32 payload cast, so it
    # matches the float64 reference exactly. The old uint16 path summed
    # log1p in float32 and changed the last bit of most stored values.
    expected = _clr_reference(values).astype(np.float32)
    np.testing.assert_array_equal(data, expected)


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


WIDE_AND_FLOAT = ["uint32", "int32", "int64", "float32", "float64"]


@pytest.mark.parametrize("block_size", [1, 3, 64])
@pytest.mark.parametrize("dtype", WIDE_AND_FLOAT)
def test_normalizers_are_bit_identical_for_wide_and_float_counts(dtype, block_size):
    rng = np.random.default_rng(5)
    raw = rng.poisson(rng.gamma(0.8, 20.0, size=17), size=(41, 17))
    raw[:, 0] += rng.integers(1_000, 100_000, size=41)
    if np.dtype(dtype).kind == "f":
        values = (raw + rng.random(raw.shape).round(2)).astype(dtype)
    else:
        values = raw.astype(dtype)
    counts = ChunkedArray.from_numpy(values, block_size=block_size)
    scalar = np.asarray(values.sum(axis=1), dtype=np.float64)
    assay = SimpleNamespace(sf=SIZE_FACTOR, scalar=scalar)

    for new, old in (
        (norm_lib_size, _old_lib_size),
        (norm_lib_size_log, _old_lib_size_log),
        (norm_clr, _old_clr),
    ):
        actual = new(assay, counts).compute()
        previous = old(assay, counts).compute()
        assert actual.dtype == previous.dtype
        np.testing.assert_array_equal(actual, previous)


@pytest.mark.parametrize(
    ("max_count", "source_dtype", "stored", "fraction"),
    [
        pytest.param(100_000, np.float64, np.uint32, 0.0, id="uint32"),
        pytest.param(3000, np.float32, np.float32, 0.25, id="float32"),
    ],
)
@pytest.mark.parametrize("log_transform", [False, True])
def test_wide_and_float_stores_keep_bit_identical_normalization(
    tmp_path, max_count, source_dtype, stored, fraction, log_transform
):
    values = _counts(60, 20, max_count) + fraction
    store = _h5ad_store(tmp_path, values, source_dtype=source_dtype)
    rna = store.RNA
    assert rna.rawData.dtype == np.dtype(stored)
    cell_idx = np.arange(60, dtype=np.int64)
    feat_idx = np.array([0, 2, 3, 8, 13], dtype=np.int64)

    counts = rna.rawData[:, feat_idx][cell_idx, :]
    scalar = np.asarray(rna.cells.fetch_all("RNA_nCounts")[cell_idx], dtype=np.float64)
    scalar[scalar == 0] = 1
    old = (_old_lib_size_log if log_transform else _old_lib_size)(
        SimpleNamespace(sf=rna.sf, scalar=scalar), counts
    ).compute()

    actual = rna.normed(cell_idx, feat_idx, log_transform=log_transform).compute()
    assert actual.dtype == old.dtype
    np.testing.assert_array_equal(actual, old)

    ref = store.run_normalization(
        store.snapshot_cell_selection(),
        _feature_mask(store, "RNA", feat_idx),
        renormalize_subset=False,
        log_transform=log_transform,
    )
    payload = np.asarray(artifact_group(store.zw, ref)["data"][:])
    np.testing.assert_array_equal(payload, old.astype(np.float32))


@contextmanager
def _counter_artifact_ids() -> Iterator[None]:
    """Draw artifact IDs from a counter so provenance hashes are repeatable.

    The provenance of an artifact includes the IDs of its inputs, which are
    otherwise random.
    """
    from scarf.storage import artifact_writer

    counter = itertools.count(1)
    original = artifact_writer.new_artifact_id
    artifact_writer.new_artifact_id = lambda: f"{next(counter):064x}"
    try:
        yield
    finally:
        artifact_writer.new_artifact_id = original


def _normalized_value_artifacts(
    ds: DataStore,
    *,
    adt: str,
    min_cells: int,
) -> dict[str, Any]:
    """Create every artifact kind whose values ``normed`` can compute."""
    from tests.fixtures_datastore import build_neighbourhood_graph

    cells = ds.snapshot_cell_selection()
    detected = np.asarray(ds.RNA.feats.fetch_all("nCells")) >= min_cells
    rna = ds.set_feature_selection(from_assay="RNA", mask=detected)
    adt_features = ds.select_all_features(from_assay=adt)
    refs: dict[str, Any] = {"cells": cells, "rna": rna, "adt": adt_features}
    for renormalize_subset, log_transform in (
        (True, True),
        (False, False),
        (False, True),
    ):
        refs[f"normalized_rna_{renormalize_subset}_{log_transform}"] = (
            ds.run_normalization(
                cells,
                rna,
                renormalize_subset=renormalize_subset,
                log_transform=log_transform,
            )
        )
    refs["normalized_adt"] = ds.run_normalization(cells, adt_features)
    graph = build_neighbourhood_graph(
        ds, cell_selection=cells, features=rna, dims=5, local_cache=False
    )
    refs["graph"] = graph
    clusters = ds.run_leiden_clustering(graph)
    refs["clusters"] = clusters
    refs["markers_rna"] = ds.run_marker_search(clusters, features=rna)
    refs["markers_rna_renormalized"] = ds.run_marker_search(
        clusters, features=rna, renormalize_subset=True
    )
    refs["markers_adt"] = ds.run_marker_search(
        clusters, from_assay=adt, features=adt_features
    )
    labels = np.asarray(artifact_group(ds.zw, clusters)["values"][:])
    present = sorted(set(labels.tolist()))
    pseudotime = ds.run_pseudotime_scoring(
        graph, source_sink=clusters, sources=[present[0]], sinks=[present[1]]
    )
    refs["pseudotime"] = pseudotime
    refs["pseudotime_markers_rna"] = ds.run_pseudotime_marker_search(
        pseudotime, features=rna
    )
    refs["pseudotime_markers_rna_renormalized"] = ds.run_pseudotime_marker_search(
        pseudotime, features=rna, renormalize_subset=True
    )
    refs["pseudotime_markers_adt"] = ds.run_pseudotime_marker_search(
        pseudotime, features=adt_features
    )
    for name, extra in (
        ("pseudotime_aggregation_rna", {}),
        ("pseudotime_aggregation_rna_renormalized", {"renormalize_subset": True}),
    ):
        refs[name] = ds.run_pseudotime_aggregation(
            pseudotime,
            features=rna,
            n_clusters=3,
            window_size=50,
            chunk_size=10,
            **extra,
        )
    ds.cells.insert(
        "narrow_count_group",
        np.where(np.arange(ds.cells.N) % 2 == 0, "even", "odd"),
        overwrite=True,
    )
    rna_names = np.asarray(ds.RNA.feats.fetch_all("names"))[detected][:2]
    refs["statistical_tests_rna"] = ds.run_statistical_testing(
        [str(name) for name in rna_names], CellField("narrow_count_group")
    ).artifact
    adt_name = str(getattr(ds, adt).feats.fetch_all("names")[0])
    refs["statistical_tests_adt"] = ds.run_statistical_testing(
        [adt_name], CellField("narrow_count_group"), from_assay=adt
    ).artifact
    return refs


def _identity_scenario(zarr_path: str) -> dict[str, str]:
    """Return the provenance hash of every normalized-value artifact kind."""
    from scarf.storage.artifacts import provenance_hash

    with _counter_artifact_ids():
        ds = DataStore(
            zarr_path,
            default_assay="RNA",
            assay_types={"assay2": "ADT"},
            nthreads=1,
        )
        refs = _normalized_value_artifacts(ds, adt="assay2", min_cells=100)
        return {
            name: provenance_hash(ds.inspect_artifact(ref).provenance)
            for name, ref in refs.items()
        }


# Provenance hashes computed by the code before narrow counts recorded a
# count-arithmetic marker. Wide-count identities must not change.
WIDE_COUNT_IDENTITIES = {
    "adt": ("6782383c63a5a3acfce86825ae70954d07d6ae74a8ce3aa14adc6ffbd2c21093"),
    "cells": ("b822a10791dd96fc4aeb1817b5abe1a153e4a6f30e4200853e456070e4de5c18"),
    "clusters": ("e1c902641853e77cdcecfbcdd545ac917d54bca44dd99910af03b3a0e54c5819"),
    "graph": ("0b87c07b3a5867933f4649eedd31f583ece0686bc986ad12b7488d309a07b3e8"),
    "markers_adt": ("ea05af821300b4ef0c6717e9212abbb7d3fbf37f7549ebf912d601dfe0c92c0c"),
    "markers_rna": ("1e6fe4227b92ace1315839fb16e4273a9c3a83dedec74c34c90a1f12846e1665"),
    "markers_rna_renormalized": (
        "aa8bf88070a68c3008410d83199998cc97bc3a725642f1f59d4fee911c51a26b"
    ),
    "normalized_adt": (
        "89045298250d7d0a784c0823a2039a4dceacc5bf44684c89f8a30befeb572d78"
    ),
    "normalized_rna_False_False": (
        "eb31610495b248361163c87e5868983c96f62972f700bb10e8dc9685110e15b7"
    ),
    "normalized_rna_False_True": (
        "1b2ffd0c669eeb5951fc6c43643863cb019a88414a3a1cd5cf561d32eac900ab"
    ),
    "normalized_rna_True_True": (
        "4bc82a5f5b64c006d4ebf97defd85012b08c02ef5abf522f4e623f0ca3830d16"
    ),
    "pseudotime": ("e979829d2dc3c2b53fd0d4d40e7471b591e236cd9af6e67d6829faf22ec4b149"),
    "pseudotime_aggregation_rna": (
        "c9331e20c8341a8277466be07952c5fc567553158de6f5ad71a2971b1e3058c5"
    ),
    "pseudotime_aggregation_rna_renormalized": (
        "15269920b40d876a6d4f04ae6bb800dd12f2baf56a8aa04d971c0f9be25ffe24"
    ),
    "pseudotime_markers_adt": (
        "9d17407e8a69728cc05720b3162b5fbfadd57ffd4a73e16d8ebf6970c3e7d8ac"
    ),
    "pseudotime_markers_rna": (
        "311438b65d2d04a97798d9687b55cb6e2f7645c2ef50ce0c552c637a09d3acf7"
    ),
    "pseudotime_markers_rna_renormalized": (
        "8e1a5e07dab2913a72f9b765d2675021259df57fa16e107db3542fc31b8eddac"
    ),
    "rna": ("35acd5a1f73cfec97cad1b69b4f4dc956a8d86388a5b0aa47d4d9c9316f33d2a"),
    "statistical_tests_adt": (
        "e84b87ba2129f7aa0bb8783de9e65f13d178908d6b2a46dc42de5902a449df55"
    ),
    "statistical_tests_rna": (
        "59b73fc6325e6a5f8bd24ecfe4ba444b7ec46ce73e57ae325e90308ef54e8a0d"
    ),
}


@pytest.mark.filterwarnings("ignore:Cell-level statistical testing:UserWarning")
def test_wide_count_identities_match_the_code_before_the_marker(
    datastore_zarr_root, tmp_path
):
    # The fixture stores uint32 counts, so no artifact records the marker.
    shutil.copytree(datastore_zarr_root, tmp_path / "store")

    assert _identity_scenario(str(tmp_path / "store")) == WIDE_COUNT_IDENTITIES


# Artifacts whose values ``normed`` computes on narrow counts. Every other
# artifact, including the RNA defaults that stream countsT or use the subset
# renormalization kernel, keeps its previous identity.
NARROW_COUNT_ARITHMETIC = {
    "normalized_rna_False_False",
    "normalized_rna_False_True",
    "normalized_adt",
    "markers_rna_renormalized",
    "markers_adt",
    "pseudotime_markers_rna_renormalized",
    "pseudotime_markers_adt",
    "pseudotime_aggregation_rna_renormalized",
    "statistical_tests_rna",
    "statistical_tests_adt",
}


@pytest.mark.filterwarnings("ignore:Cell-level statistical testing:UserWarning")
def test_narrow_counts_record_count_arithmetic_only_where_values_changed(tmp_path):
    from scarf.mapping.artifact import load_artifact_mapping_reference
    from scarf.metadata.selection import NormalizationSpec
    from tests.fixtures_datastore import _input_ref, build_neighbourhood_graph

    # The marker depends only on counts narrower than 32 bits, so one narrow
    # dtype covers it.
    store = _h5ad_store(
        tmp_path, _multimodal_counts(3000), feature_type=MULTIMODAL_TYPES
    )
    assert store.RNA.rawData.dtype == np.uint16
    assert store.ADT.rawData.dtype == np.uint16

    refs = _normalized_value_artifacts(store, adt="ADT", min_cells=20)
    recorded = {
        name
        for name, ref in refs.items()
        if "count_arithmetic" in (store.inspect_artifact(ref).parameters or {})
    }
    assert recorded == NARROW_COUNT_ARITHMETIC
    for name in recorded:
        parameters = store.inspect_artifact(refs[name]).parameters or {}
        assert parameters["count_arithmetic"] == "float64"

    raw_tests = store.run_statistical_testing(
        ["G1"],
        CellField("narrow_count_group"),
        normalization=NormalizationSpec(source="raw"),
    ).artifact
    assert "count_arithmetic" not in (
        store.inspect_artifact(raw_tests).parameters or {}
    )

    # Loaders accept the marker.
    for name in ("pseudotime_markers_rna_renormalized", "pseudotime_markers_adt"):
        store.load_pseudotime_markers(refs[name])
    store.load_pseudotime_aggregation(refs["pseudotime_aggregation_rna_renormalized"])
    assert not store.get_markers(refs["markers_adt"], min_score=-1.0).empty
    store.get_statistical_tests(refs["statistical_tests_rna"])

    # A mapping reference accepts a normalization that records the marker.
    graph = build_neighbourhood_graph(
        store,
        cell_selection=refs["cells"],
        features=refs["rna"],
        dims=5,
        renormalize_subset=False,
        local_cache=False,
    )
    neighbors = _input_ref(store, graph, "neighbors")
    reference = load_artifact_mapping_reference(
        store, store.build_mapping_reference(neighbors)
    )
    assert reference.normalization_parameters["count_arithmetic"] == "float64"


def _old_clr(_: Any, counts: ChunkedArray) -> ChunkedArray:
    return _old_clr_expression(counts)


def _old_clr_expression(counts: ChunkedArray) -> ChunkedArray:
    f = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    return np.log1p(counts / f.reshape(1, -1))


_old_clr.artifact_identity = norm_clr.artifact_identity  # type: ignore[attr-defined]


@contextmanager
def _code_before_the_fix(store: DataStore) -> Iterator[None]:
    """Compute and identify normalized values as the code before the fix did."""
    import scarf.assay.normalization as normalization
    from scarf.assay import Assay, RNAassay

    patch = pytest.MonkeyPatch()
    patch.setattr(normalization, "_library_size_scaled", _old_lib_size)
    for cls in (Assay, RNAassay):
        patch.setattr(cls, "_count_arithmetic", lambda *_, **__: None)
    adt = getattr(store, "ADT", None)
    if adt is not None:
        patch.setattr(adt, "normMethod", _old_clr)
    try:
        yield
    finally:
        patch.undo()


def test_stale_narrow_normalizations_are_not_reused(tmp_path):
    values = _multimodal_counts(3000)
    store = _h5ad_store(tmp_path, values, feature_type=MULTIMODAL_TYPES)
    assert store.RNA.rawData.dtype == np.uint16
    cells = store.snapshot_cell_selection()
    rna = store.select_all_features(from_assay="RNA")
    adt = store.select_all_features(from_assay="ADT")

    def normalize() -> tuple[Any, Any]:
        return (
            store.run_normalization(
                cells, rna, renormalize_subset=False, log_transform=False
            ),
            store.run_normalization(cells, adt),
        )

    with _code_before_the_fix(store):
        stale_rna, stale_adt = normalize()
    stale_values = np.asarray(artifact_group(store.zw, stale_rna)["data"][:])
    rna_reference = _lib_size_reference(values[:, :120])
    assert np.abs(stale_values - rna_reference).max() > 100

    fresh_rna, fresh_adt = normalize()

    assert fresh_rna != stale_rna
    assert fresh_adt != stale_adt
    np.testing.assert_allclose(
        artifact_group(store.zw, fresh_rna)["data"][:],
        rna_reference.astype(np.float32),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        artifact_group(store.zw, fresh_adt)["data"][:],
        _clr_reference(values[:, 120:]).astype(np.float32),
    )
    # The stale artifacts stay untouched, and a rerun reuses the fresh ones.
    np.testing.assert_array_equal(
        artifact_group(store.zw, stale_rna)["data"][:], stale_values
    )
    assert normalize() == (fresh_rna, fresh_adt)


def test_uint16_pipeline_reuses_every_artifact_written_before_the_fix(tmp_path):
    store = _h5ad_store(tmp_path, _multimodal_counts(3000)[:, :120])
    assert store.RNA.rawData.dtype == np.uint16
    options = {
        "cell_cycle": False,
        "hvg_count": 50,
        "pca_dims": 5,
    }

    with _code_before_the_fix(store):
        before = store.pipeline.run(label="before_fix", **options)
    after = store.pipeline.run(label="after_fix", **options)

    assert list(after) == list(before)
    assert {key: after[key] for key in after} == {key: before[key] for key in before}
    # Doublet scores always record their checked integer sums.
    for key in after:
        parameters = store.inspect_artifact(after[key]).parameters or {}
        assert parameters.get("count_arithmetic") != "float64", key


def test_count_arithmetic_is_recorded_only_when_set():
    from scarf.graph.arguments import NormalizationArguments
    from scarf.storage.artifacts import ArtifactRef

    def arguments(**extra: Any) -> NormalizationArguments:
        return NormalizationArguments(
            cell_selection=ArtifactRef("datastore", "cell_selection", "2" * 64),
            feature_selection=ArtifactRef(
                "assay", "feature_selection", "3" * 64, assay="RNA"
            ),
            dataset_fingerprint="dataset-v1",
            normalization_method="norm_lib_size",
            size_factor=1000.0,
            log_transform=True,
            renormalize_subset=False,
            **extra,
        )

    unset = arguments()
    marked = arguments(count_arithmetic="float64")

    assert "count_arithmetic" not in unset.to_record().parameters
    assert marked.to_record().parameters["count_arithmetic"] == "float64"
    assert unset.provenance_hash() == arguments(count_arithmetic=None).provenance_hash()
    assert unset.provenance_hash() != marked.provenance_hash()


_MARKER_RECORD = {
    "normalization": {"log_transform": False, "renormalize_subset": True},
    "normalization_method": {"module": "m", "qualname": "normalize"},
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


@pytest.mark.parametrize(
    ("validate", "record"),
    [
        pytest.param(validate_marker_parameters, _MARKER_RECORD, id="markers"),
        pytest.param(
            validate_aggregation_parameters, _AGGREGATION_RECORD, id="aggregation"
        ),
        pytest.param(_normalization_parameters, _REFERENCE_RECORD, id="reference"),
    ],
)
def test_loaders_validate_a_recorded_count_arithmetic(validate, record):
    assert "count_arithmetic" not in validate(record)
    marked = validate({**record, "count_arithmetic": "float64"})
    assert marked["count_arithmetic"] == "float64"
    for invalid in ("float32", None):
        with pytest.raises(ValueError, match="count_arithmetic"):
            validate({**record, "count_arithmetic": invalid})


def test_trajectory_recheck_rejects_a_changed_count_arithmetic():
    from scarf.datastore._operations import trajectory as trajectory_operations

    assay = SimpleNamespace(
        normMethod=norm_lib_size,
        sf=1000,
        _count_arithmetic=lambda *_, **__: "float64",
    )
    options = {
        "normalization_method": callable_identity(norm_lib_size),
        "size_factor": 1000.0,
        "normalization": {"log_transform": False, "renormalize_subset": True},
        "context": "Pseudotime aggregation",
    }
    trajectory_operations._validate_normalization_identity(
        assay, count_arithmetic="float64", **options
    )
    with pytest.raises(ValueError, match="normalization settings changed"):
        trajectory_operations._validate_normalization_identity(
            assay, count_arithmetic=None, **options
        )


def test_cell_cycle_records_count_arithmetic_only_on_the_normed_fallback(tmp_path):
    store = _h5ad_store(tmp_path, _counts(200, 60, 3000))
    assert store.RNA.rawData.dtype == np.dtype("uint16")
    cells = store.snapshot_cell_selection()
    genes = {
        "s_genes": [f"G{index}" for index in range(0, 10)],
        "g2m_genes": [f"G{index}" for index in range(10, 20)],
        "n_bins": 5,
        "ctrl_size": 5,
    }

    def recorded() -> object:
        ref = store.run_cell_cycle_scoring(cells, **genes)
        return (store.inspect_artifact(ref).parameters or {}).get("count_arithmetic")

    # Library-size scores average in float64 without calling normed.
    assert store.RNA.normMethod is norm_lib_size
    assert recorded() is None
    store.RNA.normMethod = norm_lib_size_log
    try:
        assert recorded() == "float64"
    finally:
        store.RNA.normMethod = norm_lib_size
