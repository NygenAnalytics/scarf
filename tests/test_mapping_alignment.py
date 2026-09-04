from typing import Any

import numpy as np
import pytest
import zarr

from scarf.assay import RNAassay, norm_lib_size
from scarf.mapping.features import AlignedFeatureStream
from scarf.matrix import ChunkedArray
from scarf.storage.artifacts import callable_identity
from scarf.storage.budget import ResourceBudget
from tests.store_probes import RecordingStore


class _MemoryMetadata:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self._values = values

    def fetch_all(self, name: str) -> np.ndarray:
        return self._values[name]

    def _get_array(self, name: str) -> np.ndarray:
        return self._values[name]


def _normalization(
    *,
    size_factor: float = 10.0,
    log_transform: bool = False,
    renormalize_subset: bool = False,
) -> dict[str, Any]:
    return {
        "normalization_method": callable_identity(norm_lib_size),
        "size_factor": size_factor,
        "log_transform": log_transform,
        "renormalize_subset": renormalize_subset,
    }


def _query_assay(
    values: np.ndarray,
    feature_ids: list[str],
    *,
    chunks: tuple[int, int] = (3, 2),
    size_factor: int = 997,
    read_only: bool = False,
) -> tuple[RNAassay, RecordingStore, zarr.Array]:
    store = RecordingStore()
    writable_root = zarr.open_group(store=store, mode="w")
    writable_counts = writable_root.create_array(
        "counts",
        shape=values.shape,
        chunks=chunks,
        dtype=values.dtype,
    )
    writable_counts[:] = values
    root = (
        zarr.open_group(store=store.with_read_only(True), mode="r")
        if read_only
        else writable_root
    )
    counts = root["counts"]
    assert isinstance(counts, zarr.Array)

    assay = object.__new__(RNAassay)
    assay.name = "RNA"
    assay.rawData = ChunkedArray(
        counts,
        nthreads=1,
        resources=ResourceBudget(1_000_000, 1),
    )
    assay.feats = _MemoryMetadata({"ids": np.asarray(feature_ids)})
    assay.cells = _MemoryMetadata({"RNA_nCounts": values.sum(axis=1, dtype=np.float64)})
    assay.sf = size_factor
    assay.scalar = np.array([123.0])
    assay.z = root
    return assay, store, writable_counts


def _stream(
    assay: RNAassay,
    *,
    cells: np.ndarray | None = None,
    reference_ids: np.ndarray | None = None,
    means: np.ndarray | None = None,
    normalization: dict[str, Any] | None = None,
    policy: str = "reference_mean",
    resources: ResourceBudget | None = None,
    reserved_resident_bytes: int = 0,
    reserved_per_row_bytes: int = 0,
) -> AlignedFeatureStream:
    if cells is None:
        cells = np.arange(assay.rawData.shape[0], dtype=np.int64)
    if reference_ids is None:
        reference_ids = np.array(["a", "missing", "b"])
    if means is None:
        means = np.array([1.5, 7.5, 2.5])
    return AlignedFeatureStream(
        query_assay=assay,
        query_cell_indices=cells,
        reference_feature_ids=reference_ids,
        reference_normalized_means=means,
        reference_normalization_parameters=normalization or _normalization(),
        missing_feature_policy=policy,
        resources=resources or ResourceBudget(1_000_000, 2),
        reserved_resident_bytes=reserved_resident_bytes,
        reserved_per_row_bytes=reserved_per_row_bytes,
    )


def _collect(stream: AlignedFeatureStream) -> np.ndarray:
    blocks = list(stream)
    assert [block.row_offset for block in blocks] == [
        start for start, _ in stream.block_boundaries
    ]
    return np.concatenate([block.values for block in blocks], axis=0)


def test_aligned_feature_stream_replays_in_reference_order() -> None:
    counts = np.array(
        [
            [9, 2, 4, 1],
            [3, 5, 7, 2],
            [8, 1, 6, 5],
        ],
        dtype=np.uint32,
    )
    assay, _, _ = _query_assay(counts, ["extra", "b", "a", "c"])
    cells = np.array([2, 0, 1])
    reference_ids = np.array(["a", "missing", "b"])
    means = np.array([1.5, 7.5, 2.5])
    stream = _stream(
        assay,
        cells=cells,
        reference_ids=reference_ids,
        means=means,
    )
    reference_ids[0] = "changed"
    means[:] = -1

    expected_present = (
        10.0
        * counts[np.ix_(cells, np.array([2, 1]))]
        / counts.sum(axis=1)[cells, np.newaxis]
    )
    expected = np.column_stack(
        (expected_present[:, 0], np.full(3, 7.5), expected_present[:, 1])
    )
    first = _collect(stream)
    second = _collect(stream)

    np.testing.assert_allclose(first, expected)
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(stream.reference_feature_ids, ["a", "missing", "b"])
    np.testing.assert_array_equal(stream.query_index_map, [2, 1])
    np.testing.assert_array_equal(stream.reference_index_map, [0, 2])
    np.testing.assert_array_equal(stream.reference_to_query_index_map, [2, -1, 1])
    np.testing.assert_array_equal(stream.query_feature_indices, [2, 1])
    np.testing.assert_array_equal(stream.reference_feature_indices, [0, 2])
    np.testing.assert_array_equal(stream.query_cell_indices, cells)
    assert stream.shape == (3, 3)
    assert stream.dtype == np.dtype(np.float64)
    assert stream.feature_coverage == pytest.approx(2 / 3)
    assert len(stream.alignment_map_fingerprint) == 64
    assert stream.alignment_map_hash == stream.alignment_map_fingerprint
    assert not stream.reference_feature_ids.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        stream.reference_feature_ids.flags.writeable = True


@pytest.mark.parametrize(
    ("policy", "fill"),
    [
        ("reference_mean", 7.5),
        ("zero", 0.0),
    ],
)
def test_aligned_feature_stream_missing_feature_fills(
    policy: str,
    fill: float,
) -> None:
    counts = np.array([[2, 3], [5, 7]], dtype=np.uint32)
    assay, _, _ = _query_assay(counts, ["a", "b"])
    values = _collect(_stream(assay, policy=policy))

    np.testing.assert_allclose(values[:, 1], fill)


def test_aligned_feature_stream_error_policy_rejects_missing_features() -> None:
    assay, _, _ = _query_assay(
        np.array([[2, 3]], dtype=np.uint32),
        ["a", "b"],
        chunks=(1, 1),
    )

    with pytest.raises(ValueError, match="missing 1 required reference feature"):
        _stream(assay, policy="error")


def test_aligned_feature_stream_uses_reference_size_factor_and_log_semantics() -> None:
    counts = np.array([[8, 2, 10], [0, 5, 5]], dtype=np.uint32)
    assay, _, _ = _query_assay(counts, ["a", "b", "extra"], size_factor=999)
    original_scalar = assay.scalar
    stream = _stream(
        assay,
        reference_ids=np.array(["b", "a"]),
        means=np.array([0.0, 0.0]),
        normalization=_normalization(size_factor=20, log_transform=True),
        policy="zero",
    )

    expected = np.log1p(
        20.0 * counts[:, [1, 0]] / counts.sum(axis=1, dtype=np.float64)[:, np.newaxis]
    )
    np.testing.assert_allclose(_collect(stream), expected)
    assert assay.sf == 999
    assert assay.scalar is original_scalar


def test_aligned_feature_stream_renormalizes_over_matched_reference_features() -> None:
    counts = np.array(
        [
            [2, 3, 100],
            [0, 0, 7],
        ],
        dtype=np.uint32,
    )
    assay, _, _ = _query_assay(counts, ["a", "b", "extra"])
    stream = _stream(
        assay,
        reference_ids=np.array(["b", "missing", "a"]),
        means=np.array([0.0, 11.0, 0.0]),
        normalization=_normalization(size_factor=10, renormalize_subset=True),
    )

    np.testing.assert_allclose(
        _collect(stream),
        np.array(
            [
                [6.0, 11.0, 4.0],
                [0.0, 11.0, 0.0],
            ]
        ),
    )


def test_raw_expression_fingerprint_includes_unmatched_query_features() -> None:
    counts = np.array([[2, 3, 100], [5, 7, 200]], dtype=np.uint32)
    assay, _, writable_counts = _query_assay(counts, ["a", "b", "extra"])
    options = {
        "reference_ids": np.array(["a", "b"]),
        "means": np.zeros(2),
        "normalization": _normalization(renormalize_subset=True),
        "policy": "zero",
    }
    before = _stream(assay, **options)
    before_values = _collect(before)
    before_fingerprint = before.raw_expression_fingerprint

    writable_counts[0, 2] = 101
    after = _stream(assay, **options)

    np.testing.assert_array_equal(_collect(after), before_values)
    assert after.alignment_map_fingerprint == before.alignment_map_fingerprint
    assert after.raw_expression_fingerprint != before_fingerprint
    assert before.raw_expression_fingerprint == before_fingerprint


def test_raw_expression_fingerprint_tracks_live_normalization_scalars() -> None:
    counts = np.array([[2, 3], [5, 7]], dtype=np.uint32)
    assay, _, _ = _query_assay(counts, ["a", "b"])
    options = {
        "reference_ids": np.array(["a", "b"]),
        "means": np.zeros(2),
        "normalization": _normalization(renormalize_subset=False),
        "policy": "zero",
    }
    before = _stream(assay, **options)
    before_values = _collect(before)
    before_fingerprint = before.raw_expression_fingerprint

    assay.cells._values["RNA_nCounts"][0] *= 2
    after = _stream(assay, **options)

    assert after.raw_expression_fingerprint != before_fingerprint
    assert not np.array_equal(_collect(after), before_values)


def test_subset_renormalization_fingerprint_ignores_unused_live_scalars() -> None:
    counts = np.array([[2, 3], [5, 7]], dtype=np.uint32)
    assay, _, _ = _query_assay(counts, ["a", "b"])
    options = {
        "reference_ids": np.array(["a", "b"]),
        "means": np.zeros(2),
        "normalization": _normalization(renormalize_subset=True),
        "policy": "zero",
    }
    before = _stream(assay, **options).raw_expression_fingerprint

    assay.cells._values["RNA_nCounts"][0] *= 2

    assert _stream(assay, **options).raw_expression_fingerprint == before


def test_aligned_feature_stream_bounds_rows_under_tiny_budget() -> None:
    counts = np.arange(1, 29, dtype=np.uint32).reshape(7, 4)
    assay, _, _ = _query_assay(
        counts,
        ["a", "b", "extra", "other"],
        chunks=(5, 2),
    )
    roomy = _stream(assay)
    two_row_budget = ResourceBudget(
        roomy.resident_bytes + roomy.decoded_chunk_bytes + 2 * roomy.stream_row_bytes,
        8,
    )
    bounded = _stream(assay, resources=two_row_budget)
    blocks = list(bounded)

    assert bounded.row_geometry.block_rows == 2
    assert max(len(block.values) for block in blocks) == 2
    assert sum(len(block.values) for block in blocks) == len(counts)
    assert bounded.block_boundaries == ((0, 2), (2, 4), (4, 6), (6, 7))


def test_aligned_feature_stream_reserves_downstream_mapping_memory() -> None:
    counts = np.arange(1, 29, dtype=np.uint32).reshape(7, 4)
    assay, _, _ = _query_assay(
        counts,
        ["a", "b", "extra", "other"],
        chunks=(5, 2),
    )
    baseline = _stream(assay)
    budget = ResourceBudget(
        baseline.resident_bytes
        + baseline.decoded_chunk_bytes
        + 4 * baseline.stream_row_bytes,
        2,
    )
    reserved = _stream(
        assay,
        resources=budget,
        reserved_resident_bytes=baseline.stream_row_bytes,
        reserved_per_row_bytes=baseline.stream_row_bytes,
    )

    assert (
        reserved.resident_bytes == baseline.resident_bytes + baseline.stream_row_bytes
    )
    assert reserved.stream_row_bytes == 2 * baseline.stream_row_bytes
    assert reserved.row_geometry.block_rows == 1
    assert sum(len(block.values) for block in reserved) == len(counts)


def test_raw_expression_fingerprint_uses_budgeted_full_width_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = np.arange(1, 141, dtype=np.uint32).reshape(7, 20)
    assay, _, _ = _query_assay(
        counts,
        ["a", "b", *(f"extra-{index}" for index in range(18))],
        chunks=(5, 4),
    )
    options = {
        "reference_ids": np.array(["a", "b"]),
        "means": np.zeros(2),
        "policy": "zero",
    }
    roomy = _stream(assay, **options)
    all_column_bytes = counts.shape[1] * np.dtype(np.int64).itemsize
    fingerprint_row_bytes = 2 * counts.shape[1] * counts.dtype.itemsize
    budget = ResourceBudget(
        roomy.resident_bytes
        + all_column_bytes
        + roomy.decoded_chunk_bytes
        + 2 * fingerprint_row_bytes,
        4,
    )
    bounded = _stream(assay, resources=budget, **options)
    observed: list[tuple[int, int, int]] = []
    read_raw = bounded._read_raw

    def observe(start: int, end: int, columns: np.ndarray) -> np.ndarray:
        observed.append((start, end, len(columns)))
        return read_raw(start, end, columns)

    monkeypatch.setattr(bounded, "_read_raw", observe)
    _ = bounded.raw_expression_fingerprint

    assert max(end - start for start, end, _ in observed) == 2
    assert all(width == counts.shape[1] for _, _, width in observed)


def test_aligned_feature_stream_reads_read_only_counts_without_zarr_writes() -> None:
    counts = np.array([[2, 3, 4], [5, 7, 11]], dtype=np.uint32)
    assay, store, _ = _query_assay(
        counts,
        ["a", "b", "extra"],
        chunks=(1, 2),
        read_only=True,
    )
    store.reset()
    stream = _stream(assay)

    _collect(stream)
    _ = stream.raw_expression_fingerprint

    assert all(operation == "get" for operation, _ in store.ops)
    assert all("normed__" not in key for _, key in store.ops)
    assert list(assay.z.array_keys()) == ["counts"]


@pytest.mark.parametrize("policy", ["intersection", "mean", "", "ERROR"])
def test_aligned_feature_stream_rejects_unsupported_policies(policy: str) -> None:
    assay, _, _ = _query_assay(
        np.array([[1, 2]], dtype=np.uint32),
        ["a", "b"],
        chunks=(1, 1),
    )

    with pytest.raises(ValueError, match="missing_feature_policy"):
        _stream(assay, policy=policy)


def test_aligned_feature_stream_rejects_duplicate_and_disjoint_ids() -> None:
    counts = np.array([[1, 2]], dtype=np.uint32)
    assay, _, _ = _query_assay(counts, ["a", "b"], chunks=(1, 1))
    with pytest.raises(
        ValueError, match="Reference feature identifiers must be unique"
    ):
        _stream(
            assay,
            reference_ids=np.array(["a", "a"]),
            means=np.zeros(2),
        )

    duplicate_query, _, _ = _query_assay(
        counts,
        ["a", "a"],
        chunks=(1, 1),
    )
    with pytest.raises(ValueError, match="Query feature identifiers must be unique"):
        _stream(duplicate_query)

    with pytest.raises(ValueError, match="No reference features overlap"):
        _stream(
            assay,
            reference_ids=np.array(["x", "y"]),
            means=np.zeros(2),
        )


@pytest.mark.parametrize(
    "means",
    [
        np.array([1.0, 2.0]),
        np.array([1.0, np.nan, 3.0]),
        np.array([1.0, np.inf, 3.0]),
        np.array([True, False, True]),
        np.array([1 + 2j, 2 + 0j, 3 + 0j]),
    ],
)
def test_aligned_feature_stream_rejects_invalid_reference_means(
    means: np.ndarray,
) -> None:
    assay, _, _ = _query_assay(
        np.array([[1, 2]], dtype=np.uint32),
        ["a", "b"],
        chunks=(1, 1),
    )

    with pytest.raises(ValueError, match="Reference normalized means"):
        _stream(assay, means=means)


@pytest.mark.parametrize("size_factor", [0.0, -1.0, np.nan, np.inf])
def test_aligned_feature_stream_rejects_invalid_reference_size_factor(
    size_factor: float,
) -> None:
    assay, _, _ = _query_assay(
        np.array([[1, 2]], dtype=np.uint32),
        ["a", "b"],
        chunks=(1, 1),
    )

    with pytest.raises(ValueError, match="size_factor must be finite and positive"):
        _stream(assay, normalization=_normalization(size_factor=size_factor))


def test_aligned_feature_stream_rejects_unknown_reference_normalizer() -> None:
    assay, _, _ = _query_assay(
        np.array([[1, 2]], dtype=np.uint32),
        ["a", "b"],
        chunks=(1, 1),
    )
    normalization = _normalization()
    normalization["normalization_method"] = {
        "module": "custom.normalization",
        "qualname": "normalize",
    }

    with pytest.raises(ValueError, match="Unsupported reference normalization method"):
        _stream(assay, normalization=normalization)
