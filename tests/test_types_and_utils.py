import gc
import weakref

import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from zarr.errors import ContainsArrayError, GroupNotFoundError
from zarr.storage import MemoryStore

from scarf.storage.types import (
    array_metadata_shards,
    as_zarr_array,
    as_zarr_group,
    read_fresh_group,
)
from scarf.utils import (
    array_digest,
    clean_array,
    configure_output,
    permute_into_chunks,
    rescale_array,
    rolling_window,
    set_verbosity,
    compute_with_progress,
    tqdmbar,
)
from scarf.utils.arguments import (
    clip_fraction_argument,
    float_argument,
    integer_argument,
)
from scarf.utils.arrays import (
    _rolling_window_kernel,
    assay_feature_ranges,
    canonicalize_sparse,
    checked_sparse_cast,
    cumulative_nnz,
    max_window_nnz,
    read_only_copy,
    sparse_matrix_bytes,
)
from scarf.utils.progress import iter_progress


def test_as_zarr_array_accepts_array():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(3,), dtype="f8")
    assert as_zarr_array(arr) is arr


def test_as_zarr_array_rejects_group():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("nested")
    with pytest.raises(TypeError, match="Expected Zarr array"):
        as_zarr_array(group, name="nested")


def test_as_zarr_group_accepts_group():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("nested")
    assert as_zarr_group(group) is group


def test_as_zarr_group_rejects_array():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(2,), dtype="f8")
    with pytest.raises(TypeError, match="Expected Zarr group"):
        as_zarr_group(arr, name="data")


def test_array_metadata_shards_returns_none_without_sharding():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arr = root.create_array("data", shape=(4,), dtype="f8")
    assert array_metadata_shards(arr) is None


def test_read_fresh_group_reads_the_stored_record(tmp_path):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    handle = root.create_group("child", attributes={"value": 1})
    zarr.open_group(store=root.store, path="child", mode="r+").attrs["value"] = 2
    root.create_array("array", shape=(1,), dtype="i4")

    # An open handle keeps the attributes it read when it was opened.
    assert handle.attrs["value"] == 1
    assert read_fresh_group(root, "child").attrs["value"] == 2
    assert read_fresh_group(handle).attrs["value"] == 2
    # A missing group raises unless the caller expects one.
    with pytest.raises(GroupNotFoundError):
        read_fresh_group(root, "missing")
    assert read_fresh_group(root, "missing", missing_ok=True) is None
    with pytest.raises(ContainsArrayError):
        read_fresh_group(root, "array", missing_ok=True)


def test_read_fresh_group_opens_as_its_parent_unless_told_otherwise(tmp_path):
    path = str(tmp_path / "store.zarr")
    zarr.open_group(path, mode="w").create_group("child")
    writable = zarr.open_group(path, mode="r+")
    read_only = zarr.open_group(path, mode="r")

    assert read_fresh_group(writable, "child").read_only is False
    assert read_fresh_group(read_only, "child").read_only is True
    assert read_fresh_group(writable, "child", mode="r").read_only is True


@pytest.mark.parametrize("fill_val", [0.0, 1.0, -1.0])
def test_clean_array_replaces_nan_inf_and_zero(fill_val):
    raw = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
    cleaned = clean_array(raw, fill_val=fill_val)
    np.testing.assert_array_equal(
        cleaned, [1.0, fill_val, fill_val, fill_val, fill_val]
    )
    np.testing.assert_array_equal(raw, [1.0, np.nan, np.inf, -np.inf, 0.0])


def test_rescale_array_trims_extreme_values():
    from scipy.stats import norm

    values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    # A normal at the median (0) with the values' spread (sqrt(40.4)) puts its
    # 10th and 90th percentiles at -/+ 1.28155 * 6.35610 = -/+ 8.14572.
    bound = norm.ppf(0.9) * np.sqrt(40.4)
    assert bound == pytest.approx(8.14572, rel=1e-5)
    trimmed = rescale_array(values, frac=0.9)
    assert trimmed is values
    np.testing.assert_allclose(trimmed, [-bound, -1.0, 0.0, 1.0, bound])


def test_rescale_array_requires_an_upper_quantile_above_one_half():
    values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    # frac=1 puts the bounds at -/+ infinity, so nothing is trimmed.
    np.testing.assert_array_equal(rescale_array(values.copy(), frac=1), values)
    for frac in (0.1, 0.5, 1.5):
        with pytest.raises(ValueError, match="frac must be greater than 0.5"):
            rescale_array(values.copy(), frac=frac)
    for frac in (True, "0.9", None):
        with pytest.raises(TypeError, match="frac must be a real number"):
            rescale_array(values.copy(), frac=frac)
    with pytest.raises(ValueError, match="frac must be finite"):
        rescale_array(values.copy(), frac=np.nan)


def test_set_verbosity_rejects_invalid_level():
    with pytest.raises(ValueError, match="Please provide a value for level"):
        set_verbosity("NOT_A_REAL_LEVEL")


def test_set_verbosity_accepts_valid_level():
    from scarf.utils.logging import _config

    try:
        set_verbosity("ERROR")
        assert (_config.level, _config.filepath) == ("ERROR", None)
        with pytest.raises(
            ValueError, match="^Please provide a value for level recognized by Loguru$"
        ):
            set_verbosity(None)
        assert _config.level == "ERROR"
    finally:
        set_verbosity("INFO")
    assert (_config.level, _config.filepath) == ("INFO", None)


def test_tqdmbar_uses_explicit_progress_independently_of_severity(monkeypatch):
    import tqdm.auto as tqdm_auto

    captured: list[bool] = []

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            captured.append(bool(kwargs.get("disable")))

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(tqdm_auto, "tqdm", FakeTqdm)
    try:
        configure_output(level="ERROR", progress=True)
        list(tqdmbar(range(1), desc="test"))
        set_verbosity("WARNING")
        list(tqdmbar(range(1), desc="test"))
        configure_output(level="DEBUG", progress=False)
        list(tqdmbar(range(1), desc="test"))
        list(tqdmbar(range(1), desc="test", disable=False))
        configure_output(progress=True)
        list(tqdmbar(range(1), desc="test", disable=True))
    finally:
        configure_output(level="INFO", progress=False, timestamps=False)

    assert captured == [False, False, True, True, True]


def test_controlled_compute_bounds_deferred_threads_and_passes_arrays_through():
    from scarf.utils.compute import controlled_compute

    calls: list[int] = []

    class Deferred:
        def compute(self, nthreads):
            calls.append(nthreads)
            return [1, 2]

    np.testing.assert_array_equal(controlled_compute(Deferred(), 3), [1, 2])
    assert calls == [3]
    values = np.arange(3)
    assert controlled_compute(values, 2) is values
    np.testing.assert_array_equal(controlled_compute([4, 5], 2), [4, 5])


def test_sort_categories_orders_numbers_then_text_then_missing_values():
    import pandas as pd

    from scarf.utils.arrays import sort_categories

    values = ["b10", np.float32("nan"), "b2", pd.NA, 3, None, "10", [1, 2], pd.NaT, 2.5]
    ordered = sort_categories(values)
    # Numbers by value, text in natural order (a list label sorts as its
    # text), and every missing marker last in input order.
    expected = [9, 4, 6, 7, 2, 0, 1, 3, 5, 8]
    assert [id(value) for value in ordered] == [id(values[index]) for index in expected]


def test_compute_with_progress_uses_explicit_progress_setting():
    calls: list[tuple[int, str | None]] = []

    class Deferred:
        def compute(self, nthreads, msg):
            calls.append((nthreads, msg))
            return np.array([1])

    try:
        configure_output(progress=False)
        compute_with_progress(Deferred(), "Computing", 2)
        configure_output(progress=True)
        compute_with_progress(Deferred(), "Computing", 3)
    finally:
        configure_output(progress=False)

    assert calls == [(2, None), (3, "Computing")]
    np.testing.assert_array_equal(
        compute_with_progress(np.array([2, 3])),
        np.array([2, 3]),
    )


def test_progress_iterator_releases_consumed_values():
    references: list[weakref.ReferenceType[object]] = []

    class Chunk:
        pass

    def source():
        for _ in range(2):
            chunk = Chunk()
            references.append(weakref.ref(chunk))
            yield chunk

    stream = iter_progress(source(), total=2, disable=True)
    first = next(stream)
    second = next(stream)
    del first
    gc.collect()

    assert references[0]() is None
    assert references[1]() is second
    stream.close()


def test_progress_iterator_closes_source_on_early_exit():
    closed = False

    def source():
        nonlocal closed
        try:
            yield object()
            yield object()
        finally:
            closed = True

    stream = iter_progress(source(), total=2, disable=True)
    next(stream)
    stream.close()

    assert closed


def test_rolling_window_smoothes_along_rows():
    data = np.arange(10, dtype=float).reshape(5, 2)
    smoothed = rolling_window(data, w=3)
    expected = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [7.0, 8.0],
        ]
    )
    assert np.array_equal(smoothed, expected)


def test_rolling_window_even_and_oversized_windows():
    data = np.arange(5, dtype=float).reshape(-1, 1)

    assert np.array_equal(rolling_window(np.array([[7.0]]), w=1), [[7.0]])
    assert np.array_equal(
        rolling_window(data, w=2).ravel(),
        [0.5, 1.5, 2.5, 3.5, 4.0],
    )
    assert np.array_equal(
        rolling_window(data, w=20).ravel(),
        [1.0, 1.5, 2.0, 2.5, 3.0],
    )
    for window_size in (0, -1):
        with pytest.raises(ValueError, match="greater than zero"):
            rolling_window(data, w=window_size)


def test_python_rolling_window_kernel_matches_public_compiled_path():
    data = np.arange(12, dtype=float).reshape(6, 2)

    np.testing.assert_allclose(
        _rolling_window_kernel(data, 4),
        rolling_window(data, 4),
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        _rolling_window_kernel(np.arange(3), 2)
    with pytest.raises(ValueError, match="greater than zero"):
        _rolling_window_kernel(data, 0)
    with pytest.raises(ValueError, match="at least one row"):
        _rolling_window_kernel(np.empty((0, 2)), 1)


def test_array_digest_is_deterministic_and_shape_sensitive():
    values = np.arange(6, dtype=np.int64)

    assert array_digest(values) == array_digest(values.copy())
    assert array_digest(values) != array_digest(values.reshape(2, 3))


def test_permute_into_chunks_preserves_all_indices():
    chunks = permute_into_chunks(10, 3, seed=7)
    # Each run of three indices, and the short remainder, is shuffled within
    # itself by one generator seeded once.
    rng = np.random.default_rng(7)
    expected = [
        rng.permutation(np.arange(start, min(start + 3, 10))) for start in (0, 3, 6, 9)
    ]
    assert [chunk.tolist() for chunk in chunks] == [
        chunk.tolist() for chunk in expected
    ]
    merged = np.concatenate(chunks)
    assert np.array_equal(np.sort(merged), np.arange(10))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            np.array([1 + 0j]),
            "Complex sparse values cannot use an integer destination",
        ),
        (
            np.array([1.5]),
            "cannot be represented by the destination dtype",
        ),
        (
            np.array([np.nan]),
            "cannot be represented by the destination dtype",
        ),
    ],
)
def test_checked_sparse_cast_rejects_lossy_integer_conversions(values, message):
    with pytest.raises(OverflowError, match=message):
        checked_sparse_cast(values, np.int32)


def test_canonicalize_sparse_handles_empty_and_already_canonical_inputs():
    empty = coo_matrix(
        (
            np.array([], dtype=np.int64),
            (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ),
        ),
        shape=(2, 2),
    )
    empty.has_canonical_format = False

    canonical_empty = canonicalize_sparse(empty)

    assert canonical_empty.shape == (2, 2)
    assert canonical_empty.nnz == 0
    assert canonical_empty.has_canonical_format

    canonical = coo_matrix(
        (
            np.array([1.0, 2.0]),
            (
                np.array([0, 1]),
                np.array([0, 1]),
            ),
        ),
        shape=(2, 2),
    )
    canonical.sum_duplicates()

    returned = canonicalize_sparse(canonical, dtype=np.int16)

    assert returned is canonical
    assert returned.dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(returned.data, [1, 2])


def test_canonicalize_sparse_detects_int64_duplicate_overflow():
    maximum = np.iinfo(np.int64).max
    duplicated = coo_matrix(
        (
            np.array([maximum, 1], dtype=np.int64),
            (
                np.array([0, 0]),
                np.array([0, 0]),
            ),
        ),
        shape=(1, 1),
    )

    with pytest.raises(OverflowError, match="Duplicate sparse values exceed"):
        canonicalize_sparse(duplicated)


def test_array_digest_rejects_object_values():
    with pytest.raises(TypeError, match="object arrays"):
        array_digest(np.array([object()], dtype=object))


def test_integer_argument_accepts_numpy_integers_and_checks_bounds():
    assert integer_argument(np.int64(3), "count", minimum=1) == 3
    assert type(integer_argument(np.uint8(3), "count")) is int
    for value in (True, np.bool_(True), 2.0, "2", None):
        with pytest.raises(TypeError, match="count must be an integer"):
            integer_argument(value, "count", minimum=1)
    with pytest.raises(ValueError, match="count must be at least 1"):
        integer_argument(0, "count", minimum=1)
    with pytest.raises(ValueError, match="count must be at most 4"):
        integer_argument(5, "count", minimum=1, maximum=4)


def test_float_argument_shares_one_value_across_numeric_spellings():
    for value in (1, 1.0, np.int64(1), np.float32(1.0)):
        resolved = float_argument(value, "ratio")
        assert type(resolved) is float
        assert resolved == 1.0
    for value in (True, np.bool_(False), "1", None, 1j):
        with pytest.raises(TypeError, match="ratio must be a real number"):
            float_argument(value, "ratio")
    for value in (float("nan"), float("inf"), np.float64(-np.inf)):
        with pytest.raises(ValueError, match="ratio must be finite"):
            float_argument(value, "ratio")


def test_clip_fraction_argument_requires_a_finite_fraction_below_one_half():
    for value, expected in ((0, 0.0), (np.int64(0), 0.0), (np.float32(0.25), 0.25)):
        resolved = clip_fraction_argument(value)
        assert type(resolved) is float
        assert resolved == expected
    for value in (True, np.bool_(False), "0.1", None, np.array(0.1)):
        with pytest.raises(TypeError, match="clip_fraction must be a real number"):
            clip_fraction_argument(value)
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="clip_fraction must be finite"):
            clip_fraction_argument(value)
    for value in (-0.1, 0.5, 0.75, 1):
        with pytest.raises(
            ValueError, match="tails must be at least 0 and less than 0.5"
        ):
            clip_fraction_argument(value, "tails")


def test_read_only_copy_owns_its_values_and_stays_read_only():
    source = np.arange(6, dtype=np.int32).reshape(2, 3).T
    copied = read_only_copy(source, np.int64)
    source[0, 0] = 99

    assert copied.dtype == np.dtype(np.int64)
    assert copied.flags.c_contiguous
    np.testing.assert_array_equal(copied, [[0, 3], [1, 4], [2, 5]])
    with pytest.raises(ValueError, match="read-only"):
        copied[0, 0] = 1
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        copied.setflags(write=True)


def test_sparse_matrix_bytes_counts_shared_arrays_once():
    matrix = csr_matrix(np.eye(3))
    single = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    sharing = matrix.copy()
    sharing.indices = matrix.indices
    sharing.indptr = matrix.indptr

    assert sparse_matrix_bytes(matrix) == single
    assert sparse_matrix_bytes(matrix, matrix) == single
    assert sparse_matrix_bytes(matrix, sharing) == single + sharing.data.nbytes


def test_cumulative_and_max_window_nnz_follow_row_prefix_sums():
    cumulative = cumulative_nnz(np.asarray([2, 0, 5, 1], dtype=np.int32))
    np.testing.assert_array_equal(cumulative, [0, 2, 2, 7, 8])
    assert cumulative.dtype == np.int64
    assert max_window_nnz(cumulative, 1) == 5
    assert max_window_nnz(cumulative, 2) == 6
    assert max_window_nnz(cumulative, 10) == 8
    assert max_window_nnz(cumulative_nnz(np.empty(0, dtype=np.int64)), 3) == 0
    with pytest.raises(ValueError, match="window_rows must be positive"):
        max_window_nnz(cumulative, 0)


def test_assay_feature_ranges_group_spans_in_first_seen_order():
    import pandas as pd

    table = pd.DataFrame(
        [["RNA", 0, 3], ["ADT", 3, 5], ["RNA", 5, 6]],
        columns=["type", "start", "end"],
        index=["RNA", "ADT", "RNA"],
    ).T
    assert assay_feature_ranges(table) == {
        "RNA": ((0, 3), (5, 6)),
        "ADT": ((3, 5),),
    }
