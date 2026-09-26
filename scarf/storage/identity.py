import hashlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import zarr

from .arrays import _decode_metadata_values
from .artifacts import ValueFingerprintBuilder, canonical_bytes
from .budget import ResourceBudget, resolve_budget
from .execution import WorkShape, plan_operation
from .geometry import array_geometry
from .parallel import stream_shards
from .partition import scan_band
from .stores import metadata_workers, run_concurrently
from .types import as_zarr_array, as_zarr_group
from .validation_scope import store_key, validated_once

REBUILD_REQUIRED = (
    "Rebuild into a fresh destination with repack_store(..., data_only=True)."
)
GENERATED_FEATURE_COLUMNS = frozenset({"nCells", "dropOuts"})


def generated_cell_columns(name: str, percentages: object) -> frozenset[str]:
    """Return the cell columns that preparation derived from an assay's counts."""
    return frozenset(
        {f"{name}_nCounts", f"{name}_nFeatures"}
        | set(percentages if isinstance(percentages, dict) else ())
    )


def fresh_group(group: zarr.Group) -> zarr.Group:
    # A known format reads one metadata document instead of probing every format.
    return zarr.open_group(
        store=group.store,
        path=group.path,
        mode="r" if group.read_only else "r+",
        zarr_format=group.metadata.zarr_format,
    )


def count_fingerprint(counts: zarr.Array) -> str:
    """Return the finalized counts fingerprint, rereading stored attributes."""
    return opened_count_fingerprint(
        zarr.open_array(
            store=counts.store,
            path=counts.path,
            mode="r",
            zarr_format=counts.metadata.zarr_format,
        )
    )


def opened_count_fingerprint(counts: zarr.Array) -> str:
    """Return the finalized fingerprint of counts just opened from the store."""
    value = counts.attrs.get("content_fingerprint")
    if (
        counts.attrs.get("complete") is not True
        or not isinstance(value, str)
        or not value
    ):
        raise ValueError(f"Raw counts are not finalized. {REBUILD_REQUIRED}")
    return value


def read_dataset_fingerprint(assay: zarr.Group) -> str:
    return validated_once(
        ("dataset_fingerprint", *store_key(assay)),
        lambda: _read_dataset_fingerprint(assay),
    )


def _read_dataset_fingerprint(assay: zarr.Group) -> str:
    attrs = fresh_group(assay).attrs
    value = attrs.get("dataset_fingerprint")
    if attrs.get("prepared") is not True or not isinstance(value, str) or not value:
        raise ValueError(f"Assay {assay.name!r} is not prepared. {REBUILD_REQUIRED}")
    return value


COUNT_SUMMARIES = "countSummaries"


class CountSummary:
    """Per-row digests and totals for a row window of one count matrix.

    Writers fill rows from bands in any order while the values are in memory.
    The fingerprint depends only on each row's nonzero values, so chunk layout,
    band size, and byte order do not change it. Row sums and positive-entry
    counts are the raw inputs for cell and feature summaries.
    """

    def __init__(self, counts: zarr.Array, rows: tuple[int, int] | None = None):
        if counts.ndim != 2 or counts.dtype.kind not in "biuf":
            raise ValueError("Counts must be a two-dimensional numeric array")
        self.shape = (int(counts.shape[0]), int(counts.shape[1]))
        self.dtype = np.dtype(counts.dtype).newbyteorder("=")
        self.start, self.stop = (0, self.shape[0]) if rows is None else rows
        n_rows = self.stop - self.start
        self.digests = np.zeros((n_rows, 2), dtype=np.uint64)
        self.rowSums = np.zeros(n_rows, dtype=np.float64)
        self.rowPositive = np.zeros(n_rows, dtype=np.int64)
        self.columnPositive = np.zeros(self.shape[1], dtype=np.int64)
        self._covered = 0
        self._lock = threading.Lock()

    @staticmethod
    def nbytes_for(n_rows: int, n_columns: int) -> int:
        """Bytes held for ``n_rows`` rows and ``n_columns`` columns.

        Each row keeps two digest words, a total, and a positive count. Column
        counts are held twice while bands add their local counts.
        """
        return 32 * int(n_rows) + 16 * int(n_columns)

    @property
    def nbytes(self) -> int:
        return self.nbytes_for(len(self.rowSums), self.shape[1])

    def update(self, start: int, block: Any) -> None:
        from ..utils.digest import summarize_rows

        values = np.ascontiguousarray(block, dtype=self.dtype)
        stop = start + values.shape[0]
        if values.ndim != 2 or values.shape[1] != self.shape[1]:
            raise ValueError("Count summary block has the wrong shape")
        rows = self._rows(start, stop)
        columns = np.zeros(self.shape[1], dtype=np.int64)
        summarize_rows(
            values,
            values.view(f"u{values.dtype.itemsize}"),
            self.digests[rows],
            self.rowSums[rows],
            self.rowPositive[rows],
            columns,
        )
        self._add(stop - start, columns)

    def merge(
        self, window: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        start, digests, row_sums, row_positive, column_positive = window
        rows = self._rows(start, start + len(digests))
        self.digests[rows] = digests
        self.rowSums[rows] = row_sums
        self.rowPositive[rows] = row_positive
        self._add(len(digests), column_positive)

    def window(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._covered != self.stop - self.start:
            raise RuntimeError("Count summary does not cover its row window")
        return (
            self.start,
            self.digests,
            self.rowSums,
            self.rowPositive,
            self.columnPositive,
        )

    def hexdigest(self) -> str:
        if (self.start, self.stop) != (0, self.shape[0]):
            raise RuntimeError("Count summary covers only part of the matrix")
        self.window()
        builder = hashlib.blake2b(digest_size=32, person=b"scarf-counts")
        builder.update(
            canonical_bytes(
                {"shape": list(self.shape), "dtype": self.dtype.newbyteorder("<").str}
            )
        )
        builder.update(np.ascontiguousarray(self.digests, dtype="<u8"))
        return builder.hexdigest()

    def _rows(self, start: int, stop: int) -> slice:
        if start < self.start or stop > self.stop or stop < start:
            raise ValueError("Count summary rows are outside their window")
        return slice(start - self.start, stop - self.start)

    def _add(self, rows: int, columns: np.ndarray) -> None:
        with self._lock:
            self._covered += rows
            self.columnPositive += columns


def finalize_counts(
    counts: zarr.Array,
    resources: ResourceBudget | None = None,
    summary: CountSummary | None = None,
) -> str:
    """Publish the content fingerprint and raw totals of finalized counts.

    Writers pass the summary they filled while writing. Without one, the
    counts are read once in chunk-aligned row bands.
    """
    if summary is None:
        counts.attrs.update({"complete": False, "content_fingerprint": None})
        summary = computed = CountSummary(counts)
        resources = resources or resolve_budget()
        itemsize = counts.dtype.itemsize
        chunk_rows = max(1, int(counts.chunks[0]))
        band_bytes = chunk_rows * max(1, summary.shape[1]) * itemsize
        rows = chunk_rows * max(1, 64 * 1024**2 // band_bytes)
        ranges = [
            (start, min(start + rows, summary.shape[0]))
            for start in range(0, summary.shape[0], rows)
        ]
        operation = plan_operation(
            resources,
            WorkShape(
                nUnits=len(ranges),
                unitBytes=rows * max(1, summary.shape[1]) * itemsize,
                residentBytes=summary.nbytes,
                decodeBytes=int(np.prod(counts.chunks)) * itemsize,
                chunksPerShard=-(-summary.shape[1] // max(1, int(counts.chunks[1]))),
            ),
        )
        for _ in stream_shards(
            ranges,
            lambda bounds: computed.update(bounds[0], counts[bounds[0] : bounds[1]]),
            workers=operation.computeWorkers,
            io_concurrency=operation.ioConcurrency,
        ):
            pass
    fingerprint = summary.hexdigest()
    matrix = zarr.open_group(
        store=counts.store, path=counts.path.rpartition("/")[0], mode="r+"
    )
    totals = matrix.create_group(COUNT_SUMMARIES, overwrite=True)
    for name, values in (
        ("rowSums", summary.rowSums),
        ("rowPositive", summary.rowPositive),
        ("columnPositive", summary.columnPositive),
    ):
        totals.create_array(
            name, data=values, chunks=(max(1, min(len(values), 1 << 20)),)
        )
    totals.attrs["source_fingerprint"] = fingerprint
    counts.attrs.update({"content_fingerprint": fingerprint, "complete": True})
    return fingerprint


FEATURE_SUMS = "featureSums"


def feature_sums_key(features: np.ndarray) -> str:
    """Name the persisted per-cell sums of one sorted feature index set."""
    return hashlib.blake2b(
        np.ascontiguousarray(features, dtype="<i8").tobytes(), digest_size=16
    ).hexdigest()


def write_feature_sums(
    matrix: zarr.Group,
    counts: zarr.Array,
    sums: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Persist per-cell sums of feature sets next to the count summaries."""
    summaries = as_zarr_group(matrix[COUNT_SUMMARIES], name=COUNT_SUMMARIES)
    fingerprint = count_fingerprint(counts)
    group = summaries.require_group(FEATURE_SUMS)
    for key, (features, values) in sums.items():
        array = group.create_array(
            key, shape=values.shape, dtype=np.float64, overwrite=True
        )
        array[:] = values
        array.attrs.update(
            {"features": features.tolist(), "source_fingerprint": fingerprint}
        )


def load_feature_sums(
    matrix: zarr.Group, counts: zarr.Array, features: np.ndarray
) -> np.ndarray | None:
    """Return persisted per-cell sums for exactly ``features``, if present."""
    summaries = matrix.get(COUNT_SUMMARIES)
    group = summaries.get(FEATURE_SUMS) if isinstance(summaries, zarr.Group) else None
    array = (
        group.get(feature_sums_key(features)) if isinstance(group, zarr.Group) else None
    )
    if (
        not isinstance(array, zarr.Array)
        or array.shape != (counts.shape[0],)
        or array.attrs.get("source_fingerprint") != count_fingerprint(counts)
        or array.attrs.get("features") != features.tolist()
    ):
        return None
    return np.asarray(array[:])


def load_count_summaries(
    matrix: zarr.Group, counts: zarr.Array
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row sums, row positive counts, and column positive counts."""
    fingerprint = count_fingerprint(counts)
    totals = matrix.get(COUNT_SUMMARIES)
    if (
        not isinstance(totals, zarr.Group)
        or totals.attrs.get("source_fingerprint") != fingerprint
    ):
        raise ValueError(f"Count summaries do not match the counts. {REBUILD_REQUIRED}")
    n_rows, n_columns = counts.shape
    loaded = []
    for name, length in (
        ("rowSums", n_rows),
        ("rowPositive", n_rows),
        ("columnPositive", n_columns),
    ):
        array = as_zarr_array(totals[name], name=name)
        if array.shape != (length,):
            raise ValueError(f"Count summaries are malformed. {REBUILD_REQUIRED}")
        loaded.append(np.asarray(array[:]))
    return loaded[0], loaded[1], loaded[2]


def _hash_column(
    builder: ValueFingerprintBuilder, name: str, column: zarr.Array
) -> None:
    rows = scan_band(array_geometry(column), fallback=100_000)
    if column.dtype.kind in "OUS" or column.dtype.hasobject:
        builder.update_bytes(name, canonical_bytes({"shape": list(column.shape)}))
        for start in range(0, column.shape[0], rows):
            for value in _decode_metadata_values(column[start : start + rows]):
                builder.update_bytes("value", str(value).encode("utf-8"))
    else:
        builder.begin_array(name, column.shape, column.dtype)
        for start in range(0, column.shape[0], rows):
            builder.update_array_block(
                name, (start,), np.asarray(column[start : start + rows])
            )
        builder.end_array(name)


def calculate_dataset_fingerprint(
    assay: zarr.Group, cells: zarr.Group, counts: zarr.Array
) -> str:
    name = assay.name.rsplit("/", 1)[-1]
    builder = ValueFingerprintBuilder()
    builder.update_bytes("counts", count_fingerprint(counts).encode("ascii"))
    features = as_zarr_group(assay["featureData"], name="featureData")
    for label, group, column in (
        ("cell_ids", cells, "ids"),
        ("feature_ids", features, "ids"),
        ("cell_n_counts", cells, f"{name}_nCounts"),
        ("cell_n_features", cells, f"{name}_nFeatures"),
        ("feature_n_cells", features, "nCells"),
    ):
        _hash_column(builder, label, as_zarr_array(group[column], name=column))
    return builder.hexdigest()


def validate_preparation(
    assay: zarr.Group,
    cells: zarr.Group,
    matrix: zarr.Group,
    *,
    require_transpose: bool,
    require_prepared: bool = True,
) -> str | None:
    if not require_prepared:
        # Preparation is being published, so the state is still changing.
        return _validate_preparation(
            assay,
            cells,
            matrix,
            require_transpose=require_transpose,
            require_prepared=False,
        )
    # Prepared data is terminal, so one operation validates it once.
    return validated_once(
        (
            "preparation",
            *store_key(assay),
            *store_key(cells),
            *store_key(matrix),
            require_transpose,
        ),
        lambda: _validate_preparation(
            assay,
            cells,
            matrix,
            require_transpose=require_transpose,
            require_prepared=True,
        ),
    )


def _validate_preparation(
    assay: zarr.Group,
    cells: zarr.Group,
    matrix: zarr.Group,
    *,
    require_transpose: bool,
    require_prepared: bool,
) -> str | None:
    from .counts_t_contract import validate_count_matrix

    assay = fresh_group(assay)
    state = assay.attrs.get("prepared")
    if state is not True and (require_prepared or state is not False):
        raise ValueError(f"Assay {assay.name!r} is not prepared. {REBUILD_REQUIRED}")
    counts, _ = validate_count_matrix(matrix, require_transpose=require_transpose)
    name = assay.name.rsplit("/", 1)[-1]
    features = as_zarr_group(assay["featureData"], name="featureData")
    percentages = assay.attrs.get("percentFeatures", {})
    if not isinstance(percentages, dict):
        raise ValueError(f"Percentage definitions are malformed. {REBUILD_REQUIRED}")

    def column_check(group: zarr.Group, column: str, length: int) -> Callable[[], None]:
        def check() -> None:
            try:
                array = as_zarr_array(group[column], name=column)
            except KeyError:
                raise ValueError(
                    f"Required column {column!r} is missing. {REBUILD_REQUIRED}"
                ) from None
            if array.shape != (length,) or (
                column != "ids" and array.dtype.kind not in "iuf"
            ):
                raise ValueError(
                    f"Required column {column!r} is inconsistent. {REBUILD_REQUIRED}"
                )

        return check

    def percentage_check(column: str, pattern: Any) -> Callable[[], None]:
        def check() -> None:
            try:
                if not isinstance(pattern, str):
                    raise KeyError(column)
                array = as_zarr_array(cells[column], name=column)
            except KeyError:
                raise ValueError(
                    f"Required percentage {column!r} is missing. {REBUILD_REQUIRED}"
                ) from None
            if array.shape != (counts.shape[0],) or not array.attrs.get(
                "feature_selection_fingerprint"
            ):
                raise ValueError(
                    f"Required percentage {column!r} is inconsistent. {REBUILD_REQUIRED}"
                )

        return check

    checks = [
        column_check(group, column, length)
        for group, columns, length in (
            (cells, ("ids", f"{name}_nCounts", f"{name}_nFeatures"), counts.shape[0]),
            (features, ("ids", *sorted(GENERATED_FEATURE_COLUMNS)), counts.shape[1]),
        )
        for column in columns
    ]
    checks += [
        percentage_check(column, pattern) for column, pattern in percentages.items()
    ]
    # Each check is one metadata read; object stores overlap them.
    run_concurrently(checks, workers=metadata_workers(assay))
    if state is not True:
        return None
    fingerprint = assay.attrs.get("dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not fingerprint
        or assay.attrs.get("counts_fingerprint") != opened_count_fingerprint(counts)
    ):
        raise ValueError(
            f"Assay {name!r} has an inconsistent dataset identity. {REBUILD_REQUIRED}"
        )
    return fingerprint


def publish_preparation(
    assay: zarr.Group,
    cells: zarr.Group,
    matrix: zarr.Group,
    *,
    require_transpose: bool,
    expected_fingerprint: str | None = None,
) -> str:
    validate_preparation(
        assay,
        cells,
        matrix,
        require_transpose=require_transpose,
        require_prepared=False,
    )
    counts = as_zarr_array(matrix["counts"], name="counts")
    fingerprint = calculate_dataset_fingerprint(assay, cells, counts)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError(
            "Copied dataset identity differs from the source; saved results cannot be used"
        )
    fresh_group(assay).attrs.update(
        {
            "prepared": True,
            "dataset_fingerprint": fingerprint,
            "counts_fingerprint": count_fingerprint(counts),
        }
    )
    return fingerprint


def protect_metadata_column(group: zarr.Group, column: str) -> None:
    root = zarr.open_group(
        store=group.store, mode="r", zarr_format=group.metadata.zarr_format
    )
    path = group.path.strip("/")
    if path.endswith("/featureData"):
        if column != "ids" and column not in GENERATED_FEATURE_COLUMNS:
            return
        assay = as_zarr_group(root[path.rsplit("/", 1)[0]], name="assay")
        protected = assay.attrs.get("prepared") is True
    elif path == "cellData" or path.endswith("/cellData"):
        parent_path = path.rpartition("/")[0]
        parent = (
            root
            if not parent_path
            else as_zarr_group(root[parent_path], name=parent_path)
        )
        protected = any(
            assay.attrs.get("prepared") is True
            and (
                column == "ids"
                or column
                in generated_cell_columns(name, assay.attrs.get("percentFeatures"))
            )
            for name, assay in parent.groups()
            if assay.attrs.get("is_assay") is True
        )
    else:
        return
    if protected:
        raise ValueError(
            f"Column {column!r} belongs to prepared data and cannot be changed. {REBUILD_REQUIRED}"
        )


def clear_column(group: zarr.Group, column: str) -> None:
    # A missing column has nothing to protect or clear; one lookup answers both.
    try:
        array = as_zarr_array(group[column], name=column)
    except KeyError:
        return
    protect_metadata_column(group, column)
    missing: Any = array.attrs.get("missing_mask")
    if missing is not None:
        if (
            not isinstance(missing, str)
            or "/" in missing
            or missing == column
            or missing not in group
        ):
            raise ValueError(
                f"Column {column!r} has a malformed missing-value dependency"
            )
        mask = as_zarr_array(group[missing], name=missing)
        if mask.dtype != np.dtype(bool) or mask.shape != array.shape:
            raise ValueError(f"Column {column!r} has a malformed missing-value array")
    del group[column]
    if missing is not None:
        del group[missing]
