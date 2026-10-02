"""Normalization of stored counts.

Library-size, CLR, and TF-IDF values are computed in float64 from the counts
and float64 totals for bool, integer, and floating-point counts alike, so the
count storage dtype never changes the arithmetic. A value that is persisted,
or that marker search ranks under ``norm_lib_size``, is the single float32
rounding of that float64 value; marker search ranks the values of every other
normalization unrounded. Totals accumulate in float64. Sums over cells, such
as the CLR log means, combine the partial sums of stored row blocks, so their
last float64 bits follow the count layout, which byte targets make differ
between dtypes of different widths.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import zarr
from numba import njit
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ..matrix import ChunkedArray
from ..storage.arrays import create_numeric_array
from ..storage.layout import normed_array_spec
from ..storage.profiles import resolve_storage_profile
from ..storage.sharding import (
    plan_dense_write,
    write_dense_from_row_batches,
    write_dense_in_shard_rows,
)
from ..storage.budget import ResourceBudget
from ..storage.io_policy import StorageIoPolicy
from ..storage.layout import array_shard_rows
from ..storage.materialize import (
    _feature_summary,
    _merge_feature_summaries,
    _write_feature_summaries,
)
from ..utils.compute import controlled_compute
from ..storage.artifacts import ArtifactRef, artifact_group, require_complete_artifact
from ..storage.errors import ArtifactResolutionError
from ..storage.feature_selection import validate_feature_selection
from ..storage.identity import read_dataset_fingerprint
from ..storage.selections import (
    ValidatedStoredSelection,
    validate_stored_selection_integrity,
)
from ..storage.types import as_zarr_array, as_zarr_group

if TYPE_CHECKING:
    from .base import Assay

type NormMethod = Callable[["Assay", ChunkedArray], ChunkedArray]

NORMALIZATION_PARAM_NAMES = frozenset({"log_transform", "renormalize_subset"})


@dataclass(frozen=True, slots=True)
class NormalizationSelections:
    cells: ValidatedStoredSelection
    features: ArtifactRef
    featureMask: np.ndarray


def load_normalization_selections(
    root: zarr.Group, assay: str, cells: ArtifactRef, features: ArtifactRef
) -> NormalizationSelections:
    read_dataset_fingerprint(as_zarr_group(root[assay], name=assay))
    validated_cells = validate_stored_selection_integrity(
        root,
        cells,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    validated_features = validate_feature_selection(root, assay, features)
    mask = validated_features.mask
    if validated_cells.selected_count < 1:
        raise ValueError("Normalization requires selected cells and features")
    return NormalizationSelections(validated_cells, features, mask)


def load_normalized_inputs(
    root: zarr.Group, normalized: ArtifactRef
) -> tuple[zarr.Group, NormalizationSelections]:
    """Return a normalized artifact's group and its validated selections.

    Callers have already resolved ``normalized`` as an assay-scoped
    ``normalized`` artifact.
    """
    assert normalized.assay is not None
    status = require_complete_artifact(root, normalized)
    inputs = status.inputs or {}
    current = read_dataset_fingerprint(
        as_zarr_group(root[normalized.assay], name=normalized.assay)
    )
    if inputs.get("dataset_fingerprint") != current:
        raise ValueError("Normalized data does not match the current prepared dataset")
    selections = load_normalization_selections(
        root,
        normalized.assay,
        ArtifactRef.from_dict(inputs["cell_selection"]),
        ArtifactRef.from_dict(inputs["feature_selection"]),
    )
    group = artifact_group(root, normalized)
    data = as_zarr_array(group["data"], name="data")
    shape = (selections.cells.selected_count, int(selections.featureMask.sum()))
    if data.shape != shape or data.dtype != np.dtype(np.float32):
        raise ArtifactResolutionError(
            f"Normalized {'rows' if data.shape[0] != shape[0] else 'columns'} do not match its selections",
            code="row_mismatch" if data.shape[0] != shape[0] else "column_mismatch",
            context={"assay": normalized.assay, "artifact_id": normalized.artifact_id},
        )
    return group, selections


def reject_unknown_normalization_params(
    params: dict[str, Any],
    *,
    caller: str,
) -> None:
    """Reject execution and other unknown keywords before they enter provenance."""
    for name in params:
        if name not in NORMALIZATION_PARAM_NAMES:
            raise TypeError(f"{caller}() got an unexpected keyword argument {name!r}")


def library_size_values(
    counts: ArrayLike,
    totals: ArrayLike,
    size_factor: float,
    *,
    dtype: DTypeLike,
    log_transform: bool = False,
) -> NDArray[Any]:
    """Scale each row of counts to ``size_factor`` over its library total.

    Every step runs in ``dtype``: ``size_factor * counts / totals``, then
    ``log1p`` when ``log_transform`` is true. The result is one new array.

    Args:
        counts: Counts with one row per cell.
        totals: Nonzero library total of each row.
        size_factor: Library size that each row is scaled to.
        dtype: Floating-point dtype of the arithmetic and the result.
        log_transform: Whether to return ``log1p`` of the scaled values.

    Returns:
        Normalized values with the shape of ``counts``.
    """
    values: NDArray[Any] = np.multiply(counts, size_factor, dtype=dtype)
    values /= np.asarray(totals, dtype=dtype).reshape(-1, 1)
    if log_transform:
        np.log1p(values, out=values)
    return values


def clr_values[T: (np.ndarray, ChunkedArray)](counts: T) -> T:
    """Return the centered log-ratio of each feature over the cells.

    Each column is divided by the exponential of its mean ``log1p`` value and
    then ``log1p`` transformed, so ``counts`` must hold every cell of the
    normalization. Every step runs in float64. A ``ChunkedArray`` is reduced
    once and returned lazily.

    Args:
        counts: Counts with cells as rows and features as columns.

    Returns:
        Float64 CLR values with the shape of ``counts``.
    """
    scale = np.exp(np.log1p(counts, dtype=np.float64).sum(axis=0) / len(counts))
    # A ufunc of a ChunkedArray stays lazy, though NumPy types it as an array.
    values: Any = np.log1p(counts / scale.reshape(1, -1))
    return cast(T, values)


def term_frequencies[T: (np.ndarray, ChunkedArray)](
    counts: T,
    term_totals: ArrayLike,
) -> T:
    """Return TF-IDF term frequencies: each row of counts over its total.

    Args:
        counts: Counts with documents (cells) as rows.
        term_totals: Nonzero term-frequency denominator of each row.

    Returns:
        Float64 term frequencies with the shape of ``counts``.
    """
    frequencies: Any = counts / np.asarray(term_totals, dtype=np.float64).reshape(-1, 1)
    return cast(T, frequencies)


def inverse_document_frequency(
    n_docs: int,
    document_frequency: ArrayLike,
) -> NDArray[np.float64]:
    """Return the TF-IDF weight ``log2(1 + n_docs / (df + 1))`` of each feature.

    Args:
        n_docs: Number of documents (cells) the frequencies were counted over.
        document_frequency: Number of those documents in which each feature
            occurs.

    Returns:
        The float64 inverse document frequency of each feature.
    """
    frequency = np.asarray(document_frequency, dtype=np.float64)
    return np.log2(1 + (n_docs / (frequency + 1)))


def tfidf_values[T: (np.ndarray, ChunkedArray)](
    counts: T,
    term_totals: ArrayLike,
    idf: ArrayLike,
) -> T:
    """Return TF-IDF values: term frequencies weighted by feature IDF.

    Args:
        counts: Counts with documents (cells) as rows and features as columns.
        term_totals: Nonzero term-frequency denominator of each row.
        idf: Inverse document frequency of each column, as returned by
            ``inverse_document_frequency``.

    Returns:
        TF-IDF values with the shape of ``counts``.
    """
    weights = np.asarray(idf).reshape(1, -1)
    values: Any = term_frequencies(counts, term_totals) * weights
    return cast(T, values)


def stream_document_frequency(
    counts: ChunkedArray,
    *,
    memory_bytes: int,
    nthreads: int,
    msg: str,
    operation: str,
    resident_bytes: int = 0,
    row_mask: np.ndarray | None = None,
    term_totals: np.ndarray | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64] | None]:
    """Count the rows in which each column of ``counts`` is nonzero.

    This is the TF-IDF document frequency, counted in one pass over row
    blocks. ``row_mask`` counts only the rows it selects. With
    ``term_totals``, one denominator per counted row, the pass also sums each
    column's term frequencies. Blocks hold at most the rows of one block of
    ``counts`` and fewer when the pass would exceed ``memory_bytes`` beside
    the ``resident_bytes`` that the caller keeps.

    Args:
        counts: Counts with documents (cells) as rows, at least one of them.
        memory_bytes: Memory limit of the operation.
        nthreads: Threads that read row blocks.
        msg: Progress message.
        operation: Name of the operation in errors.
        resident_bytes: Bytes that the caller keeps during the pass.
        row_mask: Optional boolean mask of the rows to count.
        term_totals: Optional term-frequency denominators of the counted rows.

    Returns:
        The document frequency of each column and, with ``term_totals``, the
        summed term frequency of each column.

    Raises:
        MemoryError: If one row does not fit the memory limit.
    """
    n_rows, n_columns = counts.shape
    document_frequency = np.zeros(n_columns, dtype=np.int64)
    term_frequency_sum = (
        None if term_totals is None else np.zeros(n_columns, dtype=np.float64)
    )
    column_bytes = n_columns * np.dtype(np.float64).itemsize
    # The running counts and one column reduction per block stay resident.
    static_bytes = int(resident_bytes) + document_frequency.nbytes + column_bytes
    scratch_bytes_per_row = 0
    if row_mask is not None:
        static_bytes += row_mask.nbytes
        # Each block's selected rows are copied.
        scratch_bytes_per_row += n_columns * counts.dtype.itemsize
    if term_totals is not None and term_frequency_sum is not None:
        static_bytes += term_totals.nbytes + term_frequency_sum.nbytes + column_bytes
        # Each block's term frequencies are computed in float64.
        scratch_bytes_per_row += column_bytes
    decode_bytes = counts._max_decode_bytes()
    current_rows = min(int(counts.chunksize[0]), n_rows)
    working_bytes_per_row = (
        counts._block_owned_bytes() // current_rows + scratch_bytes_per_row
    )
    available_bytes = int(memory_bytes) - static_bytes - decode_bytes
    if available_bytes < working_bytes_per_row:
        required_bytes = static_bytes + decode_bytes + working_bytes_per_row
        raise MemoryError(
            f"{operation} needs about {required_bytes} bytes for one row, but "
            f"the operation limit is {memory_bytes} bytes"
        )
    block_rows = min(current_rows, available_bytes // working_bytes_per_row)
    row_offset = 0
    for block in counts._with_block_size(block_rows)._stream_blocks(
        nthreads=nthreads,
        msg=msg,
        prefetch=1,
        row_mask=row_mask,
        resident_bytes=static_bytes + block_rows * scratch_bytes_per_row,
    ):
        row_stop = row_offset + block.shape[0]
        document_frequency += np.count_nonzero(block, axis=0)
        if term_totals is not None and term_frequency_sum is not None:
            term_frequency_sum += term_frequencies(
                block, term_totals[row_offset:row_stop]
            ).sum(axis=0)
        row_offset = row_stop
        # Release the block before the stream reads the next one.
        del block
    return document_frequency, term_frequency_sum


def _library_size_scaled(assay: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Scale each cell's counts by the size factor over its total in float64."""
    assert assay.sf is not None and assay.scalar is not None
    totals = assay.scalar.reshape(-1, 1)
    scale = partial(library_size_values, size_factor=assay.sf, dtype=np.float64)
    if isinstance(counts, ChunkedArray):
        return counts._binary(scale, totals, "left")
    return cast(ChunkedArray, scale(counts, totals))


def norm_dummy(_: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """A dummy normalizer. Doesn't perform any normalization. This is useful
    when the 'raw data' is already normalized.

    Args:
        _:
        counts: A chunked array with 'raw' counts data

    Returns: A chunked array
    """
    return counts


def norm_lib_size(assay: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs library size normalization on the data. This is the default
    method for RNA assays.

    Args:
        assay: An instance of the assay object
        counts: A chunked array with raw counts data

    Returns:  A chunked array (delayed matrix) containing normalized data.
    """
    return _library_size_scaled(assay, counts)


def lib_size_feature_stream_eligible(
    assay: "Assay",
    *,
    renormalize_subset: bool = False,
) -> bool:
    """True when column-wise lib-size streaming matches ``normed`` semantics."""
    return (
        assay.normMethod is norm_lib_size
        and not renormalize_subset
        and getattr(assay, "sf", None) is not None
    )


def norm_lib_size_log(assay: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs library size normalization and then transforms the values into
    log scale.

    Args:
        assay: An instance of the assay object
        counts: A chunked array with raw counts data

    Returns: A chunked array (delayed matrix) containing normalized data.
    """
    return cast(ChunkedArray, np.log1p(_library_size_scaled(assay, counts)))


def norm_clr(_: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs centered log-ratio normalization (ADT). This is the default
    method for ADT assays.

    Args:
        _:
        counts: A chunked array with raw counts data

    Returns: A chunked array (delayed matrix) containing normalized data.
    """
    return clr_values(counts)


norm_clr.artifact_identity = "scarf.assay.norm_clr:feature-axis"  # type: ignore[attr-defined]


def norm_tf_idf(assay: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs TF-IDF normalization This is the default method for ATAC
    assays.

    Args:
        assay: An instance of the assay object
        counts: A chunked array with raw counts data

    Returns: A chunked array (delayed matrix) containing normalized data.
    """
    assert (
        assay.n_term_per_doc is not None
        and assay.n_docs is not None
        and assay.n_docs_per_term is not None
    )
    idf = inverse_document_frequency(assay.n_docs, assay.n_docs_per_term)
    return tfidf_values(counts, assay.n_term_per_doc, idf)


norm_tf_idf.artifact_identity = (  # type: ignore[attr-defined]
    "scarf.assay.norm_tf_idf:selected-cell-df:total-count-tf"
)


def _feature_group_positions(
    feature_groups: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the sorted union of the groups and each group's union positions."""
    groups = [np.asarray(group, dtype=np.int64) for group in feature_groups]
    if not groups or any(group.ndim != 1 or group.size == 0 for group in groups):
        raise ValueError("Feature groups must be non-empty one-dimensional indices")
    union = np.unique(np.concatenate(groups))
    return union, [np.searchsorted(union, group) for group in groups]


def iter_feature_group_means(
    assay: "Assay",
    cell_idx: np.ndarray,
    feature_groups: Sequence[np.ndarray],
    *,
    block_rows: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield the per-cell mean normalized value of each feature group.

    The assay normalization is fitted once over all of ``cell_idx``, for
    example ATAC document frequency or ADT CLR geometric means, and is then
    applied to row blocks. Values therefore do not depend on the block size.
    Each block reads the union of the group features once.

    Args:
        assay: Assay whose ``normed`` defines the normalization.
        cell_idx: Ordered cells to normalize and summarize.
        feature_groups: Feature indices of each group.
        block_rows: Optional rows per block. Defaults to the block size of the
            assay's normalized matrix.

    Yields:
        Arrays with one row per cell, in ``cell_idx`` order, and one column
        per feature group.
    """
    union, positions = _feature_group_positions(feature_groups)
    cell_idx = np.asarray(cell_idx, dtype=np.int64)
    if cell_idx.size == 0:
        return
    normalized = assay.normed(cell_idx=cell_idx, feat_idx=union)
    if block_rows is not None:
        normalized = normalized._with_block_size(max(1, int(block_rows)))
    for block in normalized.stream_blocks(
        nthreads=assay.nthreads,
        msg=f"({assay.name}) Averaging feature groups",
    ):
        values = np.asarray(block, dtype=np.float64)
        means = np.column_stack(
            [values[:, position].mean(axis=1) for position in positions]
        )
        # The generator is suspended while its consumer writes, so release the
        # blocks before yielding.
        del block, values
        yield means


@njit(cache=True, nogil=True)
def _normalize_rows(
    block: np.ndarray,
    row_sum: np.ndarray,
    scale: float,
    log_transform: bool,
    out: np.ndarray,
) -> None:
    """Write library-size normalized counts of any dtype into ``out``.

    Each value is computed in float64, as ``library_size_values`` computes
    it, and rounded once to the dtype of ``out``. Zero counts stay zero, so
    the transform runs only on detected values.
    """
    for row in range(block.shape[0]):
        total = np.float64(row_sum[row])
        for column in range(block.shape[1]):
            count = block[row, column]
            if count == 0:
                out[row, column] = 0.0
            else:
                value = (scale * np.float64(count)) / total
                out[row, column] = np.log1p(value) if log_transform else value


def _normalize_count_block(
    block: np.ndarray,
    *,
    scaleFactor: float,
    logTransform: bool,
) -> np.ndarray:
    """Return float32 library-size values of ``block`` over its own row totals."""
    row_sum = block.sum(axis=1, dtype=np.float64)
    row_sum[row_sum == 0] = 1
    normalized = np.empty(block.shape, dtype=np.float32)
    _normalize_rows(block, row_sum, float(scaleFactor), bool(logTransform), normalized)
    return normalized


def _counts_t_renormalized_batches(
    assay: "Assay",
    cellIdx: np.ndarray,
    featIdx: np.ndarray,
    *,
    scaleFactor: float,
    logTransform: bool,
    resources: ResourceBudget | None = None,
) -> Iterator[np.ndarray]:
    from ..storage.feature_stream import map_feature_cell_bands, selected_feature_values

    counts_t = assay.rawDataT
    if counts_t is None:
        raise ValueError("Feature-major normalization requires countsT")
    selected_cells = np.asarray(cellIdx, dtype=np.int64)
    selected_features = np.asarray(featIdx, dtype=np.int64)
    if selected_cells.size > 1 and np.any(np.diff(selected_cells) <= 0):
        raise ValueError("Feature-major normalization requires sorted unique cells")

    n_features = int(selected_features.shape[0])
    feature_destinations = np.full(int(counts_t.shape[0]), -1, dtype=np.int64)
    feature_destinations[selected_features] = np.arange(n_features, dtype=np.int64)
    cell_chunk = max(1, int(counts_t.chunks[1]))
    raw_band_bytes = (
        cell_chunk * n_features * max(1, int(np.dtype(counts_t.dtype).itemsize))
    )
    normalized_band_bytes = cell_chunk * n_features * np.dtype(np.float32).itemsize
    scratch_bytes = (
        2 * raw_band_bytes
        + normalized_band_bytes
        # Float64 row totals and their zero mask.
        + cell_chunk * (np.dtype(np.float64).itemsize + 1)
        + feature_destinations.nbytes
        + selected_cells.nbytes
        + selected_features.nbytes
    )

    raw: np.ndarray | None = None
    completed_groups = 0
    metrics: dict[str, Any] = {}

    # The ordered cell-major stream hands over every feature group of a cell
    # band before the next band, and the bands in cell order.
    def fill_band(band: Any) -> np.ndarray | None:
        nonlocal raw, completed_groups
        local_dest = feature_destinations[band.featStart + band.featureRows()]
        keep = local_dest >= 0
        if not np.any(keep):
            raise RuntimeError(
                "Planned countsT group did not contain selected features"
            )
        row_destinations = np.asarray(band.selectedDestinations, dtype=np.int64)
        row_start = int(row_destinations[0])
        expected_rows = np.arange(
            row_start,
            row_start + int(row_destinations.shape[0]),
            dtype=np.int64,
        )
        if not np.array_equal(row_destinations, expected_rows):
            raise ValueError(
                "Feature-major normalization requires contiguous selected-cell bands"
            )
        if raw is None:
            raw = np.empty(
                (int(row_destinations.shape[0]), n_features), dtype=counts_t.dtype
            )
        selected = selected_feature_values(band.values, keep)
        destinations = local_dest[keep]
        raw[:, destinations] = selected[:, band.selectedLocal].T
        completed_groups += 1
        if completed_groups != int(metrics["featureGroupCount"]):
            return None
        result = raw
        raw = None
        completed_groups = 0
        return result

    next_row = 0
    for raw_values in map_feature_cell_bands(
        counts_t,
        fill_band,
        cell_idx=selected_cells,
        feat_idx=selected_features,
        resources=resources or assay.resources,
        io=assay.storageIo,
        metrics=metrics,
        scratchBytes=scratch_bytes,
        orderedCompute=True,
        cellMajorOrder=True,
    ):
        if raw_values is None:
            continue
        normalized = _normalize_count_block(
            raw_values,
            scaleFactor=scaleFactor,
            logTransform=logTransform,
        )
        next_row += int(normalized.shape[0])
        yield normalized
    if raw is not None or completed_groups or next_row != int(selected_cells.shape[0]):
        raise RuntimeError(
            "Feature-major normalization did not cover every selected cell"
        )


def write_renorm_subset_to_zarr(
    assay: "Assay",
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
    root: zarr.Group,
    loc: str,
    nthreads: int,
    log_transform: bool = False,
    msg: str | None = None,
    mirror: zarr.Array | None = None,
    stats_group: zarr.Group | None = None,
) -> None:
    scale_factor = assay.sf
    if scale_factor is None:
        raise ValueError("Library-size normalization requires a size factor")
    read_dataset_fingerprint(assay.z)
    resources = ResourceBudget(
        assay.resources.memoryBytes, min(max(1, nthreads), assay.resources.workers)
    )
    counts = assay.rawData[:, feat_idx][cell_idx, :]
    if msg is None:
        msg = f"Writing data to {loc}"
    spec = normed_array_spec(
        counts.shape[0],
        counts.shape[1],
        profile=resolve_storage_profile(root.store),
    )
    output = create_numeric_array(root, loc, spec)

    if assay.rawDataT is not None and mirror is None:
        summary: tuple[np.ndarray, np.ndarray] | None = None
        summary_bytes = 2 * len(feat_idx) * np.dtype(np.float64).itemsize
        writer_resident = summary_bytes if stats_group is not None else 0
        single_writer = plan_dense_write(
            output,
            resources,
            1,
            io=StorageIoPolicy(readWorkers=1, computeWorkers=1, writeWorkers=1),
            residentBytes=writer_resident,
        )
        # The producer holds only a few in-flight bands. Give the writer up to a
        # quarter of the budget, never less than one writer, so output chunks
        # encode and upload in parallel instead of stalling the ordered stream.
        writer_memory = min(
            resources.memoryBytes,
            max(single_writer.reservedBytes, resources.memoryBytes // 4),
        )
        n_bands = -(-int(output.shape[0]) // array_shard_rows(output))
        writer_plan = plan_dense_write(
            output,
            ResourceBudget(writer_memory, resources.workers),
            n_bands,
            io=assay.storageIo,
            residentBytes=writer_resident,
        )
        producer_memory = resources.memoryBytes - writer_plan.reservedBytes
        if producer_memory < 1:
            raise MemoryError(
                "Normalization needs memory for both its producer and writer"
            )
        producer_resources = ResourceBudget(
            producer_memory, max(1, resources.workers - 1)
        )

        def normalized_batches() -> Iterator[np.ndarray]:
            nonlocal summary
            for block in _counts_t_renormalized_batches(
                assay,
                cell_idx,
                feat_idx,
                scaleFactor=float(scale_factor),
                logTransform=log_transform,
                resources=producer_resources,
            ):
                if stats_group is not None:
                    current = _feature_summary(block)
                    summary = (
                        current
                        if summary is None
                        else _merge_feature_summaries(summary, current)
                    )
                yield block

        write_dense_from_row_batches(
            output,
            normalized_batches(),
            resources=resources,
            producerReserveBytes=producer_memory,
            residentBytes=summary_bytes if stats_group is not None else 0,
            io=assay.storageIo,
            msg=msg,
        )
        _write_feature_summaries(stats_group, summary)
        return

    def normalize_block(block: Any) -> np.ndarray:
        return _normalize_count_block(
            np.asarray(block),
            scaleFactor=float(scale_factor),
            logTransform=log_transform,
        )

    summary = write_dense_in_shard_rows(
        output,
        lambda start, end: normalize_block(
            controlled_compute(counts[start:end, :], nthreads)
        ),
        msg=msg,
        also_write_to=mirror,
        resources=resources,
        residentBytes=counts._resident_bytes(),
        producerBytes=(
            counts._with_block_size(array_shard_rows(output))._block_task_bytes()
            # Float64 row totals and their zero mask.
            + array_shard_rows(output) * (np.dtype(np.float64).itemsize + 1)
        ),
        resultBytes=2 * len(feat_idx) * 8 if stats_group is not None else 0,
        summarize=_feature_summary if stats_group is not None else None,
        merge_summary=(_merge_feature_summaries if stats_group is not None else None),
        io=assay.storageIo,
    )
    _write_feature_summaries(stats_group, summary)
