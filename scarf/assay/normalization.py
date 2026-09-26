from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import zarr
from numba import njit
from numpy.typing import NDArray

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
_NORMALIZATION_WORK_BYTES = 256 * 1024**2


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
    if validated_cells.selected_count < 1 or not np.any(mask):
        raise ValueError("Normalization requires selected cells and features")
    return NormalizationSelections(validated_cells, features, mask)


def load_normalized_inputs(
    root: zarr.Group, normalized: ArtifactRef
) -> tuple[zarr.Group, NormalizationSelections]:
    if (
        normalized.kind != "normalized"
        or normalized.scope != "assay"
        or normalized.assay is None
    ):
        raise ValueError("Expected an assay-scoped normalized artifact")
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
    assert assay.sf is not None and assay.scalar is not None
    return assay.sf * counts / assay.scalar.reshape(-1, 1)


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
    assert assay.sf is not None and assay.scalar is not None
    return cast(ChunkedArray, np.log1p(assay.sf * counts / assay.scalar.reshape(-1, 1)))


def norm_clr(_: "Assay", counts: ChunkedArray) -> ChunkedArray:
    """Performs centered log-ratio normalization (ADT). This is the default
    method for ADT assays.

    Args:
        _:
        counts: A chunked array with raw counts data

    Returns: A chunked array (delayed matrix) containing normalized data.
    """
    f = np.exp(cast(NDArray[Any], np.log1p(counts).sum(axis=0)) / len(counts))
    return cast(ChunkedArray, np.log1p(counts / f.reshape(1, -1)))


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
    t_f = counts / assay.n_term_per_doc.reshape(-1, 1)
    # TODO: Split TF and IDF functionality to make it similar to norml_lib and zscaling
    idf = np.log2(1 + (assay.n_docs / (assay.n_docs_per_term + 1)))
    return t_f * idf.reshape(1, -1)


norm_tf_idf.artifact_identity = (  # type: ignore[attr-defined]
    "scarf.assay.norm_tf_idf:selected-cell-df:total-count-tf"
)


@njit(cache=True, nogil=True)
def _normalize_integer_rows(
    block: np.ndarray,
    row_sum: np.ndarray,
    scale: float,
    log_transform: bool,
    out: np.ndarray,
) -> None:
    """Write library-size normalized counts with NumPy's float64 arithmetic.

    Zero counts stay zero, so the transform runs only on detected values.
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
    row_sum = block.sum(axis=1)
    row_sum[row_sum == 0] = 1
    normalized = np.empty(block.shape, dtype=np.float32)
    if block.dtype.kind in "iu":
        _normalize_integer_rows(
            block, row_sum, float(scaleFactor), bool(logTransform), normalized
        )
        return normalized
    bytes_per_row = max(1, int(block.shape[1]) * max(8, block.dtype.itemsize))
    rows_per_batch = max(1, _NORMALIZATION_WORK_BYTES // bytes_per_row)
    for start in range(0, int(block.shape[0]), rows_per_batch):
        end = min(start + rows_per_batch, int(block.shape[0]))
        work = scaleFactor * block[start:end]
        work /= row_sum[start:end, np.newaxis]
        if logTransform:
            np.log1p(work, out=work)
        normalized[start:end] = work
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
    work_row_bytes = max(1, n_features * max(8, counts_t.dtype.itemsize))
    work_rows = min(cell_chunk, max(1, _NORMALIZATION_WORK_BYTES // work_row_bytes))
    scratch_bytes = (
        2 * raw_band_bytes
        + normalized_band_bytes
        + work_rows * work_row_bytes
        + cell_chunk * (max(8, counts_t.dtype.itemsize) + 1)
        + feature_destinations.nbytes
        + selected_cells.nbytes
        + selected_features.nbytes
    )

    raw: np.ndarray | None = None
    current_cell_start: int | None = None
    completed_groups = 0
    metrics: dict[str, Any] = {}

    def fill_band(band: Any) -> tuple[int, np.ndarray] | None:
        nonlocal raw, current_cell_start, completed_groups
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
            current_cell_start = band.cellStart
        elif current_cell_start != band.cellStart:
            raise RuntimeError(
                "A countsT cell band ended before all its features arrived"
            )
        selected = selected_feature_values(band.values, keep)
        destinations = local_dest[keep]
        raw[:, destinations] = selected[:, band.selectedLocal].T
        completed_groups += 1
        if completed_groups != int(metrics["featureGroupCount"]):
            return None
        result = raw
        raw = None
        current_cell_start = None
        completed_groups = 0
        return row_start, result

    next_row = 0
    for item in map_feature_cell_bands(
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
        if item is None:
            continue
        row_start, raw_values = item
        if row_start != next_row:
            raise RuntimeError("Normalized cell bands arrived out of order")
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
        writer_plan = plan_dense_write(
            output,
            resources,
            1,
            io=StorageIoPolicy(readWorkers=1, computeWorkers=1, writeWorkers=1),
            residentBytes=summary_bytes if stats_group is not None else 0,
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
            + min(256 * 1024**2, array_shard_rows(output) * len(feat_idx) * 8)
        ),
        resultBytes=2 * len(feat_idx) * 8 if stats_group is not None else 0,
        summarize=_feature_summary if stats_group is not None else None,
        merge_summary=(_merge_feature_summaries if stats_group is not None else None),
        io=assay.storageIo,
    )
    _write_feature_summaries(stats_group, summary)
