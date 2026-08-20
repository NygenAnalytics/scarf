from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import zarr
from scipy.sparse import coo_matrix

from ..metadata.rows import (
    metadata_row_selection_peak_bytes,
    read_metadata_rows_chunkwise,
)
from ..storage.budget import ResourceBudget, admitted_worker_split
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.layout import ZarrArraySpec
from ..storage.partition import affordable_width
from ..storage.profiles import StorageProfile
from ..storage.schema import create_zarr_count_assay, load_count_array
from ..storage.sharding import (
    accumulate_sparse_to_shards,
    resolve_sparse_import_batch,
    resolve_sparse_import_spec,
    write_counts_t,
)
from ..storage.types import array_metadata_shards, as_zarr_array, as_zarr_group
from ..utils.arrays import canonicalize_sparse
from ..utils.compute import controlled_compute
from ..utils.logging import logger
from ..utils.progress import iter_progress
from .features import FeatureAlignment
from .row_plan import RowPlan, iter_row_plan_segments, max_row_plan_block_rows


CountsTReuseOutcome = Literal[
    "reusable",
    "rewrite-layout",
    "incomplete",
    "block-shape/dtype",
]


@dataclass(frozen=True, slots=True)
class CountsTReuseAssessment:
    """Structured merge decision for an existing ``countsT`` component."""

    outcome: CountsTReuseOutcome
    reason: str | None = None


def _matrix_group_path(assay_name: str, workspace: str | None) -> str:
    return assay_name if workspace is None else f"matrices/{assay_name}"


def _assay_metadata_path(assay_name: str, workspace: str | None) -> str:
    return assay_name if workspace is None else f"{workspace}/{assay_name}"


def _cell_data_path(workspace: str | None) -> str:
    return "cellData" if workspace is None else f"{workspace}/cellData"


@dataclass(frozen=True, slots=True)
class _MergeImportRequirements:
    maxWindowNnz: Callable[[int], int]
    sourceDtype: np.dtype[Any]
    residentBytes: int
    extraProducerBytes: Callable[[int], int]
    nnzScanRows: int


class MissingAssay:
    """Carrier for a source that lacks an assay modality.

    Holds the source cell table and a reference feature space so DataStoreMerge
    can emit empty sparse blocks without allocating a dummy Zarr array.
    """

    def __init__(
        self,
        cells: Any,
        feats: Any,
        name: str,
        n_cells: int,
        dtype: np.dtype[Any],
    ) -> None:
        self.cells = cells
        self.feats = feats
        self.name = name
        self._nCells = int(n_cells)
        self._dtype = np.dtype(dtype)
        self.rawData = _EmptyRawData(self._nCells, int(feats.N), self._dtype)

    @property
    def isMissing(self) -> bool:
        return True


class _EmptyRawData:
    def __init__(self, n_cells: int, n_feats: int, dtype: np.dtype[Any]) -> None:
        self.shape = (int(n_cells), int(n_feats))
        self.dtype = np.dtype(dtype)
        self.chunksize = (max(1, int(n_cells)), max(1, int(n_feats)))

    @property
    def blocks(self) -> list[Any]:
        return []


def _is_missing(assay: Any) -> bool:
    return isinstance(assay, MissingAssay) or bool(getattr(assay, "isMissing", False))


def remap_block_to_coo(
    block: Any,
    order_map: np.ndarray,
    n_feats: int,
    nthreads: int,
    destination_dtype: np.dtype[Any] | None = None,
) -> coo_matrix:
    """Dense-or-chunked block to COO with feature remapping and summation."""
    computed = controlled_compute(block, nthreads)
    if order_map.shape[0] != computed.shape[1]:
        raise ValueError("Feature order does not match the source matrix width")
    source = coo_matrix(computed)
    mapped = coo_matrix(
        (source.data, (source.row, order_map[source.col])),
        shape=(computed.shape[0], n_feats),
    )
    if not bool(mapped.has_canonical_format):
        mapped = canonicalize_sparse(mapped, destination_dtype)
    return mapped


def empty_block_coo(n_rows: int, n_feats: int) -> coo_matrix:
    return coo_matrix((n_rows, n_feats))


def create_assay_counts(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    alignment: FeatureAlignment,
    dtype: str,
    *,
    profile: StorageProfile,
    policy: CountMatrixPolicy | None,
) -> zarr.Array:
    counts = create_zarr_count_assay(
        z=root,
        assay_name=assay_name,
        workspace=workspace,
        n_cells=n_cells,
        feat_ids=np.array(alignment.mergedFeatsMap["ids"]),
        feat_names=np.array(alignment.mergedFeatsMap["names"]),
        dtype=dtype,
        profile=profile,
        policy=policy,
    )
    matrix_group = as_zarr_group(
        root[_matrix_group_path(assay_name, workspace)],
        name=_matrix_group_path(assay_name, workspace),
    )
    matrix_group.attrs["complete"] = False
    return counts


def _nnz_profile_bytes(n_cells: int) -> int:
    return (max(0, int(n_cells)) + 1) * np.dtype(np.int64).itemsize


def _resolve_nnz_scan_rows(
    assays: list[Any],
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> int:
    """Admit the NNZ profile and choose a bounded metadata scan width."""
    profile_bytes = _nnz_profile_bytes(row_plan.nCells)
    preferred = max(1, max_row_plan_block_rows(row_plan))
    int64_bytes = np.dtype(np.int64).itemsize
    resident = max(0, int(resident_bytes)) + profile_bytes
    if resident >= int(resources.memoryBytes):
        raise MemoryError(
            "Merged assay NNZ profile cannot fit within the operation memory budget"
        )

    def fits(width: int) -> bool:
        task_bytes = [max(1, int(width)) * int64_bytes]
        task_bytes.extend(
            metadata_row_selection_peak_bytes(
                assay.cells,
                f"{assay.name}_nFeatures",
                width,
            )
            for assay in assays
            if not _is_missing(assay)
            and f"{assay.name}_nFeatures" in assay.cells.columns
        )
        try:
            admitted_worker_split(
                resources,
                nTasks=1,
                residentBytes=resident,
                taskBytes=lambda _: max(task_bytes),
                requested=1,
            )
        except MemoryError:
            return False
        return True

    rows = affordable_width(fits, preferred)
    if rows < 1:
        raise MemoryError(
            "Merged assay NNZ profile cannot fit one metadata row within the "
            "operation memory budget"
        )
    return int(rows)


def _merge_import_requirements(
    assays: list[Any],
    row_plan: RowPlan,
    alignment: FeatureAlignment,
    destination_dtype: Any,
    *,
    resources: ResourceBudget,
    additionalResidentBytes: int = 0,
) -> _MergeImportRequirements:
    base_resident = (
        row_plan.resident_bytes()
        + alignment.resident_bytes()
        + max(0, int(additionalResidentBytes))
    )
    scan_rows = _resolve_nnz_scan_rows(
        assays,
        row_plan,
        resources,
        resident_bytes=base_resident,
    )
    profile_bytes = _nnz_profile_bytes(row_plan.nCells)
    cumulative = np.empty(row_plan.nCells + 1, dtype=np.int64)
    cumulative[0] = 0
    for segment in iter_row_plan_segments(row_plan, segment_rows=scan_rows):
        assay = assays[segment.sourceIdx]
        n_rows = int(segment.localRows.size)
        dest_start = int(segment.destStart)
        dest_stop = dest_start + n_rows
        if _is_missing(assay):
            cumulative[dest_start + 1 : dest_stop + 1] = cumulative[dest_start]
            continue
        column = f"{assay.name}_nFeatures"
        if column in assay.cells.columns:
            counts = np.asarray(
                read_metadata_rows_chunkwise(
                    assay.cells,
                    column,
                    segment.localRows,
                ),
                dtype=np.int64,
            )
        else:
            counts = np.full(
                n_rows,
                int(assay.rawData.shape[1]),
                dtype=np.int64,
            )
        if counts.size != n_rows:
            raise ValueError(
                f"Source assay {assay.name!r} has an invalid {column!r} column"
            )
        segment_cumulative = cumulative[dest_start + 1 : dest_stop + 1]
        np.cumsum(counts, dtype=np.int64, out=segment_cumulative)
        segment_cumulative += cumulative[dest_start]
    n_ordered = row_plan.nCells

    def max_window_nnz(window_rows: int) -> int:
        if n_ordered == 0:
            return 0
        width = min(max(0, int(window_rows)), n_ordered)
        if width == 0:
            return 0
        return int(np.max(cumulative[width:] - cumulative[:-width]))

    present = [assay for assay in assays if not _is_missing(assay)]
    source_dtype = (
        np.result_type(*(assay.rawData.dtype for assay in present))
        if present
        else np.dtype(destination_dtype)
    )
    source_n_feats = max(
        (int(assay.rawData.shape[1]) for assay in present),
        default=0,
    )
    max_decode_bytes = max(
        (
            max(0, int(getter()))
            for assay in present
            if callable(getter := getattr(assay.rawData, "_max_decode_bytes", None))
        ),
        default=0,
    )
    value_candidates = [np.dtype(destination_dtype).itemsize]
    value_candidates.extend(
        np.dtype(assay.rawData.dtype).itemsize
        for assay in assays
        if not _is_missing(assay)
    )
    value_bytes = max(value_candidates)
    resident_bytes = base_resident + profile_bytes

    def extra_producer_bytes(width: int) -> int:
        rows = max(0, int(width))
        dense_bytes = rows * source_n_feats * value_bytes
        # Remap/canonicalize staging for source-width COO indices and values.
        remap_bytes = (
            rows
            * max(1, int(source_n_feats))
            * (value_bytes + 2 * np.dtype(np.int32).itemsize)
        )
        return int(dense_bytes + remap_bytes + max_decode_bytes)

    return _MergeImportRequirements(
        maxWindowNnz=max_window_nnz,
        sourceDtype=np.dtype(source_dtype),
        residentBytes=resident_bytes,
        extraProducerBytes=extra_producer_bytes,
        nnzScanRows=scan_rows,
    )


def preflight_assay_counts(
    spec: ZarrArraySpec,
    assays: list[Any],
    row_plan: RowPlan,
    alignment: FeatureAlignment,
    *,
    resources: ResourceBudget,
    additionalResidentBytes: int = 0,
) -> None:
    """Admit a merge counts write before creating its destination arrays."""
    requirements = _merge_import_requirements(
        assays,
        row_plan,
        alignment,
        spec.dtype,
        resources=resources,
        additionalResidentBytes=additionalResidentBytes,
    )
    resolve_sparse_import_spec(
        (spec,),
        nRows=row_plan.nCells,
        resources=resources,
        maxWindowNnz=requirements.maxWindowNnz,
        sourceDtype=requirements.sourceDtype,
        residentBytes=requirements.residentBytes,
        extraProducerBytes=requirements.extraProducerBytes,
    )


def write_assay_counts(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    assays: list[Any],
    row_plan: RowPlan,
    alignment: FeatureAlignment,
    *,
    resources: ResourceBudget,
    profile: StorageProfile,
    additionalResidentBytes: int = 0,
    io: StorageIoPolicy | None = None,
) -> int:
    """Stream remapped source blocks into the destination counts array."""
    destination = load_count_array(root, assay_name, workspace)
    _ = profile
    requirements = _merge_import_requirements(
        assays,
        row_plan,
        alignment,
        destination.dtype,
        resources=resources,
        additionalResidentBytes=additionalResidentBytes,
    )
    expected_start = 0
    for segment in iter_row_plan_segments(row_plan):
        if segment.destStart != expected_start:
            raise AssertionError(
                "ERROR: Merged block order does not match the cell metadata order."
            )
        expected_start += int(segment.localRows.size)

    plan = resolve_sparse_import_batch(
        (destination,),
        nRows=row_plan.nCells,
        resources=resources,
        maxWindowNnz=requirements.maxWindowNnz,
        sourceDtype=requirements.sourceDtype,
        residentBytes=requirements.residentBytes,
        extraProducerBytes=requirements.extraProducerBytes,
    )
    batch_rows = max(1, int(plan.batchRows))

    def convert_rows(assay_idx: int, perm_order: np.ndarray) -> coo_matrix:
        assay = assays[assay_idx]
        if _is_missing(assay) or int(assay.feats.N) == 0:
            return empty_block_coo(int(perm_order.size), alignment.nFeats)
        block = assay.rawData[np.asarray(perm_order, dtype=np.int64), :]
        return remap_block_to_coo(
            block,
            alignment.featOrderMap[assay_idx],
            alignment.nFeats,
            resources.workers,
            np.dtype(destination.dtype),
        )

    def block_stream() -> Iterator[coo_matrix]:
        # Split each row-plan block into planner-admitted batch widths so peak
        # residency matches resolve_sparse_import_batch without restacking.
        total_batches = sum(
            1
            for _ in iter_row_plan_segments(
                row_plan,
                segment_rows=batch_rows,
            )
        )
        batches = (
            convert_rows(segment.sourceIdx, segment.localRows)
            for segment in iter_row_plan_segments(
                row_plan,
                segment_rows=batch_rows,
            )
        )
        yield from iter_progress(
            batches,
            total=total_batches,
            desc=f"Writing merged assay {assay_name}",
        )

    counter = accumulate_sparse_to_shards(
        destination,
        block_stream(),
        resources=resources,
        residentBytes=requirements.residentBytes,
        producerReserveBytes=plan.producerReserveBytes,
        io=io,
    )
    if counter != row_plan.nCells or expected_start != row_plan.nCells:
        raise AssertionError(
            "ERROR: Mismatch in number of cells in the merged assay. "
            "Please report this issue."
        )
    matrix_path = _matrix_group_path(assay_name, workspace)
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    matrix_group.attrs["complete"] = True
    return counter


def write_assay_counts_t(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    profile: StorageProfile,
    resources: ResourceBudget,
    residentBytes: int = 0,
    policy: CountMatrixPolicy | None = None,
    io: StorageIoPolicy | None = None,
) -> zarr.Array:
    counts = load_count_array(root, assay_name, workspace)
    group_path = _matrix_group_path(assay_name, workspace)
    group = as_zarr_group(root[group_path], name=group_path)
    from ..storage.layout import _group_zarr_format

    if _group_zarr_format(group) < 3:
        raise ValueError(
            "countsT requires a Zarr v3 destination. Repack the store to Zarr v3."
        )
    result = write_counts_t(
        counts,
        group,
        profile=profile,
        resources=resources,
        residentBytes=residentBytes,
        policy=policy,
        io=io,
    )
    logger.debug(f"Wrote countsT for assay {assay_name}")
    return result


def matrix_group_complete(
    root: zarr.Group, assay_name: str, workspace: str | None
) -> bool:
    path = _matrix_group_path(assay_name, workspace)
    if path not in root:
        return False
    group = as_zarr_group(root[path], name=path)
    return bool(group.attrs.get("complete", False))


def counts_t_complete(root: zarr.Group, assay_name: str, workspace: str | None) -> bool:
    path = _matrix_group_path(assay_name, workspace)
    if path not in root:
        return False
    group = as_zarr_group(root[path], name=path)
    if "countsT" not in group:
        return False
    counts_t = group["countsT"]
    return bool(getattr(counts_t, "attrs", {}).get("complete", False))


def validate_assay_counts(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    n_cells: int,
    alignment: FeatureAlignment,
    dtype: str,
    chunks: tuple[int, int],
    shards: tuple[int, int] | None,
) -> str | None:
    """Return why a completed counts component cannot be reused."""
    assay_path = _assay_metadata_path(assay_name, workspace)
    matrix_path = _matrix_group_path(assay_name, workspace)
    if assay_path not in root:
        return f"assay metadata group {assay_path!r} is missing"
    if matrix_path not in root:
        return f"matrix group {matrix_path!r} is missing"

    assay_group = as_zarr_group(root[assay_path], name=assay_path)
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    if matrix_group.attrs.get("complete") is not True:
        return f"matrix group {matrix_path!r} is not complete"
    if "counts" not in matrix_group:
        return f"counts array is missing from {matrix_path!r}"
    counts = as_zarr_array(matrix_group["counts"], name=f"{matrix_path}/counts")
    expected_shape = (int(n_cells), int(alignment.nFeats))
    if tuple(int(value) for value in counts.shape) != expected_shape:
        return (
            f"counts shape for {assay_name!r} is {tuple(counts.shape)}, "
            f"expected {expected_shape}"
        )
    if np.dtype(counts.dtype) != np.dtype(dtype):
        return (
            f"counts dtype for {assay_name!r} is {np.dtype(counts.dtype)}, "
            f"expected {np.dtype(dtype)}"
        )
    actual_chunks = tuple(int(value) for value in counts.chunks)
    if actual_chunks != tuple(chunks):
        return (
            f"counts chunks for {assay_name!r} are {actual_chunks}, "
            f"expected {tuple(chunks)}"
        )
    actual_shards = array_metadata_shards(counts)
    normalized_shards = (
        None if actual_shards is None else tuple(int(value) for value in actual_shards)
    )
    if normalized_shards != shards:
        return (
            f"counts shards for {assay_name!r} are {normalized_shards}, "
            f"expected {shards}"
        )

    if "featureData" not in assay_group:
        return f"featureData is missing from {assay_path!r}"
    feature_group = as_zarr_group(
        assay_group["featureData"],
        name=f"{assay_path}/featureData",
    )
    expected_ids = np.asarray(alignment.mergedFeatsMap["ids"], dtype=str)
    expected_names = np.asarray(alignment.mergedFeatsMap["names"], dtype=str)
    for column, expected in (("ids", expected_ids), ("names", expected_names)):
        if column not in feature_group:
            return f"featureData/{column} is missing for {assay_name!r}"
        feature_array = as_zarr_array(
            feature_group[column],
            name=f"{assay_path}/featureData/{column}",
        )
        actual = np.asarray(feature_array[:], dtype=str)
        if not np.array_equal(actual, expected):
            return f"featureData/{column} does not match for {assay_name!r}"
    if "I" not in feature_group:
        return f"featureData/I is missing for {assay_name!r}"
    included = as_zarr_array(
        feature_group["I"],
        name=f"{assay_path}/featureData/I",
    )
    if tuple(int(value) for value in included.shape) != (alignment.nFeats,):
        return f"featureData/I has the wrong shape for {assay_name!r}"
    if np.dtype(included.dtype) != np.dtype(bool):
        return f"featureData/I has the wrong dtype for {assay_name!r}"
    if not bool(np.asarray(included[:], dtype=bool).all()):
        return f"featureData/I is not fully selected for {assay_name!r}"
    return None


def validate_counts_t(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    n_cells: int,
    n_features: int,
    dtype: str,
) -> str | None:
    """Return why a completed countsT component cannot be reused.

    Prefer :func:`assess_counts_t_reuse` for structured outcomes. This wrapper
    keeps a human-readable reason for blocked-plan messages.
    """
    assessment = assess_counts_t_reuse(
        root,
        assay_name,
        workspace,
        n_cells=n_cells,
        n_features=n_features,
        dtype=dtype,
    )
    if assessment.outcome == "reusable":
        return None
    return assessment.reason


def assess_counts_t_reuse(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    n_cells: int,
    n_features: int,
    dtype: str,
) -> CountsTReuseAssessment:
    """Classify whether an existing ``countsT`` can be reused by merge.

    Outcomes:
    - ``reusable``: complete paired layout matching the planned geometry
    - ``rewrite-layout``: present but not the locked rotateOnce layout
    - ``incomplete``: missing or ``complete`` is not True
    - ``block-shape/dtype``: complete array that disagrees with the merge plan
    """
    from ..storage.count_matrix import require_count_matrix_layout
    from ..storage.schema import load_count_array

    matrix_path = _matrix_group_path(assay_name, workspace)
    if matrix_path not in root:
        return CountsTReuseAssessment(
            outcome="incomplete",
            reason=f"matrix group {matrix_path!r} is missing",
        )
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    if "countsT" not in matrix_group:
        return CountsTReuseAssessment(
            outcome="incomplete",
            reason=f"countsT is missing for {assay_name!r}",
        )
    counts_t = as_zarr_array(
        matrix_group["countsT"],
        name=f"{matrix_path}/countsT",
    )
    if counts_t.attrs.get("complete") is not True:
        return CountsTReuseAssessment(
            outcome="incomplete",
            reason=f"countsT is not complete for {assay_name!r}",
        )
    expected_shape = (int(n_features), int(n_cells))
    actual_shape = tuple(int(value) for value in counts_t.shape)
    if actual_shape != expected_shape:
        return CountsTReuseAssessment(
            outcome="block-shape/dtype",
            reason=(
                f"countsT shape for {assay_name!r} is {actual_shape}, "
                f"expected {expected_shape}"
            ),
        )
    if np.dtype(counts_t.dtype) != np.dtype(dtype):
        return CountsTReuseAssessment(
            outcome="block-shape/dtype",
            reason=(
                f"countsT dtype for {assay_name!r} is {np.dtype(counts_t.dtype)}, "
                f"expected {np.dtype(dtype)}"
            ),
        )
    try:
        counts = load_count_array(root, assay_name, workspace)
        require_count_matrix_layout(matrix_group, counts, counts_t)
    except ValueError as exc:
        return CountsTReuseAssessment(
            outcome="rewrite-layout",
            reason=str(exc),
        )
    return CountsTReuseAssessment(outcome="reusable", reason=None)
