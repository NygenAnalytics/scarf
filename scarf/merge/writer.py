from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import zarr
from scipy.sparse import coo_matrix

from ..storage.budget import ResourceBudget
from ..storage.identity import CountSummary, finalize_counts, load_count_summaries
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.layout import ZarrArraySpec
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
from .row_plan import RowPlan, iter_row_plan_segments


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


def _row_nnz_load_bytes(assay: Any) -> int:
    """Bound loading one source's count summaries for the NNZ profile."""
    n_rows, n_columns = (int(value) for value in assay.rawData.shape)
    itemsize = np.dtype(np.int64).itemsize
    # Row sums, row positives, and column positives are loaded together, and
    # decoding one of them briefly holds a decoded and an encoded copy.
    return itemsize * (2 * n_rows + n_columns + 2 * max(n_rows, n_columns))


def _cumulative_row_nnz(
    assays: list[Any | None],
    row_plan: RowPlan,
    resources: ResourceBudget,
    *,
    resident_bytes: int,
) -> np.ndarray:
    """Return cumulative nonzero counts over merged rows.

    Each row count is the source row's positive-entry count, read once per
    source from its saved count summaries. Missing sources contribute zeros.
    """
    profile_bytes = (row_plan.nCells + 1) * np.dtype(np.int64).itemsize
    needed = (
        max(0, int(resident_bytes))
        + profile_bytes
        + max(
            (_row_nnz_load_bytes(assay) for assay in assays if assay is not None),
            default=0,
        )
    )
    if needed > int(resources.memoryBytes):
        raise MemoryError(
            f"Merged assay NNZ profile needs about {needed} bytes, but the "
            f"operation limit is {int(resources.memoryBytes)} bytes"
        )
    cumulative = np.zeros(row_plan.nCells + 1, dtype=np.int64)
    row_nnz = cumulative[1:]
    for source_idx, assay in enumerate(assays):
        if assay is None:
            continue
        matrix = assay.matrixGroup
        counts = as_zarr_array(matrix["counts"], name="counts")
        _, row_positive, _ = load_count_summaries(matrix, counts)
        for segment in iter_row_plan_segments(row_plan):
            if segment.sourceIdx != source_idx:
                continue
            stop = segment.destStart + int(segment.localRows.size)
            row_nnz[segment.destStart : stop] = row_positive[segment.localRows]
        del row_positive
    np.cumsum(row_nnz, out=row_nnz)
    return cumulative


def _merge_import_requirements(
    assays: list[Any | None],
    row_plan: RowPlan,
    alignment: FeatureAlignment,
    destination_dtype: Any,
    *,
    resources: ResourceBudget,
    additionalResidentBytes: int = 0,
) -> _MergeImportRequirements:
    # The destination CountSummary exists for the whole write, so planning and
    # execution both count it through this one function.
    base_resident = (
        row_plan.resident_bytes()
        + alignment.resident_bytes()
        + CountSummary.nbytes_for(row_plan.nCells, alignment.nFeats)
        + max(0, int(additionalResidentBytes))
    )
    cumulative = _cumulative_row_nnz(
        assays,
        row_plan,
        resources,
        resident_bytes=base_resident,
    )
    n_ordered = row_plan.nCells

    def max_window_nnz(window_rows: int) -> int:
        if n_ordered == 0:
            return 0
        width = min(max(0, int(window_rows)), n_ordered)
        if width == 0:
            return 0
        return int(np.max(cumulative[width:] - cumulative[:-width]))

    present = [assay for assay in assays if assay is not None]
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
    value_bytes = max(
        [
            np.dtype(destination_dtype).itemsize,
            *(np.dtype(assay.rawData.dtype).itemsize for assay in present),
        ]
    )
    resident_bytes = base_resident + int(cumulative.nbytes)

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
    )


def preflight_assay_counts(
    spec: ZarrArraySpec,
    assays: list[Any | None],
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
    assays: list[Any | None],
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
    summary = CountSummary(destination)
    requirements = _merge_import_requirements(
        assays,
        row_plan,
        alignment,
        destination.dtype,
        resources=resources,
        additionalResidentBytes=additionalResidentBytes,
    )
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
        if assay is None or int(assay.feats.N) == 0:
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
        countSummary=summary,
    )
    if counter != row_plan.nCells:
        raise AssertionError(
            "ERROR: Mismatch in number of cells in the merged assay. "
            "Please report this issue."
        )
    matrix_path = _matrix_group_path(assay_name, workspace)
    matrix_group = as_zarr_group(root[matrix_path], name=matrix_path)
    finalize_counts(destination, summary=summary)
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
    from ..assay.classification import default_feature_sets

    metadata_path = _assay_metadata_path(assay_name, workspace)
    result = write_counts_t(
        counts,
        group,
        profile=profile,
        resources=resources,
        residentBytes=residentBytes,
        policy=policy,
        io=io,
        overwrite=True,
        featureSets=default_feature_sets(
            as_zarr_group(root[metadata_path], name=metadata_path)
        ),
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
    from ..storage.counts_t_contract import validate_count_matrix

    try:
        validate_count_matrix(matrix_group, require_transpose=False)
    except ValueError as error:
        return str(error)
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
    from ..storage.counts_t_contract import validate_count_matrix

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
        validate_count_matrix(matrix_group, require_transpose=True)
    except ValueError as exc:
        return CountsTReuseAssessment(
            outcome="rewrite-layout",
            reason=str(exc),
        )
    return CountsTReuseAssessment(outcome="reusable", reason=None)
