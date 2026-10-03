"""Storage helpers for tests that build Scarf stores directly."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import zarr
from numpy.typing import DTypeLike

from scarf.storage import async_execution
from scarf.storage.identity import CountSummary, finalize_counts


def finalize_test_counts(counts: zarr.Array) -> str:
    """Summarize values already written to ``counts`` and finalize them.

    Writers fill the summary while they write. Tests that assign counts
    directly summarize the stored values here instead.
    """
    summary = CountSummary(counts)
    rows = max(1, int(counts.chunks[0]))
    for start in range(0, int(counts.shape[0]), rows):
        summary.update(start, counts[start : start + rows])
    return finalize_counts(counts, summary=summary)


def write_count_store(
    zarr_loc: str,
    counts: Mapping[str, np.ndarray],
    dtype: DTypeLike,
) -> None:
    """Write a store whose count matrices have the storage dtype ``dtype``.

    ``counts`` maps assay names to cells-by-features matrices over the same
    cells. Each assay takes the preset type of its name, such as RNA, ADT,
    or ATAC, and RNA assays also get ``countsT``. Features are named
    ``f"{assay}{index}"``. The values must be exactly representable in
    ``dtype``, so the stored counts equal them.
    """
    from scarf.storage.schema import create_cell_data, create_zarr_count_assay
    from scarf.storage.stores import load_zarr
    from scarf.writers.counts_t import finalize_writer_counts_t

    storage_dtype = np.dtype(dtype)
    root = load_zarr(zarr_loc=zarr_loc, mode="w")
    n_cells = {len(values) for values in counts.values()}
    if len(n_cells) != 1:
        raise ValueError("Every assay must have the same cells")
    cell_ids = np.array([f"cell{index}" for index in range(n_cells.pop())])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    for assay, values in counts.items():
        stored = np.asarray(values).astype(storage_dtype)
        if not np.array_equal(stored, values):
            raise ValueError(f"{assay} counts do not fit {storage_dtype}")
        feature_ids = np.array([f"{assay}{index}" for index in range(stored.shape[1])])
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            len(cell_ids),
            feat_ids=feature_ids,
            feat_names=feature_ids,
            dtype=storage_dtype.name,
        )
        array[:] = stored
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, None)


def insert_nullable_cell_column(
    datastore: Any,
    name: str,
    values: np.ndarray,
    missing: np.ndarray,
) -> None:
    """Add a cell metadata column whose ``missing`` rows carry no value."""
    cell_data = datastore.zw["cellData"]
    missing_name = f"__scarf_missing__{name}"
    cell_data.create_array(name, data=np.asarray(values))
    cell_data.create_array(missing_name, data=np.asarray(missing, dtype=bool))
    cell_data[name].attrs["missing_mask"] = missing_name


def reset_zarr_runtime() -> None:
    """Restore Zarr's process defaults so the next storage call plans afresh."""
    zarr.config.set({"threading.max_workers": None, "async.concurrency": 10})
    async_execution._HOST_THREAD_CEILING = None
