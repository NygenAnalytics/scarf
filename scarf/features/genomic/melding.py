from collections.abc import Iterator

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csc_matrix

from ...assay import Assay
from ...storage.budget import ResourceBudget
from ...storage.count_matrix import (
    DEFAULT_COUNT_MATRIX_POLICY,
    CountMatrixPolicy,
)
from ...storage.layout import array_shard_rows
from ...storage.schema import create_zarr_count_assay
from ...storage.sharding import sparse_matrix_bytes, write_dense_from_row_batches
from ...utils.arrays import array_digest
from .intervals import create_bed_from_coord_ids, get_feature_mappings

__all__ = ["create_counts_mat", "coordinate_melding"]


def _source_working_bytes(
    source_rows: int,
    n_source_features: int,
    source_itemsize: int,
    *,
    nTargetFeatures: int = 0,
    storeItemsize: int = 0,
) -> int:
    rows = max(0, int(source_rows))
    source_elements = rows * max(0, int(n_source_features))
    dest_elements = rows * max(0, int(nTargetFeatures))
    float_bytes = np.dtype(np.float64).itemsize
    dest_itemsize = max(1, int(storeItemsize)) if storeItemsize else float_bytes
    return (
        source_elements * (max(1, int(source_itemsize)) + 2 * float_bytes)
        + dest_elements * dest_itemsize
        + (rows + 1) * np.dtype(np.int64).itemsize
    )


def _dest_band_hold_bytes(n_rows: int, n_cols: int, itemsize: int) -> int:
    """Align buffer plus one dense destination write of that band."""
    dense = max(1, int(n_rows)) * max(1, int(n_cols)) * max(1, int(itemsize))
    encoded = dense + dense // 128 + 1024
    write_peak = dense + dense + 2 * encoded + 1024
    return dense + write_peak


def _meld_band_cost(
    source_rows: int,
    *,
    nSourceFeatures: int,
    nTargetFeatures: int,
    sourceItemsize: int,
    storeItemsize: int,
    destRows: int | None = None,
) -> int:
    dest_rows = int(source_rows) if destRows is None else max(1, int(destRows))
    return _source_working_bytes(
        source_rows,
        nSourceFeatures,
        sourceItemsize,
        nTargetFeatures=nTargetFeatures,
        storeItemsize=storeItemsize,
    ) + _dest_band_hold_bytes(dest_rows, nTargetFeatures, storeItemsize)


def _max_meld_band_rows(
    *,
    memoryBytes: int,
    nDocs: int,
    nSourceFeatures: int,
    nTargetFeatures: int,
    sourceItemsize: int,
    storeItemsize: int,
    mappingBytes: int,
    decodeBytes: int,
    extraResidentBytes: int,
    preferredRows: int,
    maxRows: int,
    destRows: int | None = None,
) -> int:
    resident = (
        max(0, int(mappingBytes))
        + max(0, int(decodeBytes))
        + max(0, int(extraResidentBytes))
    )
    available = int(memoryBytes) - resident
    limit = max(1, min(int(nDocs), int(maxRows)))

    def cost(rows: int) -> int:
        return _meld_band_cost(
            rows,
            nSourceFeatures=nSourceFeatures,
            nTargetFeatures=nTargetFeatures,
            sourceItemsize=sourceItemsize,
            storeItemsize=storeItemsize,
            destRows=destRows,
        )

    if available < cost(1):
        raise MemoryError(
            "Gene-score melding needs about "
            f"{resident + cost(1)} bytes, but the operation limit is "
            f"{memoryBytes} bytes"
        )
    preferred = max(1, min(int(preferredRows), limit))
    if cost(preferred) <= available:
        return preferred
    low = 1
    high = preferred
    while low < high:
        mid = (low + high + 1) // 2
        if cost(mid) <= available:
            low = mid
        else:
            high = mid - 1
    return max(1, low)


def _meld_count_matrix_policy(
    *,
    nCells: int,
    nFeats: int,
    dtype: str,
    bandRows: int,
) -> CountMatrixPolicy:
    itemsize = max(1, int(np.dtype(dtype).itemsize))
    rows = max(1, min(int(bandRows), max(1, int(nCells))))
    cols = max(1, int(nFeats))
    unit_bytes = max(1, rows * cols * itemsize)
    chunk_bytes = min(unit_bytes, DEFAULT_COUNT_MATRIX_POLICY.chunkBytes)
    return CountMatrixPolicy(unitBytes=unit_bytes, chunkBytes=max(1, chunk_bytes))


def create_counts_mat(
    assay: Assay,
    store: zarr.Array,
    mapping: csc_matrix,
    scalar_coeff: float,
    renormalization: bool,
    *,
    idf_cell_idx: np.ndarray | None = None,
) -> None:
    """Populate a melded count matrix in a Zarr array."""
    n_docs = int(store.shape[0])
    n_source_features = int(mapping.shape[0])
    n_target_features = int(store.shape[1])
    mapping_bytes = sparse_matrix_bytes(mapping)
    source_itemsize = np.dtype(assay.rawData.dtype).itemsize
    decode_bytes = assay.rawData._max_decode_bytes()
    if idf_cell_idx is None:
        selected_cells = None
        selected_cell_count = n_docs
    else:
        raw_selected_cells = np.asarray(idf_cell_idx)
        if raw_selected_cells.dtype == bool or not np.issubdtype(
            raw_selected_cells.dtype,
            np.integer,
        ):
            raise TypeError("idf_cell_idx must contain integer indices")
        selected_cells = np.asarray(raw_selected_cells, dtype=np.int64)
        if selected_cells.ndim != 1:
            raise ValueError("idf_cell_idx must be one-dimensional")
        if len(selected_cells) == 0:
            raise ValueError("Gene-score IDF requires at least one selected cell")
        if np.any(selected_cells < 0) or np.any(selected_cells >= n_docs):
            raise IndexError("idf_cell_idx contains an out-of-range cell")
        selected_cells = np.unique(selected_cells)
        selected_cell_count = len(selected_cells)

    n_term_per_doc = np.asarray(
        assay.cells.fetch_all(assay.name + "_nCounts"),
        dtype=np.float64,
    )
    n_term_per_doc[n_term_per_doc == 0] = 1
    selected_mask: np.ndarray | None = None
    if selected_cell_count == n_docs:
        n_docs_per_term = np.asarray(
            assay.feats.fetch_all("nCells"),
            dtype=np.float64,
        )
    else:
        assert selected_cells is not None
        selected_mask = np.zeros(n_docs, dtype=bool)
        selected_mask[selected_cells] = True
        n_docs_per_term = np.zeros(n_source_features, dtype=np.int64)
        feature_temporaries = n_source_features * np.dtype(np.int64).itemsize
        static_bytes = (
            mapping_bytes
            + selected_cells.nbytes
            + selected_mask.nbytes
            + n_term_per_doc.nbytes
            + n_docs_per_term.nbytes
            + feature_temporaries
        )
        current_rows = min(int(assay.rawData.chunksize[0]), n_docs)
        selected_copy_per_row = n_source_features * source_itemsize
        stream_owned_per_row = n_source_features * source_itemsize
        working_bytes_per_row = selected_copy_per_row + stream_owned_per_row
        available_bytes = int(assay.resources.memoryBytes) - static_bytes - decode_bytes
        if available_bytes < working_bytes_per_row:
            required_bytes = static_bytes + decode_bytes + working_bytes_per_row
            raise MemoryError(
                "Gene-score document frequency needs about "
                f"{required_bytes} bytes for one row, but the operation limit is "
                f"{assay.resources.memoryBytes} bytes"
            )
        document_frequency_rows = min(
            current_rows,
            available_bytes // working_bytes_per_row,
        )
        document_frequency_data = assay.rawData._with_block_size(
            document_frequency_rows
        )
        stream_resident_bytes = (
            static_bytes + document_frequency_rows * selected_copy_per_row
        )
        row_offset = 0
        for block_values in document_frequency_data._stream_blocks(
            nthreads=assay.nthreads,
            msg="Computing gene-score document frequency",
            prefetch=1,
            row_mask=None,
            resident_bytes=stream_resident_bytes,
        ):
            row_stop = row_offset + block_values.shape[0]
            block_mask = selected_mask[row_offset:row_stop]
            if block_mask.any():
                n_docs_per_term += np.count_nonzero(
                    block_values[block_mask],
                    axis=0,
                )
            row_offset = row_stop
        if row_offset != n_docs:
            raise RuntimeError(
                f"Gene-score document-frequency stream produced {row_offset} rows; "
                f"expected {n_docs}"
            )
    idf = np.log2(1 + (selected_cell_count / (n_docs_per_term + 1)))
    del n_docs_per_term, selected_cells, selected_mask

    shard_rows = array_shard_rows(store)
    store_itemsize = np.dtype(store.dtype).itemsize
    resident_bytes = mapping_bytes + n_term_per_doc.nbytes + idf.nbytes
    source_rows = _max_meld_band_rows(
        memoryBytes=int(assay.resources.memoryBytes),
        nDocs=n_docs,
        nSourceFeatures=n_source_features,
        nTargetFeatures=n_target_features,
        sourceItemsize=source_itemsize,
        storeItemsize=store_itemsize,
        mappingBytes=mapping_bytes,
        decodeBytes=decode_bytes,
        extraResidentBytes=n_term_per_doc.nbytes + idf.nbytes,
        preferredRows=min(int(assay.rawData.chunksize[0]), n_docs, shard_rows),
        maxRows=min(n_docs, shard_rows),
        destRows=shard_rows,
    )

    source_data = assay.rawData._with_block_size(source_rows)
    source_stream_resident = resident_bytes + _source_working_bytes(
        source_rows,
        n_source_features,
        source_itemsize,
        nTargetFeatures=n_target_features,
        storeItemsize=store_itemsize,
    )

    def block_stream() -> Iterator[np.ndarray]:
        start = 0
        for block_values in source_data._stream_blocks(
            nthreads=1,
            msg="Melding assay",
            prefetch=1,
            row_mask=None,
            resident_bytes=source_stream_resident,
        ):
            row = 0
            while row < block_values.shape[0]:
                stop = min(row + source_rows, block_values.shape[0])
                values = np.asarray(block_values[row:stop])
                tf = values / n_term_per_doc[start : start + values.shape[0]].reshape(
                    -1, 1
                )
                tfidf = tf * idf
                block = np.asarray(tfidf @ mapping, dtype=np.float64)
                if renormalization:
                    row_sums = block.sum(axis=1)
                    scale = np.zeros(row_sums.shape[0], dtype=np.float64)
                    nonzero = row_sums != 0
                    scale[nonzero] = scalar_coeff / row_sums[nonzero]
                    block *= scale[:, None]
                yield block
                start += values.shape[0]
                row = stop

    write_dense_from_row_batches(
        store,
        block_stream(),
        resources=ResourceBudget(assay.resources.memoryBytes, 1),
        msg="Writing gene scores",
        io=getattr(assay, "storageIo", None),
    )


def coordinate_melding(
    assay: Assay,
    workspace: str | None,
    feature_bed: pd.DataFrame,
    new_assay_name: str,
    peaks_col: str = "ids",
    scalar_coeff: float = 1e5,
    renormalization: bool = True,
    peaks_coords: np.ndarray | None = None,
    idf_cell_idx: np.ndarray | None = None,
) -> None:
    """Transfer coordinate-based assay values to overlapping external features.

    Args:
        assay: Source assay whose features have genomic coordinates.
        workspace: Workspace name. None uses the legacy layout.
        feature_bed: External interval table used as the meld target.
        new_assay_name: Name of the assay group to create.
        peaks_col: Feature-metadata column holding source coordinates.
        scalar_coeff: Scaling coefficient applied during melding.
        renormalization: If True, rescale melded values after mapping.
        peaks_coords: Optional precomputed source coordinates. When None,
                      values are read from ``peaks_col``.
        idf_cell_idx: Optional cell indices used for IDF statistics.

    Returns:
        None
    """
    if peaks_coords is None:
        peaks_coords = assay.feats.fetch_all(peaks_col)
    peaks_bed = create_bed_from_coord_ids(peaks_coords)
    feat_ids, feat_names, mapping = get_feature_mappings(peaks_bed, feature_bed)

    from ...storage.stores import zarr_group_root
    from ...storage.profiles import resolve_storage_profile

    n_cells = int(assay.rawData.shape[0])
    n_source_features = int(mapping.shape[0])
    n_target_features = int(len(feat_ids))
    store_dtype = "float"
    band_rows = _max_meld_band_rows(
        memoryBytes=int(assay.resources.memoryBytes),
        nDocs=n_cells,
        nSourceFeatures=n_source_features,
        nTargetFeatures=n_target_features,
        sourceItemsize=int(np.dtype(assay.rawData.dtype).itemsize),
        storeItemsize=int(np.dtype(store_dtype).itemsize),
        mappingBytes=sparse_matrix_bytes(mapping),
        decodeBytes=assay.rawData._max_decode_bytes(),
        extraResidentBytes=(
            n_cells * np.dtype(np.float64).itemsize
            + n_source_features * np.dtype(np.float64).itemsize
        ),
        preferredRows=min(int(assay.rawData.chunksize[0]), n_cells),
        maxRows=n_cells,
    )
    store_root = zarr_group_root(assay.z, mode="r+")
    group = create_zarr_count_assay(
        z=store_root,
        assay_name=new_assay_name,
        workspace=workspace,
        n_cells=n_cells,
        feat_ids=feat_ids,
        feat_names=feat_names,
        dtype=store_dtype,
        profile=resolve_storage_profile(store_root.store),
        policy=_meld_count_matrix_policy(
            nCells=n_cells,
            nFeats=n_target_features,
            dtype=store_dtype,
            bandRows=band_rows,
        ),
    )

    create_counts_mat(
        assay=assay,
        store=group,
        mapping=mapping,
        scalar_coeff=scalar_coeff,
        renormalization=renormalization,
        idf_cell_idx=idf_cell_idx,
    )
    selected_cells = (
        np.arange(assay.rawData.shape[0], dtype=np.int64)
        if idf_cell_idx is None
        else np.unique(np.asarray(idf_cell_idx, dtype=np.int64))
    )
    assay_group_path = (
        new_assay_name if workspace is None else f"{workspace}/{new_assay_name}"
    )
    assay_group = store_root[assay_group_path]
    assay_group.attrs["idfCellIndexDigest"] = array_digest(selected_cells)
    assay_group.attrs["idfCellCount"] = int(len(selected_cells))
    assay_group.attrs["sourceAssay"] = assay.name
    assay_group.attrs["tfDenominator"] = "total_counts"
    return None
