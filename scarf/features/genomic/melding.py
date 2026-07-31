from collections.abc import Iterator

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags

from ...assay import Assay
from ...storage.layout import array_shard_rows
from ...storage.schema import create_zarr_count_assay
from ...storage.sharding import (
    accumulate_sparse_to_shards,
    sparse_matrix_bytes,
    sparse_producer_peak_bytes,
)
from ...utils.arrays import array_digest
from .intervals import create_bed_from_coord_ids, get_feature_mappings

__all__ = ["create_counts_mat", "coordinate_melding"]


def _source_working_bytes(
    source_rows: int,
    n_source_features: int,
    source_itemsize: int,
) -> int:
    source_elements = max(0, int(source_rows)) * max(0, int(n_source_features))
    return (
        source_elements * (source_itemsize + 4 * np.dtype(np.float64).itemsize)
        + (max(0, int(source_rows)) + 1) * np.dtype(np.int64).itemsize
    )


def _producer_reserve_bytes(
    *,
    source_rows: int,
    shard_rows: int,
    n_docs: int,
    n_source_features: int,
    n_target_features: int,
    source_itemsize: int,
    store_itemsize: int,
    decode_bytes: int,
) -> int:
    buffered_rows = min(n_docs, max(0, int(source_rows)) + max(1, int(shard_rows)))
    return (
        sparse_producer_peak_bytes(
            buffered_rows * n_target_features,
            max(0, int(source_rows)) * n_target_features,
            store_itemsize,
        )
        + _source_working_bytes(
            source_rows,
            n_source_features,
            source_itemsize,
        )
        + max(0, int(decode_bytes))
    )


def _max_source_rows_for_budget(
    *,
    available_bytes: int,
    shard_rows: int,
    n_docs: int,
    n_source_features: int,
    n_target_features: int,
    source_itemsize: int,
    store_itemsize: int,
    preferred_rows: int,
    decode_bytes: int,
) -> int:
    preferred = max(1, min(int(preferred_rows), int(n_docs)))
    if preferred <= 1:
        return 1
    if (
        _producer_reserve_bytes(
            source_rows=preferred,
            shard_rows=shard_rows,
            n_docs=n_docs,
            n_source_features=n_source_features,
            n_target_features=n_target_features,
            source_itemsize=source_itemsize,
            store_itemsize=store_itemsize,
            decode_bytes=decode_bytes,
        )
        <= available_bytes
    ):
        return preferred

    low = 1
    high = preferred
    while low < high:
        mid = (low + high + 1) // 2
        reserve = _producer_reserve_bytes(
            source_rows=mid,
            shard_rows=shard_rows,
            n_docs=n_docs,
            n_source_features=n_source_features,
            n_target_features=n_target_features,
            source_itemsize=source_itemsize,
            store_itemsize=store_itemsize,
            decode_bytes=decode_bytes,
        )
        if reserve <= available_bytes:
            low = mid
        else:
            high = mid - 1
    return max(1, low)


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
    preferred_rows = min(int(assay.rawData.chunksize[0]), n_docs)
    resident_bytes = mapping_bytes + n_term_per_doc.nbytes + idf.nbytes
    available_bytes = max(1, int(assay.resources.memoryBytes) - resident_bytes)
    # Leave headroom above the static producer estimate for sparse band bytes
    # that accumulate in write_sparse_bands and for densifying one destination
    # shard when it is flushed (_sparse_task_working_bytes).
    dense_shard_bytes = max(1, int(shard_rows)) * n_target_features * store_itemsize
    write_headroom = max(64 * 1024 * 1024, 4 * dense_shard_bytes)
    producer_budget = max(1, available_bytes - write_headroom)
    source_rows = _max_source_rows_for_budget(
        available_bytes=producer_budget,
        shard_rows=shard_rows,
        n_docs=n_docs,
        n_source_features=n_source_features,
        n_target_features=n_target_features,
        source_itemsize=source_itemsize,
        store_itemsize=store_itemsize,
        preferred_rows=preferred_rows,
        decode_bytes=decode_bytes,
    )
    producer_reserve = _producer_reserve_bytes(
        source_rows=source_rows,
        shard_rows=shard_rows,
        n_docs=n_docs,
        n_source_features=n_source_features,
        n_target_features=n_target_features,
        source_itemsize=source_itemsize,
        store_itemsize=store_itemsize,
        decode_bytes=decode_bytes,
    )
    if producer_reserve > producer_budget:
        raise MemoryError(
            "Gene-score melding needs about "
            f"{resident_bytes + producer_reserve + write_headroom} bytes, but "
            f"the operation limit is {assay.resources.memoryBytes} bytes"
        )

    source_data = assay.rawData._with_block_size(source_rows)
    stream_charged_bytes = (
        source_rows * n_source_features * source_itemsize + decode_bytes
    )
    source_stream_resident = resident_bytes + max(
        0,
        producer_reserve - stream_charged_bytes,
    )

    def block_stream() -> Iterator[coo_matrix]:
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
                values = block_values[row:stop]
                tf = values / n_term_per_doc[start : start + values.shape[0]].reshape(
                    -1, 1
                )
                tfidf = tf * idf
                block = (csr_matrix(tfidf) @ mapping).tocsr()
                if renormalization:
                    row_sums = np.asarray(block.sum(axis=1)).reshape(-1)
                    scale = np.zeros(row_sums.shape[0], dtype=np.float64)
                    nonzero = row_sums != 0
                    scale[nonzero] = scalar_coeff / row_sums[nonzero]
                    block = diags(scale) @ block
                yield block.tocoo()
                start += values.shape[0]
                row = stop

    accumulate_sparse_to_shards(
        store,
        block_stream(),
        resources=assay.resources,
        residentBytes=resident_bytes,
        producerReserveBytes=producer_reserve,
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
    """Transfer coordinate-based assay values to overlapping external features."""
    if peaks_coords is None:
        peaks_coords = assay.feats.fetch_all(peaks_col)
    peaks_bed = create_bed_from_coord_ids(peaks_coords)
    feat_ids, feat_names, mapping = get_feature_mappings(peaks_bed, feature_bed)

    from ...storage.stores import zarr_group_root
    from ...storage.profiles import resolve_storage_profile

    store_root = zarr_group_root(assay.z, mode="r+")
    group = create_zarr_count_assay(
        z=store_root,
        assay_name=new_assay_name,
        workspace=workspace,
        n_cells=assay.rawData.shape[0],
        feat_ids=feat_ids,
        feat_names=feat_names,
        dtype="float",
        profile=resolve_storage_profile(store_root.store),
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
