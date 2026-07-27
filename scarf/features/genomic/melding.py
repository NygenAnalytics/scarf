from collections.abc import Iterator

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags

from ...assay import Assay
from ...storage.budget import admitted_worker_count
from ...storage.layout import array_shard_rows
from ...storage.schema import create_zarr_count_assay
from ...storage.sharding import (
    accumulate_sparse_to_shards,
    sparse_matrix_bytes,
    sparse_producer_peak_bytes,
)
from .intervals import create_bed_from_coord_ids, get_feature_mappings

__all__ = ["create_counts_mat", "coordinate_melding"]


def create_counts_mat(
    assay: Assay,
    store: zarr.Array,
    mapping: csc_matrix,
    scalar_coeff: float,
    renormalization: bool,
) -> None:
    """Populate a melded count matrix in a Zarr array."""
    n_docs = int(store.shape[0])
    n_source_features = int(mapping.shape[0])
    mapping_bytes = sparse_matrix_bytes(mapping)
    metadata_peak_bytes = (n_docs + 3 * n_source_features) * np.dtype(
        np.float64
    ).itemsize
    admitted_worker_count(
        assay.resources,
        taskBytes=max(1, metadata_peak_bytes),
        residentBytes=mapping_bytes,
        requested=1,
    )
    n_term_per_doc = assay.cells.fetch_all(assay.name + "_nFeatures")
    n_docs_per_term = assay.feats.fetch_all("nCells")
    idf = np.log2(1 + (n_docs / (n_docs_per_term + 1)))

    def block_stream() -> Iterator[coo_matrix]:
        start = 0
        for block_values in assay.rawData.stream_blocks(
            nthreads=1,
            msg="Melding assay",
            prefetch=1,
        ):
            tf = block_values / n_term_per_doc[
                start : start + block_values.shape[0]
            ].reshape(-1, 1)
            tfidf = tf * idf
            block = (csr_matrix(tfidf) @ mapping).tocsr()
            if renormalization:
                row_sums = np.asarray(block.sum(axis=1)).reshape(-1)
                scale = np.zeros(row_sums.shape[0], dtype=np.float64)
                nonzero = row_sums != 0
                scale[nonzero] = scalar_coeff / row_sums[nonzero]
                block = diags(scale) @ block
            yield block.tocoo()
            start += block_values.shape[0]

    source_rows = min(int(assay.rawData.chunksize[0]), n_docs)
    buffered_rows = min(n_docs, source_rows + array_shard_rows(store))
    n_target_features = int(store.shape[1])
    source_elements = source_rows * n_source_features
    source_itemsize = np.dtype(assay.rawData.dtype).itemsize
    source_working_bytes = (
        source_elements * (source_itemsize + 4 * np.dtype(np.float64).itemsize)
        + (source_rows + 1) * np.dtype(np.int64).itemsize
    )
    resident_bytes = (
        mapping_bytes + n_term_per_doc.nbytes + n_docs_per_term.nbytes + idf.nbytes
    )
    accumulate_sparse_to_shards(
        store,
        block_stream(),
        resources=assay.resources,
        residentBytes=resident_bytes,
        producerReserveBytes=sparse_producer_peak_bytes(
            buffered_rows * n_target_features,
            source_rows * n_target_features,
            np.dtype(store.dtype).itemsize,
        )
        + source_working_bytes,
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
    )
    return None
