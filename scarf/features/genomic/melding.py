from collections.abc import Iterator

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags

from ...assay import Assay
from ...storage.schema import create_zarr_count_assay, finalize_counts
from ...storage.sharding import accumulate_sparse_to_shards
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
    n_term_per_doc = assay.cells.fetch_all(assay.name + "_nFeatures")
    n_docs = n_term_per_doc.shape[0]
    n_docs_per_term = assay.feats.fetch_all("nCells")
    idf = np.log2(1 + (n_docs / (n_docs_per_term + 1)))

    def block_stream() -> Iterator[coo_matrix]:
        start = 0
        for block_values in assay.rawData.stream_blocks(msg="Melding assay"):
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

    accumulate_sparse_to_shards(store, block_stream())


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

    store_root = zarr_group_root(assay.z, mode="r+")
    group = create_zarr_count_assay(
        z=store_root,
        assay_name=new_assay_name,
        workspace=workspace,
        chunk_size=assay.rawData.chunksize,
        n_cells=assay.rawData.shape[0],
        feat_ids=feat_ids,
        feat_names=feat_names,
        dtype="float",
    )

    create_counts_mat(
        assay=assay,
        store=group,
        mapping=mapping,
        scalar_coeff=scalar_coeff,
        renormalization=renormalization,
    )
    finalize_counts(store_root, new_assay_name, workspace)
    return None
