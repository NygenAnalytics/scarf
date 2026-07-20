import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..utils.compute import show_dask_progress
from ..utils.logging import logger
from ..utils.progress import tqdmbar


def to_h5ad(
    assay: Any,
    h5ad_filename: str,
    embeddings_cols: list[str] | None = None,
    skip_recalc_nfeats: bool = True,
    n_threads: int = 4,
) -> None:
    """Save an assay as H5ad file.

    Args:
        assay: Assay to save in H5ad format
        h5ad_filename: Name for the H5ad file to be created.
        embeddings_cols: Columns in cell metadata to be treated as embeddings e. UMAP, tSNE
                         (Default value: ['UMAP', 'tSNE'])
        skip_recalc_nfeats: Skip recalculating nFeatures per cell. (Default value: True)
        n_threads: Number of processing threads to use (Default value: 4)

    Returns:
        None
    """
    import h5py

    def save_attr(group: str, col: str, scarf_col: str, md: Any) -> None:
        d = md.fetch_all(scarf_col)
        d_type = d.dtype
        if np.issubdtype(d_type, np.number) or np.issubdtype(d_type, bool):
            pass
        else:
            d_type = h5py.special_dtype(vlen=str)
        try:
            h5[group].create_dataset(col, data=d.astype(d_type))
        except TypeError:
            logger.warning(f"Dtype issue in {col}, {d.type} ({d_type})")

    h5 = h5py.File(h5ad_filename, "w")
    for i in ["X", "obs", "var", "obsm"]:
        h5.create_group(i)

    # Recalculating nFeature here just to avoid potential issues with stale data.
    if skip_recalc_nfeats is False:
        assay.cells.insert(
            f"{assay.name}_nFeatures",
            show_dask_progress(
                assay.rawData.count_nonzero(axis=1),
                msg="Preflight: recalculating nFeatures",
                nthreads=n_threads,
            ),
            overwrite=True,
        )

    n_feats_per_cell = assay.cells.fetch_all(f"{assay.name}_nFeatures").astype(int)
    tot_counts = int(n_feats_per_cell.sum())

    for i, s in zip(
        ["indptr", "indices", "data"], [assay.cells.N + 1, tot_counts, tot_counts]
    ):
        if i == "data":
            mat_dtype = assay.rawData.dtype
        else:
            mat_dtype = int
        h5["X"].create_dataset(
            i, (s,), chunks=True, compression="gzip", dtype=mat_dtype
        )

    h5["X/indptr"][:] = np.array([0] + list(n_feats_per_cell.cumsum())).astype(int)

    s, e = 0, 0
    for i in tqdmbar(
        assay.rawData.blocks,
        total=assay.rawData.numblocks[0],
        desc="Writing raw counts",
    ):
        i = csr_matrix(i.compute())
        e += i.data.shape[0]
        h5["X/data"][s:e] = i.data
        h5["X/indices"][s:e] = i.indices
        s = e
    attrs = {
        "encoding-type": "csr_matrix",
        "encoding-version": "0.1.0",
        "shape": np.array([assay.cells.N, assay.feats.N]),
    }
    for i, j in attrs.items():
        h5["X"].attrs[i] = j

    out_cols = []
    emb_cols = []
    if embeddings_cols is None:
        embeddings_cols = ["UMAP", "tSNE"]
    for i in assay.cells.columns:
        if i == "ids":
            save_attr("obs", "_index", "ids", assay.cells)
            out_cols.append("_index")
        else:
            is_emb = False
            if len(embeddings_cols) > 0:
                for j in embeddings_cols:
                    if i.startswith(f"{assay.name}_{j}"):
                        emb_cols.append(i)
                        is_emb = True
                        break
            if is_emb is False:
                save_attr("obs", i, i, assay.cells)
                out_cols.append(i)

    attrs = {
        "_index": "_index",
        "column-order": np.array(out_cols, dtype=object),
        "encoding-type": "dataframe",
        "encoding-version": "0.1.0",
    }
    for i, j in attrs.items():
        h5["obs"].attrs[i] = j

    out_cols = []
    for i in assay.feats.columns:
        if i == "ids":
            save_attr("var", "_index", "ids", assay.feats)
            out_cols.append("_index")
        elif i == "names":
            save_attr("var", "gene_short_name", "names", assay.feats)
            out_cols.append("gene_short_name")
        else:
            save_attr("var", i, i, assay.feats)
            out_cols.append(i)

    attrs = {
        "_index": "_index",
        "column-order": np.array(out_cols, dtype=object),
        "encoding-type": "dataframe",
        "encoding-version": "0.1.0",
    }
    for i, j in attrs.items():
        h5["var"].attrs[i] = j

    if len(emb_cols) > 0:
        emb_names = np.array(emb_cols)
        c = pd.Series([x[:-1] for x in emb_names])
        for i in c.unique():
            data = np.array([assay.cells.fetch_all(x) for x in emb_names[c == i]]).T
            h5["obsm"].create_dataset(
                i.lower().replace(f"{assay.name.lower()}_", "X_"), data=data
            )

    h5.close()
    return None


def to_mtx(assay: Any, mtx_directory: str, compress: bool = False) -> None:
    """Save an assay as a Matrix Market directory.

    Args:
        assay: Scarf assay. For example: `ds.RNA`
        mtx_directory: Out directory where MTX file will be saved along with barcodes and features file
        compress: If True, then the files are compressed and saved with .gz extension. (Default value: False).

    Returns:
        None
    """
    from scipy.sparse import coo_matrix

    import gzip

    if os.path.isdir(mtx_directory) is False:
        os.mkdir(mtx_directory)

    n_feats_per_cell = assay.cells.fetch_all(f"{assay.name}_nFeatures").astype(int)
    tot_counts = int(n_feats_per_cell.sum())
    if compress:
        barcodes_fn = "barcodes.tsv.gz"
        features_fn = "features.tsv.gz"
        h = gzip.open(os.path.join(mtx_directory, "matrix.mtx.gz"), "wt")
    else:
        barcodes_fn = "barcodes.tsv"
        features_fn = "genes.tsv"
        h = open(os.path.join(mtx_directory, "matrix.mtx"), "w")
    h.write("%%MatrixMarket matrix coordinate integer general\n% Generated by Scarf\n")
    h.write(f"{assay.feats.N} {assay.cells.N} {tot_counts}\n")
    s = 0
    for i in tqdmbar(assay.rawData.blocks, total=assay.rawData.numblocks[0]):
        i = coo_matrix((i.compute()))
        df = pd.DataFrame({"col": i.col + 1, "row": i.row + s + 1, "d": i.data})
        df.to_csv(h, sep=" ", header=False, index=False, mode="a", lineterminator="\n")
        s += i.shape[0]
    h.close()
    assay.cells.to_pandas_dataframe(["ids"]).to_csv(
        os.path.join(mtx_directory, barcodes_fn), sep="\t", header=False, index=False
    )

    assay.feats.to_pandas_dataframe(["ids", "names"]).to_csv(
        os.path.join(mtx_directory, features_fn), sep="\t", header=False, index=False
    )
