import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..utils.compute import compute_with_progress
from ..utils.logging import logger


def to_h5ad(
    assay: Any,
    h5ad_filename: str,
    embeddings_cols: list[str] | None = None,
    skip_recalc_nfeats: bool = True,
    nthreads: int = 4,
) -> None:
    """Save an assay as H5ad file.

    Args:
        assay: Assay to save in H5ad format
        h5ad_filename: Name for the H5ad file to be created.
        embeddings_cols: Cell-metadata column prefixes treated as embeddings
                         (for example UMAP, tSNE). When None, uses
                         ``["UMAP", "tSNE"]``. Pass an empty list to skip
                         embeddings.
        skip_recalc_nfeats: Skip recalculating nFeatures per cell. (Default value: True)
        nthreads: Number of processing threads to use (Default value: 4)

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
            logger.warning(
                f"Skipping metadata column {col!r} with unsupported dtype {d.dtype}"
            )

    h5 = h5py.File(h5ad_filename, "w")
    for i in ["X", "obs", "var", "obsm"]:
        h5.create_group(i)

    # Recalculating nFeature here just to avoid potential issues with stale data.
    if skip_recalc_nfeats is False:
        assay.cells.insert(
            f"{assay.name}_nFeatures",
            compute_with_progress(
                assay.rawData.count_nonzero(axis=1),
                msg="Recalculating detected feature counts",
                nthreads=nthreads,
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
    for values in assay.rawData.stream_blocks(
        nthreads=nthreads,
        msg="Writing raw counts",
    ):
        block = csr_matrix(values)
        e += block.data.shape[0]
        h5["X/data"][s:e] = block.data
        h5["X/indices"][s:e] = block.indices
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
            matched = sorted(str(name) for name in emb_names[c == i])
            data = np.array([assay.cells.fetch_all(x) for x in matched]).T
            h5["obsm"].create_dataset(
                i.lower().replace(f"{assay.name.lower()}_", "X_"), data=data
            )

    h5.close()
    logger.info(
        f"Exported {assay.cells.N} cells and {assay.feats.N} features "
        f"to {h5ad_filename}"
    )
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
    for values in assay.rawData.stream_blocks(
        nthreads=assay.nthreads,
        msg="Writing Matrix Market counts",
    ):
        block = coo_matrix(values)
        df = pd.DataFrame(
            {
                "col": block.col + 1,
                "row": block.row + s + 1,
                "d": block.data,
            }
        )
        df.to_csv(h, sep=" ", header=False, index=False, mode="a", lineterminator="\n")
        s += block.shape[0]
    h.close()
    assay.cells.to_pandas_dataframe(["ids"]).to_csv(
        os.path.join(mtx_directory, barcodes_fn), sep="\t", header=False, index=False
    )

    assay.feats.to_pandas_dataframe(["ids", "names"]).to_csv(
        os.path.join(mtx_directory, features_fn), sep="\t", header=False, index=False
    )
    logger.info(
        f"Exported {assay.cells.N} cells and {assay.feats.N} features "
        f"to {mtx_directory}"
    )
