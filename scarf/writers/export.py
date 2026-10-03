import os
import re
from collections import defaultdict
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix

from ..metadata.rows import apply_missing_mask, metadata_missing_mask
from ..utils.compute import compute_with_progress
from ..utils.logging import logger


def _encoded(node: Any, encoding: str, version: str) -> Any:
    """Record the AnnData element encoding on an HDF5 dataset or group."""
    node.attrs["encoding-type"] = encoding
    node.attrs["encoding-version"] = version
    return node


def _write_array(group: Any, name: str, values: np.ndarray) -> None:
    """Write a numeric, boolean, or text array as an AnnData element."""
    import h5py

    if values.dtype.kind in "biufc":
        _encoded(group.create_dataset(name, data=values), "array", "0.2.0")
        return
    text = values.astype(str).astype(object)
    _encoded(
        group.create_dataset(name, data=text, dtype=h5py.string_dtype()),
        "string-array",
        "0.2.0",
    )


def _write_masked_column(
    group: Any,
    name: str,
    values: np.ndarray,
    missing: np.ndarray,
) -> None:
    """Write one H5AD column whose masked rows are AnnData missing values.

    Numeric columns become float64 with NaN, boolean columns use the nullable
    boolean encoding, and other columns become categoricals whose masked rows
    have code -1, as AnnData writes a run-aware export.
    """
    if values.dtype.kind in {"f", "i", "u"}:
        _write_array(group, name, apply_missing_mask(values, missing))
        return
    column = group.create_group(name)
    if values.dtype.kind == "b":
        _encoded(column, "nullable-boolean", "0.1.0")
        _write_array(column, "values", values)
        _write_array(column, "mask", missing)
        return
    categories, observed_codes = np.unique(
        values[~missing].astype(str),
        return_inverse=True,
    )
    codes = np.full(len(values), -1, dtype=np.int32)
    codes[~missing] = observed_codes
    _encoded(column, "categorical", "0.2.0")
    column.attrs["ordered"] = False
    _write_array(column, "categories", categories)
    _write_array(column, "codes", codes)


def _embedding_groups(
    columns: list[str],
    assay_name: str,
    prefixes: list[str],
) -> dict[str, list[str]]:
    """Group live embedding columns by prefix in numeric component order."""
    components: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for column in columns:
        match = re.fullmatch(r"(.+?)(\d+)", column)
        if match is None:
            continue
        prefix, component = match.groups()
        if any(prefix.startswith(f"{assay_name}_{name}") for name in prefixes):
            components[prefix].append((int(component), column))
    return {
        prefix: [column for _component, column in sorted(items)]
        for prefix, items in components.items()
    }


def to_h5ad(
    assay: Any,
    h5ad_filename: str,
    embeddings_cols: list[str] | None = None,
    skip_recalc_nfeats: bool = True,
    nthreads: int = 4,
    *,
    run: object | None = None,
) -> None:
    """Save an assay or a completed pipeline run as an H5AD file.

    Rows that a nullable metadata column's linked missing mask flags are
    written as missing values: NaN in a float64 column for numeric columns, a
    nullable boolean for boolean columns, and a missing category for other
    columns.

    Args:
        assay: Assay to save in H5ad format
        h5ad_filename: Name for the H5ad file to be created.
        embeddings_cols: Cell-metadata column prefixes treated as embeddings
                         (for example UMAP, tSNE). When None, uses
                         ``["UMAP", "tSNE"]``. Pass an empty list to skip
                         embeddings.
        skip_recalc_nfeats: Skip a preliminary nonzero-count pass. (Default value: True)
        nthreads: Number of processing threads to use (Default value: 4)
        run: Completed pipeline run opened from the datastore that owns
             ``assay``. The frozen run selections and fields are exported.
             Consecutive frozen UMAP fields are already in ``obsm['X_umap']``
             from ``to_anndata(run=...)``; cluster labels remain in
             ``obs['clusters']``. Live embedding columns and feature-count
             recalculation options do not apply to run export.

    Returns:
        None
    """
    if run is not None:
        from ..storage.pipeline_runs import PipelineRunRecord

        if not isinstance(getattr(run, "_record", None), PipelineRunRecord):
            raise TypeError("run must be a PipelineRun")
        pipeline_run = cast(Any, run)
        pipeline_run._require_completed("H5AD export")
        owner = pipeline_run._owner
        get_assay = getattr(owner, "_get_assay", None)
        to_anndata = getattr(owner, "to_anndata", None)
        if not callable(get_assay) or not callable(to_anndata):
            raise TypeError("run must be opened from a DataStore")
        if get_assay(pipeline_run.assay) is not assay:
            raise ValueError(
                "assay must be the exact run assay owned by the run datastore"
            )
        if embeddings_cols is not None:
            raise ValueError(
                "Run-aware export uses frozen embedding fields; "
                "embeddings_cols cannot be provided"
            )
        if skip_recalc_nfeats is not True:
            raise ValueError(
                "Run-aware export uses the frozen selection; "
                "skip_recalc_nfeats cannot be disabled"
            )
        if nthreads != 4:
            raise ValueError(
                "Run-aware export uses the datastore execution settings; "
                "nthreads cannot be overridden"
            )

        adata = to_anndata(run=pipeline_run)
        if adata is None:
            return None
        adata.write_h5ad(h5ad_filename)
        logger.info(
            f"Exported pipeline run {pipeline_run.run_id} with {adata.n_obs} cells and "
            f"{adata.n_vars} features to {h5ad_filename}"
        )
        return None

    import h5py

    def save_attr(group: str, col: str, scarf_col: str, md: Any) -> bool:
        """Write one metadata column and return whether it was written."""
        d = md.fetch_all(scarf_col)
        mask = metadata_missing_mask(md, scarf_col)
        missing = None if mask is None else np.asarray(mask[:], dtype=bool)
        if missing is not None and missing.any():
            _write_masked_column(h5[group], col, d, missing)
            return True
        try:
            _write_array(h5[group], col, d)
        except TypeError:
            logger.warning(
                f"Skipping metadata column {col!r} with unsupported dtype {d.dtype}"
            )
            return False
        return True

    with h5py.File(h5ad_filename, "w") as h5:
        _encoded(h5, "anndata", "0.1.0")
        for i in ["X", "obs", "var"]:
            h5.create_group(i)
        _encoded(h5.create_group("obsm"), "dict", "0.1.0")

        # The stream defines CSR row boundaries, even when stored QC is stale.
        capacity = 0
        if not skip_recalc_nfeats:
            capacity = int(
                compute_with_progress(
                    assay.rawData.count_nonzero(),
                    msg="Counting nonzero entries",
                    nthreads=nthreads,
                )
            )
        indptr = h5["X"].create_dataset(
            "indptr",
            (assay.cells.N + 1,),
            chunks=True,
            compression="gzip",
            dtype="int64",
        )
        data = h5["X"].create_dataset(
            "data",
            (capacity,),
            maxshape=(None,),
            chunks=(65_536,),
            compression="gzip",
            dtype=assay.rawData.dtype,
        )
        indices = h5["X"].create_dataset(
            "indices",
            (capacity,),
            maxshape=(None,),
            chunks=(65_536,),
            compression="gzip",
            dtype="int64",
        )
        row = offset = 0
        indptr[0] = 0
        for values in assay.rawData.stream_blocks(
            nthreads=nthreads,
            msg="Writing raw counts",
        ):
            block = csr_matrix(values)
            end_row = row + block.shape[0]
            if end_row > assay.cells.N or block.shape[1] != assay.feats.N:
                raise ValueError("Count matrix shape does not match assay metadata")
            end = offset + block.nnz
            if end > data.shape[0]:
                data.resize((end,))
                indices.resize((end,))
            data[offset:end] = block.data
            indices[offset:end] = block.indices
            indptr[row + 1 : end_row + 1] = block.indptr[1:].astype(np.int64) + offset
            row, offset = end_row, end
        if row != assay.cells.N:
            raise ValueError("Count matrix row count does not match assay metadata")
        data.resize((offset,))
        indices.resize((offset,))
        attrs = {
            "encoding-type": "csr_matrix",
            "encoding-version": "0.1.0",
            "shape": np.array([assay.cells.N, assay.feats.N]),
        }
        for i, j in attrs.items():
            h5["X"].attrs[i] = j

        if embeddings_cols is None:
            embeddings_cols = ["UMAP", "tSNE"]
        embeddings = _embedding_groups(
            list(assay.cells.columns),
            assay.name,
            embeddings_cols,
        )
        embedding_columns = {
            column for columns in embeddings.values() for column in columns
        }
        # column-order lists only written columns: AnnData reads every
        # column it names, so a skipped one would make the file unreadable.
        out_cols = []
        for i in assay.cells.columns:
            if i == "ids":
                save_attr("obs", "_index", "ids", assay.cells)
            elif i not in embedding_columns and save_attr("obs", i, i, assay.cells):
                out_cols.append(i)

        # The index element is named by ``_index`` and is not a column.
        attrs = {
            "_index": "_index",
            "column-order": np.array(out_cols, dtype=object),
            "encoding-type": "dataframe",
            "encoding-version": "0.2.0",
        }
        for i, j in attrs.items():
            h5["obs"].attrs[i] = j

        out_cols = []
        for i in assay.feats.columns:
            if i == "ids":
                save_attr("var", "_index", "ids", assay.feats)
            elif i == "names":
                if save_attr("var", "gene_short_name", "names", assay.feats):
                    out_cols.append("gene_short_name")
            elif save_attr("var", i, i, assay.feats):
                out_cols.append(i)

        attrs = {
            "_index": "_index",
            "column-order": np.array(out_cols, dtype=object),
            "encoding-type": "dataframe",
            "encoding-version": "0.2.0",
        }
        for i, j in attrs.items():
            h5["var"].attrs[i] = j

        for prefix, columns in embeddings.items():
            data = np.array([assay.cells.fetch_all(x) for x in columns]).T
            name = prefix.lower().replace(f"{assay.name.lower()}_", "X_")
            _write_array(h5["obsm"], name, data)

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
        compress: If True, then the files are compressed and saved with .gz extension, using Cell Ranger 3
                  names; ``features.tsv.gz`` then also holds a feature-type column. (Default value: False).

    Returns:
        None
    """
    import gzip

    import pandas as pd
    from scipy.sparse import coo_matrix

    from ..assay.classification import is_rna_assay_type

    if os.path.isdir(mtx_directory) is False:
        os.mkdir(mtx_directory)

    tot_counts = int(
        compute_with_progress(
            assay.rawData.count_nonzero(),
            msg="Counting nonzero entries",
            nthreads=assay.nthreads,
        )
    )
    if compress:
        barcodes_fn = "barcodes.tsv.gz"
        features_fn = "features.tsv.gz"
        matrix_path = os.path.join(mtx_directory, "matrix.mtx.gz")
    else:
        barcodes_fn = "barcodes.tsv"
        features_fn = "genes.tsv"
        matrix_path = os.path.join(mtx_directory, "matrix.mtx")
    numeric_type = (
        "integer" if np.issubdtype(assay.rawData.dtype, np.integer) else "real"
    )
    with gzip.open(matrix_path, "wt") if compress else open(matrix_path, "w") as handle:
        handle.write(
            f"%%MatrixMarket matrix coordinate {numeric_type} general\n"
            "% Generated by Scarf\n"
        )
        handle.write(f"{assay.feats.N} {assay.cells.N} {tot_counts}\n")
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
            df.to_csv(
                handle,
                sep=" ",
                header=False,
                index=False,
                mode="a",
                lineterminator="\n",
            )
            s += block.shape[0]
    assay.cells.to_pandas_dataframe(["ids"]).to_csv(
        os.path.join(mtx_directory, barcodes_fn), sep="\t", header=False, index=False
    )

    features = assay.feats.to_pandas_dataframe(["ids", "names"])
    if compress:
        # Cell Ranger 3 feature files carry a third feature-type column.
        if "feature_type" in assay.feats.columns:
            features["feature_type"] = assay.feats.fetch_all("feature_type")
        else:
            features["feature_type"] = (
                "Gene Expression" if is_rna_assay_type(assay) else assay.name
            )
    features.to_csv(
        os.path.join(mtx_directory, features_fn), sep="\t", header=False, index=False
    )
    logger.info(
        f"Exported {assay.cells.N} cells and {assay.feats.N} features "
        f"to {mtx_directory}"
    )
