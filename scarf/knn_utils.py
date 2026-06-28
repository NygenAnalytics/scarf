"""Utility functions for running the KNN algorithm."""

from collections.abc import Generator
from typing import cast

import numpy as np
import pandas as pd
from numba import jit
from scipy.sparse import csr_matrix, coo_matrix
import zarr

from .ann import AnnStream
from .utils import tqdmbar, controlled_compute, prefetch_blocks
from .storage.zarr_store import profile_prefetch_depth
from .writers import create_zarr_dataset

__all__ = [
    "self_query_knn",
    "smoothen_dists",
    "export_knn_to_mtx",
    "run_sgtsne",
    "merge_graphs",
    "wnn_integration",
]


def self_query_knn(
    ann_obj: AnnStream, store: zarr.Group, chunk_size: int, nthreads: int
) -> float:
    """Constructs KNN graph.

    Args:
        ann_obj: Fitted AnnStream with reducer and ANN index.
        store: Zarr group where ``indices`` and ``distances`` arrays are written.
        chunk_size: Row chunk size for output Zarr arrays.
        nthreads: Number of threads to use.

    Returns:
        Approximate recall percentage (fraction of queries with a self neighbor).
    """

    def get_transformed_data() -> Generator[np.ndarray, None, None]:
        msg = "Identifying neighbors"
        if ann_obj.embeddings is not None:
            bs = ann_obj.batchSize
            n_blocks = int(np.ceil(ann_obj.nCells / bs))
            for start in tqdmbar(
                range(0, ann_obj.nCells, bs),
                desc=msg,
                total=n_blocks,
            ):
                end = min(start + bs, ann_obj.nCells)
                yield ann_obj.embeddings[start:end]
            return
        if ann_obj.harmonizedData is None:
            source = ann_obj.data

            def transform(block: np.ndarray) -> np.ndarray:
                return ann_obj.reducer(controlled_compute(block, nthreads))
        else:
            source = ann_obj.harmonizedData

            def transform(block: np.ndarray) -> np.ndarray:
                return controlled_compute(block, nthreads)

        blocks = prefetch_blocks(
            source.blocks,
            transform,
            max_ahead=profile_prefetch_depth(),
        )
        yield from tqdmbar(blocks, desc=msg, total=source.numblocks[0])

    from threadpoolctl import threadpool_limits

    n_cells, n_neighbors = ann_obj.nCells, ann_obj.k
    z_knn = create_zarr_dataset(
        store, "indices", (chunk_size,), "u8", (n_cells, n_neighbors)
    )
    z_dist = create_zarr_dataset(
        store, "distances", (chunk_size,), "f8", (n_cells, n_neighbors)
    )
    nsample_start = 0
    tnm = 0  # Number of missed recall
    with threadpool_limits(limits=nthreads):
        for i in get_transformed_data():
            nsample_end = nsample_start + i.shape[0]
            ki, kv, nm = cast(
                tuple[np.ndarray, np.ndarray, int],
                ann_obj.transform_ann(
                    i,
                    k=n_neighbors,
                    self_indices=np.arange(nsample_start, nsample_end),
                ),
            )
            z_knn[nsample_start:nsample_end, :] = ki
            z_dist[nsample_start:nsample_end, :] = kv
            nsample_start = nsample_end
            tnm += nm
    recall = ann_obj.data.shape[0] - tnm
    recall_pct = 100.0 * recall / ann_obj.data.shape[0]
    return recall_pct


def _is_umap_version_new() -> bool:
    import umap
    from packaging import version

    if version.parse(umap.__version__) >= version.parse("0.5.0"):
        return True
    else:
        return False


def _patch_null_weights(
    zgw: zarr.Array,
    null_positions: list[int],
    fill_value: float,
    patch_chunk: int,
) -> None:
    """Patch zero edge weights without loading the full weights array."""
    if not null_positions:
        return
    null_positions_arr = np.asarray(null_positions, dtype=np.int64)
    n_weights = zgw.shape[0]
    for chunk_start in range(0, n_weights, patch_chunk):
        chunk_end = min(chunk_start + patch_chunk, n_weights)
        in_chunk = (null_positions_arr >= chunk_start) & (
            null_positions_arr < chunk_end
        )
        if not np.any(in_chunk):
            continue
        local_idx = null_positions_arr[in_chunk] - chunk_start
        block = np.asarray(zgw[chunk_start:chunk_end], dtype=np.float64)
        block[local_idx] = fill_value
        zgw[chunk_start:chunk_end] = block
    return


def smoothen_dists(
    store: zarr.Group,
    z_idx: zarr.Array,
    z_dist: zarr.Array,
    lc: float,
    bw: float,
    chunk_size: int,
) -> None:
    """Smoothens KNN distances.

    Args:
        store: Zarr group for edge arrays.
        z_idx: Zarr array of KNN neighbor indices.
        z_dist: Zarr array of KNN distances.
        lc: UMAP local connectivity.
        bw: UMAP bandwidth.
        chunk_size: Row batch size for streaming smooth knn.

    Returns:
        None
    """
    from umap.umap_ import smooth_knn_dist, compute_membership_strengths

    umap_is_latest = _is_umap_version_new()

    n_cells, n_neighbors = z_idx.shape
    zge = create_zarr_dataset(
        store,
        "edges",
        (chunk_size * n_neighbors,),
        ("u8", "u8"),
        (n_cells * n_neighbors, 2),
    )
    zgw = create_zarr_dataset(
        store, "weights", (chunk_size * n_neighbors,), "f8", (n_cells * n_neighbors,)
    )
    last_row = 0
    val_counts = 0
    null_positions: list[int] = []
    global_min = 1
    for i in tqdmbar(range(0, n_cells, chunk_size), desc="Smoothening KNN distances"):
        if i + chunk_size > n_cells:
            ki, kv = z_idx[i:n_cells, :], z_dist[i:n_cells, :]
        else:
            ki, kv = z_idx[i : i + chunk_size, :], z_dist[i : i + chunk_size, :]
        ki_arr = np.asarray(ki)
        kv_arr = np.asarray(kv, dtype=np.float32)
        if kv_arr.flags.c_contiguous is False:
            kv_arr = np.ascontiguousarray(kv_arr)
        sigmas, rhos = smooth_knn_dist(
            kv_arr, k=n_neighbors, local_connectivity=lc, bandwidth=bw
        )
        if umap_is_latest:
            rows, cols, vals, _ = compute_membership_strengths(
                ki_arr, kv_arr, sigmas, rhos
            )
        else:
            rows, cols, vals = compute_membership_strengths(
                ki_arr, kv_arr, sigmas, rhos
            )
        rows = rows + last_row
        start = val_counts
        end = val_counts + len(rows)
        last_row = rows[-1] + 1
        val_counts += len(rows)
        zge[start:end, 0] = rows
        zge[start:end, 1] = cols
        zgw[start:end] = vals

        local_null = np.flatnonzero(vals == 0)
        if local_null.size > 0:
            nz_vals = vals[vals != 0]
            if nz_vals.size > 0:
                min_val = nz_vals.min()
                if min_val < global_min:
                    global_min = min_val
            null_positions.extend((start + local_null).tolist())

    _patch_null_weights(zgw, null_positions, global_min, chunk_size * n_neighbors)
    return None


def export_knn_to_mtx(mtx: str, csr_graph: csr_matrix, batch_size: int = 1000) -> None:
    """Exports KNN matrix in Matrix Market format.

    Args:
        mtx:
        csr_graph:
        batch_size:

    Returns:
        None
    """
    n_cells = csr_graph.shape[0]
    with open(mtx, "w") as h:
        h.write("%%MatrixMarket matrix coordinate real general\n% Generated by Scarf\n")
        h.write(f"{n_cells} {n_cells} {csr_graph.nnz}\n")
        s = 0
        for e in tqdmbar(
            range(batch_size, n_cells + batch_size, batch_size),
            desc="Saving KNN matrix in MTX format",
        ):
            if e > n_cells:
                e = n_cells
            sg = csr_graph[s:e].tocoo()
            df = pd.DataFrame({"row": sg.row + s + 1, "col": sg.col + 1, "d": sg.data})
            df.to_csv(h, sep=" ", header=False, index=False, mode="a")
            s = e
        if s != n_cells:
            raise ValueError(
                "ERROR: Internal loop count error in export_knn_to_mtx. Please report this bug"
            )
    return None


def run_sgtsne(
    graph: csr_matrix | coo_matrix,
    ini_embed: np.ndarray,
    *,
    tsne_dims: int = 2,
    max_iter: int = 500,
    early_iter: int = 200,
    alpha: int = 10,
    lambda_scale: float = 1.0,
    box_h: float = 0.7,
    temp_file_loc: str = ".",
    verbose: bool = True,
    parallel: bool = False,
    nthreads: int = 1,
) -> np.ndarray:
    """Run SG-t-SNE on a sparse graph.

    Uses the ``sgtsne`` executable when available, otherwise falls back to the
    ``sgtsnepi`` Python package.

    Args:
        graph: Sparse cell-neighbourhood graph.
        ini_embed: Initial embedding with shape (n_cells, tsne_dims).
        tsne_dims: Number of tSNE dimensions.
        max_iter: Maximum number of iterations.
        early_iter: Number of early exaggeration iterations.
        alpha: Early exaggeration multiplier.
        lambda_scale: Lambda rescaling parameter.
        box_h: Grid side length (accuracy control).
        temp_file_loc: Directory for temporary files used by the CLI backend.
        verbose: Whether to print SG-t-SNE logs.
        parallel: Whether to run tSNE in parallel mode (CLI backend only).
        nthreads: Number of threads for parallel CLI runs.

    Returns:
        Embedding array with shape (tsne_dims, n_cells).
    """
    import os
    import shutil
    from pathlib import Path
    from uuid import uuid4

    from loguru import logger

    from .utils import system_call

    n_cells = graph.shape[0]
    ini_embed = np.asarray(ini_embed)
    if ini_embed.shape == (n_cells * tsne_dims,):
        ini_embed = ini_embed.reshape(n_cells, tsne_dims)
    if ini_embed.shape != (n_cells, tsne_dims):
        raise ValueError(
            f"ini_embed must have shape ({n_cells}, {tsne_dims}), got {ini_embed.shape}"
        )

    if shutil.which("sgtsne") is not None:
        uid = str(uuid4())
        knn_mtx_fn = Path(temp_file_loc, f"{uid}.mtx").resolve()
        export_knn_to_mtx(str(knn_mtx_fn), graph)
        ini_emb_fn = Path(temp_file_loc, f"{uid}.txt").resolve()
        with open(ini_emb_fn, "w") as h:
            h.write("\n".join(map(str, ini_embed.flatten())))
        out_fn = Path(temp_file_loc, f"{uid}_output.txt").resolve()
        threads = nthreads if parallel else 1
        cmd = (
            f"sgtsne -m {max_iter} -l {lambda_scale} -d {tsne_dims} -e {early_iter} "
            f"-p {threads} -a {alpha} -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        )
        if verbose:
            system_call(cmd)
        else:
            os.system(cmd)
        try:
            emb = np.asarray(
                pd.read_csv(out_fn, header=None, sep=" ")[
                    list(range(tsne_dims))
                ].values.T
            )
        finally:
            for fn in (out_fn, knn_mtx_fn, ini_emb_fn):
                if fn.exists():
                    fn.unlink()
        return emb

    try:
        from sgtsnepi import sgtsnepi
    except ImportError as exc:
        raise ImportError(
            "SG-t-SNE requires the sgtsne executable on PATH or the sgtsnepi package."
        ) from exc

    if parallel:
        logger.warning(
            "parallel=True is not supported by the sgtsnepi Python backend; "
            "running single-threaded"
        )

    return np.asarray(
        sgtsnepi(
            graph,
            y0=ini_embed.T,
            d=tsne_dims,
            max_iter=max_iter,
            early_exag=early_iter,
            lambda_par=lambda_scale,
            h=box_h,
            alpha=alpha,
            silent=not verbose,
        )
    )


@jit(nopython=True)
def calc_snn(indices: np.ndarray) -> np.ndarray:
    """Calculates shared nearest neighbour between each node and its neighbour.

    Args:
        indices: KNN graph indices

    Returns: A numpy matrix of shape (n_cells, n neighbours)
    """
    ncells, nk = indices.shape
    snn = np.zeros((ncells, nk))
    for i in range(ncells):
        for j in range(nk):
            k = indices[i][j]
            snn[i][j] = len(set(indices[i]).intersection(set(indices[k])))
    return np.asarray(snn / (nk - 1))


def weight_sort_indices(
    i: np.ndarray, w: np.ndarray, wn: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sort the array i and w based on values of wn. Only keep the top n
    values.

    Args:
        i: A 1D array of indices
        w: A 1D array of weights
        wn: A 1D array of weights. These weights are used for sorting
        n: Number of neighbours to retain.

    Returns: A tuple of two 1D arrays representing sorted and filtered
             indices and their corresponding weights
    """

    idx = np.argsort(wn)[::-1]
    i = i[idx]
    w = w[idx]
    # Removing duplicate neighbours
    _, unique_idx = np.unique(i, return_index=True)
    unique_idx_arr = np.asarray(sorted(unique_idx))
    return i[unique_idx_arr][:n], w[unique_idx_arr][:n]


def merge_graphs(csr_mats: list[csr_matrix]) -> coo_matrix:
    """Merge multiple graphs of same size and shape such that the merged graph
    have the same size and shape. Edge values are sorted based on their weight
    and the shared neighbours.

    Args:
        csr_mats: A list of two or more CSR matrices representing the graphs to be merged.

    Returns:
        coo_matrix: Merged graph with the same shape and edge count as inputs.
    """
    try:
        assert len(set([x.shape for x in csr_mats])) == 1
    except AssertionError:
        raise ValueError("ERROR: All graphs do not have the same shape.")
    try:
        assert len(set([x.size for x in csr_mats])) == 1
    except AssertionError:
        raise ValueError("ERROR: All graphs do not have the same number of edges")

    nk = csr_mats[0][0].indices.shape[0]
    snns = []
    for mat in tqdmbar(csr_mats, desc="Identifying SNNs in graphs"):
        snns.append(
            calc_snn(mat.indices.reshape((mat.shape[0], mat[0].indices.shape[0])))
        )
    col: list[int] = []
    data: list[float] = []
    for i in tqdmbar(range(csr_mats[0].shape[0]), desc="Merging graph edges"):
        mi = np.hstack([mat[i].indices for mat in csr_mats])
        mwn = np.hstack([mat[i].data + snns[n][i] for n, mat in enumerate(csr_mats)])
        mw = np.hstack([mat[i].data for mat in csr_mats])
        mi, mw = weight_sort_indices(mi, mw, mwn, nk)
        col.extend(mi)
        data.extend(mw)
    s = csr_mats[0].shape
    row = np.repeat(range(s[0]), nk)
    return coo_matrix((data, (row, col)), shape=s)


def wnn_integration(
    name1: str,
    g1: csr_matrix,
    ld1: np.ndarray,
    name2: str,
    g2: csr_matrix,
    ld2: np.ndarray,
    n_threads: int,
) -> coo_matrix:
    """Build a weighted nearest-neighbor graph from two modality-specific KNN graphs.

    Args:
        name1: Label for the first modality (used in log messages).
        g1: CSR KNN graph for modality 1.
        ld1: Latent embedding matrix for modality 1, shape (n_cells, n_dims).
        name2: Label for the second modality.
        g2: CSR KNN graph for modality 2.
        ld2: Latent embedding matrix for modality 2.
        n_threads: Thread limit for affinity calculations.

    Returns:
        coo_matrix: WNN graph combining both modalities.
    """

    def make_estimates(g: csr_matrix, ld: np.ndarray, msg: str = "") -> np.ndarray:
        return np.array(
            [
                ld[g[i].indices].mean(axis=0)
                for i in tqdmbar(range(g.shape[0]), desc=msg)
            ]
        )

    def get_kth_l(g: csr_matrix, ld: np.ndarray, k: int) -> np.ndarray:
        return ld[[g[x].indices[k] for x in range(g.shape[0])]]

    def calc_theta(
        ld: np.ndarray, le: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        a = np.sqrt(((ld - le) ** 2).sum(axis=1))
        d = a - b
        d[d < 0] = 0
        return np.asarray(np.exp(((-1 * d) / (c - d)).astype(np.longdouble)))

    def calc_affinity_ratios(
        g_self: csr_matrix,
        g_other: csr_matrix,
        ld: np.ndarray,
        sigma: int = -2,
        epsilon: float = 10e-4,
        name: str = "",
    ) -> np.ndarray:
        l_self = make_estimates(
            g_self, ld, msg=f"({name}) Predicting within modality profile"
        )
        l_cross = make_estimates(
            g_other, ld, msg=f"({name}) Predicting cross modality profile"
        )

        b = np.sqrt(((ld - get_kth_l(g_self, ld, 0)) ** 2).sum(axis=1))
        c = np.sqrt(((ld - get_kth_l(g_self, ld, sigma)) ** 2).sum(axis=1))

        theta_self = calc_theta(ld, l_self, b, c)
        theta_cross = calc_theta(ld, l_cross, b, c)

        return (theta_self / (theta_cross + epsilon)).astype(np.longdouble)

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=n_threads):
        sr = calc_affinity_ratios(g1, g2, ld1, name=name1)
        sp = calc_affinity_ratios(g2, g1, ld2, name=name2)

        wr = np.exp(sr) / (np.exp(sr) + np.exp(sp))
        wp = np.exp(sp) / (np.exp(sr) + np.exp(sp))

    nk = g1[0].indices.shape[0]
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for n in tqdmbar(range(g1.shape[0]), desc="Building WNN graph"):
        mixed_k = np.array(sorted(set(g1[n].indices).union(g2[n].indices)))
        dr = np.sqrt(((ld1[n] - ld1[mixed_k]) ** 2).sum(axis=1)) * wr[n]
        dp = np.sqrt(((ld2[n] - ld2[mixed_k]) ** 2).sum(axis=1)) * wp[n]
        w_d = dr + dp
        idx = np.argsort(w_d)[:nk]
        col_parts.append(mixed_k[idx])
        v = w_d[idx]
        data_parts.append(np.exp(-((v - v[0]) / (v - v[0]).mean())))

    merged_data = np.hstack(data_parts)
    row = np.repeat(np.arange(g1.shape[0]), nk)
    merged_col = np.hstack(col_parts)
    return coo_matrix((merged_data, (row, merged_col)), shape=g1.shape)
