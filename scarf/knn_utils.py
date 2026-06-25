"""Utility functions for running the KNN algorithm."""


import numpy as np
import pandas as pd
from numba import jit
from scipy.sparse import csr_matrix, coo_matrix

from .ann import AnnStream
from .utils import tqdmbar, controlled_compute
from .writers import create_zarr_dataset

__all__ = [
    "self_query_knn",
    "smoothen_dists",
    "export_knn_to_mtx",
    "merge_graphs",
    "wnn_integration",
]


def self_query_knn(ann_obj: AnnStream, store, chunk_size: int, nthreads: int) -> float:
    """Constructs KNN graph.

    Args:
        ann_obj: Fitted AnnStream with reducer and ANN index.
        store: Zarr group where ``indices`` and ``distances`` arrays are written.
        chunk_size: Row chunk size for output Zarr arrays.
        nthreads: Number of threads to use.

    Returns:
        Approximate recall percentage (fraction of queries with a self neighbor).
    """

    def get_transformed_data():
        msg = "Identifying neighbors"
        if ann_obj.harmonizedData is None:
            for _i in tqdmbar(
                ann_obj.data.blocks, desc=msg, total=ann_obj.data.numblocks[0]
            ):
                yield ann_obj.reducer(controlled_compute(_i, nthreads))
        else:
            for _i in tqdmbar(
                ann_obj.harmonizedData.blocks,
                desc=msg,
                total=ann_obj.harmonizedData.numblocks[0],
            ):
                yield controlled_compute(_i, nthreads)

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
            ki, kv, nm = ann_obj.transform_ann(
                i,
                k=n_neighbors,
                self_indices=np.arange(nsample_start, nsample_end),
            )
            z_knn[nsample_start:nsample_end, :] = ki
            z_dist[nsample_start:nsample_end, :] = kv
            nsample_start = nsample_end
            tnm += nm
    recall = ann_obj.data.shape[0] - tnm
    recall = 100 * recall / ann_obj.data.shape[0]
    return recall


def _is_umap_version_new():
    import umap
    from packaging import version

    if version.parse(umap.__version__) >= version.parse("0.5.0"):
        return True
    else:
        return False


def smoothen_dists(store, z_idx, z_dist, lc: float, bw: float, chunk_size: int):
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
        f"edges",
        (chunk_size * n_neighbors,),
        ("u8", "u8"),
        (n_cells * n_neighbors, 2),
    )
    zgw = create_zarr_dataset(
        store, f"weights", (chunk_size * n_neighbors,), "f8", (n_cells * n_neighbors,)
    )
    last_row = 0
    val_counts = 0
    null_idx = []
    global_min = 1
    for i in tqdmbar(range(0, n_cells, chunk_size), desc="Smoothening KNN distances"):
        if i + chunk_size > n_cells:
            ki, kv = z_idx[i:n_cells, :], z_dist[i:n_cells, :]
        else:
            ki, kv = z_idx[i : i + chunk_size, :], z_dist[i : i + chunk_size, :]
        kv = kv.astype(np.float32, order="C")
        sigmas, rhos = smooth_knn_dist(
            kv, k=n_neighbors, local_connectivity=lc, bandwidth=bw
        )
        if umap_is_latest:
            rows, cols, vals, _ = compute_membership_strengths(ki, kv, sigmas, rhos)
        else:
            rows, cols, vals = compute_membership_strengths(ki, kv, sigmas, rhos)
        rows = rows + last_row
        start = val_counts
        end = val_counts + len(rows)
        last_row = rows[-1] + 1
        val_counts += len(rows)
        zge[start:end, 0] = rows
        zge[start:end, 1] = cols
        zgw[start:end] = vals

        # Fixing edges with 0 weights
        # We are doing these steps here to have minimum operations outside
        # the scope of a progress bar
        nidx = vals == 0
        if nidx.sum() > 0:
            min_val = vals[~nidx].min()
            if min_val < global_min:
                global_min = min_val
        null_idx.extend(nidx)

    # The whole zarr array needs to copied, modified and written back.
    # Or is this assumption wrong?
    w = zgw[:]
    w[null_idx] = global_min
    zgw[:] = w
    return None


def export_knn_to_mtx(mtx: str, csr_graph, batch_size: int = 1000) -> None:
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
    return snn / (nk - 1)


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
    _, idx = np.unique(i, return_index=True)
    idx = sorted(idx)
    return i[idx][:n], w[idx][:n]


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
    col, data = [], []
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
    name1: str, g1: csr_matrix, ld1, name2: str, g2: csr_matrix, ld2, n_threads: int
):
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

    def make_estimates(g, ld, msg=""):
        return np.array(
            [
                ld[g[i].indices].mean(axis=0)
                for i in tqdmbar(range(g.shape[0]), desc=msg)
            ]
        )

    def get_kth_l(g, ld, k):
        return ld[[g[x].indices[k] for x in range(g.shape[0])]]

    def calc_theta(ld, le, b, c):
        a = np.sqrt(((ld - le) ** 2).sum(axis=1))
        d = a - b
        d[d < 0] = 0
        return np.exp(((-1 * d) / (c - d)).astype(np.longdouble))

    def calc_affinity_ratios(
        g_self, g_other, ld, sigma: int = -2, epsilon: float = 10e-4, name=""
    ):
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
    col, data = [], []
    for n in tqdmbar(range(g1.shape[0]), desc="Building WNN graph"):
        mixed_k = np.array(sorted(set(g1[n].indices).union(g2[n].indices)))
        dr = np.sqrt(((ld1[n] - ld1[mixed_k]) ** 2).sum(axis=1)) * wr[n]
        dp = np.sqrt(((ld2[n] - ld2[mixed_k]) ** 2).sum(axis=1)) * wp[n]
        w_d = dr + dp
        idx = np.argsort(w_d)[:nk]
        col.append(mixed_k[idx])
        v = w_d[idx]
        data.append(np.exp(-((v - v[0]) / (v - v[0]).mean())))

    data = np.hstack(data)
    row = np.repeat(range(g1.shape[0]), nk)
    col = np.hstack(col)
    return coo_matrix((data, (row, col)), shape=g1.shape)
