import numpy as np
from .writers import create_zarr_dataset
from .ann import AnnStream
from tqdm import tqdm

__all__ = ['self_query_knn', 'smoothen_dists', 'export_knn_to_mtx']


def self_query_knn(ann_obj: AnnStream, store, chunk_size: int, nthreads: int) -> None:
    from threadpoolctl import threadpool_limits

    n_cells, n_neighbors = ann_obj.nCells, ann_obj.k
    z_knn = create_zarr_dataset(store, 'indices', (chunk_size,), 'u8',
                                (n_cells, n_neighbors))
    z_dist = create_zarr_dataset(store, 'distances', (chunk_size,), 'f8',
                                 (n_cells, n_neighbors))
    nsample_start = 0
    tnm = 0  # Number of missed recall
    with threadpool_limits(limits=nthreads):
        for i in ann_obj.iter_blocks(msg='Saving KNN graph'):
            nsample_end = nsample_start + i.shape[0]
            ki, kv, nm = ann_obj.transform_ann(ann_obj.reducer(i), k=n_neighbors,
                                           self_indices=np.arange(nsample_start, nsample_end))
            z_knn[nsample_start:nsample_end, :] = ki
            z_dist[nsample_start:nsample_end, :] = kv
            nsample_start = nsample_end
            tnm += nm
    recall = ann_obj.data.shape[0] - tnm
    recall = 100 * recall / ann_obj.data.shape[0]
    recall = "%.2f" % recall
    print(f"INFO: ANN recall: {recall}%", flush=True)
    return None


def smoothen_dists(store, z_idx, z_dist, lc: float, bw: float, chunk_size: int = 100000):
    from umap.umap_ import smooth_knn_dist, compute_membership_strengths

    n_cells, n_neighbors = z_idx.shape
    zge = create_zarr_dataset(store, f'edges', (chunk_size,), ('u8', 'u8'),
                              (n_cells * n_neighbors, 2))
    zgw = create_zarr_dataset(store, f'weights', (chunk_size,), 'f8',
                              (n_cells * n_neighbors,))
    last_row = 0
    val_counts = 0
    step = int(chunk_size / n_neighbors)
    for i in tqdm(range(0, n_cells, step), desc='Smoothening KNN distances'):
        if i + step > n_cells:
            ki, kv = z_idx[i:n_cells, :], z_dist[i:n_cells, :]
        else:
            ki, kv = z_idx[i:i+step, :], z_dist[i:i+step, :]
        kv = kv.astype(np.float32, order='C')
        sigmas, rhos = smooth_knn_dist(kv, k=n_neighbors,
                                       local_connectivity=lc, bandwidth=bw)
        rows, cols, vals = compute_membership_strengths(ki, kv, sigmas, rhos)
        rows = rows + last_row
        start = val_counts
        end = val_counts + len(rows)
        last_row = rows[-1] + 1
        val_counts += len(rows)
        zge[start:end, 0] = rows
        zge[start:end, 1] = cols
        zgw[start:end] = vals
    return None


def export_knn_to_mtx(mtx: str, csr_graph) -> None:
    n_cells = csr_graph.shape[0]
    with open(mtx, 'w') as h:
        h.write("%%MatrixMarket matrix coordinate real general\n% Generated by Scarf\n")
        h.write(f"{n_cells} {n_cells} {csr_graph.nnz}\n")
        temp = []
        n_rows = 0
        for i in tqdm(range(n_cells), desc='Saving KNN matrix in MTX format'):
            idx = csr_graph[i].indices
            w = csr_graph[i].data
            w = w / w.sum()
            for x, y in zip(idx, w):
                temp.append(f'{i + 1} {x + 1}  {y}\n')
                n_rows += 1
            if i % 1000 == 0:
                h.write(''.join(temp))
                temp = []
        if len(temp) > 0:
            h.write(''.join(temp))
    return None
