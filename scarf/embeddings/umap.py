import locale
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ..utils.logging import logger
from ..utils.numba import restore_numba_threads
from ..utils.progress import tqdm_params as default_tqdm_params

locale.setlocale(locale.LC_NUMERIC, "C")

# Recorded only by densMAP runs. It separates embeddings that read symmetric
# neighbor distances from earlier ones that read reverse-only edges as zero.
DENSMAP_ALGORITHM_VERSION = "symmetric_knn_distances_v2"


def densmap_distance_graph(
    indices: np.ndarray,
    distances: np.ndarray,
) -> csr_matrix:
    """Build the symmetric cell-by-cell KNN distance matrix used by densMAP.

    A connectivity edge can come from either endpoint's neighbor list, so each
    pair keeps the larger of its two directed distances, as umap-learn does.
    """
    neighbor_indices = np.asarray(indices)
    neighbor_distances = np.asarray(distances)
    if neighbor_indices.ndim != 2 or neighbor_indices.shape != neighbor_distances.shape:
        raise ValueError("KNN indices and distances must be matching matrices")
    n_cells, n_neighbors = neighbor_indices.shape
    directed = csr_matrix(
        (
            neighbor_distances.ravel(),
            (
                np.repeat(np.arange(n_cells), n_neighbors),
                neighbor_indices.ravel(),
            ),
        ),
        shape=(n_cells, n_cells),
    )
    return csr_matrix(directed.maximum(directed.transpose()))


def calc_dens_map_params(
    graph: Any,
    dists: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute densMAP correction terms from graph and KNN distances.

    Args:
        graph: COO graph whose edges carry membership strengths.
        dists: Dense or sparse cell-by-cell distance matrix. Each graph edge
            reads the distance stored at its own row and column.

    Returns:
        Per-cell membership sums and standardized log local radii.
    """
    n_vertices = graph.shape[0]
    head = np.asarray(graph.row)
    tail = np.asarray(graph.col)
    mu = np.asarray(graph.data, dtype=np.float64)
    if issparse(dists):
        edge_distances = np.asarray(
            csr_matrix(dists)[head, tail],
            dtype=np.float64,
        ).ravel()
    else:
        edge_distances = np.asarray(np.asarray(dists)[head, tail], dtype=np.float64)
    weighted = mu * edge_distances * edge_distances
    ro = np.bincount(head, weights=weighted, minlength=n_vertices) + np.bincount(
        tail, weights=weighted, minlength=n_vertices
    )
    mu_sum = np.bincount(head, weights=mu, minlength=n_vertices) + np.bincount(
        tail, weights=mu, minlength=n_vertices
    )

    epsilon = 1e-8
    log_ro = np.log(epsilon + (ro / mu_sum))
    standardized_ro = (log_ro - np.mean(log_ro)) / np.std(log_ro)
    return mu_sum.astype(np.float32), standardized_ro.astype(np.float32)


@restore_numba_threads
def simplicial_set_embedding(
    g: Any,
    embedding: np.ndarray,
    n_epochs: int,
    a: float,
    b: float,
    random_seed: int,
    gamma: float,
    initial_alpha: float,
    negative_sample_rate: float,
    densmap_kwds: dict[str, Any],
    parallel: bool,
    nthreads: int,
    verbose: bool,
) -> np.ndarray:
    """Run UMAP simplicial-set embedding with optional densMAP."""
    import numba
    from sklearn.utils import check_random_state
    from threadpoolctl import threadpool_limits
    from umap.layouts import optimize_layout_euclidean
    from umap.umap_ import make_epochs_per_sample

    epochs_per_sample = make_epochs_per_sample(g.data, n_epochs)
    logger.trace("calculated epochs_per_sample")
    rng_state = (
        check_random_state(random_seed)
        .randint(np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3)
        .astype(np.int64)
    )
    if numba.config.NUMBA_NUM_THREADS > nthreads:
        numba.set_num_threads(nthreads)

    if densmap_kwds != {}:
        with threadpool_limits(limits=nthreads):
            mu_sum, standardized_ro = calc_dens_map_params(
                g,
                densmap_kwds["knn_dists"],
            )
        densmap_kwds["mu_sum"] = mu_sum
        densmap_kwds["R"] = standardized_ro
        densmap_kwds["mu"] = g.data
        densmap = True
        logger.trace("calculated densmap params")
    else:
        densmap = False

    tqdm_params = dict(default_tqdm_params)
    tqdm_params["desc"] = "Training UMAP"
    if "disable" not in tqdm_params:
        tqdm_params["disable"] = not verbose

    # Numba's thread count is per thread, so the layout needs no process-wide
    # BLAS limit and can run beside other pipeline stages.
    embedding = optimize_layout_euclidean(
        head_embedding=embedding,
        tail_embedding=embedding,
        head=g.row,
        tail=g.col,
        n_epochs=n_epochs,
        n_vertices=g.shape[1],
        epochs_per_sample=epochs_per_sample,
        a=a,
        b=b,
        rng_state=rng_state,
        gamma=gamma,
        initial_alpha=initial_alpha,
        negative_sample_rate=negative_sample_rate,
        parallel=parallel,
        verbose=False,
        densmap=densmap,
        densmap_kwds=densmap_kwds,
        tqdm_kwds=tqdm_params,
        move_other=True,
    )
    return np.asarray(embedding)


def fuzzy_simplicial_set(g: Any, set_op_mix_ratio: float) -> Any:
    """Combine directed and undirected graph components for UMAP."""
    transposed_graph = g.transpose()
    product = g.multiply(transposed_graph)
    result = (
        set_op_mix_ratio * (g + transposed_graph - product)
        + (1.0 - set_op_mix_ratio) * product
    )
    result.eliminate_zeros()
    return result.tocoo()


def fit_transform(
    graph: Any,
    ini_embed: np.ndarray,
    spread: float,
    min_dist: float,
    n_epochs: int,
    random_seed: int,
    repulsion_strength: float,
    initial_alpha: float,
    negative_sample_rate: float,
    densmap_kwds: dict[str, Any],
    parallel: bool,
    nthreads: int,
    verbose: bool,
) -> tuple[np.ndarray, float, float]:
    """Fit UMAP embedding from a fuzzy simplicial set graph."""
    import warnings

    from umap.umap_ import find_ab_params

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a, b = find_ab_params(spread=spread, min_dist=min_dist)
    logger.trace("Found ab params")

    embedding = simplicial_set_embedding(
        fuzzy_simplicial_set(graph, 1.0),
        ini_embed,
        n_epochs,
        a,
        b,
        random_seed,
        repulsion_strength,
        initial_alpha,
        negative_sample_rate,
        densmap_kwds,
        parallel,
        nthreads,
        verbose,
    )
    return embedding, a, b
