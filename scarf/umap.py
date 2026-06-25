# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Modified file
#
# License: BSD 3 clause

import locale
from typing import Any

import numpy as np

from .utils import logger, tqdm_params as default_tqdm_params

locale.setlocale(locale.LC_NUMERIC, "C")

__all__ = ["fit_transform"]


def calc_dens_map_params(graph: Any, dists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute densMAP correction terms from graph and KNN distances."""
    n_vertices = graph.shape[0]
    mu_sum = np.zeros(n_vertices, dtype=np.float32)
    ro = np.zeros(n_vertices, dtype=np.float32)
    head = graph.row
    tail = graph.col
    for i in range(len(head)):
        j = head[i]
        k = tail[i]

        D = dists[j, k] * dists[j, k]  # match sq-Euclidean used for embedding
        mu = graph.data[i]

        ro[j] += mu * D
        ro[k] += mu * D
        mu_sum[j] += mu
        mu_sum[k] += mu

    epsilon = 1e-8
    ro = np.log(epsilon + (ro / mu_sum))
    R = (ro - np.mean(ro)) / np.std(ro)
    return mu_sum, R


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
    from threadpoolctl import threadpool_limits
    from umap.umap_ import make_epochs_per_sample
    from umap.layouts import optimize_layout_euclidean
    from sklearn.utils import check_random_state

    # g.data[g.data < (g.data.max() / float(n_epochs))] = 0.0
    # g.eliminate_zeros()
    epochs_per_sample = make_epochs_per_sample(g.data, n_epochs)
    logger.trace("calculated epochs_per_sample")
    rng_state = (
        check_random_state(random_seed)
        .randint(np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3)
        .astype(np.int64)
    )
    # Since threadpool_limits doesnt work well with numba. We will use numba's set_num_threads to limit threads
    if numba.config.NUMBA_NUM_THREADS > nthreads:
        numba.set_num_threads(nthreads)

    if densmap_kwds != {}:
        with threadpool_limits(limits=nthreads):
            mu_sum, R = calc_dens_map_params(g, densmap_kwds["knn_dists"])
        densmap_kwds["mu_sum"] = mu_sum
        densmap_kwds["R"] = R
        densmap_kwds["mu"] = g.data
        densmap = True
        logger.trace("calculated densmap params")
    else:
        densmap = False

    # tqdm will be activated if https://github.com/lmcinnes/umap/pull/739
    # is merged and when it is released
    tqdm_params = dict(default_tqdm_params)
    tqdm_params["desc"] = "Training UMAP"

    with threadpool_limits(limits=nthreads):
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
            verbose=verbose,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
            tqdm_kwds=tqdm_params,
            move_other=True,
        )
    return np.asarray(embedding)


def fuzzy_simplicial_set(g: Any, set_op_mix_ratio: float) -> Any:
    """Combine directed and undirected graph components for UMAP."""
    tg = g.transpose()
    prod = g.multiply(tg)
    res = set_op_mix_ratio * (g + tg - prod) + (1.0 - set_op_mix_ratio) * prod
    res.eliminate_zeros()
    return res.tocoo()


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
    """Fit UMAP embedding from a fuzzy simplicial set graph.

    Args:
        graph: Sparse COO adjacency graph.
        ini_embed: Initial embedding coordinates.
        spread: UMAP spread parameter.
        min_dist: UMAP min_dist parameter.
        n_epochs: Number of optimization epochs.
        random_seed: Random seed.
        repulsion_strength: UMAP repulsion strength (gamma).
        initial_alpha: Initial learning rate.
        negative_sample_rate: Negative sampling rate.
        densmap_kwds: Extra densMAP kwargs (empty dict disables densMAP).
        parallel: Whether to run layout optimization in parallel.
        nthreads: Thread limit for layout optimization.
        verbose: Verbose UMAP output.

    Returns:
        Tuple of (embedding, a, b) UMAP parameters.
    """
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
