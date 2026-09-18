import random
from threading import Lock
from typing import Literal

import numpy as np
from scipy.sparse import spmatrix


type LeidenBackend = Literal["igraph", "leidenalg"]

_IGRAPH_RNG_LOCK = Lock()


def _igraph_membership(
    graph: spmatrix,
    resolution: float,
    random_seed: int,
) -> np.ndarray:
    try:
        import igraph
    except ImportError:
        raise ImportError(
            "ERROR: 'igraph' package is not installed. Install Scarf's required "
            "dependencies before running Leiden clustering."
        ) from None

    coo = graph.tocoo(copy=True)
    coo.eliminate_zeros()
    igraph_graph = igraph.Graph(
        n=graph.shape[0],
        edges=zip(coo.row, coo.col),
        directed=False,
    )
    with _IGRAPH_RNG_LOCK:
        igraph.set_random_number_generator(random.Random(random_seed))
        try:
            partition = igraph_graph.community_leiden(
                weights=coo.data,
                objective_function="modularity",
                resolution=resolution,
                n_iterations=2,
            )
        finally:
            igraph.set_random_number_generator(None)
    return np.array(partition.membership) + 1


def _leidenalg_membership(
    graph: spmatrix,
    resolution: float,
    random_seed: int,
) -> np.ndarray:
    import igraph

    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            "ERROR: 'leidenalg' package is not installed. Please find the "
            "installation instructions here: "
            "https://github.com/vtraag/leidenalg#installation. Also, consider "
            "running Paris with the `run_paris_clustering` method"
        ) from None

    coo = graph.tocoo(copy=True)
    coo.eliminate_zeros()
    igraph_graph = igraph.Graph()
    igraph_graph.add_vertices(graph.shape[0])
    igraph_graph.add_edges(list(zip(coo.row, coo.col)))
    igraph_graph.es["weight"] = coo.data
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_seed,
    )
    return np.array(partition.membership) + 1


def leiden_membership(
    graph: spmatrix,
    resolution: float,
    random_seed: int,
    backend: LeidenBackend = "igraph",
) -> np.ndarray:
    """Cluster a sparse graph with the Leiden algorithm."""
    if backend == "igraph":
        return _igraph_membership(graph, resolution, random_seed)
    if backend == "leidenalg":
        return _leidenalg_membership(graph, resolution, random_seed)
    raise ValueError("backend must be 'igraph' or 'leidenalg'")
