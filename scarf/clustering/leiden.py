import numpy as np
from scipy.sparse import spmatrix


def leiden_membership(
    graph: spmatrix,
    resolution: float,
    random_seed: int,
) -> np.ndarray:
    """Cluster a sparse graph with the Leiden algorithm."""
    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            "ERROR: 'leidenalg' package is not installed. Please find the "
            "installation instructions here: "
            "https://github.com/vtraag/leidenalg#installation. Also, consider "
            "running Paris with the `run_paris_clustering` method"
        )
    import igraph

    sources, targets = graph.nonzero()
    igraph_graph = igraph.Graph()
    igraph_graph.add_vertices(graph.shape[0])
    igraph_graph.add_edges(list(zip(sources, targets)))
    igraph_graph.es["weight"] = graph[sources, targets].A1
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_seed,
    )
    return np.array(partition.membership) + 1
