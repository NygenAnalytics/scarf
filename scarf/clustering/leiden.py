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
            "running Paris instead of Leiden clustering using `run_clustering` method"
        )
    import igraph

    def _probe(msg: str) -> None:
        print(f"[leiden] {msg}", flush=True)

    n_cells = int(graph.shape[0])
    nnz = int(graph.nnz)
    _probe(f"ENTER n_cells={n_cells} nnz={nnz} resolution={resolution}")
    sources, targets = graph.nonzero()
    _probe(f"nonzero done sources={len(sources)}")
    igraph_graph = igraph.Graph()
    igraph_graph.add_vertices(graph.shape[0])
    _probe("add_vertices done; building edge list")
    edges = list(zip(sources, targets, strict=True))
    _probe(f"edge list done n_edges={len(edges)}; add_edges")
    igraph_graph.add_edges(edges)
    igraph_graph.es["weight"] = graph[sources, targets].A1
    _probe("weights assigned; find_partition")
    partition = leidenalg.find_partition(
        igraph_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_seed,
    )
    membership = np.array(partition.membership) + 1
    _probe(f"DONE n_clusters={int(membership.max()) if membership.size else 0}")
    return membership
