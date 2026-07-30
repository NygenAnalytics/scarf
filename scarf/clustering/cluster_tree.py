from collections.abc import Iterator

import networkx as nx
import numpy as np
import pandas as pd

from ..utils.logging import logger
from ..utils.progress import iter_progress


def make_digraph(
    d: np.ndarray,
    clust_info: np.ndarray | None = None,
) -> nx.DiGraph:
    """Convert a scipy linkage matrix into a directed tree graph."""
    graph = nx.DiGraph()
    node = d.shape[0] + 1
    if clust_info is not None:
        if len(clust_info) != d.shape[0] + 1:
            raise ValueError(
                "ERROR: cluster information doesn't match number of leaves in dendrogram"
            )
    else:
        clust_info = np.ones(d.shape[0] + 1) * -1
    for row_values in iter_progress(
        d,
        desc="Constructing graph from dendrogram",
        total=d.shape[0],
    ):
        distance = row_values[2]
        row = row_values.astype(int)
        graph.add_node(node, nleaves=row[3], dist=distance)
        if row[0] <= d.shape[0]:
            graph.add_node(
                row[0],
                nleaves=0,
                dist=distance,
                cluster=clust_info[row[0]],
            )
        if row[1] <= d.shape[0]:
            graph.add_node(
                row[1],
                nleaves=0,
                dist=distance,
                cluster=clust_info[row[1]],
            )
        graph.add_edge(node, row[0])
        graph.add_edge(node, row[1])
        node += 1
    if graph.number_of_edges() != d.shape[0] * 2:
        logger.warning(
            "Number of edges in directed graph not twice the dendrogram shape"
        )
    return graph


def CoalesceTree(graph: nx.DiGraph, clusters: np.ndarray) -> nx.DiGraph:
    """Coalesce a hierarchy graph to each cluster's holding node."""

    def calc_steps_to_top(g: nx.DiGraph, c: np.ndarray) -> pd.Series:
        steps: dict[int, int] = {}
        for node in range(len(c)):
            steps[node] = 0
            queue = [node]
            while len(queue) > 0:
                for predecessor in g.predecessors(queue.pop(0)):
                    steps[node] += 1
                    queue.append(predecessor)
        return pd.Series(steps).sort_values()

    def iter_predecessors(g: nx.DiGraph, node: int) -> Iterator[int]:
        queue = [node]
        while len(queue) > 0:
            for predecessor in g.predecessors(queue.pop(0)):
                yield predecessor
                queue.append(predecessor)

    def aggregate_leaves(g: nx.DiGraph, node: int) -> list[int]:
        queue = [node]
        leaves: list[int] = []
        while len(queue) > 0:
            for successor in g.successors(queue.pop(0)):
                if g.nodes[successor]["nleaves"] == 0:
                    leaves.append(successor)
                else:
                    queue.append(successor)
        return leaves

    def get_holding_nodes(g: nx.DiGraph, c: np.ndarray) -> dict[int, int]:
        holding_nodes: dict[int, int] = {}
        steps = calc_steps_to_top(g, c)
        for cluster in iter_progress(
            set(c),
            desc="Identifying the top node for cluster",
        ):
            cluster_nodes = set(np.where(c == cluster)[0])
            n_cluster_nodes = len(cluster_nodes)
            start_node = steps.reindex(cluster_nodes).idxmin()
            if n_cluster_nodes == 1:
                holding_nodes[start_node] = cluster
                continue
            for predecessor in iter_predecessors(g, start_node):
                if g.nodes[predecessor]["nleaves"] >= n_cluster_nodes:
                    leaves = set(aggregate_leaves(g, predecessor))
                    if cluster_nodes.issubset(leaves):
                        if leaves != cluster_nodes or predecessor in holding_nodes:
                            raise ValueError(
                                "Cluster labels are incompatible with the hierarchy: "
                                f"cluster {cluster!r} is not monophyletic"
                            )
                        holding_nodes[predecessor] = cluster
                        break
            else:
                raise ValueError(
                    "Cluster labels are incompatible with the hierarchy: "
                    f"cluster {cluster!r} is not monophyletic"
                )
        return holding_nodes

    def aggregate_predecessors(g: nx.DiGraph, node: int) -> list[int]:
        return list(iter_predecessors(g, node))

    def make_subgraph(g: nx.DiGraph, nodes: dict[int, int]) -> nx.DiGraph:
        subgraph_nodes: list[int] = list(nodes.keys())
        for node in nodes:
            subgraph_nodes.extend(aggregate_predecessors(g, node))
        subgraph = nx.DiGraph(nx.subgraph(g, list(set(subgraph_nodes))))
        for node in nodes:
            subgraph.nodes[node]["partition_id"] = nodes[node]
        return subgraph

    return make_subgraph(graph, get_holding_nodes(graph, clusters))
