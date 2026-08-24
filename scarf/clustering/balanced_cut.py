import networkx as nx
import numpy as np

from ..utils.logging import logger
from ..utils.progress import tqdmbar
from .cluster_tree import make_digraph


class BalancedCut:
    """Identify balanced cluster splits from a hierarchy."""

    graph: nx.DiGraph
    branchpoints: dict[int, list[int]]

    def __init__(
        self,
        dendrogram: np.ndarray,
        max_size: int,
        min_size: int,
        max_distance_fc: float,
    ):
        self.nCells = dendrogram.shape[0] + 1
        self.graph = make_digraph(dendrogram)
        self.maxSize = max_size
        self.minSize = min_size
        self.maxDistFc = max_distance_fc
        self.branchpoints = self._get_branchpoints()

    def _successors(self, start: int, min_leaves: int) -> list[int]:
        """Get tree downstream of a node."""
        queue = [start]
        descendants: list[int] = []
        while len(queue) > 0:
            node = queue.pop(0)
            if self.graph.nodes[node]["nleaves"] > min_leaves:
                descendants.append(node)
                queue.extend(list(self.graph.successors(node)))
        return descendants[1:]

    def _get_mean_dist(self, start_node: int) -> float:
        """Get mean distances in downstream tree of a node."""
        nodes = self._successors(start_node, -1)
        return float(
            np.array([self.graph.nodes[node]["dist"] for node in nodes]).mean()
        )

    def _are_subtrees_mergeable(self, s1: int, s2: int) -> bool:
        n1, n2 = self.graph.nodes[s1]["nleaves"], self.graph.nodes[s2]["nleaves"]
        if n1 > self.minSize and n2 > self.minSize:
            d1, d2 = self.graph.nodes[s1]["dist"], self.graph.nodes[s2]["dist"]
            if d1 / d2 > self.maxDistFc or d2 / d1 > self.maxDistFc:
                logger.trace(f"Will not merge {s1} and {s2} because of high distance")
                return False
            md1, md2 = self._get_mean_dist(s1), self._get_mean_dist(s2)
            if md1 / md2 > self.maxDistFc or md2 / md1 > self.maxDistFc:
                logger.trace(
                    f"Will not merge {s1} and {s2} because of high distance of successors"
                )
                return False
        return True

    def _get_branchpoints(self) -> dict[int, list[int]]:
        """Aggregate leaves bottom up until target size is reached."""
        n_leaves = int((self.graph.number_of_nodes() + 1) / 2)
        leaves: dict[int, None] = {node: None for node in range(n_leaves)}
        branchpoints: dict[int, list[int]] = {}
        progress = tqdmbar(total=len(leaves), desc="Identifying nodes to split")
        try:
            while len(leaves) > 0:
                leaf, _ = leaves.popitem()
                progress.update(1)
                logger.trace(f"FRESH STEP: Leaf {leaf} plucked as base leaf")
                current = leaf
                while True:
                    parent = next(self.graph.predecessors(current), None)
                    if parent is None:
                        break
                    if parent in branchpoints:
                        logger.trace(
                            f"Will not climb to {parent} as already a branchpoint"
                        )
                        break
                    if self.graph.nodes[parent]["nleaves"] > self.maxSize:
                        logger.trace(
                            f"Will not climb to {parent} because too many leaves exist"
                        )
                        break
                    successor1, successor2 = list(self.graph.successors(parent))
                    if not self._are_subtrees_mergeable(successor1, successor2):
                        break
                    current = parent
                logger.trace(f"Aggregating from branch {current} for leaf {leaf}")
                branchpoints[current] = [leaf]
                stack = [current]
                while len(stack) > 0:
                    node = stack.pop()
                    if node in leaves:
                        branchpoints[current].append(node)
                        leaves.pop(node)
                        logger.trace(f"Leaf {node} plucked in aggregation step")
                        progress.update(1)
                    elif node in branchpoints and node != current:
                        logger.trace(
                            f"Skipping branch {node} because its already taken"
                        )
                    elif (
                        self.graph.nodes[node]["nleaves"] >= self.maxSize
                        and node != current
                    ):
                        logger.trace(
                            f"Skipping branch {node} to prevent greedy behaviour"
                        )
                    else:
                        stack.extend(list(self.graph.successors(node)))
        finally:
            progress.close()
        return branchpoints

    def _valid_names_in_branchpoints(self) -> None:
        leaves: list[int] = []
        for branchpoint in self.branchpoints:
            leaves.extend(self.branchpoints[branchpoint])
        n_leaves = len(leaves)
        if n_leaves != self.nCells:
            raise ValueError(
                "ERROR: Not all leaves present in branchpoints. This bug must be reported"
            )
        minimum_leaf = min(leaves)
        if minimum_leaf != 0:
            raise ValueError(
                f"ERROR: minimum leaf label is {minimum_leaf} rather than 0"
            )
        maximum_leaf = max(leaves)
        if n_leaves != maximum_leaf + 1:
            raise ValueError(
                f"ERROR: maximum leaf label is {maximum_leaf} "
                f"while total estimated leaves are {n_leaves}"
            )

    def get_clusters(self) -> np.ndarray:
        """Make cluster labels from the identified branchpoints."""
        self._valid_names_in_branchpoints()
        clusters = np.zeros(self.nCells).astype(int)
        for cluster, branchpoint in enumerate(self.branchpoints, start=1):
            clusters[self.branchpoints[branchpoint]] = cluster
        if (clusters == 0).sum() > 0:
            logger.warning(
                f"{(clusters == 0).sum()} samples were not assigned a cluster"
            )
            clusters[clusters == 0] = -1
        return clusters
