"""Graph artifact kinds accepted by clustering and graph consumers."""

from ..storage.artifacts import ArtifactRef

GRAPH_KINDS = frozenset({"connectivity_map", "integrated_graph"})


def require_graph_kind(graph: ArtifactRef) -> None:
    """Reject a reference that is not a connectivity map or integrated graph."""
    if graph.kind not in GRAPH_KINDS:
        raise ValueError(
            "graph must be a connectivity_map or integrated_graph artifact, "
            f"not {graph.kind!r}; build one with build_connectivity_map or "
            "integrate_assays"
        )
