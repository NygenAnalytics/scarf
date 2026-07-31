"""Read-side contract for persisted nearest-neighbor distances."""

from collections.abc import Mapping
from typing import Any

from ..storage.artifacts import ArtifactRef, inspect_artifact, parse_artifact_path

NEIGHBOR_DISTANCE_METRICS = frozenset({"l2", "cosine"})


def validate_distance_provenance(zw: Any, knn_loc: str) -> None:
    """Check that the distances stored at ``knn_loc`` have a known metric.

    Encoded locations carry no provenance and are read as written. Artifact
    locations must name the metric of their stored distances and agree with the
    index those distances were queried from.
    """
    try:
        ref = parse_artifact_path(knn_loc)
    except ValueError:
        return
    if ref.kind != "neighbors":
        return
    status = inspect_artifact(zw, ref)
    metric = (status.parameters or {}).get("distance_metric")
    if metric not in NEIGHBOR_DISTANCE_METRICS:
        raise ValueError(
            "Neighbors artifact does not name the metric of its stored "
            "distances; recompute neighbors"
        )
    source = (status.inputs or {}).get("ann_index")
    if not isinstance(source, Mapping):
        raise ValueError("Neighbors artifact has no ANN index input")
    source_metric = (
        inspect_artifact(zw, ArtifactRef.from_dict(source)).parameters or {}
    ).get("ann_metric")
    if source_metric != metric:
        raise ValueError("Neighbors distance metric does not match its ANN index input")
