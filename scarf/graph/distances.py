"""Read-side contract for persisted nearest-neighbor distances."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from ..storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ..storage.errors import ArtifactResolutionError
from ..storage.geometry import array_geometry
from ..storage.partition import row_band
from ..storage.types import as_zarr_array

NEIGHBOR_DISTANCE_METRICS = frozenset({"l2", "cosine"})


@dataclass(frozen=True, slots=True)
class ValidatedNeighborsPayload:
    indices: zarr.Array
    distances: zarr.Array
    n_cells: int
    n_neighbors: int


def _neighbors_payload_error(
    ref: ArtifactRef,
    message: str,
) -> ArtifactResolutionError:
    return ArtifactResolutionError(
        message,
        code="corrupt_payload",
        context={
            "assay": ref.assay,
            "artifact_id": ref.artifact_id,
            "actual_kind": ref.kind,
        },
    )


def validate_distance_provenance(zw: Any, ref: ArtifactRef) -> None:
    """Check that a neighbors artifact stores distances in its named metric."""
    if ref.kind != "neighbors":
        raise ValueError("Distance provenance requires a neighbors artifact")
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


def validate_neighbors_payload(
    root: zarr.Group,
    ref: ArtifactRef,
) -> ValidatedNeighborsPayload:
    """Validate a persisted neighbor matrix in bounded row blocks."""
    if ref.kind != "neighbors":
        raise ValueError("Neighbor payload validation requires a neighbors artifact")
    try:
        group = artifact_group(root, ref)
        indices = as_zarr_array(group["indices"], name="indices")
        distances = as_zarr_array(group["distances"], name="distances")
    except Exception as error:
        raise _neighbors_payload_error(
            ref,
            "Neighbors artifact payload is unreadable",
        ) from error

    raw_cells = group.attrs.get("n_cells")
    raw_neighbors = group.attrs.get("n_neighbors")
    raw_self_hit_rate = group.attrs.get("self_hit_rate")
    if (
        isinstance(raw_cells, bool)
        or not isinstance(raw_cells, int | np.integer)
        or int(raw_cells) < 1
        or int(raw_cells) > np.iinfo(np.uint32).max
        or isinstance(raw_neighbors, bool)
        or not isinstance(raw_neighbors, int | np.integer)
        or int(raw_neighbors) < 1
        or int(raw_neighbors) >= int(raw_cells)
        or isinstance(raw_self_hit_rate, bool)
        or not isinstance(raw_self_hit_rate, int | float | np.integer | np.floating)
        or not math.isfinite(float(raw_self_hit_rate))
        or not 0 <= float(raw_self_hit_rate) <= 100
    ):
        raise _neighbors_payload_error(
            ref,
            "Neighbors artifact has invalid dimensions or metadata",
        )
    n_cells = int(raw_cells)
    n_neighbors = int(raw_neighbors)
    expected_shape = (n_cells, n_neighbors)
    if (
        indices.ndim != 2
        or tuple(map(int, indices.shape)) != expected_shape
        or np.dtype(indices.dtype) != np.dtype(np.uint32)
        or distances.ndim != 2
        or tuple(map(int, distances.shape)) != expected_shape
        or np.dtype(distances.dtype) != np.dtype(np.float32)
    ):
        raise _neighbors_payload_error(
            ref,
            "Neighbors arrays do not match their stored dimensions",
        )

    block_rows = min(
        row_band(array_geometry(indices), unit="chunk", fallback=1),
        row_band(array_geometry(distances), unit="chunk", fallback=1),
    )
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        try:
            index_block = np.asarray(indices[start:stop])
            distance_block = np.asarray(distances[start:stop])
        except Exception as error:
            raise _neighbors_payload_error(
                ref,
                "Neighbors arrays are unreadable",
            ) from error
        row_ids = np.arange(start, stop, dtype=np.uint32)[:, None]
        if (
            np.any(index_block >= n_cells)
            or np.any(index_block == row_ids)
            or not np.all(np.isfinite(distance_block))
            or np.any(distance_block < 0)
        ):
            raise _neighbors_payload_error(
                ref,
                "Neighbors arrays contain invalid indices or distances",
            )
    return ValidatedNeighborsPayload(
        indices=indices,
        distances=distances,
        n_cells=n_cells,
        n_neighbors=n_neighbors,
    )
