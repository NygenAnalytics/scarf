"""Read-side contracts for persisted neighbor and connectivity-map payloads."""

import math
from collections.abc import Iterator
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


def payload_error(ref: ArtifactRef, message: str) -> ArtifactResolutionError:
    """Return the error for a graph payload that breaks its stored contract."""
    return ArtifactResolutionError(
        message,
        code="corrupt_payload",
        context={
            "assay": ref.assay,
            "artifact_id": ref.artifact_id,
            "actual_kind": ref.kind,
        },
    )


def _payload_arrays(
    root: zarr.Group,
    ref: ArtifactRef,
    names: tuple[str, str],
) -> tuple[zarr.Group, zarr.Array, zarr.Array]:
    try:
        group = artifact_group(root, ref)
        first = as_zarr_array(group[names[0]], name=names[0])
        second = as_zarr_array(group[names[1]], name=names[1])
    except Exception as error:
        raise payload_error(
            ref, f"{ref.kind} artifact payload is unreadable"
        ) from error
    return group, first, second


def _payload_dimensions(group: zarr.Group, ref: ArtifactRef) -> tuple[int, int]:
    """Return the cell and neighbor counts a graph payload records."""
    raw_cells = group.attrs.get("n_cells")
    raw_neighbors = group.attrs.get("n_neighbors")
    if (
        isinstance(raw_cells, bool)
        or not isinstance(raw_cells, int | np.integer)
        or not 1 <= int(raw_cells) <= np.iinfo(np.uint32).max
        or isinstance(raw_neighbors, bool)
        or not isinstance(raw_neighbors, int | np.integer)
        or int(raw_neighbors) < 1
    ):
        raise payload_error(
            ref,
            f"{ref.kind} artifact has invalid n_cells or n_neighbors metadata",
        )
    n_cells = int(raw_cells)
    n_neighbors = int(raw_neighbors)
    if n_neighbors >= n_cells:
        raise payload_error(ref, f"{ref.kind} artifact has an invalid neighbor count")
    return n_cells, n_neighbors


def _payload_block_rows(*arrays: zarr.Array) -> int:
    return min(
        row_band(array_geometry(array), unit="chunk", fallback=1) for array in arrays
    )


def _payload_blocks(
    ref: ArtifactRef,
    first: zarr.Array,
    second: zarr.Array,
    n_rows: int,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield aligned row blocks of two payload arrays in bounded reads."""
    block_rows = _payload_block_rows(first, second)
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        try:
            first_block = np.asarray(first[start:stop])
            second_block = np.asarray(second[start:stop])
        except Exception as error:
            raise payload_error(ref, f"{ref.kind} arrays are unreadable") from error
        yield start, stop, first_block, second_block


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
    source_metric = (
        inspect_artifact(zw, status.input_ref("ann_index")).parameters or {}
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
    group, indices, distances = _payload_arrays(root, ref, ("indices", "distances"))
    n_cells, n_neighbors = _payload_dimensions(group, ref)
    raw_self_hit_rate = group.attrs.get("self_hit_rate")
    if (
        isinstance(raw_self_hit_rate, bool)
        or not isinstance(raw_self_hit_rate, int | float | np.integer | np.floating)
        or not math.isfinite(float(raw_self_hit_rate))
        or not 0 <= float(raw_self_hit_rate) <= 100
    ):
        raise payload_error(ref, "neighbors artifact has an invalid self_hit_rate")
    expected_shape = (n_cells, n_neighbors)
    if (
        indices.ndim != 2
        or tuple(map(int, indices.shape)) != expected_shape
        or np.dtype(indices.dtype) != np.dtype(np.uint32)
        or distances.ndim != 2
        or tuple(map(int, distances.shape)) != expected_shape
        or np.dtype(distances.dtype) != np.dtype(np.float32)
    ):
        raise payload_error(
            ref, "neighbors arrays do not match their stored dimensions"
        )

    for start, stop, index_block, distance_block in _payload_blocks(
        ref, indices, distances, n_cells
    ):
        row_ids = np.arange(start, stop, dtype=np.uint32)[:, None]
        if (
            np.any(index_block >= n_cells)
            or np.any(index_block == row_ids)
            or not np.all(np.isfinite(distance_block))
            or np.any(distance_block < 0)
        ):
            raise payload_error(
                ref,
                "neighbors arrays contain invalid indices or distances",
            )
    return ValidatedNeighborsPayload(
        indices=indices,
        distances=distances,
        n_cells=n_cells,
        n_neighbors=n_neighbors,
    )


def validate_connectivity_payload(root: zarr.Group, ref: ArtifactRef) -> int:
    """Validate a persisted connectivity map in bounded blocks; return its cells."""
    if ref.kind != "connectivity_map":
        raise ValueError(
            "Connectivity payload validation requires a connectivity_map artifact"
        )
    group, edges, weights = _payload_arrays(root, ref, ("edges", "weights"))
    n_cells, n_neighbors = _payload_dimensions(group, ref)
    n_edges = n_cells * n_neighbors
    if (
        edges.ndim != 2
        or tuple(map(int, edges.shape)) != (n_edges, 2)
        or np.dtype(edges.dtype) != np.dtype(np.uint32)
        or weights.ndim != 1
        or tuple(map(int, weights.shape)) != (n_edges,)
        or np.dtype(weights.dtype) != np.dtype(np.float32)
    ):
        raise payload_error(
            ref,
            "connectivity_map arrays do not match their stored dimensions",
        )

    row_counts = np.zeros(n_cells, dtype=np.uint64)
    for _start, _stop, edge_block, weight_block in _payload_blocks(
        ref, edges, weights, n_edges
    ):
        if (
            np.any(edge_block >= n_cells)
            or not np.all(np.isfinite(weight_block))
            or np.any(weight_block < 0)
        ):
            raise payload_error(
                ref,
                "connectivity_map arrays contain invalid edge or weight values",
            )
        row_counts += np.bincount(
            edge_block[:, 0],
            minlength=n_cells,
        ).astype(np.uint64, copy=False)
    if np.any(row_counts != n_neighbors):
        raise payload_error(ref, "connectivity_map rows do not match n_neighbors")
    return n_cells


def validate_integration_source_payload(root: zarr.Group, ref: ArtifactRef) -> int:
    """Validate a connectivity-map or neighbors integration source; return its cells."""
    if ref.kind == "connectivity_map":
        return validate_connectivity_payload(root, ref)
    if ref.kind == "neighbors":
        return validate_neighbors_payload(root, ref).n_cells
    raise payload_error(ref, "Integration source has an unsupported artifact kind")
