from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import zarr

from ..graph.feature_projection import (
    resolve_coordinate_inputs,
    resolve_native_graph_inputs,
)
from ..metrics.cluster_selection import (
    DEFAULT_MIN_CLUSTER_QUOTA,
    SHARED_CLUSTER_QUOTA_STRATEGY,
    ClusterSelectionResult,
    select_clusters_by_silhouette,
    shared_cluster_quota_sample_indices,
)
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.artifacts import (
    ArtifactRef,
    artifact_group,
    require_complete_artifact,
)
from ..storage.arrays import create_zarr_dataset
from ..storage.errors import ArtifactResolutionError
from ..storage.types import as_zarr_array
from ..utils.shutdown import shutdown_checkpoint

_COORDINATE_OPERATIONS = {
    "reduction": "run_pca",
    "batch_correction": "run_harmony",
}


def cluster_label_array(root: zarr.Group, ref: ArtifactRef) -> zarr.Array:
    group = artifact_group(root, ref)
    name = "labels" if ref.kind == "cluster_cut" else "values"
    if name not in group:
        raise ValueError(f"Cluster candidate {ref!r} has no {name!r} array")
    values = as_zarr_array(group[name], name=name)
    if values.ndim != 1:
        raise ValueError(f"Cluster candidate {ref!r} is not one-dimensional")
    if np.dtype(values.dtype).kind not in {"i", "u"}:
        raise TypeError(f"Cluster candidate {ref!r} labels must be integers")
    return values


def cluster_label_values(root: zarr.Group, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(cluster_label_array(root, ref)[:])


def _artifact_input_ref(
    status_inputs: Mapping[str, Any] | None,
    name: str,
    *,
    owner: str,
) -> ArtifactRef:
    raw = (status_inputs or {}).get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{owner} has no {name!r} artifact input")
    try:
        return ArtifactRef.from_dict(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{owner} has a malformed {name!r} artifact input") from error


def _lineage_error(error: ArtifactResolutionError) -> ValueError:
    return ValueError(str(error))


def _candidate_graph(
    status_inputs: Mapping[str, Any] | None,
    *,
    key: str,
) -> ArtifactRef:
    owner = f"Cluster candidate {key!r}"
    return _artifact_input_ref(status_inputs, "graph", owner=owner)


def _validate_inputs(
    store: Any,
    *,
    coordinates: ArtifactRef,
    connectivity_map: ArtifactRef,
    cell_selection: ArtifactRef,
    candidates: Sequence[tuple[str, ArtifactRef]],
) -> tuple[zarr.Array, tuple[tuple[str, ArtifactRef, zarr.Array], ...]]:
    if not isinstance(coordinates, ArtifactRef):
        raise TypeError("coordinates must be an ArtifactRef")
    expected_operation = _COORDINATE_OPERATIONS.get(coordinates.kind)
    if (
        expected_operation is None
        or coordinates.scope != "assay"
        or coordinates.assay is None
    ):
        raise ValueError(
            "coordinates must be an assay-scoped PCA reduction or Harmony "
            "batch-correction artifact"
        )
    coordinate_status = require_complete_artifact(store.zw, coordinates)
    if coordinate_status.operation != expected_operation:
        raise ValueError(
            "coordinates must reference a "
            f"{'PCA' if coordinates.kind == 'reduction' else 'Harmony'} artifact"
        )
    if not isinstance(connectivity_map, ArtifactRef):
        raise TypeError("connectivity_map must be an ArtifactRef")
    if not isinstance(cell_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef")
    try:
        coordinate_inputs = resolve_coordinate_inputs(store.zw, coordinates)
        graph_inputs = resolve_native_graph_inputs(store.zw, connectivity_map)
    except ArtifactResolutionError as error:
        raise _lineage_error(error) from error
    if coordinate_inputs.cell_selection != cell_selection:
        raise ValueError("coordinates do not use the requested cell selection")
    if graph_inputs.coordinates != coordinates:
        raise ValueError("connectivity map was not built from the scored coordinates")
    if graph_inputs.cell_selection != cell_selection:
        raise ValueError("connectivity map does not use the requested cell selection")
    coordinate_group = artifact_group(store.zw, coordinates)
    if "data" not in coordinate_group:
        raise ValueError("Coordinate artifact is missing its data array")
    scored = as_zarr_array(coordinate_group["data"], name="coordinates")
    if scored.ndim != 2 or scored.shape[0] < 1 or scored.shape[1] < 1:
        raise ValueError("Coordinates must be a non-empty two-dimensional array")

    resolved_candidates = tuple(candidates)
    if not resolved_candidates:
        raise ValueError("Cluster selection requires at least one candidate")
    candidate_keys: list[str] = []
    validated: list[tuple[str, ArtifactRef, zarr.Array]] = []
    for candidate in resolved_candidates:
        if not isinstance(candidate, tuple) or len(candidate) != 2:
            raise TypeError("candidates must contain (key, ArtifactRef) tuples")
        key, ref = candidate
        if not isinstance(key, str) or not key:
            raise TypeError("Cluster selection keys must be non-empty strings")
        if not isinstance(ref, ArtifactRef):
            raise TypeError(f"Cluster candidate {key!r} must be an ArtifactRef")
        if (
            ref.kind != "cluster_labels"
            or ref.scope != "assay"
            or ref.assay != coordinates.assay
        ):
            raise ValueError(
                f"Cluster candidate {key!r} must be an assay-scoped Leiden "
                "cluster-label artifact for the coordinate assay"
            )
        status = require_complete_artifact(store.zw, ref)
        if status.operation != "run_leiden_clustering":
            raise ValueError(
                f"Cluster candidate {key!r} must reference a Leiden clustering artifact"
            )
        candidate_selection = _artifact_input_ref(
            status.inputs,
            "cell_selection",
            owner=f"Cluster candidate {key!r}",
        )
        if candidate_selection != cell_selection:
            raise ValueError(
                f"Cluster candidate {key!r} does not use the requested cell selection"
            )
        candidate_graph = _candidate_graph(status.inputs, key=key)
        if candidate_graph != connectivity_map:
            raise ValueError(
                f"Cluster candidate {key!r} was not partitioned from the "
                "requested connectivity map"
            )
        labels = cluster_label_array(store.zw, ref)
        if labels.shape[0] != scored.shape[0]:
            raise ValueError(
                f"Cluster candidate {key!r} does not align with coordinate rows"
            )
        candidate_keys.append(key)
        validated.append((key, ref, labels))
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Cluster selection candidate keys must be unique")
    return scored, tuple(validated)


def _validated_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return resolved


def _cluster_selection_reuse_validator(
    *,
    candidate_keys: tuple[str, ...],
    candidate_refs: tuple[ArtifactRef, ...],
    candidate_labels: tuple[zarr.Array, ...],
    seed: int,
    population_size: int,
    max_sample_size: int,
    min_cluster_quota: int,
    working_memory_mib: int,
) -> Callable[[ArtifactRef, zarr.Group], bool]:
    sample_size = min(population_size, max_sample_size)
    expected_indices = shared_cluster_quota_sample_indices(
        tuple(zip(candidate_keys, candidate_labels, strict=True)),
        n_cells=population_size,
        seed=seed,
        max_sample_size=max_sample_size,
        min_cluster_quota=min_cluster_quota,
    )
    expected_refs = [ref.to_dict() for ref in candidate_refs]
    expected_sample_definition = {
        "seed": seed,
        "populationSize": population_size,
        "sampleSize": sample_size,
        "maxSampleSize": max_sample_size,
        "sampleStrategy": SHARED_CLUSTER_QUOTA_STRATEGY,
        "minClusterQuota": min_cluster_quota,
    }

    def validate(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            raw_keys = group.attrs["candidateKeys"]
            raw_tie_order = group.attrs["tieOrder"]
            raw_refs = group.attrs["candidateRefs"]
            raw_reasons = group.attrs["invalidReasons"]
            selected_key = group.attrs["selectedKey"]
            if not isinstance(raw_keys, list | tuple) or any(
                not isinstance(key, str) for key in raw_keys
            ):
                return False
            if tuple(raw_keys) != candidate_keys:
                return False
            if not isinstance(raw_tie_order, list | tuple) or any(
                not isinstance(key, str) for key in raw_tie_order
            ):
                return False
            if tuple(raw_tie_order) != candidate_keys:
                return False
            if not isinstance(raw_refs, list | tuple) or any(
                not isinstance(ref, Mapping) for ref in raw_refs
            ):
                return False
            if list(raw_refs) != expected_refs:
                return False
            if not isinstance(raw_reasons, list | tuple) or any(
                reason is not None and not isinstance(reason, str)
                for reason in raw_reasons
            ):
                return False
            if not isinstance(selected_key, str):
                return False
            sample_definition = group.attrs["sampleDefinition"]
            if (
                not isinstance(sample_definition, Mapping)
                or dict(sample_definition) != expected_sample_definition
            ):
                return False
            sample_indices = np.asarray(
                as_zarr_array(group["sample_indices"], name="sample_indices")[:]
            )
            if not np.array_equal(sample_indices, expected_indices):
                return False
            scores = np.asarray(as_zarr_array(group["scores"], name="scores")[:])
            result = ClusterSelectionResult(
                candidate_keys=candidate_keys,
                sample_indices=sample_indices,
                scores=scores,
                invalid_reasons=tuple(raw_reasons),
                selected_key=selected_key,
                seed=seed,
                population_size=population_size,
                max_sample_size=max_sample_size,
                working_memory_mib=working_memory_mib,
                min_cluster_quota=min_cluster_quota,
            )
            return result.tie_order == candidate_keys
        except (KeyError, TypeError, ValueError):
            return False

    return validate


def run_cluster_selection(
    store: Any,
    *,
    coordinates: ArtifactRef,
    connectivity_map: ArtifactRef,
    cell_selection: ArtifactRef,
    candidates: Sequence[tuple[str, ArtifactRef]],
    seed: int = 4466,
    max_sample_size: int = 10_000,
    min_cluster_quota: int = DEFAULT_MIN_CLUSTER_QUOTA,
) -> tuple[ArtifactRef, str, ArtifactRef]:
    """Validate, score, and persist one bounded cluster-selection decision."""
    seed = _validated_integer(seed, "seed", minimum=0)
    max_sample_size = _validated_integer(
        max_sample_size,
        "max_sample_size",
        minimum=1,
    )
    min_cluster_quota = _validated_integer(
        min_cluster_quota,
        "min_cluster_quota",
        minimum=1,
    )
    scored, validated = _validate_inputs(
        store,
        coordinates=coordinates,
        connectivity_map=connectivity_map,
        cell_selection=cell_selection,
        candidates=candidates,
    )
    candidate_keys = tuple(key for key, _ref, _labels in validated)
    candidate_refs = tuple(ref for _key, ref, _labels in validated)
    candidate_labels = tuple(labels for _key, _ref, labels in validated)
    population_size = int(scored.shape[0])
    sample_size = min(population_size, max_sample_size)
    working_memory_mib = max(
        1,
        min(1024, int(store.memoryBytes // 4 // (1024**2))),
    )
    planned = plan_artifact(
        store.zw,
        scope="assay",
        assay=coordinates.assay,
        kind="cluster_selection",
        operation="select_clusters_by_silhouette",
        parameters={
            "candidateKeys": list(candidate_keys),
            "seed": seed,
            "maxSampleSize": max_sample_size,
            "metric": "euclidean",
            "tieOrder": list(candidate_keys),
            "sampleStrategy": SHARED_CLUSTER_QUOTA_STRATEGY,
            "minClusterQuota": min_cluster_quota,
        },
        inputs={
            "coordinates": coordinates,
            "connectivityMap": connectivity_map,
            "cellSelection": cell_selection,
            "candidates": {key: ref for key, ref, _labels in validated},
        },
        execution_options={"workingMemoryMiB": working_memory_mib},
        required_arrays=(
            ArrayRequirement(
                "sample_indices",
                shape=(sample_size,),
                dtype=np.int64,
            ),
            ArrayRequirement(
                "scores",
                shape=(len(validated),),
                dtype=np.float64,
            ),
        ),
        required_attributes=(
            AttributeRequirement("candidateKeys", expected_types=(list, tuple)),
            AttributeRequirement("candidateRefs", expected_types=(list, tuple)),
            AttributeRequirement("invalidReasons", expected_types=(list, tuple)),
            AttributeRequirement("selectedKey", expected_types=(str,)),
            AttributeRequirement("sampleDefinition", expected_types=(Mapping,)),
            AttributeRequirement("tieOrder", expected_types=(list, tuple)),
        ),
        reuse_validator=_cluster_selection_reuse_validator(
            candidate_keys=candidate_keys,
            candidate_refs=candidate_refs,
            candidate_labels=candidate_labels,
            seed=seed,
            population_size=population_size,
            max_sample_size=max_sample_size,
            min_cluster_quota=min_cluster_quota,
            working_memory_mib=working_memory_mib,
        ),
    )
    refs_by_key = {key: ref for key, ref, _labels in validated}
    if planned.reused:
        selected_key = artifact_group(store.zw, planned.ref).attrs["selectedKey"]
        if not isinstance(selected_key, str) or selected_key not in refs_by_key:
            raise ValueError("Stored cluster selection has an invalid selected key")
        return planned.ref, selected_key, refs_by_key[selected_key]

    result = select_clusters_by_silhouette(
        scored,
        tuple((key, labels) for key, _ref, labels in validated),
        seed=seed,
        max_sample_size=max_sample_size,
        working_memory_mib=working_memory_mib,
        min_cluster_quota=min_cluster_quota,
        checkpoint=shutdown_checkpoint,
    )
    group = start_artifact(store.zw, planned)
    sample_array = create_zarr_dataset(
        group,
        "sample_indices",
        (min(result.sample_size, 100_000),),
        np.int64,
        result.sample_indices.shape,
    )
    sample_array[:] = result.sample_indices
    score_array = create_zarr_dataset(
        group,
        "scores",
        (max(1, len(result.candidate_keys)),),
        np.float64,
        result.scores.shape,
    )
    score_array[:] = result.scores
    group.attrs.update(
        {
            "candidateKeys": list(result.candidate_keys),
            "candidateRefs": [ref.to_dict() for ref in candidate_refs],
            "invalidReasons": list(result.invalid_reasons),
            "selectedKey": result.selected_key,
            "sampleDefinition": dict(result.sample_definition),
            "tieOrder": list(result.tie_order),
        }
    )
    finish_artifact(group, planned)
    return planned.ref, result.selected_key, refs_by_key[result.selected_key]
