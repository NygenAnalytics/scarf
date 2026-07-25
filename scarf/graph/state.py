import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import zarr

from ..storage.artifacts import (
    ArtifactRef,
    artifact_path,
    fingerprint_strings,
    inspect_artifact,
)
from ..storage.types import as_zarr_array, as_zarr_group
from .paths import AssayGraphPaths, StoredAssayGraph

_GRAPH_REF_KINDS = {
    "normalized": "normalized",
    "feature_scaling": "feature_scaling",
    "reduction": "reduction",
    "batch_correction": "batch_correction",
    "ann_index": "ann_index",
    "embedding_initialization": "embedding_initialization",
    "neighbors": "neighbors",
    "connectivity_map": "connectivity_map",
}
_RESULT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_NAMED_RESULT_KINDS = {"mapping_reference": "mapping_reference"}


def _legacy_subset_hash(
    cell_indices: np.ndarray,
    feature_indices: np.ndarray,
) -> int:
    cells = tuple(np.asarray(cell_indices))
    features = tuple(np.asarray(feature_indices))
    return hash((hash(cells), hash(features)))


def validate_legacy_graph_selection(
    store: Any,
    graph_loc: str,
    from_assay: str,
    cell_key: str,
    feat_key: str,
) -> None:
    from .encoded_paths import (
        nearest_neighbor_paths_from_loc,
        parse_assay_graph_paths,
    )

    try:
        normalized_path = parse_assay_graph_paths(graph_loc).paths.normalized_group_path
    except ValueError:
        try:
            normalized_path = nearest_neighbor_paths_from_loc(
                graph_loc
            ).normalized_group_path
        except ValueError as exc:
            raise ValueError(
                "Legacy graph selection provenance cannot be resolved"
            ) from exc
    if normalized_path not in store.zw:
        raise ValueError("Legacy graph normalized data is missing")
    normalized_group = as_zarr_group(
        store.zw[normalized_path],
        name=normalized_path,
    )
    stored_hash = normalized_group.attrs.get("subset_hash")
    assay = store._get_assay(from_assay)
    feature_column = "I" if feat_key == "I" else f"{cell_key}__{feat_key}"
    try:
        cell_indices = store.cells.active_index(cell_key)
        feature_indices = assay.feats.active_index(feature_column)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Legacy graph selection columns cannot be validated") from exc
    if stored_hash is None:
        if "data" not in normalized_group:
            raise ValueError("Legacy graph selection provenance is missing")
        normalized_data = as_zarr_array(
            normalized_group["data"],
            name=f"{normalized_path}/data",
        )
        if normalized_data.shape != (len(cell_indices), len(feature_indices)):
            raise ValueError(
                "Legacy graph dimensions do not match the current selection"
            )
        warnings.warn(
            "Legacy graph predates exact selection provenance; validation is "
            "limited to cell and feature counts.",
            DeprecationWarning,
            stacklevel=2,
        )
        return
    if isinstance(stored_hash, str):
        current_hash: str | int = assay._create_subset_hash(
            cell_indices,
            feature_indices,
        )
    elif not isinstance(stored_hash, bool) and isinstance(
        stored_hash,
        (int, np.integer),
    ):
        current_hash = _legacy_subset_hash(cell_indices, feature_indices)
        stored_hash = int(stored_hash)
    else:
        raise ValueError("Legacy graph selection provenance has an invalid type")
    if current_hash != stored_hash:
        raise ValueError(
            "cell_key or feat_key no longer matches the legacy graph selection"
        )


@dataclass(frozen=True, slots=True)
class AssayState:
    assay: str
    cell_key: str
    feat_key: str
    normalized: ArtifactRef | None = None
    feature_scaling: ArtifactRef | None = None
    reduction: ArtifactRef | None = None
    batch_correction: ArtifactRef | None = None
    ann_index: ArtifactRef | None = None
    embedding_initialization: ArtifactRef | None = None
    neighbors: ArtifactRef | None = None
    connectivity_map: ArtifactRef | None = None
    named_results: Mapping[str, ArtifactRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.assay or "/" in self.assay:
            raise ValueError("AssayState requires a valid assay name")
        if not self.cell_key or not self.feat_key:
            raise ValueError("AssayState requires cell_key and feat_key")
        for field_name, expected_kind in _GRAPH_REF_KINDS.items():
            ref = getattr(self, field_name)
            if ref is not None:
                self._validate_ref(ref, expected_kind, field_name)
        for name, ref in self.named_results.items():
            if _RESULT_NAME_PATTERN.fullmatch(name) is None:
                raise ValueError("named_results keys must be snake_case identifiers")
            self._validate_ref(ref, ref.kind, f"named_results[{name!r}]")
            expected_named_kind = _NAMED_RESULT_KINDS.get(name)
            if expected_named_kind is not None and ref.kind != expected_named_kind:
                raise ValueError(
                    f"named_results[{name!r}] requires kind {expected_named_kind!r}"
                )
        object.__setattr__(
            self,
            "named_results",
            MappingProxyType(dict(self.named_results)),
        )

    def _validate_ref(
        self,
        ref: ArtifactRef,
        expected_kind: str,
        field_name: str,
    ) -> None:
        if ref.scope != "assay" or ref.assay != self.assay:
            raise ValueError(f"{field_name} must reference assay {self.assay!r}")
        if ref.kind != expected_kind:
            raise ValueError(
                f"{field_name} requires kind {expected_kind!r}, got {ref.kind!r}"
            )

    def matches(self, cell_key: str, feat_key: str) -> bool:
        return self.cell_key == cell_key and self.feat_key == feat_key

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "assay": self.assay,
            "cell_key": self.cell_key,
            "feat_key": self.feat_key,
            "named_results": {
                name: ref.to_dict() for name, ref in sorted(self.named_results.items())
            },
        }
        for field_name in _GRAPH_REF_KINDS:
            ref = getattr(self, field_name)
            value[field_name] = ref.to_dict() if ref is not None else None
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssayState":
        refs: dict[str, ArtifactRef | None] = {}
        for field_name in _GRAPH_REF_KINDS:
            raw_ref = value.get(field_name)
            if raw_ref is None:
                refs[field_name] = None
            elif isinstance(raw_ref, Mapping):
                refs[field_name] = ArtifactRef.from_dict(raw_ref)
            else:
                raise TypeError(f"{field_name} must be an artifact reference or null")
        raw_named = value.get("named_results", {})
        if not isinstance(raw_named, Mapping):
            raise TypeError("named_results must be a mapping")
        named_results = {}
        for name, raw_ref in raw_named.items():
            if not isinstance(name, str) or not isinstance(raw_ref, Mapping):
                raise TypeError("Every named result must be a named artifact reference")
            named_results[name] = ArtifactRef.from_dict(raw_ref)
        assay = value.get("assay")
        cell_key = value.get("cell_key")
        feat_key = value.get("feat_key")
        if (
            not isinstance(assay, str)
            or not isinstance(cell_key, str)
            or not isinstance(feat_key, str)
        ):
            raise TypeError("Assay state assay, cell_key, and feat_key must be strings")
        return cls(
            assay=assay,
            cell_key=cell_key,
            feat_key=feat_key,
            named_results=named_results,
            **refs,
        )


def assay_state_path(assay: str) -> str:
    if not assay or "/" in assay:
        raise ValueError("Invalid assay name")
    return f"{assay}/state"


def read_assay_state(root: zarr.Group, assay: str) -> AssayState | None:
    path = assay_state_path(assay)
    if path not in root:
        return None
    group = as_zarr_group(root[path], name=path)
    if "state" not in group.attrs:
        return None
    value = group.attrs["state"]
    if not isinstance(value, Mapping):
        raise TypeError(f"Assay state at {path} must be a mapping")
    state = AssayState.from_dict(value)
    if state.assay != assay:
        raise ValueError(f"Assay state at {path} names assay {state.assay!r}")
    return state


def write_assay_state(root: zarr.Group, state: AssayState) -> None:
    for field_name in _GRAPH_REF_KINDS:
        ref = getattr(state, field_name)
        if ref is None:
            continue
        status = inspect_artifact(root, ref)
        if not status.exists or not status.complete:
            raise RuntimeError(
                f"Cannot select incomplete {field_name} artifact: {status.path}"
            )
    for name, ref in state.named_results.items():
        status = inspect_artifact(root, ref)
        if not status.exists or not status.complete:
            raise RuntimeError(
                f"Cannot select incomplete named result {name!r}: {status.path}"
            )
        if name == "mapping_reference":
            expected = {
                "reduction": state.reduction,
                "batch_correction": state.batch_correction,
                "ann_index": state.ann_index,
                "neighbors": state.neighbors,
            }
            missing = [
                input_name
                for input_name, input_ref in expected.items()
                if input_ref is None
            ]
            if missing:
                raise ValueError(
                    "Mapping reference state is missing " + ", ".join(missing)
                )
            inputs = status.inputs or {}
            for input_name, input_ref in expected.items():
                assert input_ref is not None
                if inputs.get(input_name) != input_ref.to_dict():
                    raise ValueError(
                        "Mapping reference input "
                        f"{input_name!r} does not match AssayState"
                    )
    if state.connectivity_map is not None:
        _stored_assay_graph(root, state)
    path = assay_state_path(state.assay)
    group = (
        as_zarr_group(root[path], name=path)
        if path in root
        else root.create_group(path)
    )
    group.attrs["state"] = state.to_dict()


def _complete_path(root: zarr.Group, ref: ArtifactRef, field_name: str) -> str:
    status = inspect_artifact(root, ref)
    if not status.exists:
        raise KeyError(
            f"Assay state {field_name} artifact does not exist: {status.path}"
        )
    if not status.complete:
        raise RuntimeError(
            f"Assay state {field_name} artifact is incomplete: {status.path}"
        )
    return status.path


def normalized_path_from_state(
    root: zarr.Group,
    assay: str,
    cell_key: str,
    feat_key: str,
) -> str | None:
    state = read_assay_state(root, assay)
    if state is None or not state.matches(cell_key, feat_key):
        return None
    if state.normalized is None:
        raise KeyError("AssayState has no selected normalized artifact")
    return _complete_path(root, state.normalized, "normalized")


def embedding_initialization_path_from_state(
    root: zarr.Group,
    assay: str,
    cell_key: str,
    feat_key: str,
) -> str | None:
    state = read_assay_state(root, assay)
    if state is None or not state.matches(cell_key, feat_key):
        return None
    if state.embedding_initialization is None:
        raise KeyError("AssayState has no selected embedding initialization")
    return _complete_path(
        root,
        state.embedding_initialization,
        "embedding_initialization",
    )


def _parameters(root: zarr.Group, ref: ArtifactRef | None) -> dict[str, Any]:
    if ref is None:
        return {}
    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        return {}
    return status.parameters or {}


def _input_ref(root: zarr.Group, ref: ArtifactRef, name: str) -> ArtifactRef:
    status = inspect_artifact(root, ref)
    if not status.exists:
        raise KeyError(f"Artifact input owner does not exist: {status.path}")
    if not status.complete:
        raise RuntimeError(f"Artifact input owner is incomplete: {status.path}")
    inputs = status.inputs or {}
    value = inputs.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{ref.kind} artifact has no {name!r} artifact input")
    input_ref = ArtifactRef.from_dict(value)
    input_status = inspect_artifact(root, input_ref)
    if not input_status.exists:
        raise KeyError(f"Artifact input does not exist: {input_status.path}")
    if not input_status.complete:
        raise RuntimeError(f"Artifact input is incomplete: {input_status.path}")
    return input_ref


def _require_input(
    root: zarr.Group,
    ref: ArtifactRef,
    name: str,
    expected: ArtifactRef,
) -> None:
    actual = _input_ref(root, ref, name)
    if actual != expected:
        raise ValueError(
            f"{ref.kind} artifact input {name!r} does not match AssayState"
        )


def _stored_assay_graph(root: zarr.Group, state: AssayState) -> StoredAssayGraph:
    required = {
        "normalized": state.normalized,
        "feature_scaling": state.feature_scaling,
        "reduction": state.reduction,
        "ann_index": state.ann_index,
        "neighbors": state.neighbors,
        "connectivity_map": state.connectivity_map,
    }
    if any(ref is None for ref in required.values()):
        missing = [name for name, ref in required.items() if ref is None]
        raise KeyError(
            f"AssayState has no complete selected graph; missing {', '.join(missing)}"
        )
    assert state.normalized is not None
    assert state.feature_scaling is not None
    assert state.reduction is not None
    assert state.ann_index is not None
    assert state.neighbors is not None
    assert state.connectivity_map is not None
    _require_input(
        root,
        state.reduction,
        "normalized",
        state.normalized,
    )
    _require_input(
        root,
        state.feature_scaling,
        "normalized",
        state.normalized,
    )
    _require_input(
        root,
        state.reduction,
        "feature_scaling",
        state.feature_scaling,
    )
    coordinates = state.reduction
    if state.batch_correction is not None:
        _require_input(
            root,
            state.batch_correction,
            "reduction",
            state.reduction,
        )
        coordinates = state.batch_correction
    _require_input(root, state.ann_index, "coordinates", coordinates)
    _require_input(root, state.neighbors, "ann_index", state.ann_index)
    _require_input(root, state.neighbors, "coordinates", coordinates)
    _require_input(
        root,
        state.connectivity_map,
        "neighbors",
        state.neighbors,
    )
    if state.embedding_initialization is not None:
        _require_input(
            root,
            state.embedding_initialization,
            "reduction",
            state.reduction,
        )
    paths = {
        name: _complete_path(root, ref, name)
        for name, optional_ref in required.items()
        if (ref := optional_ref) is not None
    }
    initialization_path = (
        _complete_path(
            root,
            state.embedding_initialization,
            "embedding_initialization",
        )
        if state.embedding_initialization is not None
        else None
    )
    reduction_status = (
        inspect_artifact(root, state.reduction) if state.reduction is not None else None
    )
    reduction = (
        reduction_status.parameters or {} if reduction_status is not None else {}
    )
    reduction_execution = (
        inspect_artifact(root, state.reduction).execution_options or {}
        if state.reduction is not None
        else {}
    )
    ann_index = _parameters(root, state.ann_index)
    neighbors = _parameters(root, state.neighbors)
    connectivity = _parameters(root, state.connectivity_map)
    initialization = _parameters(root, state.embedding_initialization)
    reduction_operation = (
        reduction_status.operation if reduction_status is not None else None
    )
    reduction_method = (
        {
            "run_pca": "pca",
            "run_lsi": "lsi",
            "run_custom_reduction": "custom",
        }.get(reduction_operation)
        if reduction_operation is not None
        else None
    )
    if reduction_method is None:
        reduction_method = _optional_str(reduction.get("reduction_method"))
    reduction_dims: int | None
    if reduction_operation in {"run_pca", "run_lsi"}:
        reduction_dims = _required_int(
            reduction,
            "dims",
            "reduction",
        )
    else:
        reduction_dims = _optional_int(reduction.get("dims"))
    if reduction_dims is None and state.reduction is not None:
        reduction_group = as_zarr_group(
            root[artifact_path(state.reduction)],
            name=artifact_path(state.reduction),
        )
        if "loadings" in reduction_group:
            reduction_dims = int(
                as_zarr_array(
                    reduction_group["loadings"],
                    name="loadings",
                ).shape[1]
            )
    pca_cell_key: str | None
    feat_scaling: bool | None
    if reduction_operation == "run_pca":
        pca_cell_key = _required_str(
            reduction_execution,
            "pca_cell_key",
            "reduction execution",
        )
        feat_scaling = _required_bool(
            reduction,
            "feat_scaling",
            "reduction",
        )
    elif reduction_operation in {"run_lsi", "run_custom_reduction"}:
        pca_cell_key = None
        feat_scaling = False
    else:
        pca_cell_key = _optional_str(
            reduction_execution.get(
                "pca_cell_key",
                reduction.get("pca_cell_key"),
            )
        )
        feat_scaling = _optional_bool(reduction.get("feat_scaling"))
    stored = StoredAssayGraph(
        paths=AssayGraphPaths(
            normalized_group_path=paths["normalized"],
            reduction_group_path=paths["reduction"],
            neighbor_index_group_path=paths["ann_index"],
            nearest_neighbors_group_path=paths["neighbors"],
            cell_graph_group_path=paths["connectivity_map"],
            kmeans_initialization_group_path=initialization_path,
        ),
        from_assay=state.assay,
        cell_key=state.cell_key,
        feat_key=state.feat_key,
        reduction_method=reduction_method,
        dims=reduction_dims,
        pca_cell_key=pca_cell_key,
        ann_metric=_required_str(ann_index, "ann_metric", "ann_index"),
        ann_efc=_required_int(ann_index, "ann_efc", "ann_index"),
        ann_ef=_required_int(ann_index, "ann_ef", "ann_index"),
        ann_m=_required_int(ann_index, "ann_m", "ann_index"),
        rand_state=_required_int(ann_index, "rand_state", "ann_index"),
        k=_required_int(neighbors, "k", "neighbors"),
        local_connectivity=_required_float(
            connectivity,
            "local_connectivity",
            "connectivity_map",
        ),
        bandwidth=_required_float(
            connectivity,
            "bandwidth",
            "connectivity_map",
        ),
        feat_scaling=feat_scaling,
        n_centroids=(
            _required_int(
                initialization,
                "n_centroids",
                "embedding_initialization",
            )
            if state.embedding_initialization is not None
            else None
        ),
    )
    return stored


def stored_assay_graph_from_state(
    root: zarr.Group,
    assay: str,
    cell_key: str,
    feat_key: str,
) -> StoredAssayGraph | None:
    state = read_assay_state(root, assay)
    if state is None or not state.matches(cell_key, feat_key):
        return None
    stored = _stored_assay_graph(root, state)
    assert state.connectivity_map is not None
    validate_artifact_graph_selection(
        root,
        state.connectivity_map,
        state.cell_key,
        state.feat_key,
    )
    return stored


def _require_artifact_ref(
    ref: ArtifactRef,
    *,
    kind: str,
    scope: str,
    assay: str | None,
) -> None:
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise ValueError(f"Expected {scope}-scoped {kind} artifact")


def _validate_selection_artifact(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: str,
    assay: str | None,
    table_path: str,
    column: str,
) -> None:
    _require_artifact_ref(
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
    )
    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        raise ValueError(f"{kind} artifact is incomplete")
    if table_path not in root:
        raise ValueError(f"Selection table {table_path!r} is unavailable")
    table = as_zarr_group(root[table_path], name=table_path)
    if column not in table or "ids" not in table:
        raise ValueError(f"Selection source column {column!r} is unavailable")
    selection_group = as_zarr_group(
        root[artifact_path(ref)],
        name=artifact_path(ref),
    )
    if "values" not in selection_group:
        raise ValueError(f"{kind} artifact has no values")
    stored_values = np.asarray(
        as_zarr_array(selection_group["values"], name="values")[:]
    )
    current_values = np.asarray(as_zarr_array(table[column], name=column)[:])
    row_ids = np.asarray(as_zarr_array(table["ids"], name="ids")[:])
    expected_row_ids = (status.inputs or {}).get("ordered_row_ids_fingerprint")
    if not isinstance(expected_row_ids, str) or expected_row_ids != fingerprint_strings(
        row_ids
    ):
        raise ValueError(f"{kind} row identity does not match its metadata table")
    if (
        stored_values.ndim != 1
        or stored_values.dtype != np.dtype(bool)
        or current_values.ndim != 1
        or current_values.dtype != np.dtype(bool)
        or stored_values.shape != current_values.shape
        or not np.array_equal(stored_values, current_values)
    ):
        raise ValueError(
            f"Selection source column {column!r} no longer matches its artifact"
        )


def validate_cell_selection_artifact(
    root: zarr.Group,
    selection: ArtifactRef,
    cell_key: str,
) -> None:
    _validate_selection_artifact(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
        column=cell_key,
    )


def validate_artifact_graph_selection(
    root: zarr.Group,
    connectivity_map: ArtifactRef,
    cell_key: str,
    feat_key: str,
) -> None:
    if connectivity_map.assay is None:
        raise ValueError("Connectivity-map artifact has no assay")
    assay = connectivity_map.assay
    _require_artifact_ref(
        connectivity_map,
        kind="connectivity_map",
        scope="assay",
        assay=assay,
    )
    neighbors = _input_ref(root, connectivity_map, "neighbors")
    _require_artifact_ref(
        neighbors,
        kind="neighbors",
        scope="assay",
        assay=assay,
    )
    ann_index = _input_ref(root, neighbors, "ann_index")
    _require_artifact_ref(
        ann_index,
        kind="ann_index",
        scope="assay",
        assay=assay,
    )
    coordinates = _input_ref(root, neighbors, "coordinates")
    if coordinates.kind == "batch_correction":
        _require_artifact_ref(
            coordinates,
            kind="batch_correction",
            scope="assay",
            assay=assay,
        )
        reduction = _input_ref(root, coordinates, "reduction")
    elif coordinates.kind == "reduction":
        reduction = coordinates
    else:
        raise ValueError("Neighbor coordinates must be reduction or batch_correction")
    _require_artifact_ref(
        reduction,
        kind="reduction",
        scope="assay",
        assay=assay,
    )
    if _input_ref(root, ann_index, "coordinates") != coordinates:
        raise ValueError("ANN index and neighbors use different coordinates")
    normalized = _input_ref(root, reduction, "normalized")
    _require_artifact_ref(
        normalized,
        kind="normalized",
        scope="assay",
        assay=assay,
    )
    validate_normalized_artifact_selection(
        root,
        normalized,
        cell_key,
        feat_key,
    )


def validate_normalized_artifact_selection(
    root: zarr.Group,
    normalized: ArtifactRef,
    cell_key: str,
    feat_key: str,
) -> None:
    if normalized.assay is None:
        raise ValueError("Normalized artifact has no assay")
    assay = normalized.assay
    _require_artifact_ref(
        normalized,
        kind="normalized",
        scope="assay",
        assay=assay,
    )
    cell_selection = _input_ref(root, normalized, "cell_selection")
    feature_selection = _input_ref(root, normalized, "feature_selection")
    feature_column = "I" if feat_key == "I" else f"{cell_key}__{feat_key}"
    validate_cell_selection_artifact(root, cell_selection, cell_key)
    _validate_selection_artifact(
        root,
        feature_selection,
        kind="feature_selection",
        scope="assay",
        assay=assay,
        table_path=f"{assay}/featureData",
        column=feature_column,
    )


def stored_assay_graph_from_ref(
    root: zarr.Group,
    connectivity_map: ArtifactRef,
) -> StoredAssayGraph:
    if connectivity_map.scope != "assay" or connectivity_map.assay is None:
        raise ValueError("Connectivity-map reference must be assay-scoped")
    if connectivity_map.kind != "connectivity_map":
        raise ValueError("Expected a connectivity_map artifact reference")
    neighbors = _input_ref(root, connectivity_map, "neighbors")
    ann_index = _input_ref(root, neighbors, "ann_index")
    coordinates = _input_ref(root, neighbors, "coordinates")
    if coordinates.kind == "batch_correction":
        batch_correction = coordinates
        reduction = _input_ref(root, batch_correction, "reduction")
    elif coordinates.kind == "reduction":
        batch_correction = None
        reduction = coordinates
    else:
        raise ValueError("Neighbor coordinates must be reduction or batch_correction")
    if _input_ref(root, ann_index, "coordinates") != coordinates:
        raise ValueError("ANN index and neighbors use different coordinates")
    normalized = _input_ref(root, reduction, "normalized")
    feature_scaling = _input_ref(root, reduction, "feature_scaling")
    _require_artifact_ref(
        feature_scaling,
        kind="feature_scaling",
        scope="assay",
        assay=connectivity_map.assay,
    )
    normalized_status = inspect_artifact(root, normalized)
    normalized_execution = normalized_status.execution_options or {}
    cell_key = normalized_execution.get("cell_key")
    feat_key = normalized_execution.get("feat_key")
    if not isinstance(cell_key, str) or not isinstance(feat_key, str):
        raise ValueError("Normalized artifact selection keys are missing")
    state = AssayState(
        assay=connectivity_map.assay,
        cell_key=cell_key,
        feat_key=feat_key,
        normalized=normalized,
        feature_scaling=feature_scaling,
        reduction=reduction,
        batch_correction=batch_correction,
        ann_index=ann_index,
        neighbors=neighbors,
        connectivity_map=connectivity_map,
    )
    stored = _stored_assay_graph(root, state)
    validate_artifact_graph_selection(
        root,
        connectivity_map,
        cell_key,
        feat_key,
    )
    return stored


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Optional provenance value must be a string")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("Optional provenance value must be an integer")
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("Optional provenance value must be numeric")
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("Optional provenance value must be boolean")
    return bool(value)


def _required_value(
    values: Mapping[str, Any],
    name: str,
    owner: str,
) -> Any:
    if name not in values:
        raise ValueError(f"{owner} provenance is missing {name!r}")
    return values[name]


def _required_str(
    values: Mapping[str, Any],
    name: str,
    owner: str,
) -> str:
    value = _required_value(values, name, owner)
    if not isinstance(value, str):
        raise TypeError(f"{owner} provenance {name!r} must be a string")
    return value


def _required_int(
    values: Mapping[str, Any],
    name: str,
    owner: str,
) -> int:
    value = _required_value(values, name, owner)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{owner} provenance {name!r} must be an integer")
    return int(value)


def _required_float(
    values: Mapping[str, Any],
    name: str,
    owner: str,
) -> float:
    value = _required_value(values, name, owner)
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{owner} provenance {name!r} must be numeric")
    return float(value)


def _required_bool(
    values: Mapping[str, Any],
    name: str,
    owner: str,
) -> bool:
    value = _required_value(values, name, owner)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{owner} provenance {name!r} must be boolean")
    return bool(value)
