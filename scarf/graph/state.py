import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import zarr

from ..storage.arrays import create_metadata_column, create_numeric_array
from ..storage.artifact_writer import (
    ArrayRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    artifact_group,
    artifact_path,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    fingerprint_strings,
    group_at,
    inspect_artifact,
    parse_artifact_path,
    require_complete_artifact,
)
from ..storage.geometry import array_geometry
from ..storage.layout import _group_zarr_format, row_sharded_array_spec
from ..storage.partition import row_band
from ..storage.profiles import resolve_storage_profile
from ..storage.selections import fingerprint_selected_stored_strings
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


type _SelectionContextValue = str | int | float | bool | None


class ArtifactSelectionError(ValueError):
    """A machine-readable failure to validate an artifact's stored selection."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, _SelectionContextValue],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _restore_artifact_selection_error,
            (type(self), str(self), self.code, self.context),
        )


def _restore_artifact_selection_error(
    error_type: type[ArtifactSelectionError],
    message: str,
    code: str,
    context: Mapping[str, _SelectionContextValue],
) -> ArtifactSelectionError:
    return error_type(message, code=code, context=context)


@dataclass(frozen=True, slots=True)
class ImportedArtifactStorage:
    """Narrow storage adapter for imported embedding-domain artifacts."""

    root: zarr.Group

    @staticmethod
    def requirement(
        name: str,
        *,
        shape: tuple[int, ...],
        dtype_kind: str | None,
    ) -> ArrayRequirement:
        return ArrayRequirement(name, shape=shape, dtype_kind=dtype_kind)

    def plan(
        self,
        *,
        assay: str,
        kind: str,
        parameters: dict[str, Any],
        inputs: dict[str, Any],
        execution_options: dict[str, Any],
        invalidate_cache: bool,
        required_arrays: tuple[ArrayRequirement, ...],
        reuse_validator: Callable[[ArtifactRef, zarr.Group], bool],
    ) -> PlannedArtifact:
        return plan_artifact(
            self.root,
            scope="assay",
            assay=assay,
            kind=kind,
            operation="import_dimreduc",
            parameters=parameters,
            inputs=inputs,
            execution_options=execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=required_arrays,
            reuse_validator=reuse_validator,
        )

    def start(self, planned: PlannedArtifact) -> zarr.Group:
        return start_artifact(self.root, planned)

    @staticmethod
    def finish(group: zarr.Group, planned: PlannedArtifact) -> None:
        finish_artifact(group, planned)

    def artifact_group(self, ref: ArtifactRef) -> zarr.Group:
        return artifact_group(self.root, ref)

    def require_complete(self, ref: ArtifactRef) -> Any:
        return require_complete_artifact(self.root, ref)

    @staticmethod
    def as_array(node: zarr.Array | zarr.Group, name: str) -> zarr.Array:
        return as_zarr_array(node, name=name)

    @staticmethod
    def as_group(node: zarr.Array | zarr.Group, name: str) -> zarr.Group:
        return as_zarr_group(node, name=name)

    @staticmethod
    def fingerprint_builder() -> ValueFingerprintBuilder:
        return ValueFingerprintBuilder()

    @staticmethod
    def fingerprint_strings(values: np.ndarray) -> str:
        return fingerprint_strings(values)

    @staticmethod
    def fingerprint_stored_strings(array: zarr.Array) -> str:
        return fingerprint_stored_strings(array)

    @staticmethod
    def block_rows(array: zarr.Array) -> int:
        return row_band(array_geometry(array), unit="chunk", fallback=1)

    @staticmethod
    def create_numeric(
        group: zarr.Group,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        block_rows: int,
    ) -> zarr.Array:
        return create_numeric_array(
            group,
            name,
            row_sharded_array_spec(
                shape,
                dtype,
                profile=resolve_storage_profile(group.store),
                band_rows=min(max(shape[0], 1), block_rows),
                zarr_format=_group_zarr_format(group),
                fill_value=0.0,
            ),
        )

    @staticmethod
    def create_metadata(
        group: zarr.Group,
        name: str,
        *,
        dtype: Any,
        shape: int,
        block_rows: int,
    ) -> zarr.Array:
        return create_metadata_column(
            group,
            name,
            dtype=dtype,
            shape=shape,
            chunkSize=min(max(shape, 1), block_rows),
            overwrite=True,
        )


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
        raise ValueError(
            "Legacy graph selection provenance is missing, so it cannot be "
            "selected safely"
        )
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


def named_result_mismatch(
    root: zarr.Group,
    name: str,
    ref: ArtifactRef,
    state: AssayState,
) -> str | None:
    """Report why a named result cannot belong to ``state``.

    Returns None when the handle fits the chain. Both the write path and the
    carry-forward in ``_artifact_chain_state`` consult this, so a handle can
    never be preserved into a state that the write would then reject.
    """
    if name != "mapping_reference":
        return None
    status = require_complete_artifact(root, ref)
    expected = {
        "reduction": state.reduction,
        "ann_index": state.ann_index,
        "neighbors": state.neighbors,
    }
    inputs = status.inputs or {}
    if "batch_correction" in inputs:
        expected["batch_correction"] = state.batch_correction
    elif state.batch_correction is not None:
        return "Plain PCA mapping reference cannot select batch correction"
    missing = [
        input_name for input_name, input_ref in expected.items() if input_ref is None
    ]
    if missing:
        return "Mapping reference state is missing " + ", ".join(missing)
    for input_name, input_ref in expected.items():
        assert input_ref is not None
        if inputs.get(input_name) != input_ref.to_dict():
            return f"Mapping reference input {input_name!r} does not match AssayState"
    return None


def write_assay_state(root: zarr.Group, state: AssayState) -> None:
    for field_name in _GRAPH_REF_KINDS:
        ref = getattr(state, field_name)
        if ref is None:
            continue
        require_complete_artifact(root, ref)
    for name, ref in state.named_results.items():
        require_complete_artifact(root, ref)
        if ref.kind == "imported_coordinates":
            validate_imported_coordinates_artifact(root, ref)
        reason = named_result_mismatch(root, name, ref, state)
        if reason is not None:
            raise ValueError(reason)
    if state.connectivity_map is not None:
        _stored_assay_graph(root, state)
    path = assay_state_path(state.assay)
    group = group_at(root, path) if path in root else root.create_group(path)
    group.attrs["state"] = state.to_dict()


def _complete_path(root: zarr.Group, ref: ArtifactRef, field_name: str) -> str:
    try:
        return require_complete_artifact(root, ref).path
    except KeyError as exc:
        raise KeyError(
            f"Assay state {field_name} artifact does not exist: {artifact_path(ref)}"
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            f"Assay state {field_name} artifact is incomplete: {artifact_path(ref)}"
        ) from exc


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
        raise KeyError(
            "AssayState has no selected embedding initialization; call "
            "build_embedding_initialization or pass ini_embed"
        )
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


def _input_ref(
    root: zarr.Group,
    ref: ArtifactRef,
    name: str,
    *,
    require_input_complete: bool = True,
) -> ArtifactRef:
    status = require_complete_artifact(root, ref)
    inputs = status.inputs or {}
    value = inputs.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{ref.kind} artifact has no {name!r} artifact input")
    input_ref = ArtifactRef.from_dict(value)
    if require_input_complete:
        require_complete_artifact(root, input_ref)
    return input_ref


def resolve_stored_graph_input(
    root: zarr.Group,
    graph_loc: str,
) -> ArtifactRef | dict[str, str]:
    try:
        return parse_artifact_path(graph_loc)
    except ValueError:
        graph_group = group_at(root, graph_loc)
        return {
            "legacy_graph_fingerprint": fingerprint_stored_arrays(
                graph_group,
                ("edges", "weights"),
            )
        }


@dataclass(frozen=True, slots=True)
class GraphSelection:
    """A resolved graph and the keys that name the columns written from it."""

    graph_loc: str
    graph_input: ArtifactRef | dict[str, str]
    from_assay: str
    cell_key: str
    feat_key: str
    integrated_label: str | None

    @property
    def output_assay(self) -> str:
        """Prefix used for cell-metadata columns written from this graph."""
        return self.integrated_label or self.from_assay


def integrated_graph_label(store: Any, ref: ArtifactRef) -> str:
    """Return the label an integrated-graph artifact is registered under."""
    index_path = store._integratedGraphsLoc
    labels: list[str] = []
    if index_path in store.zw:
        index_group = as_zarr_group(store.zw[index_path], name=index_path)
        raw_artifacts = index_group.attrs.get("artifacts", {})
        if "artifacts" in index_group.attrs and not isinstance(raw_artifacts, dict):
            raise RuntimeError("Integrated graph artifact index is invalid")
        if isinstance(raw_artifacts, dict):
            for label, raw_ref in raw_artifacts.items():
                if not isinstance(raw_ref, dict):
                    raise RuntimeError(
                        f"Integrated graph index for {label!r} is invalid"
                    )
                if ArtifactRef.from_dict(raw_ref) == ref:
                    labels.append(str(label))
    if not labels:
        raise KeyError("Integrated graph artifact is not registered under a label")
    if len(labels) > 1:
        raise ValueError(
            "Integrated graph artifact is shared by labels "
            f"{', '.join(sorted(labels))}; pass integrated_graph to choose one"
        )
    return labels[0]


def resolve_graph_selection(
    store: Any,
    graph: ArtifactRef | None,
    *,
    from_assay: str | None,
    cell_key: str | None,
    feat_key: str | None,
    integrated_graph: str | None = None,
) -> GraphSelection:
    """Resolve which graph an operation reads and how it names its outputs.

    Omitting ``graph`` keeps the implicit contract: the current analysis chain
    of the assay, or the integrated graph named by ``integrated_graph``.
    Passing ``graph`` reads that connectivity map or integrated graph instead
    and takes the assay, cell key, and feature key from the artifact.
    """
    if graph is not None and integrated_graph is not None:
        raise ValueError("Pass either graph or integrated_graph, not both")
    integrated_label = integrated_graph
    if graph is None:
        resolved_assay, resolved_cell_key, resolved_feat_key = store._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if integrated_graph is None:
            graph_loc = store.get_latest_graph_loc(
                resolved_assay,
                resolved_cell_key,
                resolved_feat_key,
            )
        else:
            graph_loc = store._resolve_integrated_graph_path(integrated_graph)
            if graph_loc not in store.zw:
                raise KeyError(
                    f"An integrated graph with label {integrated_graph!r} does not exist"
                )
    elif not isinstance(graph, ArtifactRef):
        raise TypeError("graph must be an artifact reference")
    elif graph.kind == "integrated_graph":
        graph_loc = str(require_complete_artifact(store.zw, graph).path)
        integrated_label = integrated_graph_label(store, graph)
        resolved_assay, resolved_cell_key, resolved_feat_key = store._get_latest_keys(
            from_assay, cell_key, feat_key
        )
    elif graph.kind == "connectivity_map":
        stored = stored_assay_graph_from_ref(store.zw, graph)
        for name, requested, resolved in (
            ("from_assay", from_assay, stored.from_assay),
            ("cell_key", cell_key, stored.cell_key),
            ("feat_key", feat_key, stored.feat_key),
        ):
            if requested is not None and requested != resolved:
                raise ValueError(
                    f"{name} {requested!r} does not match the graph, "
                    f"which was built with {resolved!r}"
                )
        resolved_assay = stored.from_assay
        resolved_cell_key = stored.cell_key
        resolved_feat_key = stored.feat_key
        graph_loc = stored.paths.cell_graph_group_path
    else:
        raise ValueError(
            "graph must reference a connectivity map or an integrated graph"
        )
    return GraphSelection(
        graph_loc=graph_loc,
        graph_input=resolve_stored_graph_input(store.zw, graph_loc),
        from_assay=resolved_assay,
        cell_key=resolved_cell_key,
        feat_key=resolved_feat_key,
        integrated_label=integrated_label,
    )


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
        reduction_group = artifact_group(root, state.reduction)
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
        raise ArtifactSelectionError(
            f"Expected {scope}-scoped {kind} artifact",
            code="artifact_reference_mismatch",
            context={
                "expected_scope": scope,
                "expected_assay": assay,
                "expected_kind": kind,
                "actual_scope": ref.scope,
                "actual_assay": ref.assay,
                "actual_kind": ref.kind,
                "artifact_id": ref.artifact_id,
            },
        )


def _selection_error_context(
    ref: ArtifactRef,
    *,
    table_path: str,
    column: str,
) -> dict[str, _SelectionContextValue]:
    return {
        "scope": ref.scope,
        "assay": ref.assay,
        "kind": ref.kind,
        "artifact_id": ref.artifact_id,
        "table": table_path,
        "column": column,
    }


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
    context = _selection_error_context(
        ref,
        table_path=table_path,
        column=column,
    )
    if not status.exists:
        raise ArtifactSelectionError(
            f"{kind} artifact does not exist",
            code="artifact_missing",
            context=context,
        )
    if not status.complete:
        raise ArtifactSelectionError(
            f"{kind} artifact is incomplete",
            code="artifact_incomplete",
            context=context,
        )
    if table_path not in root:
        raise ArtifactSelectionError(
            f"Selection table {table_path!r} is unavailable",
            code="selection_table_missing",
            context=context,
        )
    table = group_at(root, table_path)
    if column not in table:
        raise ArtifactSelectionError(
            f"Selection source column {column!r} is unavailable",
            code="selection_column_missing",
            context=context,
        )
    if "ids" not in table:
        raise ArtifactSelectionError(
            "Selection row identifier column 'ids' is unavailable",
            code="selection_row_ids_missing",
            context=context,
        )
    selection_group = artifact_group(root, ref)
    if "values" not in selection_group:
        raise ArtifactSelectionError(
            f"{kind} artifact has no values",
            code="selection_values_missing",
            context=context,
        )
    stored_values = as_zarr_array(selection_group["values"], name="values")
    current_values = as_zarr_array(table[column], name=column)
    row_ids = as_zarr_array(table["ids"], name="ids")
    expected_row_ids = (status.inputs or {}).get("ordered_row_ids_fingerprint")
    if not isinstance(
        expected_row_ids, str
    ) or expected_row_ids != fingerprint_stored_strings(row_ids):
        raise ArtifactSelectionError(
            f"{kind} row identity does not match its metadata table",
            code="row_identity_mismatch",
            context=context,
        )
    if (
        stored_values.ndim != 1
        or np.dtype(stored_values.dtype) != np.dtype(bool)
        or current_values.ndim != 1
        or np.dtype(current_values.dtype) != np.dtype(bool)
        or stored_values.shape != current_values.shape
    ):
        raise ArtifactSelectionError(
            f"Selection source column {column!r} no longer matches its artifact",
            code="selection_values_changed",
            context=context,
        )
    block_rows = min(
        row_band(array_geometry(stored_values), unit="chunk", fallback=1),
        row_band(array_geometry(current_values), unit="chunk", fallback=1),
    )
    for start in range(0, int(stored_values.shape[0]), block_rows):
        stop = min(start + block_rows, int(stored_values.shape[0]))
        if not np.array_equal(
            np.asarray(stored_values[start:stop], dtype=bool),
            np.asarray(current_values[start:stop], dtype=bool),
        ):
            raise ArtifactSelectionError(
                f"Selection source column {column!r} no longer matches its artifact",
                code="selection_values_changed",
                context=context,
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


def _stored_value_fingerprint(array: zarr.Array) -> str:
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", array.shape, array.dtype)
    block_rows = row_band(array_geometry(array), unit="chunk", fallback=1)
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        block = np.asarray(array[start:stop])
        builder.update_array_block(
            "values",
            (start,) + (0,) * (array.ndim - 1),
            block,
        )
    builder.end_array("values")
    return builder.hexdigest()


def _payload_fingerprint(group: zarr.Group, name: str) -> str:
    array = as_zarr_array(group[name], name=name)
    if name == "feature_ids":
        return fingerprint_stored_strings(array)
    return _stored_value_fingerprint(array)


def validate_imported_coordinates_artifact(
    root: zarr.Group,
    coordinates: ArtifactRef,
    *,
    cell_key: str | None = None,
) -> None:
    if coordinates.assay is None:
        raise ValueError("Imported-coordinate artifact has no assay")
    _require_artifact_ref(
        coordinates,
        kind="imported_coordinates",
        scope="assay",
        assay=coordinates.assay,
    )
    status = require_complete_artifact(root, coordinates)
    if status.operation != "import_dimreduc":
        raise ValueError(
            "Imported-coordinate artifact operation must be 'import_dimreduc'"
        )
    execution = status.execution_options or {}
    stored_cell_key = execution.get("cell_key")
    if not isinstance(stored_cell_key, str) or not stored_cell_key:
        raise ValueError("Imported-coordinate artifact has no cell selection key")
    if cell_key is not None and cell_key != stored_cell_key:
        raise ValueError(
            f"cell_key {cell_key!r} does not match imported coordinates "
            f"built for {stored_cell_key!r}"
        )
    block_rows = execution.get("block_rows")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int | np.integer)
        or int(block_rows) < 1
    ):
        raise ValueError("Imported-coordinate block_rows is invalid")
    selection = _input_ref(
        root,
        coordinates,
        "cell_selection",
        require_input_complete=False,
    )
    validate_cell_selection_artifact(root, selection, stored_cell_key)

    inputs = status.inputs or {}
    source_digest = inputs.get("source_digest")
    if (
        not isinstance(source_digest, Mapping)
        or set(source_digest) != {"bytes_hex"}
        or not isinstance(source_digest.get("bytes_hex"), str)
        or len(source_digest["bytes_hex"]) != 64
        or source_digest["bytes_hex"].lower() != source_digest["bytes_hex"]
    ):
        raise ValueError("Imported-coordinate source digest is missing")
    try:
        bytes.fromhex(source_digest["bytes_hex"])
    except ValueError as exc:
        raise ValueError(
            "Imported-coordinate source digest is not hexadecimal"
        ) from exc

    payload_fingerprints = inputs.get("payload_fingerprints")
    if not isinstance(payload_fingerprints, Mapping) or not all(
        isinstance(name, str)
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        for name, fingerprint in payload_fingerprints.items()
    ):
        raise ValueError("Imported-coordinate payload fingerprints are malformed")

    group = artifact_group(root, coordinates)
    if "data" not in group:
        raise ValueError("Imported-coordinate artifact has no data array")
    data = as_zarr_array(group["data"], name="data")
    if data.ndim != 2 or int(data.shape[0]) < 1 or np.dtype(data.dtype).kind != "f":
        raise ValueError("Imported-coordinate data must be a floating-point matrix")

    parameters = status.parameters or {}
    dimreduc_key = parameters.get("dimreduc_key")
    if not isinstance(dimreduc_key, str) or not dimreduc_key:
        raise ValueError("Imported-coordinate source key is missing")
    dims = parameters.get("dims")
    if isinstance(dims, bool) or not isinstance(dims, int | np.integer):
        raise ValueError("Imported-coordinate dimensions are missing")
    if int(dims) < 1 or int(data.shape[1]) != int(dims):
        raise ValueError("Imported-coordinate dimensions do not match data")
    role = parameters.get("role")
    if (
        not isinstance(role, str)
        or not role
        or role.lower() != role
        or role in {"umap", "tsne"}
    ):
        raise ValueError("Imported-coordinate role is invalid")

    selection_group = artifact_group(root, selection)
    mask = as_zarr_array(selection_group["values"], name="values")
    cell_data = group_at(root, "cellData")
    row_ids = as_zarr_array(cell_data["ids"], name="ids")
    selected_fingerprint, selected_count = fingerprint_selected_stored_strings(
        row_ids,
        mask,
    )
    if int(data.shape[0]) != selected_count:
        raise ValueError(
            "Imported-coordinate rows do not match the exact cell selection"
        )
    ordered_fingerprint = inputs.get("ordered_cell_ids_fingerprint")
    if (
        not isinstance(ordered_fingerprint, str)
        or ordered_fingerprint != selected_fingerprint
    ):
        raise ValueError(
            "Imported-coordinate cell IDs do not match the selected cell order"
        )

    optional = {
        "loadings": parameters.get("loadings_stored"),
        "feature_ids": parameters.get("feature_ids_stored"),
        "stdev": parameters.get("stdev_stored"),
    }
    for name, stored in optional.items():
        if not isinstance(stored, bool) or stored != (name in group):
            raise ValueError(
                f"Imported-coordinate {name!r} storage flag does not match payload"
            )
    if optional["loadings"] != optional["feature_ids"]:
        raise ValueError(
            "Imported-coordinate loadings and feature IDs must be stored together"
        )
    if "loadings" in group:
        loadings = as_zarr_array(group["loadings"], name="loadings")
        feature_ids = as_zarr_array(group["feature_ids"], name="feature_ids")
        if (
            loadings.ndim != 2
            or np.dtype(loadings.dtype).kind != "f"
            or int(loadings.shape[0]) < 1
            or int(loadings.shape[1]) != int(dims)
            or feature_ids.ndim != 1
            or np.dtype(feature_ids.dtype).kind not in {"O", "S", "U"}
            or int(feature_ids.shape[0]) != int(loadings.shape[0])
        ):
            raise ValueError(
                "Imported-coordinate loadings and feature IDs are misaligned"
            )
    if "stdev" in group:
        stdev = as_zarr_array(group["stdev"], name="stdev")
        if (
            stdev.ndim != 1
            or np.dtype(stdev.dtype).kind != "f"
            or tuple(stdev.shape) != (int(dims),)
        ):
            raise ValueError("Imported-coordinate stdev does not match dimensions")

    payload_names = ("data", "loadings", "feature_ids", "stdev")
    stored_payloads = {name for name in payload_names if name in group}
    if set(payload_fingerprints) != stored_payloads:
        raise ValueError(
            "Imported-coordinate payload fingerprints do not match stored payloads"
        )
    for name in payload_names:
        if name not in group:
            continue
        expected = payload_fingerprints.get(name)
        if not isinstance(expected, str):
            raise ValueError(
                f"Imported-coordinate payload fingerprint for {name!r} is missing"
            )
        if _payload_fingerprint(group, name) != expected:
            raise ValueError(
                f"Imported-coordinate payload fingerprint for {name!r} does not match"
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
    validate_neighbors_artifact_selection(root, neighbors, cell_key, feat_key)


def validate_neighbors_artifact_selection(
    root: zarr.Group,
    neighbors: ArtifactRef,
    cell_key: str,
    feat_key: str,
) -> None:
    if neighbors.assay is None:
        raise ValueError("Neighbors artifact has no assay")
    assay = neighbors.assay
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
    elif coordinates.kind == "imported_coordinates":
        _require_artifact_ref(
            coordinates,
            kind="imported_coordinates",
            scope="assay",
            assay=assay,
        )
        validate_imported_coordinates_artifact(
            root,
            coordinates,
            cell_key=cell_key,
        )
        reduction = None
    else:
        raise ValueError(
            "Neighbor coordinates must be reduction, batch_correction, "
            "or imported_coordinates"
        )
    if _input_ref(root, ann_index, "coordinates") != coordinates:
        raise ValueError("ANN index and neighbors use different coordinates")
    if reduction is None:
        return
    _require_artifact_ref(
        reduction,
        kind="reduction",
        scope="assay",
        assay=assay,
    )
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
    cell_selection = _input_ref(
        root,
        normalized,
        "cell_selection",
        require_input_complete=False,
    )
    feature_selection = _input_ref(
        root,
        normalized,
        "feature_selection",
        require_input_complete=False,
    )
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
