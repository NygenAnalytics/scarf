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
    ArtifactScope,
    ValueFingerprintBuilder,
    artifact_group,
    artifact_path,
    fingerprint_stored_strings,
    fingerprint_strings,
    group_at,
    inspect_artifact,
    require_complete_artifact,
)
from ..storage.geometry import array_geometry
from ..storage.errors import ArtifactResolutionError
from ..storage.feature_selection import resolve_feature_selection
from ..storage.layout import _group_zarr_format, row_sharded_array_spec
from ..storage.partition import row_band
from ..storage.profiles import resolve_storage_profile
from ..storage.selections import fingerprint_selected_stored_strings
from ..storage.selections import validate_stored_selection_artifact
from ..storage.types import as_zarr_array, as_zarr_group
from .errors import IncompatibleAnalysisStateError

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
_STATE_FIELDS = frozenset(
    {
        "assay",
        "cell_key",
        *_GRAPH_REF_KINDS,
        "named_results",
    }
)
_LEGACY_FEATURE_FIELDS = frozenset({"feat_key", "feature_selection"})


def _state_error(
    message: str,
    *,
    code: str = "invalid_analysis_state",
    assay: str | None = None,
    field_name: str | None = None,
    keys: str | None = None,
) -> IncompatibleAnalysisStateError:
    return IncompatibleAnalysisStateError(
        message,
        code=code,
        context={
            "assay": assay,
            "field": field_name,
            "keys": keys,
        },
    )


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


@dataclass(frozen=True, slots=True)
class AssayState:
    assay: str
    cell_key: str
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
        if not isinstance(self.assay, str) or not self.assay or "/" in self.assay:
            raise _state_error(
                "AssayState requires a valid assay name",
                assay=self.assay if isinstance(self.assay, str) else None,
                field_name="assay",
            )
        if not isinstance(self.cell_key, str) or not self.cell_key:
            raise _state_error(
                "AssayState requires a non-empty cell_key",
                assay=self.assay,
                field_name="cell_key",
            )
        for field_name, expected_kind in _GRAPH_REF_KINDS.items():
            ref = getattr(self, field_name)
            if ref is not None:
                self._validate_ref(ref, expected_kind, field_name)
        if not isinstance(self.named_results, Mapping):
            raise _state_error(
                "named_results must be a mapping",
                assay=self.assay,
                field_name="named_results",
            )
        for name, ref in self.named_results.items():
            if (
                not isinstance(name, str)
                or _RESULT_NAME_PATTERN.fullmatch(name) is None
            ):
                raise _state_error(
                    "named_results keys must be snake_case identifiers",
                    assay=self.assay,
                    field_name="named_results",
                )
            if not isinstance(ref, ArtifactRef):
                raise _state_error(
                    f"named_results[{name!r}] must be an artifact reference",
                    assay=self.assay,
                    field_name=f"named_results[{name!r}]",
                )
            self._validate_ref(ref, ref.kind, f"named_results[{name!r}]")
            expected_named_kind = _NAMED_RESULT_KINDS.get(name)
            if expected_named_kind is not None and ref.kind != expected_named_kind:
                raise _state_error(
                    f"named_results[{name!r}] requires kind {expected_named_kind!r}",
                    assay=self.assay,
                    field_name=f"named_results[{name!r}]",
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
        if not isinstance(ref, ArtifactRef):
            raise _state_error(
                f"{field_name} must be an artifact reference or null",
                assay=self.assay,
                field_name=field_name,
            )
        if ref.scope != "assay" or ref.assay != self.assay:
            raise _state_error(
                f"{field_name} must reference assay {self.assay!r}",
                assay=self.assay,
                field_name=field_name,
            )
        if ref.kind != expected_kind:
            raise _state_error(
                f"{field_name} requires kind {expected_kind!r}, got {ref.kind!r}",
                assay=self.assay,
                field_name=field_name,
            )

    def matches(self, cell_key: str) -> bool:
        return self.cell_key == cell_key

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "assay": self.assay,
            "cell_key": self.cell_key,
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
        if not isinstance(value, Mapping):
            raise _state_error("Assay state must be a mapping")
        raw_keys = set(value)
        non_string_keys = [key for key in raw_keys if not isinstance(key, str)]
        if non_string_keys:
            raise _state_error("Assay state keys must be strings")
        legacy = sorted(raw_keys & _LEGACY_FEATURE_FIELDS)
        assay_value = value.get("assay")
        assay = assay_value if isinstance(assay_value, str) else None
        if legacy:
            joined = ",".join(legacy)
            raise _state_error(
                "Stored analysis state uses the removed feature-key contract",
                code="legacy_feature_contract",
                assay=assay,
                keys=joined,
            )
        unknown = sorted(raw_keys - _STATE_FIELDS)
        if unknown:
            joined = ",".join(unknown)
            raise _state_error(
                f"Assay state contains unknown fields: {joined}",
                assay=assay,
                keys=joined,
            )
        try:
            refs: dict[str, ArtifactRef | None] = {}
            for field_name in _GRAPH_REF_KINDS:
                raw_ref = value.get(field_name)
                if raw_ref is None:
                    refs[field_name] = None
                elif isinstance(raw_ref, Mapping):
                    refs[field_name] = ArtifactRef.from_dict(raw_ref)
                else:
                    raise TypeError(
                        f"{field_name} must be an artifact reference or null"
                    )
            raw_named = value.get("named_results", {})
            if not isinstance(raw_named, Mapping):
                raise TypeError("named_results must be a mapping")
            named_results = {}
            for name, raw_ref in raw_named.items():
                if not isinstance(name, str) or not isinstance(raw_ref, Mapping):
                    raise TypeError(
                        "Every named result must be a named artifact reference"
                    )
                named_results[name] = ArtifactRef.from_dict(raw_ref)
            cell_key = value.get("cell_key")
            if not isinstance(assay_value, str) or not isinstance(cell_key, str):
                raise TypeError("Assay state assay and cell_key must be strings")
            missing = sorted(_STATE_FIELDS - raw_keys)
            if missing:
                joined = ",".join(missing)
                raise _state_error(
                    f"Assay state is missing required fields: {joined}",
                    assay=assay,
                    keys=joined,
                )
            return cls(
                assay=assay_value,
                cell_key=cell_key,
                named_results=named_results,
                **refs,
            )
        except IncompatibleAnalysisStateError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise _state_error(
                f"Assay state is malformed: {exc}",
                assay=assay,
            ) from exc


def assay_state_path(assay: str) -> str:
    if not assay or "/" in assay:
        raise ValueError("Invalid assay name")
    return f"{assay}/state"


def read_assay_state_document(root: zarr.Group, assay: str) -> AssayState | None:
    """Parse an assay-state document without resolving its artifact lineage.

    Recovery-capable producers use this narrow read to reject legacy, unknown,
    or malformed state documents before writing while allowing a new chain to
    replace unavailable artifacts from an older current chain. Consumers must
    use :func:`read_assay_state`, which additionally validates every live edge.
    """

    path = assay_state_path(assay)
    if path not in root:
        return None
    try:
        group = as_zarr_group(root[path], name=path)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _state_error(
            f"Assay state node or attributes at {path} are malformed",
            assay=assay,
        ) from exc
    if "state" not in group.attrs:
        raise _state_error(
            f"Assay state at {path} is missing its required state attribute",
            assay=assay,
        )
    value = group.attrs["state"]
    if not isinstance(value, Mapping):
        raise _state_error(
            f"Assay state at {path} must be a mapping",
            assay=assay,
        )
    state = AssayState.from_dict(value)
    if state.assay != assay:
        raise _state_error(
            f"Assay state at {path} names assay {state.assay!r}",
            assay=assay,
            field_name="assay",
        )
    return state


def read_assay_state(root: zarr.Group, assay: str) -> AssayState | None:
    """Read and recursively validate the current assay artifact chain."""

    state = read_assay_state_document(root, assay)
    if state is None:
        return None
    _validate_assay_state_artifacts(
        root,
        state,
        allow_unavailable_named_results=True,
    )
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
    status = require_complete_artifact(root, ref)
    if name != "mapping_reference":
        return None
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
    validate_assay_state(root, state)
    path = assay_state_path(state.assay)
    group = group_at(root, path) if path in root else root.create_group(path)
    group.attrs["state"] = state.to_dict()


def validate_assay_state(root: zarr.Group, state: AssayState) -> None:
    """Validate that an assay-state document can be published unchanged."""

    _validate_assay_state_artifacts(root, state)


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
) -> str | None:
    state = read_assay_state(root, assay)
    if state is None or not state.matches(cell_key):
        return None
    if state.normalized is None:
        raise KeyError("AssayState has no selected normalized artifact")
    return _complete_path(root, state.normalized, "normalized")


def embedding_initialization_path_from_state(
    root: zarr.Group,
    assay: str,
    cell_key: str,
) -> str | None:
    state = read_assay_state(root, assay)
    if state is None or not state.matches(cell_key):
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


def _require_state_artifact(
    root: zarr.Group,
    state: AssayState,
    field_name: str,
    ref: ArtifactRef,
) -> None:
    try:
        status = inspect_artifact(root, ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise _state_error(
            f"AssayState {field_name} artifact record is malformed",
            assay=state.assay,
            field_name=field_name,
        ) from exc
    context = {
        "assay": state.assay,
        "kind": ref.kind,
        "artifact_id": ref.artifact_id,
        "field": field_name,
    }
    if not status.exists:
        raise ArtifactResolutionError(
            f"AssayState {field_name} artifact does not exist",
            code="missing_artifact",
            context=context,
        )
    if not status.complete:
        raise ArtifactResolutionError(
            f"AssayState {field_name} artifact is incomplete",
            code="incomplete_artifact",
            context=context,
        )


def _normalized_state_inputs(
    root: zarr.Group,
    state: AssayState,
) -> tuple[ArtifactRef, ArtifactRef] | None:
    normalized = state.normalized
    if normalized is None:
        return None
    status = inspect_artifact(root, normalized)
    inputs = status.inputs or {}
    raw_cells = inputs.get("cell_selection")
    raw_features = inputs.get("feature_selection")
    if not isinstance(raw_features, Mapping):
        legacy_sources = (
            inputs,
            status.parameters or {},
            status.execution_options or {},
        )
        if isinstance(raw_features, str) or any(
            "feat_key" in source or "feature_key" in source for source in legacy_sources
        ):
            raise _state_error(
                "Normalized artifact uses the removed feature-selection contract",
                code="legacy_feature_contract",
                assay=state.assay,
                field_name="normalized.feature_selection",
            )
        raise ArtifactResolutionError(
            "Normalized artifact has no valid feature_selection input",
            code="corrupt_payload",
            context={
                "assay": state.assay,
                "kind": normalized.kind,
                "artifact_id": normalized.artifact_id,
                "field": "normalized.feature_selection",
            },
        )
    if not isinstance(raw_cells, Mapping):
        raise ArtifactResolutionError(
            "Normalized artifact has no valid cell_selection input",
            code="corrupt_payload",
            context={
                "assay": state.assay,
                "kind": normalized.kind,
                "artifact_id": normalized.artifact_id,
                "field": "normalized.cell_selection",
            },
        )
    if "feat_key" in raw_features or "feature_key" in raw_features:
        raise _state_error(
            "Normalized artifact uses the removed feature-selection contract",
            code="legacy_feature_contract",
            assay=state.assay,
            field_name="normalized.feature_selection",
        )
    try:
        feature_selection = ArtifactRef.from_dict(raw_features)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Normalized artifact has a malformed feature_selection input",
            code="corrupt_payload",
            context={
                "assay": state.assay,
                "kind": normalized.kind,
                "artifact_id": normalized.artifact_id,
                "field": "normalized.feature_selection",
            },
        ) from exc
    if set(raw_features) != set(feature_selection.to_dict()):
        if "feat_key" in raw_features or "feature_key" in raw_features:
            raise _state_error(
                "Normalized artifact uses the removed feature-selection contract",
                code="legacy_feature_contract",
                assay=state.assay,
                field_name="normalized.feature_selection",
            )
        raise ArtifactResolutionError(
            "Normalized artifact has a malformed feature_selection input",
            code="corrupt_payload",
            context={
                "assay": state.assay,
                "kind": normalized.kind,
                "artifact_id": normalized.artifact_id,
                "field": "normalized.feature_selection",
            },
        )
    try:
        cell_selection = ArtifactRef.from_dict(raw_cells)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Normalized artifact has a malformed cell_selection input",
            code="corrupt_payload",
            context={
                "assay": state.assay,
                "kind": normalized.kind,
                "artifact_id": normalized.artifact_id,
                "field": "normalized.cell_selection",
            },
        ) from exc
    if set(raw_cells) != set(cell_selection.to_dict()):
        raise ArtifactResolutionError(
            "Normalized artifact has a malformed cell_selection input",
            code="corrupt_payload",
            context={
                "assay": state.assay,
                "kind": normalized.kind,
                "artifact_id": normalized.artifact_id,
                "field": "normalized.cell_selection",
            },
        )
    validate_cell_selection_artifact(root, cell_selection, state.cell_key)
    resolve_feature_selection(root, state.assay, feature_selection)
    return cell_selection, feature_selection


def _state_input_ref(
    root: zarr.Group,
    state: AssayState,
    owner_name: str,
    owner: ArtifactRef,
    input_name: str,
) -> ArtifactRef:
    raw_ref = (inspect_artifact(root, owner).inputs or {}).get(input_name)
    if not isinstance(raw_ref, Mapping):
        raise _state_error(
            f"{owner_name} artifact has no named {input_name} input",
            assay=state.assay,
            field_name=f"{owner_name}.{input_name}",
        )
    try:
        return ArtifactRef.from_dict(raw_ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise _state_error(
            f"{owner_name} artifact has a malformed {input_name} input",
            assay=state.assay,
            field_name=f"{owner_name}.{input_name}",
        ) from exc


def _require_matching_state_input(
    root: zarr.Group,
    state: AssayState,
    owner_name: str,
    owner: ArtifactRef | None,
    input_name: str,
    expected: ArtifactRef | None,
) -> None:
    if owner is None:
        return
    actual = _state_input_ref(root, state, owner_name, owner, input_name)
    if expected is None or actual != expected:
        raise _state_error(
            f"AssayState {owner_name}.{input_name} does not match graph lineage",
            assay=state.assay,
            field_name=f"{owner_name}.{input_name}",
        )


def _validate_ann_index_coordinates(root: zarr.Group, state: AssayState) -> None:
    ann_index = state.ann_index
    if ann_index is None:
        return
    coordinates = _state_input_ref(
        root,
        state,
        "ann_index",
        ann_index,
        "coordinates",
    )
    if coordinates.scope != "assay" or coordinates.assay != state.assay:
        raise _state_error(
            "AssayState ann_index.coordinates belongs to a different scope or assay",
            assay=state.assay,
            field_name="ann_index.coordinates",
        )
    if coordinates.kind == "imported_coordinates":
        validate_imported_coordinates_artifact(
            root,
            coordinates,
            cell_key=state.cell_key,
        )
        native_fields = (
            "normalized",
            "feature_scaling",
            "reduction",
            "batch_correction",
        )
        if any(getattr(state, field_name) is not None for field_name in native_fields):
            raise _state_error(
                "Imported-coordinate AssayState contains unrelated native ancestry",
                assay=state.assay,
                field_name="ann_index.coordinates",
            )
        return
    if coordinates.kind not in {"reduction", "batch_correction"}:
        raise _state_error(
            "AssayState ann_index.coordinates has an unsupported artifact kind",
            assay=state.assay,
            field_name="ann_index.coordinates",
        )
    _require_state_artifact(
        root,
        state,
        "ann_index.coordinates",
        coordinates,
    )
    if state.normalized is None or state.reduction is None:
        raise _state_error(
            "Native ANN-index state requires normalized and reduction ancestry",
            assay=state.assay,
            field_name="ann_index.coordinates",
        )
    expected = state.batch_correction or state.reduction
    if coordinates != expected:
        raise _state_error(
            "AssayState ann_index.coordinates does not match native ancestry",
            assay=state.assay,
            field_name="ann_index.coordinates",
        )


def _validate_assay_state_artifacts(
    root: zarr.Group,
    state: AssayState,
    *,
    allow_unavailable_named_results: bool = False,
) -> None:
    for field_name in _GRAPH_REF_KINDS:
        ref = getattr(state, field_name)
        if ref is not None:
            _require_state_artifact(root, state, field_name, ref)
    for name, ref in state.named_results.items():
        try:
            _require_state_artifact(root, state, f"named_results[{name!r}]", ref)
        except ArtifactResolutionError as exc:
            if allow_unavailable_named_results and exc.code in {
                "missing_artifact",
                "incomplete_artifact",
            }:
                continue
            raise
        if ref.kind == "imported_coordinates":
            validate_imported_coordinates_artifact(root, ref)
        reason = named_result_mismatch(root, name, ref, state)
        if reason is not None:
            raise _state_error(
                reason,
                assay=state.assay,
                field_name=f"named_results[{name!r}]",
            )

    _normalized_state_inputs(root, state)

    _require_matching_state_input(
        root,
        state,
        "feature_scaling",
        state.feature_scaling,
        "normalized",
        state.normalized,
    )
    _require_matching_state_input(
        root,
        state,
        "reduction",
        state.reduction,
        "normalized",
        state.normalized,
    )
    if state.reduction is not None:
        reduction_inputs = inspect_artifact(root, state.reduction).inputs or {}
        if "feature_scaling" in reduction_inputs:
            _require_matching_state_input(
                root,
                state,
                "reduction",
                state.reduction,
                "feature_scaling",
                state.feature_scaling,
            )
        elif state.feature_scaling is not None:
            raise _state_error(
                "AssayState feature_scaling is not named by reduction",
                assay=state.assay,
                field_name="feature_scaling",
            )
    _require_matching_state_input(
        root,
        state,
        "batch_correction",
        state.batch_correction,
        "reduction",
        state.reduction,
    )
    _require_matching_state_input(
        root,
        state,
        "embedding_initialization",
        state.embedding_initialization,
        "reduction",
        state.reduction,
    )
    _validate_ann_index_coordinates(root, state)

    source = state.connectivity_map or state.neighbors
    if source is None:
        return
    from .feature_projection import resolve_native_graph_inputs

    ancestry = resolve_native_graph_inputs(root, source)
    expected_refs = {
        "neighbors": ancestry.neighbors,
        "ann_index": ancestry.ann_index,
        "reduction": ancestry.reduction,
        "normalized": ancestry.normalized,
    }
    if ancestry.reduction is not None:
        reduction_inputs = inspect_artifact(root, ancestry.reduction).inputs or {}
        raw_scaling = reduction_inputs.get("feature_scaling")
        if raw_scaling is None:
            expected_refs["feature_scaling"] = None
        elif isinstance(raw_scaling, Mapping):
            try:
                expected_refs["feature_scaling"] = ArtifactRef.from_dict(raw_scaling)
            except (KeyError, TypeError, ValueError) as exc:
                raise _state_error(
                    "Reduction artifact has a malformed feature_scaling input",
                    assay=state.assay,
                    field_name="feature_scaling",
                ) from exc
        else:
            raise _state_error(
                "Reduction artifact has a malformed feature_scaling input",
                assay=state.assay,
                field_name="feature_scaling",
            )
    if ancestry.coordinates.kind == "batch_correction":
        expected_refs["batch_correction"] = ancestry.coordinates
    elif ancestry.coordinates.kind == "reduction":
        expected_refs["batch_correction"] = None
    else:
        expected_refs["batch_correction"] = None
        expected_refs["feature_scaling"] = None
    for field_name, expected_ref in expected_refs.items():
        actual_ref = getattr(state, field_name)
        if actual_ref != expected_ref:
            raise _state_error(
                f"AssayState {field_name} does not match graph lineage",
                assay=state.assay,
                field_name=field_name,
            )
    validate_cell_selection_artifact(
        root,
        ancestry.cell_selection,
        state.cell_key,
    )


@dataclass(frozen=True, slots=True)
class GraphSelection:
    """A resolved graph and the metadata route for columns written from it."""

    graph_loc: str
    graph_ref: ArtifactRef
    from_assay: str
    cell_key: str
    integrated_label: str | None

    @property
    def graph_input(self) -> ArtifactRef:
        """Exact artifact input used by result planners."""
        return self.graph_ref

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
            f"{', '.join(sorted(labels))}; output routing is ambiguous"
        )
    return labels[0]


def _cell_key_from_selection(root: zarr.Group, selection: ArtifactRef) -> str:
    status = require_complete_artifact(root, selection)
    source_column = (status.execution_options or {}).get("source_column")
    if not isinstance(source_column, str) or not source_column:
        raise _state_error(
            "Cell-selection artifact has no source_column",
            field_name="cell_selection",
        )
    return source_column


def resolve_graph_selection(
    store: Any,
    graph: ArtifactRef | None,
    *,
    from_assay: str | None,
    cell_key: str | None,
) -> GraphSelection:
    """Resolve an explicit graph or the assay's current connectivity map."""
    from .feature_projection import graph_cell_selection, graph_source_assays

    explicit_graph = graph is not None
    if graph is None:
        assay = store._get_assay(from_assay)
        resolved_assay = assay.name
        state = read_assay_state(store.zw, resolved_assay)
        if state is None or state.connectivity_map is None:
            raise ArtifactResolutionError(
                f"Assay {resolved_assay!r} has no current graph",
                code="missing_current_graph",
                context={"assay": resolved_assay},
            )
        graph = state.connectivity_map
        resolved_cell_key = state.cell_key
        integrated_label = None
    elif not isinstance(graph, ArtifactRef):
        raise TypeError("graph must be an artifact reference")
    elif graph.kind == "integrated_graph":
        selection = graph_cell_selection(store.zw, graph)
        resolved_cell_key = _cell_key_from_selection(store.zw, selection)
        validate_cell_selection_artifact(
            store.zw,
            selection,
            resolved_cell_key,
        )
        integrated_label = integrated_graph_label(store, graph)
        resolved_assay = store._get_assay(from_assay).name
    elif graph.kind == "connectivity_map":
        if graph.assay is None:
            raise ArtifactResolutionError(
                "Connectivity-map artifact has no assay",
                code="wrong_scope",
                context={
                    "artifact_id": graph.artifact_id,
                    "actual_scope": graph.scope,
                },
            )
        selection = graph_cell_selection(store.zw, graph)
        resolved_cell_key = _cell_key_from_selection(store.zw, selection)
        validate_cell_selection_artifact(
            store.zw,
            selection,
            resolved_cell_key,
        )
        resolved_assay = graph.assay
        integrated_label = None
    else:
        raise ArtifactResolutionError(
            "graph must reference a connectivity map or an integrated graph",
            code="unsupported_graph_kind",
            context={
                "artifact_id": graph.artifact_id,
                "actual_kind": graph.kind,
            },
        )
    if explicit_graph:
        for source_assay in graph_source_assays(store.zw, graph):
            read_assay_state(store.zw, source_assay)
    if from_assay is not None and from_assay != resolved_assay:
        raise ArtifactResolutionError(
            f"Graph belongs to assay {resolved_assay!r}, not {from_assay!r}",
            code="wrong_assay",
            context={
                "expected_assay": from_assay,
                "actual_assay": resolved_assay,
                "artifact_id": graph.artifact_id,
            },
        )
    if cell_key is not None and cell_key != resolved_cell_key:
        raise ArtifactResolutionError(
            f"cell_key {cell_key!r} does not match graph selection "
            f"{resolved_cell_key!r}",
            code="row_mismatch",
            context={
                "cell_key": cell_key,
                "expected_cell_key": resolved_cell_key,
                "artifact_id": graph.artifact_id,
            },
        )
    return GraphSelection(
        graph_loc=artifact_path(graph),
        graph_ref=graph,
        from_assay=resolved_assay,
        cell_key=resolved_cell_key,
        integrated_label=integrated_label,
    )


def _require_artifact_ref(
    ref: ArtifactRef,
    *,
    kind: str,
    scope: str,
    assay: str | None,
) -> None:
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise ArtifactResolutionError(
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


def _validate_selection_artifact(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: ArtifactScope,
    assay: str | None,
    table_path: str,
    column: str,
) -> None:
    validate_stored_selection_artifact(
        root,
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        table_path=table_path,
        column=column,
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
    _require_artifact_ref(
        coordinates,
        kind="imported_coordinates",
        scope="assay",
        assay=coordinates.assay,
    )
    if coordinates.assay is None:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has no assay",
            code="wrong_scope",
            context={
                "artifact_id": coordinates.artifact_id,
                "actual_scope": coordinates.scope,
                "actual_assay": coordinates.assay,
                "actual_kind": coordinates.kind,
            },
        )
    error_context = {
        "assay": coordinates.assay,
        "artifact_id": coordinates.artifact_id,
        "actual_kind": coordinates.kind,
    }
    try:
        status = inspect_artifact(root, coordinates)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact record is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if not status.exists:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact does not exist",
            code="missing_artifact",
            context=error_context,
        )
    if not status.complete:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact is incomplete",
            code="incomplete_artifact",
            context=error_context,
        )
    if status.operation != "import_dimreduc":
        raise ArtifactResolutionError(
            "Imported-coordinate artifact operation must be 'import_dimreduc'",
            code="corrupt_payload",
            context=error_context,
        )
    execution = status.execution_options or {}
    stored_cell_key = execution.get("cell_key")
    if not isinstance(stored_cell_key, str) or not stored_cell_key:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has no cell selection key",
            code="corrupt_payload",
            context=error_context,
        )
    if cell_key is not None and cell_key != stored_cell_key:
        raise ArtifactResolutionError(
            f"cell_key {cell_key!r} does not match imported coordinates "
            f"built for {stored_cell_key!r}",
            code="row_mismatch",
            context={
                **error_context,
                "cell_key": cell_key,
                "expected_cell_key": stored_cell_key,
            },
        )
    block_rows = execution.get("block_rows")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int | np.integer)
        or int(block_rows) < 1
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate block_rows is invalid",
            code="corrupt_payload",
            context=error_context,
        )

    inputs = status.inputs or {}
    raw_selection = inputs.get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has no 'cell_selection' artifact input",
            code="corrupt_payload",
            context=error_context,
        )
    try:
        selection = ArtifactRef.from_dict(raw_selection)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has a malformed cell_selection input",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    validate_cell_selection_artifact(root, selection, stored_cell_key)

    source_digest = inputs.get("source_digest")
    if (
        not isinstance(source_digest, Mapping)
        or set(source_digest) != {"bytes_hex"}
        or not isinstance(source_digest.get("bytes_hex"), str)
        or len(source_digest["bytes_hex"]) != 64
        or source_digest["bytes_hex"].lower() != source_digest["bytes_hex"]
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate source digest is missing",
            code="corrupt_payload",
            context=error_context,
        )
    try:
        bytes.fromhex(source_digest["bytes_hex"])
    except ValueError as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate source digest is not hexadecimal",
            code="corrupt_payload",
            context=error_context,
        ) from exc

    payload_fingerprints = inputs.get("payload_fingerprints")
    if not isinstance(payload_fingerprints, Mapping) or not all(
        isinstance(name, str)
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        for name, fingerprint in payload_fingerprints.items()
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate payload fingerprints are malformed",
            code="corrupt_payload",
            context=error_context,
        )

    try:
        group = artifact_group(root, coordinates)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if "data" not in group:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has no data array",
            code="corrupt_payload",
            context=error_context,
        )
    try:
        data = as_zarr_array(group["data"], name="data")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate data payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if data.ndim != 2 or int(data.shape[0]) < 1 or np.dtype(data.dtype).kind != "f":
        raise ArtifactResolutionError(
            "Imported-coordinate data must be a floating-point matrix",
            code="corrupt_payload",
            context=error_context,
        )

    parameters = status.parameters or {}
    dimreduc_key = parameters.get("dimreduc_key")
    if not isinstance(dimreduc_key, str) or not dimreduc_key:
        raise ArtifactResolutionError(
            "Imported-coordinate source key is missing",
            code="corrupt_payload",
            context=error_context,
        )
    dims = parameters.get("dims")
    if isinstance(dims, bool) or not isinstance(dims, int | np.integer):
        raise ArtifactResolutionError(
            "Imported-coordinate dimensions are missing",
            code="corrupt_payload",
            context=error_context,
        )
    if int(dims) < 1 or int(data.shape[1]) != int(dims):
        raise ArtifactResolutionError(
            "Imported-coordinate dimensions do not match data",
            code="corrupt_payload",
            context=error_context,
        )
    role = parameters.get("role")
    if (
        not isinstance(role, str)
        or not role
        or role.lower() != role
        or role in {"umap", "tsne"}
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate role is invalid",
            code="corrupt_payload",
            context=error_context,
        )

    try:
        selection_group = artifact_group(root, selection)
        mask = as_zarr_array(selection_group["values"], name="values")
        cell_data = group_at(root, "cellData")
        row_ids = as_zarr_array(cell_data["ids"], name="ids")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate cell alignment payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    try:
        selected_fingerprint, selected_count = fingerprint_selected_stored_strings(
            row_ids,
            mask,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate cell alignment payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if int(data.shape[0]) != selected_count:
        raise ArtifactResolutionError(
            "Imported-coordinate rows do not match the exact cell selection",
            code="dimreduc_row_count_mismatch",
            context={
                **error_context,
                "coordinate_rows": int(data.shape[0]),
                "selected_count": selected_count,
            },
        )
    ordered_fingerprint = inputs.get("ordered_cell_ids_fingerprint")
    if (
        not isinstance(ordered_fingerprint, str)
        or ordered_fingerprint != selected_fingerprint
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate cell IDs do not match the selected cell order",
            code="dimreduc_cell_identity_mismatch",
            context={
                **error_context,
                "expected_fingerprint": ordered_fingerprint
                if isinstance(ordered_fingerprint, str)
                else None,
                "actual_fingerprint": selected_fingerprint,
            },
        )

    optional = {
        "loadings": parameters.get("loadings_stored"),
        "feature_ids": parameters.get("feature_ids_stored"),
        "stdev": parameters.get("stdev_stored"),
    }
    for name, stored in optional.items():
        if not isinstance(stored, bool) or stored != (name in group):
            raise ArtifactResolutionError(
                f"Imported-coordinate {name!r} storage flag does not match payload",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            )
    if optional["loadings"] != optional["feature_ids"]:
        raise ArtifactResolutionError(
            "Imported-coordinate loadings and feature IDs must be stored together",
            code="corrupt_payload",
            context=error_context,
        )
    if "loadings" in group:
        try:
            loadings = as_zarr_array(group["loadings"], name="loadings")
            feature_ids = as_zarr_array(group["feature_ids"], name="feature_ids")
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                "Imported-coordinate loadings or feature IDs are malformed",
                code="corrupt_payload",
                context=error_context,
            ) from exc
        if (
            loadings.ndim != 2
            or np.dtype(loadings.dtype).kind != "f"
            or int(loadings.shape[0]) < 1
            or int(loadings.shape[1]) != int(dims)
            or feature_ids.ndim != 1
            or np.dtype(feature_ids.dtype).kind not in {"O", "S", "U"}
            or int(feature_ids.shape[0]) != int(loadings.shape[0])
        ):
            raise ArtifactResolutionError(
                "Imported-coordinate loadings and feature IDs are misaligned",
                code="corrupt_payload",
                context=error_context,
            )
    if "stdev" in group:
        try:
            stdev = as_zarr_array(group["stdev"], name="stdev")
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                "Imported-coordinate stdev payload is malformed",
                code="corrupt_payload",
                context=error_context,
            ) from exc
        if (
            stdev.ndim != 1
            or np.dtype(stdev.dtype).kind != "f"
            or tuple(stdev.shape) != (int(dims),)
        ):
            raise ArtifactResolutionError(
                "Imported-coordinate stdev does not match dimensions",
                code="corrupt_payload",
                context=error_context,
            )

    payload_names = ("data", "loadings", "feature_ids", "stdev")
    stored_payloads = {name for name in payload_names if name in group}
    if set(payload_fingerprints) != stored_payloads:
        raise ArtifactResolutionError(
            "Imported-coordinate payload fingerprints do not match stored payloads",
            code="corrupt_payload",
            context=error_context,
        )
    for name in payload_names:
        if name not in group:
            continue
        expected = payload_fingerprints.get(name)
        if not isinstance(expected, str):
            raise ArtifactResolutionError(
                f"Imported-coordinate payload fingerprint for {name!r} is missing",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            )
        try:
            actual = _payload_fingerprint(group, name)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                f"Imported-coordinate payload fingerprint for {name!r} is unreadable",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            ) from exc
        if actual != expected:
            raise ArtifactResolutionError(
                f"Imported-coordinate payload fingerprint for {name!r} does not match",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            )


def validate_artifact_graph_selection(
    root: zarr.Group,
    connectivity_map: ArtifactRef,
    cell_key: str,
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
    validate_neighbors_artifact_selection(root, neighbors, cell_key)


def validate_neighbors_artifact_selection(
    root: zarr.Group,
    neighbors: ArtifactRef,
    cell_key: str,
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
    )


def validate_normalized_artifact_selection(
    root: zarr.Group,
    normalized: ArtifactRef,
    cell_key: str,
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
    validate_cell_selection_artifact(root, cell_selection, cell_key)
    resolve_feature_selection(root, assay, feature_selection)
