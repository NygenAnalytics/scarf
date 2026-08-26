from collections.abc import Mapping
from dataclasses import dataclass

import zarr

from ..storage.artifacts import ArtifactRef, ArtifactStatus, inspect_artifact
from ..storage.errors import ArtifactResolutionError
from ..storage.feature_selection import resolve_feature_selection
from ..storage.selections import validate_stored_selection_artifact
from .errors import IncompatibleAnalysisStateError

_LEGACY_FEATURE_CONTRACT_KEYS = frozenset({"feat_key", "feature_key"})


@dataclass(frozen=True, slots=True)
class NativeGraphInputs:
    """Validated named inputs for one assay-scoped graph branch."""

    neighbors: ArtifactRef
    ann_index: ArtifactRef
    coordinates: ArtifactRef
    reduction: ArtifactRef | None
    normalized: ArtifactRef | None
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef | None


def _resolution_error(
    message: str,
    *,
    code: str,
    ref: ArtifactRef,
    **context: str | int | float | bool | None,
) -> ArtifactResolutionError:
    return ArtifactResolutionError(
        message,
        code=code,
        context={
            "assay": ref.assay,
            "artifact_id": ref.artifact_id,
            "actual_kind": ref.kind,
            **context,
        },
    )


def _legacy_contract_error(
    message: str,
    ref: ArtifactRef,
    *,
    input_name: str | None = None,
) -> IncompatibleAnalysisStateError:
    return IncompatibleAnalysisStateError(
        message,
        code="legacy_feature_contract",
        context={
            "assay": ref.assay,
            "artifact_id": ref.artifact_id,
            "artifact_kind": ref.kind,
            "input_name": input_name,
        },
    )


def _contains_legacy_feature_contract(value: object) -> bool:
    if isinstance(value, Mapping):
        if any(
            isinstance(key, str) and key in _LEGACY_FEATURE_CONTRACT_KEYS
            for key in value
        ):
            return True
        return any(_contains_legacy_feature_contract(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_legacy_feature_contract(item) for item in value)
    return False


def _integrated_contract_error(
    message: str,
    graph: ArtifactRef,
    *,
    value: object = None,
    input_name: str | None = None,
) -> ArtifactResolutionError | IncompatibleAnalysisStateError:
    if _contains_legacy_feature_contract(value):
        return _legacy_contract_error(message, graph, input_name=input_name)
    return _resolution_error(
        message,
        code="corrupt_payload",
        ref=graph,
        input_name=input_name,
    )


def _named_input_contract_error(
    message: str,
    owner: ArtifactRef,
    *,
    inputs: Mapping[str, object],
    input_name: str,
    value: object,
) -> ArtifactResolutionError | IncompatibleAnalysisStateError:
    """Classify a bad named edge without treating modern damage as legacy state."""

    legacy_keys = {
        key: item for key, item in inputs.items() if key in {"feat_key", "feature_key"}
    }
    if (
        legacy_keys
        or _contains_legacy_feature_contract(value)
        or isinstance(value, str)
    ):
        return _legacy_contract_error(message, owner, input_name=input_name)
    return _resolution_error(
        message,
        code="corrupt_payload",
        ref=owner,
        input_name=input_name,
    )


def _require_complete(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    expected_kind: str,
    expected_scope: str,
    expected_assay: str | None,
) -> ArtifactStatus:
    if ref.kind != expected_kind:
        raise _resolution_error(
            f"Expected a {expected_kind} artifact, got {ref.kind}",
            code="wrong_kind",
            ref=ref,
            expected_kind=expected_kind,
        )
    if ref.scope != expected_scope:
        raise _resolution_error(
            f"{expected_kind} artifact has the wrong scope",
            code="wrong_scope",
            ref=ref,
            expected_scope=expected_scope,
        )
    if expected_assay is not None and ref.assay != expected_assay:
        raise _resolution_error(
            f"{expected_kind} artifact belongs to a different assay",
            code="wrong_assay",
            ref=ref,
            expected_assay=expected_assay,
        )
    try:
        status = inspect_artifact(root, ref)
    except (KeyError, TypeError, ValueError) as error:
        raise _resolution_error(
            "Artifact record is malformed",
            code="corrupt_payload",
            ref=ref,
        ) from error
    if not status.exists:
        raise _resolution_error(
            f"Artifact does not exist: {status.path}",
            code="missing_artifact",
            ref=ref,
        )
    if not status.complete:
        raise _resolution_error(
            f"Artifact is incomplete: {status.path}",
            code="incomplete_artifact",
            ref=ref,
        )
    return status


def _input_ref(
    root: zarr.Group,
    owner: ArtifactRef,
    name: str,
    *,
    expected_kind: str,
    expected_scope: str,
    expected_assay: str | None,
) -> ArtifactRef:
    status = _require_complete(
        root,
        owner,
        expected_kind=owner.kind,
        expected_scope=owner.scope,
        expected_assay=owner.assay,
    )
    inputs = status.inputs or {}
    raw = inputs.get(name)
    if not isinstance(raw, Mapping):
        raise _named_input_contract_error(
            f"{owner.kind} artifact has no {name!r} artifact input",
            owner,
            inputs=inputs,
            input_name=name,
            value=raw,
        )
    try:
        value = ArtifactRef.from_dict(raw)
    except (TypeError, ValueError) as error:
        raise _named_input_contract_error(
            f"{owner.kind} artifact has a malformed {name!r} input",
            owner,
            inputs=inputs,
            input_name=name,
            value=raw,
        ) from error
    if set(raw) != set(value.to_dict()):
        raise _named_input_contract_error(
            f"{owner.kind} artifact has a malformed {name!r} input",
            owner,
            inputs=inputs,
            input_name=name,
            value=raw,
        )
    _require_complete(
        root,
        value,
        expected_kind=expected_kind,
        expected_scope=expected_scope,
        expected_assay=expected_assay,
    )
    return value


def _validate_cell_selection(root: zarr.Group, selection: ArtifactRef) -> None:
    status = _require_complete(
        root,
        selection,
        expected_kind="cell_selection",
        expected_scope="datastore",
        expected_assay=None,
    )
    source_column = (status.execution_options or {}).get("source_column")
    if not isinstance(source_column, str) or not source_column:
        raise _resolution_error(
            "Cell-selection artifact has no source_column",
            code="corrupt_payload",
            ref=selection,
        )
    validate_stored_selection_artifact(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
        column=source_column,
    )


def resolve_native_graph_inputs(
    root: zarr.Group,
    source: ArtifactRef,
) -> NativeGraphInputs:
    """Resolve one native connectivity or neighbor branch through named inputs."""

    if source.kind == "connectivity_map":
        _require_complete(
            root,
            source,
            expected_kind="connectivity_map",
            expected_scope="assay",
            expected_assay=source.assay,
        )
        neighbors = _input_ref(
            root,
            source,
            "neighbors",
            expected_kind="neighbors",
            expected_scope="assay",
            expected_assay=source.assay,
        )
    elif source.kind == "neighbors":
        _require_complete(
            root,
            source,
            expected_kind="neighbors",
            expected_scope="assay",
            expected_assay=source.assay,
        )
        neighbors = source
    else:
        raise _resolution_error(
            "Native graph source must be connectivity_map or neighbors",
            code="unsupported_graph_kind",
            ref=source,
            expected_kind="connectivity_map,neighbors",
        )
    if neighbors.assay is None:
        raise _resolution_error(
            "Native graph source has no assay",
            code="wrong_scope",
            ref=neighbors,
            expected_scope="assay",
        )
    assay = neighbors.assay
    ann_index = _input_ref(
        root,
        neighbors,
        "ann_index",
        expected_kind="ann_index",
        expected_scope="assay",
        expected_assay=assay,
    )
    neighbor_status = _require_complete(
        root,
        neighbors,
        expected_kind="neighbors",
        expected_scope="assay",
        expected_assay=assay,
    )
    neighbor_inputs = neighbor_status.inputs or {}
    raw_coordinates = neighbor_inputs.get("coordinates")
    if not isinstance(raw_coordinates, Mapping):
        raise _named_input_contract_error(
            "neighbors artifact has no 'coordinates' artifact input",
            neighbors,
            inputs=neighbor_inputs,
            input_name="coordinates",
            value=raw_coordinates,
        )
    try:
        coordinates = ArtifactRef.from_dict(raw_coordinates)
    except (TypeError, ValueError) as error:
        raise _named_input_contract_error(
            "neighbors artifact has a malformed 'coordinates' input",
            neighbors,
            inputs=neighbor_inputs,
            input_name="coordinates",
            value=raw_coordinates,
        ) from error
    if set(raw_coordinates) != set(coordinates.to_dict()):
        raise _named_input_contract_error(
            "neighbors artifact has a malformed 'coordinates' input",
            neighbors,
            inputs=neighbor_inputs,
            input_name="coordinates",
            value=raw_coordinates,
        )
    if coordinates.kind not in {
        "reduction",
        "batch_correction",
        "imported_coordinates",
    }:
        raise _resolution_error(
            "Neighbor coordinates have an unsupported artifact kind",
            code="unsupported_graph_kind",
            ref=coordinates,
            expected_kind="reduction,batch_correction,imported_coordinates",
        )
    _require_complete(
        root,
        coordinates,
        expected_kind=coordinates.kind,
        expected_scope="assay",
        expected_assay=assay,
    )
    ann_coordinates = _input_ref(
        root,
        ann_index,
        "coordinates",
        expected_kind=coordinates.kind,
        expected_scope="assay",
        expected_assay=assay,
    )
    if ann_coordinates != coordinates:
        raise _resolution_error(
            "ANN index and neighbors name different coordinate artifacts",
            code="corrupt_payload",
            ref=neighbors,
            input_name="coordinates",
        )

    if coordinates.kind == "imported_coordinates":
        from .state import validate_imported_coordinates_artifact

        validate_imported_coordinates_artifact(root, coordinates)
        cell_selection = _input_ref(
            root,
            coordinates,
            "cell_selection",
            expected_kind="cell_selection",
            expected_scope="datastore",
            expected_assay=None,
        )
        _validate_cell_selection(root, cell_selection)
        return NativeGraphInputs(
            neighbors=neighbors,
            ann_index=ann_index,
            coordinates=coordinates,
            reduction=None,
            normalized=None,
            cell_selection=cell_selection,
            feature_selection=None,
        )

    if coordinates.kind == "batch_correction":
        reduction = _input_ref(
            root,
            coordinates,
            "reduction",
            expected_kind="reduction",
            expected_scope="assay",
            expected_assay=assay,
        )
    else:
        reduction = coordinates
    normalized = _input_ref(
        root,
        reduction,
        "normalized",
        expected_kind="normalized",
        expected_scope="assay",
        expected_assay=assay,
    )
    cell_selection = _input_ref(
        root,
        normalized,
        "cell_selection",
        expected_kind="cell_selection",
        expected_scope="datastore",
        expected_assay=None,
    )
    _validate_cell_selection(root, cell_selection)
    feature_selection = _input_ref(
        root,
        normalized,
        "feature_selection",
        expected_kind="feature_selection",
        expected_scope="assay",
        expected_assay=assay,
    )
    feature_selection = resolve_feature_selection(root, assay, feature_selection)
    return NativeGraphInputs(
        neighbors=neighbors,
        ann_index=ann_index,
        coordinates=coordinates,
        reduction=reduction,
        normalized=normalized,
        cell_selection=cell_selection,
        feature_selection=feature_selection,
    )


def _integrated_sources(
    root: zarr.Group,
    graph: ArtifactRef,
) -> tuple[ArtifactStatus, list[ArtifactRef]]:
    status = _require_complete(
        root,
        graph,
        expected_kind="integrated_graph",
        expected_scope="datastore",
        expected_assay=None,
    )
    parameters = status.parameters or {}
    method = parameters.get("method")
    assays = parameters.get("assays")
    if status.operation != "integrate_assays" or method not in {"snn", "wnn"}:
        raise _integrated_contract_error(
            "Integrated graph has invalid source parameters",
            graph,
            value={"parameters": parameters, "inputs": status.inputs or {}},
        )
    expected_parameters = (
        {"method", "assays", "l2_normalize"}
        if method == "wnn"
        else {"method", "assays"}
    )
    if (
        not isinstance(assays, list)
        or set(parameters) != expected_parameters
        or (method == "wnn" and not isinstance(parameters.get("l2_normalize"), bool))
    ):
        raise _integrated_contract_error(
            "Integrated graph has invalid source parameters",
            graph,
            value=parameters,
        )
    if (
        len(assays) < 2
        or any(not isinstance(assay, str) or not assay for assay in assays)
        or len(set(assays)) != len(assays)
    ):
        raise _integrated_contract_error(
            "Integrated graph requires at least two unique assay names",
            graph,
            value=assays,
        )
    inputs = status.inputs or {}
    expected_inputs = {
        "cell_selection",
        *(f"source_{index}" for index in range(len(assays))),
    }
    if set(inputs) != expected_inputs:
        raise _integrated_contract_error(
            "Integrated graph source count does not match its assay order",
            graph,
            value=inputs,
        )
    sources: list[ArtifactRef] = []
    for index, assay in enumerate(assays):
        source_name = f"source_{index}"
        if method == "snn":
            raw_source = inputs.get(source_name)
            if not isinstance(raw_source, Mapping):
                raise _integrated_contract_error(
                    "Integrated SNN graph has a malformed source",
                    graph,
                    value=raw_source,
                    input_name=source_name,
                )
            try:
                source = ArtifactRef.from_dict(raw_source)
            except (TypeError, ValueError) as error:
                raise _integrated_contract_error(
                    "Integrated SNN graph has a malformed source",
                    graph,
                    value=raw_source,
                    input_name=source_name,
                ) from error
            if set(raw_source) != set(source.to_dict()):
                raise _integrated_contract_error(
                    "Integrated SNN graph has a malformed source",
                    graph,
                    value=raw_source,
                    input_name=source_name,
                )
            _require_complete(
                root,
                source,
                expected_kind="connectivity_map",
                expected_scope="assay",
                expected_assay=assay,
            )
            sources.append(source)
            continue
        raw_bundle = inputs.get(source_name)
        if not isinstance(raw_bundle, Mapping) or set(raw_bundle) != {
            "neighbors",
            "coordinates",
        }:
            raise _integrated_contract_error(
                "Integrated WNN graph has no source bundle",
                graph,
                value=raw_bundle,
                input_name=source_name,
            )
        raw_neighbors = raw_bundle.get("neighbors")
        raw_coordinates = raw_bundle.get("coordinates")
        if not isinstance(raw_neighbors, Mapping) or not isinstance(
            raw_coordinates, Mapping
        ):
            raise _integrated_contract_error(
                "Integrated WNN source bundle is incomplete",
                graph,
                value=raw_bundle,
                input_name=source_name,
            )
        try:
            neighbors = ArtifactRef.from_dict(raw_neighbors)
            coordinates = ArtifactRef.from_dict(raw_coordinates)
        except (TypeError, ValueError) as error:
            raise _integrated_contract_error(
                "Integrated WNN source bundle is malformed",
                graph,
                value=raw_bundle,
                input_name=source_name,
            ) from error
        if set(raw_neighbors) != set(neighbors.to_dict()) or set(
            raw_coordinates
        ) != set(coordinates.to_dict()):
            raise _integrated_contract_error(
                "Integrated WNN source bundle is malformed",
                graph,
                value=raw_bundle,
                input_name=source_name,
            )
        _require_complete(
            root,
            neighbors,
            expected_kind="neighbors",
            expected_scope="assay",
            expected_assay=assay,
        )
        if coordinates.kind not in {"reduction", "batch_correction"}:
            raise _resolution_error(
                "WNN coordinates must be reduction or batch_correction",
                code="wrong_kind",
                ref=coordinates,
                expected_kind="reduction,batch_correction",
            )
        _require_complete(
            root,
            coordinates,
            expected_kind=coordinates.kind,
            expected_scope="assay",
            expected_assay=assay,
        )
        ancestry = resolve_native_graph_inputs(root, neighbors)
        if ancestry.coordinates != coordinates:
            raise _integrated_contract_error(
                "Integrated WNN source names coordinates that differ from neighbors",
                graph,
                value=raw_bundle,
                input_name=source_name,
            )
        sources.append(neighbors)
    return status, sources


def graph_cell_selection(root: zarr.Group, graph: ArtifactRef) -> ArtifactRef:
    """Return the exact cell-selection input for a native or integrated graph."""

    if graph.kind in {"connectivity_map", "neighbors"}:
        return resolve_native_graph_inputs(root, graph).cell_selection
    if graph.kind == "integrated_graph":
        status, sources = _integrated_sources(root, graph)
        raw_selection = (status.inputs or {}).get("cell_selection")
        if not isinstance(raw_selection, Mapping):
            raise _integrated_contract_error(
                "Integrated graph has a malformed cell-selection input",
                graph,
                value=raw_selection,
                input_name="cell_selection",
            )
        try:
            selection = ArtifactRef.from_dict(raw_selection)
        except (TypeError, ValueError) as error:
            raise _integrated_contract_error(
                "Integrated graph has a malformed cell-selection input",
                graph,
                value=raw_selection,
                input_name="cell_selection",
            ) from error
        if set(raw_selection) != set(selection.to_dict()):
            raise _integrated_contract_error(
                "Integrated graph has a malformed cell-selection input",
                graph,
                value=raw_selection,
                input_name="cell_selection",
            )
        _require_complete(
            root,
            selection,
            expected_kind="cell_selection",
            expected_scope="datastore",
            expected_assay=None,
        )
        _validate_cell_selection(root, selection)
        for source in sources:
            if resolve_native_graph_inputs(root, source).cell_selection != selection:
                raise _integrated_contract_error(
                    "Integrated graph sources do not name its shared cell selection",
                    graph,
                    value=status.inputs or {},
                    input_name="cell_selection",
                )
        return selection
    raise _resolution_error(
        "Graph must be connectivity_map, neighbors, or integrated_graph",
        code="unsupported_graph_kind",
        ref=graph,
        expected_kind="connectivity_map,neighbors,integrated_graph",
    )


def graph_source_assays(
    root: zarr.Group,
    graph: ArtifactRef,
) -> tuple[str, ...]:
    """Return validated source-assay names in persisted graph order."""
    if graph.kind in {"connectivity_map", "neighbors"}:
        ancestry = resolve_native_graph_inputs(root, graph)
        assay = ancestry.neighbors.assay
        if assay is None:
            raise _resolution_error(
                "Native graph source has no assay",
                code="wrong_scope",
                ref=graph,
                expected_scope="assay",
            )
        return (assay,)
    if graph.kind == "integrated_graph":
        _status, sources = _integrated_sources(root, graph)
        assays: list[str] = []
        for source in sources:
            ancestry = resolve_native_graph_inputs(root, source)
            assay = ancestry.neighbors.assay
            if assay is None:
                raise _resolution_error(
                    "Integrated graph source has no assay",
                    code="wrong_scope",
                    ref=source,
                    expected_scope="assay",
                )
            assays.append(assay)
        return tuple(assays)
    raise _resolution_error(
        "Graph must be connectivity_map, neighbors, or integrated_graph",
        code="unsupported_graph_kind",
        ref=graph,
        expected_kind="connectivity_map,neighbors,integrated_graph",
    )


def resolve_graph_assay_inputs(
    root: zarr.Group,
    graph: ArtifactRef,
    assay: str,
) -> NativeGraphInputs:
    """Resolve one assay branch captured by a native or integrated graph."""
    if graph.kind in {"connectivity_map", "neighbors"}:
        ancestry = resolve_native_graph_inputs(root, graph)
        if ancestry.neighbors.assay != assay:
            raise _resolution_error(
                f"Graph has no source for assay {assay!r}",
                code="wrong_assay",
                ref=graph,
                expected_assay=assay,
            )
        return ancestry
    if graph.kind != "integrated_graph":
        raise _resolution_error(
            "Graph must be connectivity_map, neighbors, or integrated_graph",
            code="unsupported_graph_kind",
            ref=graph,
            expected_kind="connectivity_map,neighbors,integrated_graph",
        )

    _status, sources = _integrated_sources(root, graph)
    matches: list[NativeGraphInputs] = []
    for source in sources:
        ancestry = resolve_native_graph_inputs(root, source)
        if ancestry.neighbors.assay == assay:
            matches.append(ancestry)
    if not matches:
        raise _resolution_error(
            f"Integrated graph has no source for assay {assay!r}",
            code="wrong_assay",
            ref=graph,
            expected_assay=assay,
        )
    if len(matches) != 1:
        raise _resolution_error(
            f"Integrated graph has multiple sources for assay {assay!r}",
            code="corrupt_payload",
            ref=graph,
            expected_assay=assay,
        )
    return matches[0]


def project_normalized_feature_selections(
    root: zarr.Group,
    graph: ArtifactRef,
) -> tuple[ArtifactRef, ...]:
    """Project exact normalized feature-selection ancestry from a graph."""

    if graph.kind in {"connectivity_map", "neighbors"}:
        selection = resolve_native_graph_inputs(root, graph).feature_selection
        return () if selection is None else (selection,)
    if graph.kind != "integrated_graph":
        raise _resolution_error(
            "Graph must be connectivity_map, neighbors, or integrated_graph",
            code="unsupported_graph_kind",
            ref=graph,
            expected_kind="connectivity_map,neighbors,integrated_graph",
        )
    _status, sources = _integrated_sources(root, graph)
    graph_cell_selection(root, graph)
    projected: list[ArtifactRef] = []
    seen: set[ArtifactRef] = set()
    for source in sources:
        selection = resolve_native_graph_inputs(root, source).feature_selection
        if selection is not None and selection not in seen:
            seen.add(selection)
            projected.append(selection)
    return tuple(projected)
