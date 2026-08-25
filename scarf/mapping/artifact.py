"""Persistence and validation for mapping reference artifacts."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import zarr

from ..assay import RNAassay
from ..graph.distances import validate_distance_provenance
from ..graph.state import validate_normalized_artifact_selection
from ..storage.arrays import create_zarr_dataset, create_zarr_obj_array
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    artifact_group,
    inspect_artifact,
)
from ..storage.feature_selection import resolve_feature_selection
from ..storage.geometry import array_geometry
from ..storage.partition import row_band
from ..storage.types import as_zarr_array, as_zarr_group
from .models import ScaledPCAProjectionModel, SymphonyCorrectionModel
from .reference import MappingReference

MAPPING_REFERENCE_REBUILD_MESSAGE = (
    "Rebuild it with build_mapping_reference(neighbors)."
)
_COMMON_ARRAYS = frozenset(
    {
        "feature_ids",
        "feature_means",
        "feature_scales",
        "loadings",
        "reference_distance_quantiles",
        "reference_distance_values",
    }
)
_SYMPHONY_ARRAYS = frozenset(
    {
        "centroids",
        "raw_centroids",
        "corrected_centroids",
        "cluster_mass",
        "sigma",
    }
)
_COMMON_METADATA = frozenset(
    {
        "method",
        "assay",
        "cell_key",
        "selected_cell_count",
        "ann_metric",
        "normalization_parameters",
        "dataset_fingerprint",
    }
)
_SYMPHONY_METADATA = frozenset(
    {
        "batch_columns",
        "harmony_parameters",
        "batch_levels",
    }
)


def _selected_feature_ids(
    root: zarr.Group,
    assay: str,
    feature_selection: ArtifactRef,
) -> np.ndarray:
    """Read selected feature IDs in exact assay row order, blockwise."""
    resolve_feature_selection(root, assay, feature_selection)
    feature_data = as_zarr_group(
        root[f"{assay}/featureData"],
        name=f"{assay}/featureData",
    )
    ids = as_zarr_array(feature_data["ids"], name="ids")
    values = as_zarr_array(
        artifact_group(root, feature_selection)["values"],
        name="values",
    )
    block_rows = min(
        row_band(array_geometry(ids), unit="chunk", fallback=1),
        row_band(array_geometry(values), unit="chunk", fallback=1),
    )
    selected: list[np.ndarray] = []
    for start in range(0, int(values.shape[0]), block_rows):
        stop = min(start + block_rows, int(values.shape[0]))
        mask = np.asarray(values[start:stop], dtype=bool)
        if np.any(mask):
            selected.append(np.asarray(ids[start:stop])[mask])
    if not selected:
        return np.asarray(ids[:0])
    return np.concatenate(selected)


def write_artifact_mapping_reference(
    group: zarr.Group,
    model: ScaledPCAProjectionModel,
    symphony_state: SymphonyCorrectionModel | None,
    feature_ids: np.ndarray,
    metadata: dict[str, Any],
    reference_distance_quantiles: np.ndarray,
    reference_distance_values: np.ndarray,
) -> None:
    """Write the conditional payload of one planned mapping reference."""
    create_zarr_obj_array(group, "feature_ids", np.asarray(feature_ids))
    _write_array(group, "feature_means", model.feature_means)
    _write_array(group, "feature_scales", model.feature_scales)
    _write_array(group, "loadings", model.loadings)
    _write_array(
        group,
        "reference_distance_quantiles",
        reference_distance_quantiles,
    )
    _write_array(
        group,
        "reference_distance_values",
        reference_distance_values,
    )
    if symphony_state is not None:
        _write_array(group, "centroids", symphony_state.centroids)
        _write_array(group, "raw_centroids", symphony_state.raw_centroids)
        _write_array(
            group,
            "corrected_centroids",
            symphony_state.corrected_centroids,
        )
        _write_array(group, "cluster_mass", symphony_state.cluster_mass)
        _write_array(group, "sigma", symphony_state.sigma)
    group.attrs["reference_metadata"] = metadata


def load_artifact_mapping_reference(
    datastore: Any,
    ref: ArtifactRef,
) -> MappingReference:
    """Load one mapping reference after validating its immutable graph chain."""
    if (
        not isinstance(ref, ArtifactRef)
        or ref.scope != "assay"
        or ref.assay is None
        or ref.kind != "mapping_reference"
    ):
        raise _contract_error("Expected an assay-scoped mapping reference artifact")
    status = inspect_artifact(datastore.zw, ref)
    if not status.exists or not status.complete:
        raise _contract_error("Mapping reference artifact is missing or incomplete")
    if status.operation != "build_mapping_reference":
        raise _contract_error("Mapping reference artifact has an old operation")

    parameters = status.parameters or {}
    method = parameters.get("method")
    if method not in {"pca", "symphony"}:
        raise _contract_error("Mapping reference method is missing or unsupported")

    inputs = status.inputs or {}
    expected_inputs = {
        "reduction",
        "ann_index",
        "neighbors",
        "cell_selection",
        "feature_selection",
    }
    if method == "symphony":
        expected_inputs.add("batch_correction")
    if set(inputs) != expected_inputs:
        raise _contract_error(
            "Mapping reference artifact inputs do not match the current contract"
        )
    reduction = _input_ref(
        datastore.zw,
        inputs,
        "reduction",
        kind="reduction",
        scope="assay",
        assay=ref.assay,
    )
    ann_index = _input_ref(
        datastore.zw,
        inputs,
        "ann_index",
        kind="ann_index",
        scope="assay",
        assay=ref.assay,
    )
    neighbors = _input_ref(
        datastore.zw,
        inputs,
        "neighbors",
        kind="neighbors",
        scope="assay",
        assay=ref.assay,
    )
    cell_selection = _input_ref(
        datastore.zw,
        inputs,
        "cell_selection",
        kind="cell_selection",
        scope="datastore",
        assay=None,
    )
    feature_selection = _input_ref(
        datastore.zw,
        inputs,
        "feature_selection",
        kind="feature_selection",
        scope="assay",
        assay=ref.assay,
    )
    batch_correction = (
        _input_ref(
            datastore.zw,
            inputs,
            "batch_correction",
            kind="batch_correction",
            scope="assay",
            assay=ref.assay,
        )
        if method == "symphony"
        else None
    )
    if method == "pca" and "batch_correction" in inputs:
        raise _contract_error("Plain PCA reference includes batch correction")

    reduction_status = inspect_artifact(datastore.zw, reduction)
    ann_status = inspect_artifact(datastore.zw, ann_index)
    neighbors_status = inspect_artifact(datastore.zw, neighbors)
    if reduction_status.operation != "run_pca":
        raise _contract_error("Mapping reference reduction is not PCA")
    if (reduction_status.parameters or {}).get("feat_scaling") is not True:
        raise _contract_error("Mapping reference PCA did not enable feature scaling")
    if ann_status.operation != "build_ann_index":
        raise _contract_error("Mapping reference ANN input has an old operation")
    if neighbors_status.operation != "query_neighbors":
        raise _contract_error("Mapping reference neighbors input has an old operation")

    coordinates = batch_correction or reduction
    if _ref_from_input(ann_status, "coordinates") != coordinates:
        raise _contract_error("ANN index uses different coordinates")
    if (
        _ref_from_input(neighbors_status, "ann_index") != ann_index
        or _ref_from_input(neighbors_status, "coordinates") != coordinates
    ):
        raise _contract_error("Neighbors use a different ANN coordinate chain")
    if batch_correction is not None:
        correction_status = inspect_artifact(datastore.zw, batch_correction)
        if correction_status.operation != "run_harmony":
            raise _contract_error("Symphony reference correction is not Harmony")
        if _ref_from_input(correction_status, "reduction") != reduction:
            raise _contract_error("Batch correction uses a different PCA reduction")

    normalized = _ref_from_input(reduction_status, "normalized")
    feature_scaling = _ref_from_input(reduction_status, "feature_scaling")
    _require_ref(
        normalized,
        kind="normalized",
        scope="assay",
        assay=ref.assay,
        label="normalized",
    )
    _require_ref(
        feature_scaling,
        kind="feature_scaling",
        scope="assay",
        assay=ref.assay,
        label="feature_scaling",
    )
    normalized_status = _complete_status(datastore.zw, normalized, "normalized")
    scaling_status = _complete_status(
        datastore.zw,
        feature_scaling,
        "feature_scaling",
    )
    if normalized_status.operation != "run_normalization":
        raise _contract_error("Normalized input has an old operation")
    if (
        scaling_status.operation != "calculate_feature_scaling"
        or (scaling_status.parameters or {}).get("enabled") is not True
        or _ref_from_input(scaling_status, "normalized") != normalized
    ):
        raise _contract_error("Feature scaling does not match the PCA input")
    if (
        _ref_from_input(normalized_status, "cell_selection") != cell_selection
        or _ref_from_input(normalized_status, "feature_selection") != feature_selection
    ):
        raise _contract_error("Stored selections do not match normalized data")

    group = artifact_group(datastore.zw, ref)
    _validate_payload_names(group, method)
    raw_metadata = group.attrs.get("reference_metadata")
    if not isinstance(raw_metadata, Mapping):
        raise _contract_error("Mapping reference metadata is missing")
    metadata = dict(raw_metadata)
    if any(
        name in metadata
        for name in ("schemaVersion", "schema_version", "modelVersion", "model_version")
    ):
        raise _contract_error("Mapping reference metadata uses a versioned contract")
    if "feature_key" in metadata:
        raise _contract_error(
            "Mapping reference metadata uses the removed feature-key contract"
        )
    expected_metadata = set(_COMMON_METADATA)
    if method == "symphony":
        expected_metadata.update(_SYMPHONY_METADATA)
    if set(metadata) != expected_metadata:
        raise _contract_error(
            "Mapping reference metadata does not match the current contract"
        )
    assay_name = _metadata_string(metadata, "assay")
    cell_key = _metadata_string(metadata, "cell_key")
    if assay_name != ref.assay or metadata.get("method") != method:
        raise _contract_error("Mapping reference metadata does not match its artifact")

    normalization_parameters = metadata.get("normalization_parameters")
    if not isinstance(normalization_parameters, Mapping):
        raise _contract_error("Mapping reference normalization parameters are missing")
    normalization_parameters = dict(normalization_parameters)
    if normalization_parameters != (normalized_status.parameters or {}):
        raise _contract_error("Mapping reference normalization parameters changed")
    size_factor = normalization_parameters.get("size_factor")
    if (
        isinstance(size_factor, bool)
        or not isinstance(size_factor, int | float)
        or not np.isfinite(size_factor)
        or float(size_factor) <= 0
    ):
        raise _contract_error(
            "Mapping reference normalization size_factor must be finite and positive"
        )

    ann_metric = (ann_status.parameters or {}).get("ann_metric")
    if ann_metric not in {"l2", "cosine"}:
        raise _contract_error("Mapping reference ANN metric is unsupported")
    if (
        metadata.get("ann_metric") != ann_metric
        or (neighbors_status.parameters or {}).get("distance_metric") != ann_metric
    ):
        raise _contract_error("Mapping reference distance metrics do not agree")
    dataset_fingerprint = metadata.get("dataset_fingerprint")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise _contract_error("Mapping reference dataset fingerprint is missing")
    assay = datastore._get_assay(assay_name)
    if not isinstance(assay, RNAassay):
        raise _contract_error("Mapping references currently support RNA assays only")
    if assay.attrs.get("dataset_fingerprint") != dataset_fingerprint:
        raise _contract_error(
            "Live assay dataset fingerprint does not match the mapping reference"
        )

    selected_cell_count = metadata.get("selected_cell_count")
    if (
        isinstance(selected_cell_count, bool)
        or not isinstance(selected_cell_count, int)
        or selected_cell_count < 1
    ):
        raise _contract_error("Selected reference cell count is missing")
    selected_count = int(
        np.count_nonzero(
            np.asarray(
                as_zarr_array(
                    artifact_group(datastore.zw, cell_selection)["values"],
                    name="values",
                )[:],
                dtype=bool,
            )
        )
    )
    if selected_count != selected_cell_count:
        raise _contract_error("Selected reference cell count does not match its input")
    try:
        validate_normalized_artifact_selection(
            datastore.zw,
            normalized,
            cell_key,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error(
            "Mapping reference cell or feature selection no longer matches"
        ) from exc

    feature_ids = _values(group, "feature_ids")
    try:
        model = ScaledPCAProjectionModel(
            feature_means=_values(group, "feature_means"),
            feature_scales=_values(group, "feature_scales"),
            loadings=_values(group, "loadings"),
        )
    except ValueError as exc:
        raise _contract_error("Mapping reference PCA model is invalid") from exc
    if feature_ids.ndim != 1 or len(feature_ids) != model.n_features:
        raise _contract_error("Mapping reference feature IDs do not match PCA loadings")
    if np.asarray(feature_ids).dtype.kind not in {"O", "S", "U"}:
        raise _contract_error("Mapping reference feature IDs must contain strings")
    if not np.array_equal(
        feature_ids.astype(str),
        _selected_feature_ids(
            datastore.zw,
            assay_name,
            feature_selection,
        ).astype(str),
    ):
        raise _contract_error(
            "Mapping reference feature IDs do not match the selected features"
        )

    symphony_state = None
    if method == "symphony":
        try:
            symphony_state = SymphonyCorrectionModel(
                centroids=_values(group, "centroids"),
                raw_centroids=_values(group, "raw_centroids"),
                corrected_centroids=_values(group, "corrected_centroids"),
                cluster_mass=_values(group, "cluster_mass"),
                sigma=_values(group, "sigma"),
            )
        except ValueError as exc:
            raise _contract_error(
                "Mapping reference Symphony correction model is invalid"
            ) from exc
        if symphony_state.n_dims != model.n_dims:
            raise _contract_error(
                "Symphony correction dimensions do not match PCA loadings"
            )

    distance_quantiles = _values(group, "reference_distance_quantiles")
    distance_values = _values(group, "reference_distance_values")
    if (
        distance_quantiles.ndim != 1
        or distance_values.shape != distance_quantiles.shape
        or distance_quantiles.size < 1
        or not np.all(np.isfinite(distance_quantiles))
        or not np.all(np.isfinite(distance_values))
        or np.any(distance_quantiles < 0)
        or np.any(distance_quantiles > 1)
        or np.any(distance_values < 0)
        or np.any(np.diff(distance_quantiles) < 0)
        or np.any(np.diff(distance_values) < 0)
    ):
        raise _contract_error("Mapping reference distance summary is invalid")
    try:
        validate_distance_provenance(datastore.zw, neighbors)
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error(
            "Mapping reference neighbor distances are invalid"
        ) from exc

    return MappingReference(
        datastore=datastore,
        ref=ref,
        assay_name=assay_name,
        cell_key=cell_key,
        reduction=reduction,
        ann_index=ann_index,
        neighbors=neighbors,
        cell_selection=cell_selection,
        feature_selection=feature_selection,
        batch_correction=batch_correction,
        dataset_fingerprint=dataset_fingerprint,
        selected_cell_count=selected_cell_count,
        model=model,
        symphony_state=symphony_state,
        feature_ids=feature_ids,
        metadata=metadata,
        reference_distance_quantiles=distance_quantiles,
        reference_distance_values=distance_values,
    )


def _input_ref(
    root: zarr.Group,
    inputs: Mapping[str, Any],
    name: str,
    *,
    kind: str,
    scope: str,
    assay: str | None,
) -> ArtifactRef:
    raw_ref = inputs.get(name)
    if not isinstance(raw_ref, Mapping):
        raise _contract_error(f"Mapping reference input {name!r} is missing")
    try:
        ref = ArtifactRef.from_dict(raw_ref)
    except (TypeError, ValueError) as exc:
        raise _contract_error(f"Mapping reference input {name!r} is malformed") from exc
    _require_ref(
        ref,
        kind=kind,
        scope=scope,
        assay=assay,
        label=name,
    )
    _complete_status(root, ref, name)
    return ref


def _require_ref(
    ref: ArtifactRef,
    *,
    kind: str,
    scope: str,
    assay: str | None,
    label: str,
) -> None:
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise _contract_error(
            f"Mapping reference input {label!r} has the wrong artifact kind or scope"
        )


def _complete_status(
    root: zarr.Group,
    ref: ArtifactRef,
    label: str,
) -> ArtifactStatus:
    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        raise _contract_error(
            f"Mapping reference input {label!r} is missing or incomplete"
        )
    return status


def _ref_from_input(status: ArtifactStatus, name: str) -> ArtifactRef:
    raw_ref = (status.inputs or {}).get(name)
    if not isinstance(raw_ref, Mapping):
        raise _contract_error(
            f"{status.ref.kind} input {name!r} is missing from the graph chain"
        )
    try:
        return ArtifactRef.from_dict(raw_ref)
    except (TypeError, ValueError) as exc:
        raise _contract_error(f"{status.ref.kind} input {name!r} is malformed") from exc


def _validate_payload_names(group: zarr.Group, method: str) -> None:
    if set(group.group_keys()):
        raise _contract_error(
            "Mapping reference contains groups outside the current contract"
        )
    arrays = set(group.array_keys())
    expected = set(_COMMON_ARRAYS)
    if method == "symphony":
        expected.update(_SYMPHONY_ARRAYS)
    missing = expected - arrays
    unexpected = arrays - expected
    if missing:
        raise _contract_error(
            "Mapping reference is missing required arrays: "
            + ", ".join(sorted(missing))
        )
    if unexpected:
        raise _contract_error(
            "Mapping reference contains arrays outside the current contract: "
            + ", ".join(sorted(unexpected))
        )
    if any(
        name in group.attrs
        for name in ("schemaVersion", "schema_version", "modelVersion", "model_version")
    ):
        raise _contract_error("Mapping reference uses an old versioned contract")


def _metadata_string(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise _contract_error(f"Mapping reference metadata {name!r} is missing")
    return value


def _values(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[:])


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    array_values = np.asarray(values, dtype=np.float64)
    if array_values.ndim not in {1, 2}:
        raise ValueError(f"Mapping reference array {name!r} must have one or two axes")
    chunks = (
        (min(max(array_values.shape[0], 1), 10_000),)
        if array_values.ndim == 1
        else (
            min(max(array_values.shape[0], 1), 1_000),
            max(array_values.shape[1], 1),
        )
    )
    array = create_zarr_dataset(
        group,
        name,
        chunks,
        "f8",
        array_values.shape,
    )
    array[...] = array_values


def _contract_error(detail: str) -> ValueError:
    return ValueError(f"{detail}. {MAPPING_REFERENCE_REBUILD_MESSAGE}")
