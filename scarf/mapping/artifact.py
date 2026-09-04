"""Persistence and validation for mapping reference artifacts."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import zarr

from ..assay import RNAassay
from ..graph.distances import (
    validate_distance_provenance,
    validate_neighbors_payload,
)
from ..storage.ann_index import validate_ann_index_payload
from ..storage.arrays import create_zarr_dataset, create_zarr_obj_array
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    ValueFingerprintBuilder,
    artifact_group,
    canonical_bytes,
    fingerprint_stored_arrays,
    inspect_artifact,
)
from ..storage.feature_selection import resolve_feature_selection
from ..storage.geometry import array_geometry
from ..storage.partition import row_band
from ..storage.selections import validate_stored_selection_integrity
from ..storage.types import as_zarr_array, as_zarr_group
from .confidence import _distance_quantile_summary
from .features import _normalization_parameters
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
_COMMON_ARRAY_ORDER = (
    "feature_ids",
    "feature_means",
    "feature_scales",
    "loadings",
    "reference_distance_quantiles",
    "reference_distance_values",
)
_SYMPHONY_ARRAY_ORDER = (
    "centroids",
    "raw_centroids",
    "corrected_centroids",
    "cluster_mass",
    "sigma",
)
_COMMON_METADATA = frozenset(
    {
        "method",
        "assay",
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
    method = metadata.get("method")
    if method not in {"pca", "symphony"}:
        raise ValueError("Mapping reference metadata has an unsupported method")
    group.attrs["payload_fingerprint"] = _payload_fingerprint(
        group,
        method,
        metadata,
    )


def write_artifact_mapping_reference_from_sources(
    group: zarr.Group,
    *,
    feature_means: zarr.Array,
    feature_scales: zarr.Array,
    loadings: zarr.Array,
    symphony_sources: Mapping[str, zarr.Array] | None,
    feature_ids: np.ndarray,
    metadata: dict[str, Any],
    reference_distance_quantiles: np.ndarray,
    reference_distance_values: np.ndarray,
) -> None:
    """Stream an immutable mapping reference from its upstream artifacts."""
    create_zarr_obj_array(group, "feature_ids", np.asarray(feature_ids))
    _write_array_from_source(group, "feature_means", feature_means)
    _write_array_from_source(group, "feature_scales", feature_scales)
    _write_array_from_source(group, "loadings", loadings)
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
    if symphony_sources is not None:
        expected = set(_SYMPHONY_ARRAY_ORDER)
        if set(symphony_sources) != expected:
            raise ValueError(
                "Mapping reference Symphony sources do not match the payload contract"
            )
        for name in _SYMPHONY_ARRAY_ORDER:
            _write_array_from_source(
                group,
                name,
                symphony_sources[name],
                transpose=name == "centroids",
            )
    group.attrs["reference_metadata"] = metadata
    method = metadata.get("method")
    if method not in {"pca", "symphony"}:
        raise ValueError("Mapping reference metadata has an unsupported method")
    if (method == "symphony") != (symphony_sources is not None):
        raise ValueError("Mapping reference method and Symphony sources do not match")
    group.attrs["payload_fingerprint"] = _payload_fingerprint(
        group,
        method,
        metadata,
    )


def load_artifact_mapping_reference(
    datastore: Any,
    ref: ArtifactRef,
    *,
    require_complete: bool = True,
) -> MappingReference:
    """Load one mapping reference after validating its immutable graph chain."""
    result = _validate_and_load_artifact_mapping_reference(
        datastore,
        ref,
        require_complete=require_complete,
        build_result=True,
    )
    assert isinstance(result, MappingReference)
    return result


def validate_artifact_mapping_reference(
    datastore: Any,
    ref: ArtifactRef,
    *,
    require_complete: bool = True,
) -> None:
    """Validate a mapping-reference artifact without materializing its model."""
    _validate_and_load_artifact_mapping_reference(
        datastore,
        ref,
        require_complete=require_complete,
        build_result=False,
    )


def _validate_and_load_artifact_mapping_reference(
    datastore: Any,
    ref: ArtifactRef,
    *,
    require_complete: bool,
    build_result: bool,
) -> MappingReference | None:
    if (
        not isinstance(ref, ArtifactRef)
        or ref.scope != "assay"
        or ref.assay is None
        or ref.kind != "mapping_reference"
    ):
        raise _contract_error("Expected an assay-scoped mapping reference artifact")
    status = inspect_artifact(datastore.zw, ref)
    if not status.exists or (require_complete and not status.complete):
        raise _contract_error("Mapping reference artifact is missing or incomplete")
    if status.operation != "build_mapping_reference":
        raise _contract_error("Mapping reference artifact has an old operation")

    parameters = status.parameters or {}
    if set(parameters) != {"method"}:
        raise _contract_error(
            "Mapping-reference parameters do not match the current contract"
        )
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
    normalized_dataset_fingerprint = (normalized_status.inputs or {}).get(
        "dataset_fingerprint"
    )
    if (
        not isinstance(normalized_dataset_fingerprint, str)
        or not normalized_dataset_fingerprint
    ):
        raise _contract_error("Normalized input has no dataset fingerprint")
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
    expected_metadata = set(_COMMON_METADATA)
    if method == "symphony":
        expected_metadata.update(_SYMPHONY_METADATA)
    if set(metadata) != expected_metadata:
        raise _contract_error(
            "Mapping reference metadata does not match the current contract"
        )
    assay_name = _metadata_string(metadata, "assay")
    if assay_name != ref.assay or metadata.get("method") != method:
        raise _contract_error("Mapping reference metadata does not match its artifact")

    normalization_parameters = metadata.get("normalization_parameters")
    if not isinstance(normalization_parameters, Mapping):
        raise _contract_error("Mapping reference normalization parameters are missing")
    normalization_parameters = dict(normalization_parameters)
    if normalization_parameters != (normalized_status.parameters or {}):
        raise _contract_error("Mapping reference normalization parameters changed")
    try:
        normalization_parameters = _normalization_parameters(normalization_parameters)
    except (TypeError, ValueError) as exc:
        raise _contract_error(
            f"Mapping reference normalization is unsupported: {exc}"
        ) from exc

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
    if dataset_fingerprint != normalized_dataset_fingerprint:
        raise _contract_error(
            "Mapping reference dataset fingerprint does not match normalized data"
        )
    assay = datastore._get_assay(assay_name)
    if not isinstance(assay, RNAassay):
        raise _contract_error("Mapping references currently support RNA assays only")
    stored_dataset_fingerprint = assay.attrs.get("dataset_fingerprint")
    live_dataset_fingerprint = (
        stored_dataset_fingerprint
        if isinstance(stored_dataset_fingerprint, str) and stored_dataset_fingerprint
        else datastore._calculate_dataset_fingerprint(assay_name)
    )
    if live_dataset_fingerprint != dataset_fingerprint:
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
    try:
        validated_cells = validate_stored_selection_integrity(
            datastore.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error(
            "Mapping reference cell selection no longer matches"
        ) from exc
    selected_count = validated_cells.selected_count
    if selected_count != selected_cell_count:
        raise _contract_error("Selected reference cell count does not match its input")

    feature_ids_array = as_zarr_array(group["feature_ids"], name="feature_ids")
    feature_means_array = _numeric_payload_array(group, "feature_means", ndim=1)
    feature_scales_array = _numeric_payload_array(group, "feature_scales", ndim=1)
    loadings_array = _numeric_payload_array(group, "loadings", ndim=2)
    n_features = int(loadings_array.shape[0])
    n_dims = int(loadings_array.shape[1])
    if (
        n_features < 1
        or n_dims < 1
        or feature_ids_array.ndim != 1
        or np.dtype(feature_ids_array.dtype).kind not in {"O", "S", "T", "U"}
        or feature_ids_array.shape != (n_features,)
        or not _stored_string_values_are_unique(feature_ids_array)
        or feature_means_array.shape != (n_features,)
        or feature_scales_array.shape != (n_features,)
        or not _numeric_values_are_valid(feature_means_array)
        or not _numeric_values_are_valid(feature_scales_array, positive=True)
        or not _numeric_values_are_valid(loadings_array)
    ):
        raise _contract_error("Mapping reference PCA model is invalid")
    if not _stored_feature_ids_match_selection(
        datastore.zw,
        assay_name,
        feature_selection,
        feature_ids_array,
    ):
        raise _contract_error(
            "Mapping reference feature IDs do not match the selected features"
        )
    reduction_group = artifact_group(datastore.zw, reduction)
    scaling_group = artifact_group(datastore.zw, feature_scaling)
    source_feature_means = as_zarr_array(scaling_group["mean"], name="mean")
    source_feature_scales = as_zarr_array(scaling_group["scale"], name="scale")
    source_loadings = as_zarr_array(reduction_group["loadings"], name="loadings")
    reduction_data = as_zarr_array(reduction_group["data"], name="data")
    if (
        reduction_data.ndim != 2
        or reduction_data.shape != (selected_cell_count, n_dims)
        or np.dtype(reduction_data.dtype) != np.dtype(np.float32)
    ):
        raise _contract_error(
            "Mapping reference PCA coordinates do not match selected cells"
        )
    try:
        validate_ann_index_payload(
            artifact_group(datastore.zw, ann_index),
            str(ann_metric),
            n_dims,
            selected_cell_count,
            require_metadata=True,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error("Mapping reference ANN index payload is invalid") from exc
    if not all(
        (
            _stored_array_matches_array(
                feature_means_array,
                source_feature_means,
            ),
            _stored_array_matches_array(
                feature_scales_array,
                source_feature_scales,
            ),
            _stored_array_matches_array(
                loadings_array,
                source_loadings,
            ),
        )
    ):
        raise _contract_error("Mapping reference PCA model changed from its inputs")

    symphony_arrays: dict[str, zarr.Array] = {}
    source_arrays: dict[str, zarr.Array] | None = None
    if method == "symphony":
        for name, ndim in (
            ("centroids", 2),
            ("raw_centroids", 2),
            ("corrected_centroids", 2),
            ("cluster_mass", 1),
            ("sigma", 1),
        ):
            symphony_arrays[name] = _numeric_payload_array(group, name, ndim=ndim)
        centroids_array = symphony_arrays["centroids"]
        n_clusters = int(centroids_array.shape[0])
        expected_matrix_shape = (n_clusters, n_dims)
        if (
            n_clusters < 1
            or centroids_array.shape != expected_matrix_shape
            or symphony_arrays["raw_centroids"].shape != expected_matrix_shape
            or symphony_arrays["corrected_centroids"].shape != expected_matrix_shape
            or symphony_arrays["cluster_mass"].shape != (n_clusters,)
            or symphony_arrays["sigma"].shape != (n_clusters,)
            or any(
                not _numeric_values_are_valid(symphony_arrays[name])
                for name in (
                    "centroids",
                    "raw_centroids",
                    "corrected_centroids",
                )
            )
            or not _numeric_values_are_valid(
                symphony_arrays["cluster_mass"],
                positive=True,
            )
            or not _numeric_values_are_valid(
                symphony_arrays["sigma"],
                positive=True,
            )
        ):
            raise _contract_error(
                "Mapping reference Symphony correction model is invalid"
            )
        assert batch_correction is not None
        correction_status = inspect_artifact(datastore.zw, batch_correction)
        correction_parameters = correction_status.parameters or {}
        correction_group = artifact_group(datastore.zw, batch_correction)
        correction_data = as_zarr_array(correction_group["data"], name="data")
        if (
            correction_data.ndim != 2
            or correction_data.shape != (selected_cell_count, n_dims)
            or np.dtype(correction_data.dtype) != np.dtype(np.float32)
        ):
            raise _contract_error(
                "Mapping reference Harmony coordinates do not match selected cells"
            )
        expected_batch_columns = list(correction_parameters.get("batch_columns", []))
        raw_harmony_parameters = correction_parameters.get("harmony_parameters", {})
        expected_harmony_parameters = (
            dict(raw_harmony_parameters)
            if isinstance(raw_harmony_parameters, Mapping)
            else None
        )
        expected_batch_levels = correction_group.attrs.get("batch_levels", [])
        if (
            expected_harmony_parameters is None
            or not isinstance(expected_batch_levels, list)
            or metadata.get("batch_columns") != expected_batch_columns
            or metadata.get("harmony_parameters") != expected_harmony_parameters
            or metadata.get("batch_levels") != expected_batch_levels
        ):
            raise _contract_error(
                "Mapping reference Symphony metadata does not match batch correction"
            )
        source_arrays = {
            name: as_zarr_array(correction_group[name], name=name)
            for name in (
                "centroids",
                "raw_centroids",
                "corrected_centroids",
                "cluster_mass",
                "sigma",
            )
        }
        if not all(
            _stored_array_matches_array(
                symphony_arrays[name],
                source_arrays[name],
                transpose=(name == "centroids"),
            )
            for name in symphony_arrays
        ):
            raise _contract_error(
                "Mapping reference Symphony model changed from its input"
            )
    try:
        validate_mapping_reference_sources(
            feature_means=source_feature_means,
            feature_scales=source_feature_scales,
            loadings=source_loadings,
            symphony_sources=source_arrays,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error(
            "Mapping reference upstream model arrays are invalid"
        ) from exc

    distance_quantiles_array = _numeric_payload_array(
        group,
        "reference_distance_quantiles",
        ndim=1,
    )
    distance_values_array = _numeric_payload_array(
        group,
        "reference_distance_values",
        ndim=1,
    )
    if (
        distance_quantiles_array.shape != distance_values_array.shape
        or int(distance_quantiles_array.shape[0]) < 1
        or not _numeric_values_are_valid(
            distance_quantiles_array,
            minimum=0.0,
            maximum=1.0,
            nondecreasing=True,
        )
        or not _numeric_values_are_valid(
            distance_values_array,
            minimum=0.0,
            nondecreasing=True,
        )
    ):
        raise _contract_error("Mapping reference distance summary is invalid")
    try:
        validate_distance_provenance(datastore.zw, neighbors)
        neighbor_payload = validate_neighbors_payload(datastore.zw, neighbors)
        if neighbor_payload.n_cells != selected_cell_count:
            raise ValueError("Neighbors do not match the selected reference cell count")
        expected_quantiles, expected_distances = _distance_quantile_summary(
            neighbor_payload.distances
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error(
            "Mapping reference neighbor distances are invalid"
        ) from exc
    if not (
        _stored_array_matches_values(
            distance_quantiles_array,
            expected_quantiles,
        )
        and _stored_array_matches_values(
            distance_values_array,
            expected_distances,
        )
    ):
        raise _contract_error(
            "Mapping reference distance summary changed from its neighbor input"
        )
    stored_payload_fingerprint = group.attrs.get("payload_fingerprint")
    if (
        not isinstance(stored_payload_fingerprint, str)
        or not stored_payload_fingerprint
        or stored_payload_fingerprint != _payload_fingerprint(group, method, metadata)
    ):
        raise _contract_error(
            "Mapping reference payload fingerprint does not match stored output"
        )

    if not build_result:
        return None

    feature_ids = _values(group, "feature_ids")
    model = ScaledPCAProjectionModel(
        feature_means=_values(group, "feature_means"),
        feature_scales=_values(group, "feature_scales"),
        loadings=_values(group, "loadings"),
    )
    symphony_state = (
        SymphonyCorrectionModel(
            centroids=_values(group, "centroids"),
            raw_centroids=_values(group, "raw_centroids"),
            corrected_centroids=_values(group, "corrected_centroids"),
            cluster_mass=_values(group, "cluster_mass"),
            sigma=_values(group, "sigma"),
        )
        if method == "symphony"
        else None
    )
    distance_quantiles = _values(group, "reference_distance_quantiles")
    distance_values = _values(group, "reference_distance_values")

    return MappingReference(
        datastore=datastore,
        ref=ref,
        assay_name=assay_name,
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


def validate_mapping_reference_binding(
    reference: MappingReference,
) -> MappingReference:
    """Reject an in-memory handle that differs from its stored artifact."""
    if not isinstance(reference, MappingReference):
        raise TypeError("reference must be a MappingReference")
    validate_artifact_mapping_reference(reference.datastore, reference.ref)
    status = inspect_artifact(reference.datastore.zw, reference.ref)
    method = (status.parameters or {}).get("method")
    group = artifact_group(reference.datastore.zw, reference.ref)
    raw_metadata = group.attrs.get("reference_metadata")
    assert isinstance(raw_metadata, Mapping)
    metadata = dict(raw_metadata)
    expected_scalars = {
        "ref": reference.ref,
        "assay_name": metadata["assay"],
        "reduction": _ref_from_input(status, "reduction"),
        "ann_index": _ref_from_input(status, "ann_index"),
        "neighbors": _ref_from_input(status, "neighbors"),
        "cell_selection": _ref_from_input(status, "cell_selection"),
        "feature_selection": _ref_from_input(status, "feature_selection"),
        "batch_correction": (
            _ref_from_input(status, "batch_correction")
            if method == "symphony"
            else None
        ),
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "selected_cell_count": metadata["selected_cell_count"],
    }
    matches = all(
        getattr(reference, name) == expected
        for name, expected in expected_scalars.items()
    )
    matches = (
        matches
        and isinstance(reference.model, ScaledPCAProjectionModel)
        and all(
            _stored_array_matches_values(group[name], getattr(reference.model, name))
            for name in ("feature_means", "feature_scales", "loadings")
        )
        and _stored_array_matches_values(
            group["feature_ids"],
            reference.feature_ids,
            strings=True,
            require_dtype=False,
        )
        and _stored_array_matches_values(
            group["reference_distance_quantiles"],
            reference.reference_distance_quantiles,
        )
        and _stored_array_matches_values(
            group["reference_distance_values"],
            reference.reference_distance_values,
        )
    )
    if method == "symphony":
        matches = (
            matches
            and isinstance(reference.symphony_state, SymphonyCorrectionModel)
            and all(
                _stored_array_matches_values(
                    group[name],
                    getattr(reference.symphony_state, name),
                )
                for name in (
                    "centroids",
                    "raw_centroids",
                    "corrected_centroids",
                    "cluster_mass",
                    "sigma",
                )
            )
        )
    else:
        matches = matches and reference.symphony_state is None
    try:
        matches = matches and canonical_bytes(reference.metadata) == canonical_bytes(
            metadata
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError(
            "MappingReference handle does not match its stored artifact. "
            "Reload it with get_mapping_reference(reference.ref)."
        )
    return reference


def mapping_reference_payload_matches_expected(
    group: zarr.Group,
    *,
    model: ScaledPCAProjectionModel,
    symphony_state: SymphonyCorrectionModel | None,
    feature_ids: np.ndarray,
    metadata: Mapping[str, Any],
    reference_distance_quantiles: np.ndarray,
    reference_distance_values: np.ndarray,
) -> bool:
    try:
        raw_metadata = group.attrs.get("reference_metadata")
        matches = isinstance(raw_metadata, Mapping) and canonical_bytes(
            raw_metadata
        ) == canonical_bytes(metadata)
        matches = (
            matches
            and all(
                _stored_array_matches_values(group[name], getattr(model, name))
                for name in ("feature_means", "feature_scales", "loadings")
            )
            and _stored_array_matches_values(
                group["feature_ids"],
                feature_ids,
                strings=True,
                require_dtype=False,
            )
            and _stored_array_matches_values(
                group["reference_distance_quantiles"],
                reference_distance_quantiles,
            )
            and _stored_array_matches_values(
                group["reference_distance_values"],
                reference_distance_values,
            )
        )
        if symphony_state is None:
            return matches and not any(name in group for name in _SYMPHONY_ARRAYS)
        return matches and all(
            _stored_array_matches_values(
                group[name],
                getattr(symphony_state, name),
            )
            for name in (
                "centroids",
                "raw_centroids",
                "corrected_centroids",
                "cluster_mass",
                "sigma",
            )
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def mapping_reference_payload_matches_sources(
    group: zarr.Group,
    *,
    feature_means: zarr.Array,
    feature_scales: zarr.Array,
    loadings: zarr.Array,
    symphony_sources: Mapping[str, zarr.Array] | None,
    feature_ids: np.ndarray,
    metadata: Mapping[str, Any],
    reference_distance_quantiles: np.ndarray,
    reference_distance_values: np.ndarray,
    expected_source_fingerprint: str,
) -> bool:
    """Compare a stored mapping reference with upstream arrays blockwise."""
    try:
        raw_metadata = group.attrs.get("reference_metadata")
        matches = isinstance(raw_metadata, Mapping) and canonical_bytes(
            raw_metadata
        ) == canonical_bytes(metadata)
        matches = (
            matches
            and mapping_reference_source_fingerprint(
                feature_means=feature_means,
                feature_scales=feature_scales,
                loadings=loadings,
                symphony_sources=symphony_sources,
            )
            == expected_source_fingerprint
        )
        matches = (
            matches
            and _stored_array_matches_array(
                as_zarr_array(group["feature_means"], name="feature_means"),
                feature_means,
            )
            and _stored_array_matches_array(
                as_zarr_array(group["feature_scales"], name="feature_scales"),
                feature_scales,
            )
            and _stored_array_matches_array(
                as_zarr_array(group["loadings"], name="loadings"),
                loadings,
            )
            and _stored_array_matches_values(
                group["feature_ids"],
                feature_ids,
                strings=True,
                require_dtype=False,
            )
            and _stored_array_matches_values(
                group["reference_distance_quantiles"],
                reference_distance_quantiles,
            )
            and _stored_array_matches_values(
                group["reference_distance_values"],
                reference_distance_values,
            )
        )
        if symphony_sources is None:
            return matches and not any(name in group for name in _SYMPHONY_ARRAYS)
        if set(symphony_sources) != set(_SYMPHONY_ARRAY_ORDER):
            return False
        return matches and all(
            _stored_array_matches_array(
                as_zarr_array(group[name], name=name),
                symphony_sources[name],
                transpose=name == "centroids",
            )
            for name in _SYMPHONY_ARRAY_ORDER
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def validate_mapping_reference_sources(
    *,
    feature_means: zarr.Array,
    feature_scales: zarr.Array,
    loadings: zarr.Array,
    symphony_sources: Mapping[str, zarr.Array] | None,
) -> tuple[int, int]:
    """Validate upstream model arrays blockwise before publication."""
    if loadings.ndim != 2 or int(loadings.shape[0]) < 1 or int(loadings.shape[1]) < 1:
        raise ValueError("Reference PCA loadings have incompatible dimensions")
    n_features = int(loadings.shape[0])
    n_dims = int(loadings.shape[1])
    pca_arrays = (feature_means, feature_scales, loadings)
    if (
        feature_means.ndim != 1
        or feature_scales.ndim != 1
        or feature_means.shape != (n_features,)
        or feature_scales.shape != (n_features,)
        or any(np.dtype(array.dtype) != np.dtype(np.float64) for array in pca_arrays)
        or not _numeric_values_are_valid(feature_means)
        or not _numeric_values_are_valid(feature_scales, positive=True)
        or not _numeric_values_are_valid(loadings)
    ):
        raise ValueError("Reference PCA model arrays are invalid")

    if symphony_sources is None:
        return n_features, n_dims
    if set(symphony_sources) != set(_SYMPHONY_ARRAY_ORDER):
        raise ValueError(
            "Mapping reference Symphony sources do not match the payload contract"
        )
    if any(
        np.dtype(array.dtype) != np.dtype(np.float64)
        for array in symphony_sources.values()
    ):
        raise ValueError("Harmony correction arrays must use float64 storage")
    centroids = symphony_sources["centroids"]
    if centroids.ndim != 2 or int(centroids.shape[0]) != n_dims:
        raise ValueError("Harmony correction dimensions do not match PCA loadings")
    n_clusters = int(centroids.shape[1])
    if (
        n_clusters < 1
        or symphony_sources["raw_centroids"].shape != (n_clusters, n_dims)
        or symphony_sources["corrected_centroids"].shape != (n_clusters, n_dims)
        or symphony_sources["cluster_mass"].shape != (n_clusters,)
        or symphony_sources["sigma"].shape != (n_clusters,)
        or any(
            not _numeric_values_are_valid(symphony_sources[name])
            for name in ("centroids", "raw_centroids", "corrected_centroids")
        )
        or not _numeric_values_are_valid(
            symphony_sources["cluster_mass"],
            positive=True,
        )
        or not _numeric_values_are_valid(
            symphony_sources["sigma"],
            positive=True,
        )
    ):
        raise ValueError("Harmony correction arrays are invalid")
    return n_features, n_dims


def mapping_reference_source_fingerprint(
    *,
    feature_means: zarr.Array,
    feature_scales: zarr.Array,
    loadings: zarr.Array,
    symphony_sources: Mapping[str, zarr.Array] | None,
) -> str:
    """Fingerprint the exact upstream model values in bounded row blocks."""
    sources: list[tuple[str, zarr.Array]] = [
        ("feature_means", feature_means),
        ("feature_scales", feature_scales),
        ("loadings", loadings),
    ]
    if symphony_sources is not None:
        if set(symphony_sources) != set(_SYMPHONY_ARRAY_ORDER):
            raise ValueError(
                "Mapping reference Symphony sources do not match the payload contract"
            )
        sources.extend((name, symphony_sources[name]) for name in _SYMPHONY_ARRAY_ORDER)
    builder = ValueFingerprintBuilder()
    for name, source in sources:
        shape = tuple(int(value) for value in source.shape)
        builder.begin_array(name, shape, source.dtype)
        block_rows = row_band(array_geometry(source), unit="chunk", fallback=1)
        for start in range(0, shape[0], block_rows):
            stop = min(start + block_rows, shape[0])
            builder.update_array_block(
                name,
                (start,) + (0,) * (source.ndim - 1),
                np.asarray(source[start:stop]),
            )
        builder.end_array(name)
    return builder.hexdigest()


def _numeric_payload_array(
    group: zarr.Group,
    name: str,
    *,
    ndim: int,
) -> zarr.Array:
    array = as_zarr_array(group[name], name=name)
    if array.ndim != ndim or np.dtype(array.dtype) != np.dtype(np.float64):
        raise _contract_error(
            f"Mapping reference array {name!r} has an invalid dtype or shape"
        )
    return array


def _numeric_values_are_valid(
    array: zarr.Array,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    nondecreasing: bool = False,
) -> bool:
    block_rows = row_band(array_geometry(array), unit="chunk", fallback=1)
    previous: float | None = None
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        values = np.asarray(array[start:stop], dtype=np.float64)
        if not np.isfinite(values).all():
            return False
        if positive and np.any(values <= 0.0):
            return False
        if minimum is not None and np.any(values < minimum):
            return False
        if maximum is not None and np.any(values > maximum):
            return False
        if nondecreasing and values.size:
            flat = values.reshape(-1)
            if (previous is not None and float(flat[0]) < previous) or np.any(
                np.diff(flat) < 0.0
            ):
                return False
            previous = float(flat[-1])
    return True


def _stored_array_matches_values(
    stored: Any,
    values: Any,
    *,
    strings: bool = False,
    require_dtype: bool = True,
) -> bool:
    array = as_zarr_array(stored, name="stored mapping-reference array")
    expected = np.asarray(values)
    if array.shape != expected.shape or (
        require_dtype and np.dtype(array.dtype) != expected.dtype
    ):
        return False
    block_rows = row_band(array_geometry(array), unit="chunk", fallback=1)
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        received = np.asarray(array[start:stop])
        wanted = expected[start:stop]
        if strings:
            received = received.astype(str)
            wanted = wanted.astype(str)
        if not np.array_equal(received, wanted):
            return False
    return True


def _stored_array_matches_array(
    stored: zarr.Array,
    source: zarr.Array,
    *,
    transpose: bool = False,
) -> bool:
    expected_shape = (
        (int(source.shape[1]), int(source.shape[0]))
        if transpose and source.ndim == 2
        else tuple(int(value) for value in source.shape)
    )
    if tuple(int(value) for value in stored.shape) != expected_shape:
        return False
    block_rows = row_band(array_geometry(stored), unit="chunk", fallback=1)
    for start in range(0, int(stored.shape[0]), block_rows):
        stop = min(start + block_rows, int(stored.shape[0]))
        source_values = (
            np.asarray(source[:, start:stop]).T
            if transpose
            else np.asarray(source[start:stop])
        )
        if not np.array_equal(
            np.asarray(stored[start:stop]),
            np.asarray(source_values, dtype=stored.dtype),
        ):
            return False
    return True


def _stored_feature_ids_match_selection(
    root: zarr.Group,
    assay: str,
    feature_selection: ArtifactRef,
    stored_feature_ids: zarr.Array,
) -> bool:
    resolve_feature_selection(root, assay, feature_selection)
    feature_data = as_zarr_group(
        root[f"{assay}/featureData"],
        name=f"{assay}/featureData",
    )
    live_ids = as_zarr_array(feature_data["ids"], name="ids")
    selection = as_zarr_array(
        artifact_group(root, feature_selection)["values"],
        name="values",
    )
    if live_ids.shape != selection.shape:
        return False
    block_rows = min(
        row_band(array_geometry(live_ids), unit="chunk", fallback=1),
        row_band(array_geometry(selection), unit="chunk", fallback=1),
    )
    output_offset = 0
    for start in range(0, int(selection.shape[0]), block_rows):
        stop = min(start + block_rows, int(selection.shape[0]))
        mask = np.asarray(selection[start:stop], dtype=bool)
        selected_ids = np.asarray(live_ids[start:stop])[mask].astype(str)
        output_stop = output_offset + len(selected_ids)
        if output_stop > int(stored_feature_ids.shape[0]) or not np.array_equal(
            np.asarray(stored_feature_ids[output_offset:output_stop]).astype(str),
            selected_ids,
        ):
            return False
        output_offset = output_stop
    return output_offset == int(stored_feature_ids.shape[0])


def _stored_string_values_are_unique(array: zarr.Array) -> bool:
    block_rows = row_band(array_geometry(array), unit="chunk", fallback=1)
    seen: set[str] = set()
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        raw = np.asarray(array[start:stop])
        if raw.dtype.kind == "O" and any(
            not isinstance(value, str | bytes | np.str_ | np.bytes_)
            for value in raw.reshape(-1)
        ):
            return False
        identifiers = raw.astype(str).reshape(-1)
        for identifier in identifiers:
            value = str(identifier)
            if value in seen:
                return False
            seen.add(value)
    return True


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
    if any(set(as_zarr_array(group[name], name=name).attrs) for name in expected):
        raise _contract_error(
            "Mapping reference array attributes do not match the current contract"
        )
    expected_attributes = {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "scarf_version",
        "complete",
        "reference_metadata",
        "payload_fingerprint",
    }
    if "reference_metadata" not in group.attrs:
        raise _contract_error("Mapping reference metadata is missing")
    if "payload_fingerprint" not in group.attrs:
        raise _contract_error("Mapping reference payload fingerprint is missing")
    if set(group.attrs) != expected_attributes:
        raise _contract_error(
            "Mapping reference attributes do not match the current contract"
        )


def _metadata_string(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise _contract_error(f"Mapping reference metadata {name!r} is missing")
    return value


def _values(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(as_zarr_array(group[name], name=name)[:])


def _payload_fingerprint(
    group: zarr.Group,
    method: str,
    metadata: Mapping[str, Any],
) -> str:
    names = (
        _COMMON_ARRAY_ORDER + _SYMPHONY_ARRAY_ORDER
        if method == "symphony"
        else _COMMON_ARRAY_ORDER
    )
    builder = ValueFingerprintBuilder()
    builder.update_bytes(
        "arrays",
        fingerprint_stored_arrays(group, names).encode(),
    )
    builder.update_bytes("reference_metadata", canonical_bytes(metadata))
    return builder.hexdigest()


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


def _write_array_from_source(
    group: zarr.Group,
    name: str,
    source: zarr.Array,
    *,
    transpose: bool = False,
) -> None:
    if source.ndim not in {1, 2} or (transpose and source.ndim != 2):
        raise ValueError(f"Mapping reference array {name!r} must have one or two axes")
    shape = (
        (int(source.shape[1]), int(source.shape[0]))
        if transpose
        else tuple(int(value) for value in source.shape)
    )
    chunks = (
        (min(max(shape[0], 1), 10_000),)
        if len(shape) == 1
        else (
            min(max(shape[0], 1), 1_000),
            max(shape[1], 1),
        )
    )
    target = create_zarr_dataset(group, name, chunks, "f8", shape)
    block_rows = int(chunks[0])
    for start in range(0, shape[0], block_rows):
        stop = min(start + block_rows, shape[0])
        values = (
            np.asarray(source[:, start:stop]).T
            if transpose
            else np.asarray(source[start:stop])
        )
        target[start:stop] = np.asarray(values, dtype=np.float64)


def _contract_error(detail: str) -> ValueError:
    return ValueError(f"{detail}. {MAPPING_REFERENCE_REBUILD_MESSAGE}")
