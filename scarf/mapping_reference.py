"""Persistent handles for immutable Symphony-style mapping references."""

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, cast
import warnings

import numpy as np
import zarr

from ._types import as_zarr_array, as_zarr_group
from .symphony import SYMPHONY_STYLE_VARIANT, SymphonyReferenceModel
from .writers import create_zarr_dataset, create_zarr_obj_array

if TYPE_CHECKING:
    import pandas as pd

    from .assay import Assay


MAPPING_REFERENCE_SCHEMA_VERSION = 2
MAPPING_REFERENCE_GROUP = "mappingReference"
MAPPING_REFERENCES_GROUP = "mappingReferences"
LATEST_MAPPING_REFERENCE_ATTRIBUTE = "latestMappingReference"


@dataclass(frozen=True)
class MappingResult:
    projection_path: str
    n_cells: int
    correction_method: str
    diagnostics: dict[str, float | str]
    indices: np.ndarray | None = None
    distances: np.ndarray | None = None
    uncorrected_latent: np.ndarray | None = None
    corrected_latent: np.ndarray | None = None
    uninformative: np.ndarray | None = None


@dataclass(frozen=True)
class MappingReference:
    """An immutable Symphony-style reference loaded from a SCARF Zarr store."""

    datastore: Any
    assay_name: str
    cell_key: str
    feature_key: str
    reduction_path: str
    ann_path: str
    artifact_path: str
    model: SymphonyReferenceModel
    feature_ids: np.ndarray
    metadata: dict[str, Any]
    reference_distance_quantiles: np.ndarray | None = None
    reference_distance_values: np.ndarray | None = None

    def map_query(
        self,
        target_assay: "Assay",
        target_name: str,
        target_feat_key: str,
        target_cell_key: str = "I",
        save_k: int = 3,
        query_batches: "pd.DataFrame | None" = None,
        correction_method: str = "symphony",
        missing_feature_policy: str = "reference_mean",
        result_store: zarr.Group | None = None,
    ) -> MappingResult:
        """Map a query into the fixed reference coordinate system.

        A writable reference stores results under its projection group. A
        read-only reference returns arrays in memory unless ``result_store`` is
        supplied.
        """
        return cast(
            MappingResult,
            self.datastore._map_with_mapping_reference(
                self,
                target_assay=target_assay,
                target_name=target_name,
                target_feat_key=target_feat_key,
                target_cell_key=target_cell_key,
                save_k=save_k,
                query_batches=query_batches,
                correction_method=correction_method,
                missing_feature_policy=missing_feature_policy,
                result_store=result_store,
            ),
        )


def persist_mapping_reference(
    reduction_group: zarr.Group,
    model: SymphonyReferenceModel,
    feature_ids: np.ndarray,
    metadata: dict[str, Any],
    reference_distance_quantiles: np.ndarray | None = None,
    reference_distance_values: np.ndarray | None = None,
) -> str:
    """Write a complete immutable reference artifact into a reduction group."""
    model = SymphonyReferenceModel(
        feature_means=np.asarray(model.feature_means, dtype=np.float64),
        feature_scales=np.asarray(model.feature_scales, dtype=np.float64),
        loadings=np.asarray(model.loadings, dtype=np.float64),
        centroids=np.asarray(model.centroids, dtype=np.float64),
        raw_centroids=np.asarray(model.raw_centroids, dtype=np.float64),
        corrected_centroids=np.asarray(model.corrected_centroids, dtype=np.float64),
        cluster_mass=np.asarray(model.cluster_mass, dtype=np.float64),
        sigma=np.asarray(model.sigma, dtype=np.float64),
        correction_ridge=float(model.correction_ridge),
    )
    if reference_distance_quantiles is not None:
        reference_distance_quantiles = np.asarray(
            reference_distance_quantiles, dtype=np.float64
        )
    if reference_distance_values is not None:
        reference_distance_values = np.asarray(
            reference_distance_values, dtype=np.float64
        )
    metadata = {**metadata, "algorithmVariant": SYMPHONY_STYLE_VARIANT}
    artifact_hash = mapping_reference_hash(
        model,
        feature_ids,
        metadata,
        reference_distance_quantiles,
        reference_distance_values,
    )
    if MAPPING_REFERENCES_GROUP not in reduction_group:
        reduction_group.create_group(MAPPING_REFERENCES_GROUP)
    references = as_zarr_group(
        reduction_group[MAPPING_REFERENCES_GROUP], name=MAPPING_REFERENCES_GROUP
    )
    if artifact_hash in references:
        existing = as_zarr_group(references[artifact_hash], name=artifact_hash)
        if bool(existing.attrs.get("complete", False)):
            _validate_mapping_reference_hash(existing, artifact_hash)
            reduction_group.attrs[LATEST_MAPPING_REFERENCE_ATTRIBUTE] = artifact_hash
            return f"{MAPPING_REFERENCES_GROUP}/{artifact_hash}"
        del references[artifact_hash]
    group = references.create_group(artifact_hash)
    group.attrs["schemaVersion"] = MAPPING_REFERENCE_SCHEMA_VERSION
    group.attrs["complete"] = False
    group.attrs["artifactHash"] = artifact_hash
    for key, value in metadata.items():
        group.attrs[key] = value
    try:
        create_zarr_obj_array(group, "featureIds", np.asarray(feature_ids))
        _write_array(group, "featureMeans", model.feature_means)
        _write_array(group, "featureScales", model.feature_scales)
        _write_array(group, "loadings", model.loadings)
        _write_array(group, "centroids", model.centroids)
        _write_array(group, "rawCentroids", model.raw_centroids)
        _write_array(group, "correctedCentroids", model.corrected_centroids)
        _write_array(group, "clusterMass", model.cluster_mass)
        _write_array(group, "sigma", model.sigma)
        if reference_distance_quantiles is not None:
            _write_array(
                group,
                "referenceDistanceQuantiles",
                reference_distance_quantiles,
            )
        if reference_distance_values is not None:
            _write_array(
                group,
                "referenceDistanceValues",
                reference_distance_values,
            )
        group.attrs["correctionRidge"] = float(model.correction_ridge)
        group.attrs["complete"] = True
        reduction_group.attrs[LATEST_MAPPING_REFERENCE_ATTRIBUTE] = artifact_hash
    except Exception:
        group.attrs["complete"] = False
        raise
    return f"{MAPPING_REFERENCES_GROUP}/{artifact_hash}"


def load_mapping_reference(
    datastore: Any,
    assay_name: str,
    cell_key: str,
    feature_key: str,
    reduction_path: str,
    ann_path: str,
) -> MappingReference:
    """Load a persisted reference after the caller has validated its provenance."""
    reduction_group = as_zarr_group(datastore.zw[reduction_path], name=reduction_path)
    group, relative_path, is_legacy = resolve_mapping_reference_group(reduction_group)
    group_path = f"{reduction_path}/{relative_path}"
    group = as_zarr_group(datastore.zw[group_path], name=group_path)
    if not bool(group.attrs.get("complete", False)):
        raise ValueError(
            "Mapping reference is incomplete. Rebuild the harmonized reference."
        )
    schema_version = group.attrs.get("schemaVersion")
    if is_legacy:
        if schema_version != 1:
            raise ValueError(
                "Legacy mapping reference schema is incompatible. Rebuild the reference."
            )
        warnings.warn(
            "This mapping reference uses the legacy overwrite-in-place layout. "
            "Rebuild it to create a content-addressed artifact.",
            DeprecationWarning,
            stacklevel=3,
        )
    elif schema_version != MAPPING_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            "Mapping reference schema is incompatible. Rebuild the harmonized reference."
        )
    if not is_legacy:
        validate_mapping_reference_artifact(group)
    model = SymphonyReferenceModel(
        feature_means=np.asarray(
            as_zarr_array(group["featureMeans"], name="featureMeans")[:]
        ),
        feature_scales=np.asarray(
            as_zarr_array(group["featureScales"], name="featureScales")[:]
        ),
        loadings=np.asarray(as_zarr_array(group["loadings"], name="loadings")[:]),
        centroids=np.asarray(as_zarr_array(group["centroids"], name="centroids")[:]),
        raw_centroids=np.asarray(
            as_zarr_array(group["rawCentroids"], name="rawCentroids")[:]
        ),
        corrected_centroids=np.asarray(
            as_zarr_array(group["correctedCentroids"], name="correctedCentroids")[:]
        ),
        cluster_mass=np.asarray(
            as_zarr_array(group["clusterMass"], name="clusterMass")[:]
        ),
        sigma=np.asarray(as_zarr_array(group["sigma"], name="sigma")[:]),
        correction_ridge=_number_attribute(group, "correctionRidge"),
    )
    return MappingReference(
        datastore=datastore,
        assay_name=assay_name,
        cell_key=cell_key,
        feature_key=feature_key,
        reduction_path=reduction_path,
        ann_path=ann_path,
        artifact_path=group_path,
        model=model,
        feature_ids=np.asarray(
            as_zarr_array(group["featureIds"], name="featureIds")[:]
        ),
        metadata=dict(group.attrs),
        reference_distance_quantiles=(
            np.asarray(
                as_zarr_array(
                    group["referenceDistanceQuantiles"],
                    name="referenceDistanceQuantiles",
                )[:]
            )
            if "referenceDistanceQuantiles" in group
            else None
        ),
        reference_distance_values=(
            np.asarray(
                as_zarr_array(
                    group["referenceDistanceValues"],
                    name="referenceDistanceValues",
                )[:]
            )
            if "referenceDistanceValues" in group
            else None
        ),
    )


def validate_mapping_reference_artifact(group: zarr.Group) -> None:
    """Validate a content-addressed mapping-reference artifact."""
    if not bool(group.attrs.get("complete", False)):
        raise ValueError(
            "Mapping reference is incomplete. Rebuild the harmonized reference."
        )
    if group.attrs.get("schemaVersion") != MAPPING_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            "Mapping reference schema is incompatible. Rebuild the harmonized reference."
        )
    artifact_hash = group.attrs.get("artifactHash")
    if not isinstance(artifact_hash, str):
        raise ValueError("Mapping reference is missing its artifact hash.")
    group_name = str(getattr(group, "path", "")).rstrip("/").rsplit("/", 1)[-1]
    if group_name and group_name != artifact_hash:
        raise ValueError("Mapping reference path does not match its artifact hash.")
    _validate_mapping_reference_hash(group, artifact_hash)


def resolve_mapping_reference_group(
    reduction_group: zarr.Group,
) -> tuple[zarr.Group, str, bool]:
    artifact_hash = reduction_group.attrs.get(LATEST_MAPPING_REFERENCE_ATTRIBUTE)
    if artifact_hash is not None and MAPPING_REFERENCES_GROUP in reduction_group:
        relative_path = f"{MAPPING_REFERENCES_GROUP}/{artifact_hash}"
        if relative_path in reduction_group:
            return (
                as_zarr_group(reduction_group[relative_path], name=relative_path),
                relative_path,
                False,
            )
    if MAPPING_REFERENCE_GROUP in reduction_group:
        return (
            as_zarr_group(
                reduction_group[MAPPING_REFERENCE_GROUP],
                name=MAPPING_REFERENCE_GROUP,
            ),
            MAPPING_REFERENCE_GROUP,
            True,
        )
    raise KeyError("No mapping-reference artifact exists in this reduction")


def mapping_reference_hash(
    model: SymphonyReferenceModel,
    feature_ids: np.ndarray,
    metadata: dict[str, Any],
    reference_distance_quantiles: np.ndarray | None = None,
    reference_distance_values: np.ndarray | None = None,
) -> str:
    from .mapping_utils import array_hash

    digest = hashlib.sha256()
    for values in (
        feature_ids,
        model.feature_means,
        model.feature_scales,
        model.loadings,
        model.centroids,
        model.raw_centroids,
        model.corrected_centroids,
        model.cluster_mass,
        model.sigma,
    ):
        digest.update(array_hash(np.asarray(values)).encode())
    for optional_values in (
        reference_distance_quantiles,
        reference_distance_values,
    ):
        if optional_values is not None:
            digest.update(array_hash(np.asarray(optional_values)).encode())
    digest.update(repr(float(model.correction_ridge)).encode())
    digest.update(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    )
    return digest.hexdigest()


def _validate_mapping_reference_hash(group: zarr.Group, expected_hash: str) -> None:
    model = SymphonyReferenceModel(
        feature_means=np.asarray(
            as_zarr_array(group["featureMeans"], name="featureMeans")[:]
        ),
        feature_scales=np.asarray(
            as_zarr_array(group["featureScales"], name="featureScales")[:]
        ),
        loadings=np.asarray(as_zarr_array(group["loadings"], name="loadings")[:]),
        centroids=np.asarray(as_zarr_array(group["centroids"], name="centroids")[:]),
        raw_centroids=np.asarray(
            as_zarr_array(group["rawCentroids"], name="rawCentroids")[:]
        ),
        corrected_centroids=np.asarray(
            as_zarr_array(group["correctedCentroids"], name="correctedCentroids")[:]
        ),
        cluster_mass=np.asarray(
            as_zarr_array(group["clusterMass"], name="clusterMass")[:]
        ),
        sigma=np.asarray(as_zarr_array(group["sigma"], name="sigma")[:]),
        correction_ridge=_number_attribute(group, "correctionRidge"),
    )
    feature_ids = np.asarray(as_zarr_array(group["featureIds"], name="featureIds")[:])
    distance_quantiles = (
        np.asarray(
            as_zarr_array(
                group["referenceDistanceQuantiles"],
                name="referenceDistanceQuantiles",
            )[:]
        )
        if "referenceDistanceQuantiles" in group
        else None
    )
    distance_values = (
        np.asarray(
            as_zarr_array(
                group["referenceDistanceValues"],
                name="referenceDistanceValues",
            )[:]
        )
        if "referenceDistanceValues" in group
        else None
    )
    metadata = {
        key: value
        for key, value in dict(group.attrs).items()
        if key
        not in {
            "artifactHash",
            "complete",
            "correctionRidge",
            "schemaVersion",
        }
    }
    actual_hash = mapping_reference_hash(
        model,
        feature_ids,
        metadata,
        distance_quantiles,
        distance_values,
    )
    if actual_hash != expected_hash:
        raise ValueError(
            "Mapping reference content does not match its artifact hash. "
            "Rebuild the reference."
        )


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    chunks = (
        (min(max(values.shape[0], 1), 10_000),)
        if values.ndim == 1
        else (min(max(values.shape[0], 1), 1_000), values.shape[1])
    )
    array = create_zarr_dataset(group, name, chunks, "f8", values.shape)
    array[...] = values


def _number_attribute(group: zarr.Group, name: str) -> float:
    value = group.attrs[name]
    if not isinstance(value, int | float):
        raise ValueError(f"Mapping reference attribute {name!r} must be numeric")
    return float(value)
