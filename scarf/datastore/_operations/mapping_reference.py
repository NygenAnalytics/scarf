from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...assay import RNAassay
from ...graph.distances import validate_distance_provenance
from ...mapping.artifact import (
    _selected_feature_ids,
    load_artifact_mapping_reference,
    write_artifact_mapping_reference,
)
from ...mapping.confidence import _distance_quantile_summary
from ...mapping.models import (
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
)
from ...mapping.reference import MappingReference
from ...storage.artifact_writer import (
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
)
from ...storage.types import as_zarr_array
from ...storage.selections import validate_stored_selection_integrity

if TYPE_CHECKING:
    from .graph import _GraphOperationsMixin as _MappingReferenceOperationsBase
else:
    _MappingReferenceOperationsBase = object


class _MappingReferenceOperationsMixin(_MappingReferenceOperationsBase):
    def get_mapping_reference(
        self,
        reference: ArtifactRef,
    ) -> MappingReference:
        """Load one mapping reference from an explicit artifact."""
        if not isinstance(reference, ArtifactRef):
            raise TypeError("reference must be an ArtifactRef")
        if (
            reference.scope != "assay"
            or reference.assay is None
            or reference.kind != "mapping_reference"
        ):
            raise ValueError(
                "reference must identify an assay-scoped mapping_reference artifact"
            )
        return load_artifact_mapping_reference(self, reference)

    def build_mapping_reference(
        self,
        neighbors: ArtifactRef,
        *,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Package a scaled-PCA neighbor chain as an immutable artifact."""
        if self.zarr_mode != "r+":
            raise ValueError("Building a mapping reference requires a read-write store")
        if not isinstance(neighbors, ArtifactRef):
            raise TypeError("neighbors must be an ArtifactRef")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        if (
            neighbors.scope != "assay"
            or neighbors.assay is None
            or neighbors.kind != "neighbors"
        ):
            raise ValueError(
                "neighbors must identify an assay-scoped neighbors artifact"
            )
        assay_name = neighbors.assay
        assay = self._get_assay(assay_name)
        if not isinstance(assay, RNAassay):
            raise TypeError("Mapping references currently support RNA assays only")
        neighbors_status = self._require_complete_artifact(
            neighbors,
            "neighbors",
            assay=assay_name,
        )
        if neighbors_status.operation != "query_neighbors":
            raise ValueError("Mapping references require query_neighbors artifacts")
        ann_index = self._artifact_input_ref(
            neighbors,
            "ann_index",
            "ann_index",
        )
        ann_status = self._require_complete_artifact(
            ann_index,
            "ann_index",
            assay=assay_name,
        )
        if ann_status.operation != "build_ann_index":
            raise ValueError("Mapping references require build_ann_index artifacts")
        coordinates = _artifact_input(neighbors_status.inputs, "coordinates")
        if (
            coordinates.scope != "assay"
            or coordinates.assay != assay_name
            or coordinates.kind not in {"reduction", "batch_correction"}
        ):
            raise ValueError(
                "Neighbor coordinates must be a PCA reduction or batch correction"
            )
        self._require_complete_artifact(
            coordinates,
            coordinates.kind,
            assay=assay_name,
        )
        if (
            self._artifact_input_ref(
                ann_index,
                "coordinates",
                coordinates.kind,
            )
            != coordinates
        ):
            raise ValueError("Neighbors and ANN index use different coordinates")

        batch_correction = None
        if coordinates.kind == "batch_correction":
            batch_correction = coordinates
            correction_status = self._require_complete_artifact(
                batch_correction,
                "batch_correction",
                assay=assay_name,
            )
            if correction_status.operation != "run_harmony":
                raise ValueError(
                    "Symphony mapping references require run_harmony correction"
                )
            reduction = self._artifact_input_ref(
                batch_correction,
                "reduction",
                "reduction",
            )
            method = "symphony"
        else:
            correction_status = None
            reduction = coordinates
            method = "pca"

        reduction_status = self._require_complete_artifact(
            reduction,
            "reduction",
            assay=assay_name,
        )
        if reduction_status.operation != "run_pca":
            raise ValueError("Mapping references require a run_pca reduction")
        if (reduction_status.parameters or {}).get("feat_scaling") is not True:
            raise ValueError("Mapping references require PCA with feature scaling")
        normalized = self._artifact_input_ref(
            reduction,
            "normalized",
            "normalized",
        )
        feature_scaling = self._artifact_input_ref(
            reduction,
            "feature_scaling",
            "feature_scaling",
        )
        normalized_status = self._require_complete_artifact(
            normalized,
            "normalized",
            assay=assay_name,
        )
        scaling_status = self._require_complete_artifact(
            feature_scaling,
            "feature_scaling",
            assay=assay_name,
        )
        if normalized_status.operation != "run_normalization":
            raise ValueError("Mapping references require a run_normalization artifact")
        normalized_inputs = normalized_status.inputs or {}
        lineage_dataset_fingerprint = normalized_inputs.get("dataset_fingerprint")
        if (
            not isinstance(lineage_dataset_fingerprint, str)
            or not lineage_dataset_fingerprint
        ):
            raise ValueError("Normalized artifact is missing its dataset fingerprint")
        if (
            scaling_status.operation != "calculate_feature_scaling"
            or (scaling_status.parameters or {}).get("enabled") is not True
            or self._artifact_input_ref(
                feature_scaling,
                "normalized",
                "normalized",
            )
            != normalized
        ):
            raise ValueError(
                "Mapping references require enabled scaling for the same normalized data"
            )
        cell_selection = self._artifact_input_ref(
            normalized,
            "cell_selection",
            "cell_selection",
        )
        feature_selection = self._artifact_input_ref(
            normalized,
            "feature_selection",
            "feature_selection",
        )

        validated_cells = validate_stored_selection_integrity(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

        ann_metric = (ann_status.parameters or {}).get("ann_metric")
        if ann_metric not in {"l2", "cosine"}:
            raise ValueError(
                "Mapping references support only l2 and cosine ANN metrics"
            )
        if (neighbors_status.parameters or {}).get("distance_metric") != ann_metric:
            raise ValueError("Neighbor and ANN distance metrics do not match")
        normalization_parameters = normalized_status.parameters or {}
        size_factor = normalization_parameters.get("size_factor")
        if (
            isinstance(size_factor, bool)
            or not isinstance(size_factor, int | float)
            or not np.isfinite(size_factor)
            or float(size_factor) <= 0
        ):
            raise ValueError(
                "Mapping reference normalization size_factor must be finite and positive"
            )

        reduction_group = artifact_group(self.zw, reduction)
        scaling_group = artifact_group(self.zw, feature_scaling)
        neighbors_group = artifact_group(self.zw, neighbors)
        loadings = np.asarray(
            as_zarr_array(reduction_group["loadings"], name="loadings")[:]
        )
        model = ScaledPCAProjectionModel(
            feature_means=np.asarray(
                as_zarr_array(scaling_group["mean"], name="mean")[:]
            ),
            feature_scales=np.asarray(
                as_zarr_array(scaling_group["scale"], name="scale")[:]
            ),
            loadings=loadings,
        )
        feature_ids = _selected_feature_ids(
            self.zw,
            assay_name,
            feature_selection,
        )
        if len(feature_ids) != model.n_features:
            raise ValueError("Selected reference features do not match PCA loadings")
        selected_cell_count = validated_cells.selected_count
        if selected_cell_count < 1:
            raise ValueError("Mapping references require at least one selected cell")
        reduction_data = as_zarr_array(reduction_group["data"], name="data")
        distances = as_zarr_array(neighbors_group["distances"], name="distances")
        if (
            int(reduction_data.shape[0]) != selected_cell_count
            or int(distances.shape[0]) != selected_cell_count
        ):
            raise ValueError(
                "PCA and neighbor rows must match the selected reference cells"
            )

        symphony_state = None
        if correction_status is not None and batch_correction is not None:
            correction_group = artifact_group(self.zw, batch_correction)
            symphony_state = SymphonyCorrectionModel(
                centroids=np.asarray(
                    as_zarr_array(
                        correction_group["centroids"],
                        name="centroids",
                    )[:]
                ).T,
                raw_centroids=np.asarray(
                    as_zarr_array(
                        correction_group["raw_centroids"],
                        name="raw_centroids",
                    )[:]
                ),
                corrected_centroids=np.asarray(
                    as_zarr_array(
                        correction_group["corrected_centroids"],
                        name="corrected_centroids",
                    )[:]
                ),
                cluster_mass=np.asarray(
                    as_zarr_array(
                        correction_group["cluster_mass"],
                        name="cluster_mass",
                    )[:]
                ),
                sigma=np.asarray(
                    as_zarr_array(correction_group["sigma"], name="sigma")[:]
                ),
            )
            if symphony_state.n_dims != model.n_dims:
                raise ValueError(
                    "Harmony correction dimensions do not match PCA loadings"
                )

        validate_distance_provenance(self.zw, neighbors)
        distance_quantiles, distance_values = _distance_quantile_summary(distances)
        stored_dataset_fingerprint = assay.attrs.get("dataset_fingerprint")
        live_dataset_fingerprint = (
            stored_dataset_fingerprint
            if isinstance(stored_dataset_fingerprint, str)
            and stored_dataset_fingerprint
            else self._calculate_dataset_fingerprint(assay_name)
        )
        if live_dataset_fingerprint != lineage_dataset_fingerprint:
            raise ValueError(
                "Normalized artifact does not match the current reference dataset"
            )
        dataset_fingerprint = lineage_dataset_fingerprint
        metadata: dict[str, Any] = {
            "method": method,
            "assay": assay_name,
            "selected_cell_count": selected_cell_count,
            "ann_metric": ann_metric,
            "normalization_parameters": dict(normalization_parameters),
            "dataset_fingerprint": dataset_fingerprint,
        }
        if correction_status is not None and batch_correction is not None:
            correction_parameters = correction_status.parameters or {}
            correction_group = artifact_group(self.zw, batch_correction)
            batch_levels = correction_group.attrs.get("batch_levels", [])
            if not isinstance(batch_levels, list):
                raise ValueError("Harmony batch levels must be a list")
            metadata.update(
                {
                    "batch_columns": list(
                        correction_parameters.get("batch_columns", [])
                    ),
                    "harmony_parameters": dict(
                        correction_parameters.get("harmony_parameters", {})
                    ),
                    "batch_levels": batch_levels,
                }
            )

        inputs: dict[str, ArtifactRef] = {
            "reduction": reduction,
            "ann_index": ann_index,
            "neighbors": neighbors,
            "cell_selection": cell_selection,
            "feature_selection": feature_selection,
        }
        if batch_correction is not None:
            inputs["batch_correction"] = batch_correction
        required_arrays: tuple[str, ...] = (
            "feature_ids",
            "feature_means",
            "feature_scales",
            "loadings",
            "reference_distance_quantiles",
            "reference_distance_values",
        )
        required_attributes: tuple[str | AttributeRequirement, ...] = (
            AttributeRequirement(
                "reference_metadata",
                expected_types=(dict,),
            ),
        )
        if symphony_state is not None:
            required_arrays += (
                "centroids",
                "raw_centroids",
                "corrected_centroids",
                "cluster_mass",
                "sigma",
            )

        def valid_reference(candidate: ArtifactRef, _group: Any) -> bool:
            if not bool(_group.attrs.get("complete", False)):
                return True
            try:
                load_artifact_mapping_reference(self, candidate)
            except (KeyError, TypeError, ValueError):
                return False
            return True

        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=assay_name,
            kind="mapping_reference",
            operation="build_mapping_reference",
            parameters={"method": method},
            inputs=inputs,
            execution_options=dict(getattr(self, "_artifactExecutionContext", {})),
            invalidate_cache=invalidate_cache,
            required_arrays=required_arrays,
            required_attributes=required_attributes,
            reuse_validator=valid_reference,
        )
        if not planned.reused:
            group = start_artifact(self.zw, planned)
            write_artifact_mapping_reference(
                group,
                model,
                symphony_state,
                feature_ids,
                metadata,
                distance_quantiles,
                distance_values,
            )
            finish_artifact(group, planned)
        return planned.ref


def _artifact_input(
    inputs: Mapping[str, Any] | None,
    name: str,
) -> ArtifactRef:
    raw_ref = (inputs or {}).get(name)
    if not isinstance(raw_ref, Mapping):
        raise ValueError(f"Artifact has no {name!r} input")
    return ArtifactRef.from_dict(raw_ref)
