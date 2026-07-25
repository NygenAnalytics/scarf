from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...assay import Assay, RNAassay
from ...graph.encoded_paths import (
    lookup_latest_neighbor_index_group_path,
    lookup_latest_reduction_group_path,
    make_normalized_group_path,
    parse_neighbor_index_group_path,
)
from ...graph.state import read_assay_state
from ...mapping.artifact import (
    load_artifact_mapping_reference,
    load_mapping_reference,
    resolve_mapping_reference_group,
    write_artifact_mapping_reference,
)
from ...mapping.confidence import _distance_quantile_summary
from ...mapping.hashing import array_hash, array_store_hash
from ...mapping.models import SymphonyReferenceModel
from ...mapping.reference import MappingReference
from ...mapping.symphony import weighted_centroids
from ...storage.artifact_writer import (
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
)
from ...storage.types import as_zarr_array, as_zarr_group

if TYPE_CHECKING:
    from ..base_datastore import BaseDataStore as _MappingReferenceOperationsBase
else:
    _MappingReferenceOperationsBase = object


class _MappingReferenceOperationsMixin(_MappingReferenceOperationsBase):
    def _mapping_reference_metadata(
        self,
        assay: Assay,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        reduction_loc: str,
        ann_loc: str,
        batch_columns: list[str],
    ) -> dict[str, Any]:

        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        normed_group = as_zarr_group(self.zw[normed_loc], name=normed_loc)
        reduction_group = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
        feature_key = f"{cell_key}__{feat_key}" if feat_key != "I" else "I"
        feature_ids = assay.feats.fetch("ids", key=feature_key)
        batch_values = (
            np.column_stack(
                [self.cells.fetch(column, key=cell_key) for column in batch_columns]
            )
            if batch_columns
            else np.empty((len(self.cells.active_index(cell_key)), 0), dtype=str)
        )
        loadings = np.asarray(
            as_zarr_array(reduction_group["reduction"], name="reduction")[:]
        )
        corrected_hash = ""
        if "harmonizedData" in reduction_group:
            corrected_hash = array_store_hash(
                as_zarr_array(reduction_group["harmonizedData"], name="harmonizedData")
            )
        ann_group = as_zarr_group(self.zw[ann_loc], name=ann_loc)
        (
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            ann_rand_state,
            _,
            _,
        ) = parse_neighbor_index_group_path(ann_loc)
        return {
            "assay": from_assay,
            "cellKey": cell_key,
            "featureKey": feat_key,
            "reductionPath": reduction_loc,
            "annPath": ann_loc,
            "featureHash": array_hash(feature_ids),
            "cellHash": array_hash(self.cells.fetch("ids", key=cell_key)),
            "batchValueHash": array_hash(batch_values),
            "batchColumns": batch_columns,
            "subsetHash": normed_group.attrs["subset_hash"],
            "subsetParams": normed_group.attrs["subset_params"],
            "loadingsHash": array_hash(loadings),
            "correctedCoordinatesHash": corrected_hash,
            "reductionMethod": "pca",
            "annContract": {
                "metric": ann_metric,
                "efConstruction": ann_efc,
                "ef": ann_ef,
                "m": ann_m,
                "randomState": ann_rand_state,
                "featureScaling": bool(ann_group.attrs.get("featureScaling", True)),
            },
        }

    def get_mapping_reference(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
    ) -> MappingReference:
        """Load a validated immutable RNA/PCA Symphony-style mapping reference."""

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("Mapping references currently support RNA assays only")
        state = read_assay_state(self.zw, from_assay)
        if state is not None and state.matches(cell_key, feat_key):
            mapping_ref = state.named_results.get("mapping_reference")
            if mapping_ref is None:
                raise KeyError(
                    "AssayState has no mapping reference for the selected graph"
                )

            return load_artifact_mapping_reference(
                self,
                mapping_ref,
                from_assay,
                cell_key,
                feat_key,
            )
        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        if normed_loc not in self.zw:
            raise KeyError("No normalized reference data exists for the requested keys")
        try:
            reduction_loc = lookup_latest_reduction_group_path(self.zw, normed_loc)
        except KeyError:
            reduction_loc = ""
        if not reduction_loc or reduction_loc not in self.zw:
            raise KeyError("No reduction exists for the requested reference")
        reduction_group = as_zarr_group(self.zw[reduction_loc], name=reduction_loc)
        try:
            ann_loc = lookup_latest_neighbor_index_group_path(self.zw, reduction_loc)
        except KeyError:
            ann_loc = ""
        if not ann_loc or ann_loc not in self.zw:
            raise KeyError("No ANN index exists for the requested reference")
        try:
            artifact, _, is_legacy = resolve_mapping_reference_group(reduction_group)
        except KeyError:
            raise ValueError(
                "This harmonized graph predates the mapping-reference artifact. "
                "Rebuild it with build_mapping_reference."
            ) from None
        artifact_ann_path = artifact.attrs.get("annPath")
        if isinstance(artifact_ann_path, str):
            ann_loc = artifact_ann_path
        if ann_loc not in self.zw:
            raise ValueError(
                "The mapping-reference ANN index is missing. Rebuild the reference."
            )
        ann_group = as_zarr_group(self.zw[ann_loc], name=ann_loc)
        if not bool(ann_group.attrs.get("isHarmonized", False)):
            raise ValueError(
                "The requested graph is not harmonized. Build a harmonized mapping reference first."
            )
        batch_columns = [
            str(column) for column in cast(list[Any], artifact.attrs["batchColumns"])
        ]
        expected = self._mapping_reference_metadata(
            assay,
            from_assay,
            cell_key,
            feat_key,
            reduction_loc,
            ann_loc,
            batch_columns,
        )
        legacy_omissions = {"correctedCoordinatesHash", "annContract"}
        for key, value in expected.items():
            if is_legacy and key in legacy_omissions:
                continue
            if artifact.attrs.get(key) != value:
                raise ValueError(
                    f"Mapping reference is stale because {key} no longer matches. "
                    "Rebuild it with build_mapping_reference."
                )
        return load_mapping_reference(
            self, from_assay, cell_key, feat_key, reduction_loc, ann_loc
        )

    def build_mapping_reference(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        batch_columns: list[str] | None = None,
        **graph_kwargs: Any,
    ) -> MappingReference:
        """Build and return an RNA/PCA Symphony mapping reference."""
        if self.zarr_mode != "r+":
            raise ValueError("Building a mapping reference requires a read-write store")
        if batch_columns is None:
            raise ValueError("batch_columns is required to build a mapping reference")
        if graph_kwargs.get("feat_scaling", True) is False:
            raise ValueError(
                "Mapping references require feat_scaling=True because query "
                "projection uses the stored reference mean and scale."
            )
        reduction_method = graph_kwargs.get("reduction_method", "pca")
        if reduction_method not in {"auto", "pca"}:
            raise ValueError("Mapping references require PCA reduction")
        force_harmony_refit = False
        try:
            current = self.get_mapping_reference(from_assay, cell_key, feat_key)
        except (KeyError, ValueError):
            force_harmony_refit = True
        else:
            current_columns = [
                str(column)
                for column in cast(
                    list[Any],
                    current.metadata.get(
                        "batch_columns",
                        current.metadata.get("batchColumns", []),
                    ),
                )
            ]
            force_harmony_refit = (
                current_columns != batch_columns
                or not bool(current.metadata.get("complete", False))
                or (
                    "artifact_id" not in current.metadata
                    and "artifactHash" not in current.metadata
                )
            )
        graph_kwargs["reduction_method"] = "pca"
        graph_kwargs["feat_scaling"] = True
        plan = self._resolve_graph_plan(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            harmonize=True,
            batch_columns=batch_columns,
            force_harmony_refit=force_harmony_refit,
            **graph_kwargs,
        )
        self._run_resolved_graph_plan(plan)
        reference = self.get_mapping_reference(from_assay, cell_key, feat_key)
        if not bool(reference.metadata.get("complete", False)):
            raise RuntimeError(
                "Mapping reference build did not produce a complete artifact"
            )
        return reference

    def _build_mapping_reference_artifact(
        self,
        *,
        reduction: ArtifactRef,
        batch_correction: ArtifactRef,
        ann_index: ArtifactRef,
        neighbors: ArtifactRef,
        invalidate_cache: bool,
    ) -> ArtifactRef:

        if reduction.assay is None:
            raise ValueError("Reduction artifact has no assay")
        reduction_status = self._require_complete_artifact(
            reduction,
            "reduction",
        )
        if reduction_status.operation != "run_pca" or not bool(
            (reduction_status.parameters or {}).get(
                "feat_scaling",
                False,
            )
        ):
            raise ValueError("Mapping references require PCA with feature scaling")
        if (
            self._artifact_input_ref(
                batch_correction,
                "reduction",
                "reduction",
            )
            != reduction
        ):
            raise ValueError("Batch correction uses another reduction")
        if (
            self._artifact_input_ref(
                ann_index,
                "coordinates",
                "batch_correction",
            )
            != batch_correction
        ):
            raise ValueError("ANN index uses another batch correction")
        if (
            self._artifact_input_ref(
                neighbors,
                "ann_index",
                "ann_index",
            )
            != ann_index
            or self._artifact_input_ref(
                neighbors,
                "coordinates",
                "batch_correction",
            )
            != batch_correction
        ):
            raise ValueError("Neighbors use another corrected graph chain")
        normalized = self._artifact_input_ref(
            reduction,
            "normalized",
            "normalized",
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
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=reduction.assay,
            kind="mapping_reference",
            operation="build_mapping_reference",
            parameters={"method": "symphony"},
            inputs={
                "reduction": reduction,
                "batch_correction": batch_correction,
                "ann_index": ann_index,
                "neighbors": neighbors,
                "cell_selection": cell_selection,
                "feature_selection": feature_selection,
            },
            execution_options=dict(getattr(self, "_artifactExecutionContext", {})),
            invalidate_cache=invalidate_cache,
            required_arrays=(
                "feature_ids",
                "feature_means",
                "feature_scales",
                "loadings",
                "centroids",
                "raw_centroids",
                "corrected_centroids",
                "cluster_mass",
                "sigma",
                "reference_distance_quantiles",
                "reference_distance_values",
            ),
            required_attributes=(
                AttributeRequirement(
                    "correction_ridge",
                    expected_types=(int, float),
                    predicate=lambda value: (
                        not isinstance(value, bool) and np.isfinite(value)
                    ),
                ),
                AttributeRequirement(
                    "reference_metadata",
                    expected_types=(dict,),
                ),
            ),
        )
        if planned.reused:
            return planned.ref
        reduction_parameters = reduction_status.parameters or {}
        batch_size = int(reduction_parameters.get("batch_size", 1000))
        transform, stream = self._load_reduction_stream(
            reduction,
            batch_size=batch_size,
        )
        if transform.loadings is None:
            raise RuntimeError("Mapping reference requires persisted PCA loadings")
        original = np.vstack(
            stream.parallel_blocks(
                "Loading reference coordinates",
            )
        )
        correction_status = self._require_complete_artifact(
            batch_correction,
            "batch_correction",
        )
        correction_group = as_zarr_group(
            self.zw[correction_status.path],
            name=correction_status.path,
        )
        corrected = np.asarray(as_zarr_array(correction_group["data"], name="data")[:])
        assignments = np.asarray(
            as_zarr_array(
                correction_group["assignments"],
                name="assignments",
            )[:]
        )
        cluster_mass, raw_centroids = weighted_centroids(
            original,
            assignments,
        )
        _, corrected_centroids = weighted_centroids(
            corrected,
            assignments,
        )
        ridge = np.asarray(as_zarr_array(correction_group["ridge"], name="ridge")[:])
        ridge_values = np.diag(ridge)[1:]
        correction_ridge = (
            float(np.mean(ridge_values[ridge_values > 0]))
            if np.any(ridge_values > 0)
            else 1.0
        )
        scaling = self._artifact_input_ref(
            reduction,
            "feature_scaling",
            "feature_scaling",
        )
        scaling_group = artifact_group(self.zw, scaling)
        model = SymphonyReferenceModel(
            feature_means=np.asarray(
                as_zarr_array(scaling_group["mean"], name="mean")[:]
            ),
            feature_scales=np.asarray(
                as_zarr_array(scaling_group["scale"], name="scale")[:]
            ),
            loadings=transform.loadings,
            centroids=np.asarray(
                as_zarr_array(
                    correction_group["centroids"],
                    name="centroids",
                )[:]
            ).T,
            raw_centroids=raw_centroids,
            corrected_centroids=corrected_centroids,
            cluster_mass=cluster_mass,
            sigma=np.asarray(
                as_zarr_array(
                    correction_group["sigma"],
                    name="sigma",
                )[:]
            ),
            correction_ridge=correction_ridge,
        )
        normalized_execution = (
            inspect_artifact(self.zw, normalized).execution_options or {}
        )
        feature_column = (
            normalized_execution.get("feat_key")
            if normalized_execution.get("feat_key") == "I"
            else (
                f"{normalized_execution.get('cell_key')}__"
                f"{normalized_execution.get('feat_key')}"
            )
        )
        assay = self._get_assay(reduction.assay)
        feature_ids = assay.feats.fetch("ids", key=str(feature_column))
        correction_parameters = correction_status.parameters or {}
        metadata = {
            "assay": reduction.assay,
            "cell_key": normalized_execution.get("cell_key"),
            "feature_key": normalized_execution.get("feat_key"),
            "batch_columns": correction_parameters.get(
                "batch_columns",
                [],
            ),
            "harmony_parameters": correction_parameters.get(
                "harmony_parameters",
                {},
            ),
            "batch_levels": correction_group.attrs.get(
                "batch_levels",
                [],
            ),
            "method": "symphony",
            "normalization_parameters": (
                inspect_artifact(self.zw, normalized).parameters or {}
            ),
        }
        neighbors_group = artifact_group(self.zw, neighbors)
        distance_quantiles, distance_values = _distance_quantile_summary(
            as_zarr_array(
                neighbors_group["distances"],
                name="distances",
            )
        )
        group = start_artifact(self.zw, planned)
        write_artifact_mapping_reference(
            group,
            model,
            feature_ids,
            metadata,
            distance_quantiles,
            distance_values,
        )
        finish_artifact(group, planned)
        return planned.ref
