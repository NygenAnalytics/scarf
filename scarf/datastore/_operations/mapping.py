from collections.abc import Generator
from typing import TYPE_CHECKING, Any, cast
import os
import warnings

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix

from ...graph.encoded_paths import (
    lookup_latest_neighbor_index_group_path,
    lookup_latest_nearest_neighbors_group_path,
    lookup_latest_reduction_group_path,
    make_normalized_group_path,
    make_normalized_leaf_name,
    parse_neighbor_index_group_path,
    reduction_group_path_from_neighbor_index,
)
from ...graph.distances import validate_distance_provenance
from ...graph.state import read_assay_state
from ...storage.artifacts import (
    ArtifactRef,
    ValueFingerprintBuilder,
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
    inspect_artifact,
    parse_artifact_path,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...assay import ATACassay, Assay, RNAassay
from ...matrix import ChunkedArray
from ...mapping.models import MappingResult
from ...mapping.reference import MappingReference
from ...mapping.symphony import SYMPHONY_ALGORITHM
from ...neighbors.stream import AnnStream
from ...neighbors.stages import (
    AnnIndexStage,
    LazyTransformStream,
    NeighborQueryStage,
    ReductionTransform,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.ann_index import serialize_ann_index
from ...storage.geometry import array_geometry
from ...storage.partition import row_band
from ...storage.artifact_writer import (
    ArrayRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from ...utils.compute import controlled_compute
from ...utils.logging import logger, progress_enabled

if TYPE_CHECKING:
    from ..graph_datastore import GraphDataStore as _MappingOperationsBase
else:
    _MappingOperationsBase = object


class _MappingOperationsMixin(_MappingOperationsBase):
    # Mapping artifacts in this module intentionally preserve the current
    # reference-store workflow, including query-owned projections written into
    # that store. This is a compatibility bridge, not the final mapping model.
    # A separate refactor should keep query data in its source store and expose
    # a lazy, virtually aligned feature stream for the reference ANN index.
    _PROJECTION_PROVENANCE_ATTRS = frozenset(
        {
            "assay",
            "cell_key",
            "feature_key",
            "save_k",
            "reference_cell_fingerprint",
            "reference_feature_fingerprint",
            "reference_path",
            "reduction_path",
            "ann_path",
            "normalization_fingerprint",
            "reduction_fingerprint",
        }
    )

    def _select_projection_artifact(
        self,
        from_assay: str,
        target_name: str,
        ref: ArtifactRef,
    ) -> None:
        assay = self._get_assay(from_assay)
        projections = (
            as_zarr_group(assay.z["projections"], name="projections")
            if "projections" in assay.z
            else assay.z.create_group("projections")
        )
        raw_artifacts = projections.attrs.get("artifacts", {})
        if "artifacts" in projections.attrs and not isinstance(
            raw_artifacts,
            dict,
        ):
            raise RuntimeError("Projection artifact index is invalid")
        artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
        artifacts[target_name] = ref.to_dict()
        projections.attrs["artifacts"] = artifacts

    def _projection_artifact_path(
        self,
        from_assay: str,
        target_name: str,
    ) -> str | None:
        assay = self._get_assay(from_assay)
        if "projections" not in assay.z:
            return None
        projections = as_zarr_group(assay.z["projections"], name="projections")
        raw_artifacts = projections.attrs.get("artifacts", {})
        if "artifacts" in projections.attrs and not isinstance(
            raw_artifacts,
            dict,
        ):
            raise RuntimeError("Projection artifact index is invalid")
        if not isinstance(raw_artifacts, dict):
            return None
        raw_ref = raw_artifacts.get(target_name)
        if target_name in raw_artifacts and not isinstance(raw_ref, dict):
            raise RuntimeError(f"Projection index for {target_name!r} is invalid")
        if not isinstance(raw_ref, dict):
            return None
        try:
            ref = ArtifactRef.from_dict(raw_ref)
            if (
                ref.scope != "assay"
                or ref.assay != from_assay
                or ref.kind != "projection"
            ):
                raise ValueError("Projection index has an invalid reference")
            status = inspect_artifact(self.zw, ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Projection index for {target_name!r} is invalid"
            ) from exc
        if not status.exists or not status.complete:
            raise RuntimeError(f"Projection index for {target_name!r} is incomplete")
        return status.path

    def _delete_projection_artifact(
        self,
        from_assay: str,
        target_name: str,
    ) -> None:
        assay = self._get_assay(from_assay)
        if "projections" not in assay.z:
            return
        projections = as_zarr_group(
            assay.z["projections"],
            name="projections",
        )
        raw_artifacts = projections.attrs.get("artifacts", {})
        if not isinstance(raw_artifacts, dict):
            raise RuntimeError("Projection artifact index is invalid")
        raw_ref = raw_artifacts.get(target_name)
        if not isinstance(raw_ref, dict):
            return
        ref = ArtifactRef.from_dict(raw_ref)
        if ref.kind != "projection" or ref.scope != "assay" or ref.assay != from_assay:
            raise RuntimeError(f"Projection index for {target_name!r} is invalid")
        path = artifact_path(ref)
        if path in self.zw:
            del self.zw[path]
        artifacts = dict(raw_artifacts)
        artifacts.pop(target_name, None)
        projections.attrs["artifacts"] = artifacts

    @staticmethod
    def _validate_projection_arrays(store: zarr.Group, target_name: str) -> None:
        if "indices" not in store or "distances" not in store:
            raise ValueError(
                f"Projection {target_name!r} is missing indices or distances."
            )
        indices = as_zarr_array(store["indices"], name="indices")
        distances = as_zarr_array(store["distances"], name="distances")
        if (
            len(indices.shape) != 2
            or len(distances.shape) != 2
            or indices.shape != distances.shape
        ):
            raise ValueError(
                f"Projection {target_name!r} has incompatible neighbor arrays."
            )
        if indices.shape[1] < 1:
            raise ValueError(
                f"Projection {target_name!r} does not contain any neighbors."
            )

    def _projection_has_provenance(self, store: zarr.Group) -> bool:
        attrs = store.attrs
        if not bool(attrs.get("complete", False)):
            return False
        if any(attrs.get(name) is None for name in self._PROJECTION_PROVENANCE_ATTRS):
            return False
        return "reference_feature_indices" in store

    def _load_complete_projection(
        self,
        target_name: str,
        from_assay: str,
        cell_key: str,
        feat_key: str | None = None,
    ) -> zarr.Group:
        from ...mapping.hashing import array_hash

        store_loc = self._projection_artifact_path(from_assay, target_name)
        if store_loc is None:
            store_loc = f"{from_assay}/projections/{target_name}"
        if store_loc not in self.zw:
            raise KeyError(
                f"Projections have not been computed for {target_name}. Run run_mapping first."
            )
        store = as_zarr_group(self.zw[store_loc], name=store_loc)
        attrs = store.attrs
        self._validate_projection_arrays(store, target_name)
        if "complete" in attrs and not bool(attrs["complete"]):
            raise ValueError(
                f"Projection {target_name!r} is incomplete. Run run_mapping again."
            )
        if not self._projection_has_provenance(store):
            # The feature-index array is the provenance-era marker; genuine
            # legacy projections lack it entirely. A projection that carries the
            # marker but is missing required provenance is a partial or corrupt
            # current write and must not silently downgrade to the legacy path.
            if "reference_feature_indices" in store:
                raise ValueError(
                    f"Projection {target_name!r} has incomplete provenance metadata. "
                    "Re-run run_mapping to rebuild it."
                )
            warnings.warn(
                f"Projection {target_name!r} predates projection provenance. "
                "Re-run run_mapping to remove this compatibility warning.",
                DeprecationWarning,
                stacklevel=3,
            )
            return store
        stored_cell_key = attrs.get("cell_key")
        if attrs.get("assay") != from_assay or stored_cell_key != cell_key:
            raise ValueError(
                f"Projection {target_name!r} does not match the selected reference assay or cells."
            )
        stored_feat_key = attrs.get("feature_key")
        if feat_key is not None and stored_feat_key != feat_key:
            logger.warning(
                f"Projection {target_name!r} uses feature key {stored_feat_key!r}, "
                f"not the current key {feat_key!r}; validating its stored provenance."
            )
        reference_cells = self.cells.fetch("ids", key=stored_cell_key)
        if attrs.get("reference_cell_fingerprint") != array_hash(reference_cells):
            raise ValueError(
                f"Projection {target_name!r} was built from a different reference cell set."
            )
        self._validate_projection_provenance(store, target_name)
        return store

    def _validate_projection_provenance(
        self, store: zarr.Group, target_name: str
    ) -> None:
        from ...mapping.hashing import array_hash

        attrs = store.attrs
        save_k = attrs.get("save_k")
        if (
            isinstance(save_k, bool)
            or not isinstance(save_k, int | np.integer)
            or int(save_k)
            != int(as_zarr_array(store["indices"], name="indices").shape[1])
        ):
            raise ValueError(
                f"Projection {target_name!r} has inconsistent saved-neighbor provenance."
            )
        assay_name = cast(str, attrs["assay"])
        cell_key = cast(str, attrs.get("cell_key"))
        feature_key = cast(str, attrs.get("feature_key"))
        source_assay = self._get_assay(assay_name)
        feature_column = (
            feature_key if feature_key == "I" else f"{cell_key}__{feature_key}"
        )
        reference_features = source_assay.feats.fetch("ids", key=feature_column)
        if attrs.get("reference_feature_fingerprint") != array_hash(reference_features):
            raise ValueError(
                f"Projection {target_name!r} was built from a different reference feature set."
            )

        reference_path = cast(
            str,
            attrs.get("reference_path", ""),
        )
        if reference_path not in self.zw:
            raise ValueError(
                f"Projection {target_name!r} references missing normalized data."
            )
        normed = as_zarr_group(self.zw[reference_path], name=reference_path)
        try:
            normalized_ref = parse_artifact_path(reference_path)
        except ValueError:
            expected_normalized_identity = normed.attrs.get("subset_hash")
        else:
            expected_normalized_identity = normalized_ref.artifact_id
        if attrs.get("normalization_fingerprint") != expected_normalized_identity:
            raise ValueError(
                f"Projection {target_name!r} references changed normalized data."
            )

        reduction_path = cast(
            str,
            attrs.get("reduction_path", ""),
        )
        if reduction_path not in self.zw:
            raise ValueError(
                f"Projection {target_name!r} references a missing reduction."
            )
        reduction = as_zarr_group(self.zw[reduction_path], name=reduction_path)
        reduction_array_name = (
            "loadings"
            if "loadings" in reduction
            else "reduction"
            if "reduction" in reduction
            else None
        )
        if reduction_array_name is None:
            raise ValueError(
                f"Projection {target_name!r} references a reduction without loadings."
            )
        loadings = np.asarray(
            as_zarr_array(
                reduction[reduction_array_name],
                name=reduction_array_name,
            )[:]
        )
        if attrs.get("reduction_fingerprint") != array_hash(loadings):
            raise ValueError(
                f"Projection {target_name!r} references changed reduction loadings."
            )

        ann_path = cast(str, attrs.get("ann_path", ""))
        if ann_path not in self.zw:
            raise ValueError(
                f"Projection {target_name!r} references a missing ANN index."
            )
        ann = as_zarr_group(self.zw[ann_path], name=ann_path)
        expected_scaling = bool(attrs.get("ann_feature_scaling"))
        artifact_intersection = False
        intersection_source_path: str | None = None
        intersection_feature_fingerprint: str | None = None
        try:
            ann_ref = parse_artifact_path(ann_path)
        except ValueError:
            actual_scaling = bool(ann.attrs.get("featureScaling", True))
            actual_harmonized = bool(ann.attrs.get("isHarmonized", False))
            artifact_ann_parameters = None
        else:
            ann_status = inspect_artifact(self.zw, ann_ref)
            artifact_ann_parameters = ann_status.parameters or {}
            ann_inputs = ann_status.inputs or {}
            if ann_ref.kind == "intersection_ann_index":
                artifact_intersection = True
                raw_fingerprint = ann_inputs.get("selected_feature_fingerprint")
                if not isinstance(raw_fingerprint, str):
                    raise ValueError(
                        "Intersection ANN selected-feature fingerprint is missing"
                    )
                intersection_feature_fingerprint = raw_fingerprint
                raw_source = ann_inputs.get("source_ann_index")
                if not isinstance(raw_source, dict):
                    raise ValueError("Intersection ANN source is missing")
                try:
                    source_ann_ref = ArtifactRef.from_dict(raw_source)
                except (KeyError, TypeError, ValueError):
                    actual_harmonized = False
                    actual_scaling = bool(
                        artifact_ann_parameters.get("feat_scaling", True)
                    )
                    coordinates_ref = None
                else:
                    if (
                        source_ann_ref.kind != "ann_index"
                        or source_ann_ref.assay != ann_ref.assay
                    ):
                        raise ValueError("Intersection ANN source is invalid")
                    source_status = inspect_artifact(self.zw, source_ann_ref)
                    if not source_status.exists or not source_status.complete:
                        raise ValueError("Intersection ANN source is incomplete")
                    intersection_source_path = artifact_path(source_ann_ref)
                    ann_inputs = source_status.inputs or {}
                    coordinates_ref = None
            else:
                coordinates_ref = None
            if coordinates_ref is None and "coordinates" in ann_inputs:
                raw_coordinates = ann_inputs.get("coordinates")
                if not isinstance(raw_coordinates, dict):
                    raise ValueError("ANN coordinate provenance is missing")
                coordinates_ref = ArtifactRef.from_dict(raw_coordinates)
            if coordinates_ref is not None:
                actual_harmonized = coordinates_ref.kind == "batch_correction"
                if actual_harmonized:
                    correction_inputs = (
                        inspect_artifact(self.zw, coordinates_ref).inputs or {}
                    )
                    raw_reduction = correction_inputs.get("reduction")
                    if not isinstance(raw_reduction, dict):
                        raise ValueError("Batch correction reduction input is missing")
                    reduction_ref = ArtifactRef.from_dict(raw_reduction)
                else:
                    reduction_ref = coordinates_ref
                reduction_parameters = (
                    inspect_artifact(self.zw, reduction_ref).parameters or {}
                )
                actual_scaling = bool(reduction_parameters.get("feat_scaling", True))
        if actual_scaling != expected_scaling:
            raise ValueError(
                f"Projection {target_name!r} references an ANN index with changed scaling."
            )
        if actual_harmonized != bool(attrs.get("ann_is_harmonized")):
            raise ValueError(
                f"Projection {target_name!r} references a changed ANN coordinate space."
            )
        settings_ann_path = ann_path
        if "__intersection_" in ann_path:
            source_ann_path = ann.attrs.get(
                "ann_source_path", ann.attrs.get("sourceAnnPath")
            )
            if not isinstance(source_ann_path, str) or not source_ann_path:
                raise ValueError(
                    f"Projection {target_name!r} has invalid intersection ANN provenance."
                )
            settings_ann_path = source_ann_path
        if artifact_ann_parameters is None:
            try:
                (
                    path_ann_metric,
                    path_ann_efc,
                    path_ann_ef,
                    path_ann_m,
                    path_ann_random_state,
                    _,
                    _,
                ) = parse_neighbor_index_group_path(settings_ann_path)
            except ValueError as exc:
                raise ValueError(
                    f"Projection {target_name!r} has an invalid ANN provenance path."
                ) from exc
        else:
            path_ann_metric = str(artifact_ann_parameters["ann_metric"])
            path_ann_efc = int(artifact_ann_parameters["ann_efc"])
            path_ann_ef = int(artifact_ann_parameters["ann_ef"])
            path_ann_m = int(artifact_ann_parameters["ann_m"])
            path_ann_random_state = int(artifact_ann_parameters["rand_state"])
        stored_ann_values = (
            attrs.get("ann_efc"),
            attrs.get("ann_ef"),
            attrs.get("ann_m"),
            attrs.get("ann_random_state"),
        )
        if not all(isinstance(value, int | float) for value in stored_ann_values):
            raise ValueError(
                f"Projection {target_name!r} is missing numeric ANN settings."
            )
        ann_efc, ann_ef, ann_m, ann_random_state = cast(
            tuple[int | float, ...], stored_ann_values
        )
        if (
            attrs.get("ann_metric") != path_ann_metric
            or int(ann_efc) != path_ann_efc
            or int(ann_ef) != path_ann_ef
            or int(ann_m) != path_ann_m
            or int(ann_random_state) != path_ann_random_state
        ):
            raise ValueError(
                f"Projection {target_name!r} references incompatible ANN settings."
            )

        if "reference_feature_indices" not in store:
            raise ValueError(
                f"Projection {target_name!r} is missing selected-feature provenance."
            )
        feature_indices = np.asarray(
            as_zarr_array(
                store["reference_feature_indices"],
                name="reference_feature_indices",
            )[:],
            dtype=np.int64,
        )
        all_feature_ids = source_assay.feats.fetch_all("ids")
        if np.any(feature_indices < 0) or np.any(
            feature_indices >= len(all_feature_ids)
        ):
            raise ValueError(
                f"Projection {target_name!r} contains invalid reference feature indices."
            )
        if attrs.get("selected_feature_fingerprint") != array_hash(
            all_feature_ids[feature_indices]
        ):
            raise ValueError(
                f"Projection {target_name!r} references a changed selected feature set."
            )
        if artifact_intersection:
            if intersection_feature_fingerprint != attrs.get(
                "selected_feature_fingerprint"
            ):
                raise ValueError(
                    f"Projection {target_name!r} references a changed "
                    "intersection ANN index."
                )
            if (
                intersection_source_path is not None
                and attrs.get("ann_source_path") != intersection_source_path
            ):
                raise ValueError(
                    f"Projection {target_name!r} references a changed "
                    "intersection ANN source."
                )
        if "__intersection_" in ann_path and ann.attrs.get(
            "selectedFeatureHash"
        ) != attrs.get("selected_feature_fingerprint"):
            raise ValueError(
                f"Projection {target_name!r} references a changed intersection ANN index."
            )
        if "__intersection_" in ann_path:
            source_ann_path = attrs.get("ann_source_path")
            if (
                not isinstance(source_ann_path, str)
                or not source_ann_path
                or ann.attrs.get("sourceAnnPath") != source_ann_path
                or source_ann_path not in self.zw
            ):
                raise ValueError(
                    f"Projection {target_name!r} has invalid intersection ANN provenance."
                )
            source_ann = as_zarr_group(self.zw[source_ann_path], name=source_ann_path)
            if bool(source_ann.attrs.get("featureScaling", True)) != expected_scaling:
                raise ValueError(
                    f"Projection {target_name!r} references a changed source ANN space."
                )
        if attrs.get("correction_method") == "symphony":
            mapping_path = attrs.get("mapping_reference_path")
            if not isinstance(mapping_path, str) or mapping_path not in self.zw:
                raise ValueError(
                    f"Projection {target_name!r} references a missing mapping artifact."
                )
            try:
                mapping_ref = parse_artifact_path(mapping_path)
            except ValueError:
                from ...mapping.artifact import validate_mapping_reference_artifact

                validate_mapping_reference_artifact(
                    as_zarr_group(self.zw[mapping_path], name=mapping_path)
                )
            else:
                mapping_status = inspect_artifact(self.zw, mapping_ref)
                if not mapping_status.exists or not mapping_status.complete:
                    raise ValueError(
                        f"Projection {target_name!r} mapping artifact is incomplete."
                    )

    @staticmethod
    def _same_assay_store(source_assay: Assay, target_assay: Assay) -> bool:
        if source_assay is target_assay:
            return True
        source_group = source_assay.z
        target_group = target_assay.z

        def normalized_store_path(group: zarr.Group) -> str:
            value = str(getattr(group, "store_path", "")).rstrip("/")
            if value.startswith("file://"):
                return os.path.realpath(value[7:])
            return value

        source_store_path = normalized_store_path(source_group)
        target_store_path = normalized_store_path(target_group)
        if source_store_path and source_store_path == target_store_path:
            return True
        return getattr(source_group, "store", None) is getattr(
            target_group, "store", None
        ) and getattr(source_group, "path", None) == getattr(target_group, "path", None)

    def _guard_mapping_target_path(
        self,
        source_assay: Assay,
        target_assay: Assay,
        source_cell_key: str,
        source_feat_key: str,
        target_cell_key: str,
        target_feat_key: str,
    ) -> None:
        if not self._same_assay_store(source_assay, target_assay):
            return
        source_path = make_normalized_leaf_name(source_cell_key, source_feat_key)
        target_path = make_normalized_leaf_name(target_cell_key, target_feat_key)
        if source_path == target_path:
            raise ValueError(
                "The mapping target normalization path matches the reference path. "
                "Choose a distinct target_feat_key so reference data cannot be overwritten."
            )

    @staticmethod
    def _projection_block_size(indices: Any) -> int:
        return row_band(
            array_geometry(indices),
            unit="chunk",
            fallback=min(max(int(indices.shape[0]), 1), 10_000),
        )

    def _fingerprint_mapping_matrix(self, data: Any) -> str:
        shape = tuple(int(value) for value in data.shape)
        builder = ValueFingerprintBuilder()
        builder.begin_array("target_normalized", shape, np.dtype(data.dtype))
        offset = 0
        for delayed_block in data.blocks:
            block = np.asarray(controlled_compute(delayed_block, self.nthreads))
            builder.update_array_block(
                "target_normalized",
                (offset,) + (0,) * (block.ndim - 1),
                block,
            )
            offset += block.shape[0]
        builder.end_array("target_normalized")
        return builder.hexdigest()

    def _iter_projection_neighbor_rows(
        self, store: zarr.Group
    ) -> Generator[tuple[int, np.ndarray, np.ndarray, np.ndarray, bool], None, None]:
        from ...mapping.confidence import distance_weights

        indices = as_zarr_array(store["indices"], name="indices")
        distances = as_zarr_array(store["distances"], name="distances")
        uninformative = (
            as_zarr_array(store["uninformative"], name="uninformative")
            if "uninformative" in store
            else None
        )
        block_size = self._projection_block_size(indices)
        for start in range(0, indices.shape[0], block_size):
            stop = min(start + block_size, indices.shape[0])
            block_indices = np.asarray(indices[start:stop])
            block_distances = np.asarray(distances[start:stop])
            block_weights = distance_weights(block_distances)
            block_uninformative = (
                np.asarray(uninformative[start:stop], dtype=bool)
                if uninformative is not None
                else np.zeros(stop - start, dtype=bool)
            )
            rows = zip(
                block_indices,
                block_weights,
                block_distances,
                block_uninformative,
                strict=True,
            )
            for offset, row in enumerate(rows):
                neighbors, weights, row_distances, force_unknown = row
                yield (
                    start + offset,
                    neighbors,
                    weights,
                    row_distances,
                    bool(force_unknown),
                )

    def _projection_provenance(
        self,
        source_assay: Assay,
        target_assay: Assay,
        source_assay_name: str,
        target_name: str,
        cell_key: str,
        feat_key: str,
        target_cell_key: str,
        target_feat_key: str,
        feature_indices: np.ndarray,
        correction_method: str,
        ann_obj: AnnStream,
    ) -> dict[str, Any]:
        from ...mapping.hashing import array_hash

        reference_features = source_assay.feats.fetch(
            "ids", key=f"{cell_key}__{feat_key}" if feat_key != "I" else "I"
        )
        selected_feature_ids = source_assay.feats.fetch_all("ids")[feature_indices]
        if correction_method == "intersection":
            feature_coverage = float(len(feature_indices) / len(reference_features))
        else:
            target_feature_ids = target_assay.feats.fetch_all("ids")
            feature_coverage = float(
                np.isin(reference_features, target_feature_ids).sum()
                / len(reference_features)
            )
        reference_path = make_normalized_group_path(
            source_assay_name, cell_key, feat_key
        )
        reduction_path: str | None = None
        ann_source_path = ""
        ann_path = self._ann_stream_path(ann_obj)
        if ann_path is not None:
            try:
                ann_ref = parse_artifact_path(ann_path)
            except ValueError:
                reduction_source_path = ann_path
                if "__intersection_" in ann_path and ann_path in self.zw:
                    intersection_group = as_zarr_group(self.zw[ann_path], name=ann_path)
                    stored_source_path = intersection_group.attrs.get("sourceAnnPath")
                    if isinstance(stored_source_path, str):
                        reduction_source_path = stored_source_path
                reduction_path = reduction_group_path_from_neighbor_index(
                    reduction_source_path
                )
            else:
                ann_status = inspect_artifact(self.zw, ann_ref)
                ann_inputs = ann_status.inputs or {}
                if ann_ref.kind == "intersection_ann_index":
                    raw_source = ann_inputs.get("source_ann_index")
                    if not isinstance(raw_source, dict):
                        raise ValueError("Intersection ANN source is missing")
                    source_ann_ref = ArtifactRef.from_dict(raw_source)
                    ann_source_path = artifact_path(source_ann_ref)
                    ann_inputs = (
                        inspect_artifact(
                            self.zw,
                            source_ann_ref,
                        ).inputs
                        or {}
                    )
                raw_coordinates = ann_inputs.get("coordinates")
                if not isinstance(raw_coordinates, dict):
                    raise ValueError("ANN coordinate provenance is missing")
                coordinates_ref = ArtifactRef.from_dict(raw_coordinates)
                if coordinates_ref.kind == "batch_correction":
                    correction_inputs = (
                        inspect_artifact(self.zw, coordinates_ref).inputs or {}
                    )
                    raw_reduction = correction_inputs.get("reduction")
                    if not isinstance(raw_reduction, dict):
                        raise ValueError("Batch correction reduction is missing")
                    reduction_ref = ArtifactRef.from_dict(raw_reduction)
                else:
                    reduction_ref = coordinates_ref
                reduction_path = artifact_path(reduction_ref)
                reduction_inputs = inspect_artifact(self.zw, reduction_ref).inputs or {}
                raw_normalized = reduction_inputs.get("normalized")
                if not isinstance(raw_normalized, dict):
                    raise ValueError("Reduction normalized input is missing")
                reference_path = artifact_path(ArtifactRef.from_dict(raw_normalized))
        if reference_path in self.zw:
            if reduction_path is None:
                try:
                    reduction_path = lookup_latest_reduction_group_path(
                        self.zw, reference_path
                    )
                except KeyError:
                    reduction_path = None
            if reduction_path is not None and reduction_path in self.zw:
                if ann_path is None:
                    try:
                        ann_path = lookup_latest_neighbor_index_group_path(
                            self.zw, reduction_path
                        )
                    except KeyError:
                        ann_path = None
        reduction_fingerprint = ""
        normalization_fingerprint: int | str = ""
        ann_feature_scaling = ann_obj.featureScaling
        ann_is_harmonized = ann_obj.harmonize
        if ann_path is not None and ann_path in self.zw:
            ann_group = as_zarr_group(self.zw[ann_path], name=ann_path)
            stored_source_path = ann_group.attrs.get("sourceAnnPath")
            if isinstance(stored_source_path, str):
                ann_source_path = stored_source_path
        if reference_path in self.zw:
            normed = as_zarr_group(self.zw[reference_path], name=reference_path)
            normalization_fingerprint = cast(
                int | str,
                normed.attrs.get("subset_hash", ""),
            )
            if not normalization_fingerprint:
                try:
                    normalized_ref = parse_artifact_path(reference_path)
                except ValueError:
                    pass
                else:
                    normalization_fingerprint = normalized_ref.artifact_id
        if reduction_path is not None and reduction_path in self.zw:
            reduction = as_zarr_group(self.zw[reduction_path], name=reduction_path)
            reduction_array_name = (
                "loadings"
                if "loadings" in reduction
                else "reduction"
                if "reduction" in reduction
                else None
            )
            if reduction_array_name is not None:
                reduction_fingerprint = array_hash(
                    np.asarray(
                        as_zarr_array(
                            reduction[reduction_array_name],
                            name=reduction_array_name,
                        )[:]
                    )
                )
        return {
            "complete": False,
            "assay": source_assay_name,
            "target_name": target_name,
            "cell_key": cell_key,
            "feature_key": feat_key,
            "target_cell_key": target_cell_key,
            "target_feature_key": target_feat_key,
            "reference_path": reference_path,
            "reduction_path": reduction_path or "",
            "ann_path": ann_path or "",
            "reference_cell_fingerprint": array_hash(
                self.cells.fetch("ids", key=cell_key)
            ),
            "target_cell_fingerprint": array_hash(
                target_assay.cells.fetch("ids", key=target_cell_key)
            ),
            "reference_feature_fingerprint": array_hash(reference_features),
            "selected_feature_fingerprint": array_hash(selected_feature_ids),
            "normalization_fingerprint": normalization_fingerprint,
            "reduction_fingerprint": reduction_fingerprint,
            "feature_coverage": feature_coverage,
            "reduction_method": ann_obj.method,
            "reduction_dimensions": int(
                ann_obj.dims if ann_obj.dims is not None else ann_obj.nFeats
            ),
            "ann_metric": ann_obj.annMetric,
            "ann_efc": int(ann_obj.annEfc),
            "ann_ef": int(ann_obj.annEf),
            "ann_m": int(ann_obj.annM),
            "ann_random_state": int(ann_obj.randState),
            "ann_feature_scaling": ann_feature_scaling,
            "ann_is_harmonized": ann_is_harmonized,
            "ann_source_path": ann_source_path,
            "correction_method": correction_method,
            "algorithm_variant": (
                SYMPHONY_ALGORITHM
                if correction_method == "symphony"
                else correction_method
            ),
        }

    def _build_intersection_ann(
        self,
        ann_obj: AnnStream,
        source_assay: Assay,
        cell_key: str,
        feat_key: str,
        feature_indices: np.ndarray,
    ) -> AnnStream:
        from ...mapping.confidence import _distance_quantile_summary
        from ...mapping.hashing import array_hash

        if ann_obj.method != "pca" or ann_obj.loadings is None:
            raise ValueError(
                "missing_feature_policy='intersection' only supports PCA references"
            )
        if ann_obj.harmonize:
            raise ValueError(
                "missing_feature_policy='intersection' is not supported for harmonized references"
            )
        active_indices = source_assay.feats.active_index(
            f"{cell_key}__{feat_key}" if feat_key != "I" else "I"
        )
        positions = np.searchsorted(active_indices, feature_indices)
        if len(positions) != len(feature_indices) or not np.array_equal(
            active_indices[positions], feature_indices
        ):
            raise ValueError("Failed to align selected reference feature positions")
        ann_path = self._ann_stream_path(ann_obj)
        if ann_path is None:
            raise ValueError("The reference ANN path is unavailable")
        selected_feature_fingerprint = array_hash(
            source_assay.feats.fetch_all("ids")[feature_indices]
        )
        try:
            source_ann_ref = parse_artifact_path(ann_path)
        except ValueError:
            source_ann_ref = None
        if source_ann_ref is None:
            source_input: object = {
                "legacy_ann_index_fingerprint": fingerprint_array(
                    serialize_ann_index(ann_obj.annIdx)
                )
            }
            artifact_assay = str(source_assay.name)
        else:
            source_input = source_ann_ref
            if source_ann_ref.assay is None:
                raise ValueError("Assay-scoped ANN artifact has no assay")
            artifact_assay = source_ann_ref.assay
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=artifact_assay,
            kind="intersection_ann_index",
            operation="build_intersection_ann_index",
            parameters={
                "ann_metric": ann_obj.annMetric,
                "ann_efc": int(ann_obj.annEfc),
                "ann_ef": int(ann_obj.annEf),
                "ann_m": int(ann_obj.annM),
                "rand_state": int(ann_obj.randState),
                "feat_scaling": bool(ann_obj.featureScaling),
            },
            inputs={
                "source_ann_index": source_input,
                "selected_feature_fingerprint": selected_feature_fingerprint,
            },
            execution_options={},
            required_arrays=(
                "ann_idx_bytes",
                "reference_distance_quantiles",
                "reference_distance_values",
            ),
        )
        intersection_path = artifact_path(planned.ref)
        source_data = ann_obj.data[:, positions]
        selected_mu = ann_obj.mu[positions] if ann_obj.featureScaling else ann_obj.mu
        selected_sigma = (
            ann_obj.sigma[positions] if ann_obj.featureScaling else ann_obj.sigma
        )
        reduction = ReductionTransform(
            data=source_data,
            method="pca",
            dims=ann_obj.loadings.shape[1],
            loadings=ann_obj.loadings[positions, :],
            use_for_pca=np.ones(ann_obj.nCells, dtype=bool),
            mu=selected_mu,
            sigma=selected_sigma,
            batch_size=ann_obj.batchSize,
            nthreads=self.nthreads,
            rand_state=ann_obj.randState,
            disable_scaling=not ann_obj.featureScaling,
            lsi_skip_first=True,
            lsi_params={},
        )
        stream = LazyTransformStream(
            data=source_data,
            transform=reduction.transform,
            nthreads=self.nthreads,
            batch_size=ann_obj.batchSize,
        )
        intersection_idx = None
        if planned.reused:
            intersection_idx = self._resolve_ann_index(
                intersection_path,
                ann_obj.annMetric,
                ann_obj.loadings.shape[1],
                expected_count=ann_obj.nCells,
            )
            if intersection_idx is None:
                raise RuntimeError("Reusable intersection ANN artifact has no index")
            intersection_idx = AnnIndexStage.configure(
                intersection_idx,
                ef=ann_obj.annEf,
                threads=1,
            )
        else:
            intersection_idx = AnnIndexStage.fit(
                coordinates=stream,
                metric=ann_obj.annMetric,
                dims=ann_obj.loadings.shape[1],
                n_cells=ann_obj.nCells,
                ef_construction=ann_obj.annEfc,
                ef=ann_obj.annEf,
                m=ann_obj.annM,
                rand_state=ann_obj.randState,
                ann_threads=1,
            )
            intersection_group = start_artifact(self.zw, planned)
            self._persist_ann_index(
                intersection_path,
                intersection_idx,
                ann_metric=ann_obj.annMetric,
                dimensions=ann_obj.loadings.shape[1],
                element_count=ann_obj.nCells,
            )
            sample_stride = max(
                int(np.ceil(ann_obj.nCells / 100_000)),
                1,
            )
            query = NeighborQueryStage(
                intersection_idx,
                ann_obj.k,
                ann_obj.annMetric,
            )
            sampled_distances: list[np.ndarray] = []
            entry_start = 0
            for block in stream.iter_raw():
                entry_end = entry_start + len(block)
                transformed = reduction.transform(block)
                _indices, distances, _missed = cast(
                    tuple[np.ndarray, np.ndarray, int],
                    query.query(
                        transformed,
                        self_indices=np.arange(entry_start, entry_end),
                    ),
                )
                sample_mask = (
                    np.arange(
                        entry_start,
                        entry_end,
                        dtype=np.int64,
                    )
                    % sample_stride
                    == 0
                )
                sampled_distances.append(
                    np.asarray(
                        distances[sample_mask, 0],
                        dtype=np.float64,
                    )
                )
                entry_start = entry_end
            distance_quantiles, distance_values = _distance_quantile_summary(
                np.concatenate(sampled_distances)
            )
            for name, values in (
                ("reference_distance_quantiles", distance_quantiles),
                ("reference_distance_values", distance_values),
            ):
                output = create_zarr_dataset(
                    intersection_group,
                    name,
                    (min(len(values), 1_001),),
                    "f8",
                    values.shape,
                )
                output[:] = values
            finish_artifact(intersection_group, planned)
        intersection_ann = AnnStream(
            data=source_data,
            k=ann_obj.k,
            n_cluster=2,
            reduction_method="pca",
            dims=ann_obj.loadings.shape[1],
            loadings=reduction.loadings,
            use_for_pca=np.ones(ann_obj.nCells, dtype=bool),
            mu=reduction.mu,
            sigma=reduction.sigma,
            ann_metric=ann_obj.annMetric,
            ann_efc=ann_obj.annEfc,
            ann_ef=ann_obj.annEf,
            ann_m=ann_obj.annM,
            nthreads=self.nthreads,
            ann_parallel=False,
            rand_state=ann_obj.randState,
            do_kmeans_fit=False,
            disable_scaling=not ann_obj.featureScaling,
            ann_idx=intersection_idx,
            lsi_skip_first=True,
            lsi_params={},
            harmonize=False,
        )
        self._remember_ann_stream_path(intersection_ann, intersection_path)
        self._remember_ann_stream_neighbors(
            intersection_ann,
            intersection_path,
        )
        return intersection_ann

    def run_mapping(
        self,
        target_assay: Assay,
        target_name: str,
        target_feat_key: str,
        target_cell_key: str = "I",
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        save_k: int = 3,
        ref_mu: bool = True,
        ref_sigma: bool = True,
        run_coral: bool = False,
        exclude_missing: bool = False,
        filter_null: bool = False,
        feat_scaling: bool = True,
        missing_feature_policy: str | None = None,
        query_batches: pd.DataFrame | None = None,
        invalidate_cache: bool = False,
    ) -> None:
        """Projects cells from external assays into the cell-neighbourhood
        graph using existing PCA loadings and ANN index. For each external cell
        (target) nearest neighbours are identified and save within the Zarr
        hierarchy group `projections`.

        Args:
            target_assay: Assay object of the target dataset.
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            target_feat_key: This will be used to name wherein the normalized target data will be saved in its own
                             zarr hierarchy.
            target_cell_key: Cell key for the target data. (Default value: 'I')
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            save_k: Number of the nearest neighbours to identify for each target cell (Default value: 3)
            ref_mu: Deprecated compatibility flag. Reference means are always used.
            ref_sigma: Deprecated compatibility flag. Reference scales are always used.
            run_coral: Deprecated experimental feature-space correction. If True,
                       CORAL aligns target features to the reference distribution.
                       It creates an m by m matrix where m is the number of
                       features, so it is not suitable for very large feature
                       sets. Build a Symphony-style mapping reference for new
                       harmonized atlas workflows. (Default value: False)
            exclude_missing: If set to True then only those features that are present in both reference and
                             target are used. If not all reference features from `feat_key` are present in target data
                             then a compatibility graph key is retained while mapping uses an isolated intersection
                             index. Deprecated; use ``missing_feature_policy='intersection'``.
            filter_null: If True then those features that have a total sum of 0 in the target cells are removed.
                         This has an affect only when `exclude_missing` is True. (Default value: False)
            feat_scaling: If False then features from target cells are not scaled.
                          Setting this to False is not recommended.
            missing_feature_policy: Handling for reference features absent from the
                target. ``'zero'`` fills them with zero values, ``'intersection'``
                constructs an isolated overlap-only ANN index, and ``'error'``
                rejects incomplete overlap. Harmonized mapping references also
                support ``'reference_mean'``, their neutral default.
                ``exclude_missing=True`` is retained as a deprecated alias for
                ``'intersection'``.
            query_batches: Optional query batch metadata for Symphony-style correction
                when mapping into a reusable harmonized reference.
        Returns:
            None
        """
        from ...mapping.coral import coral
        from ...mapping.features import align_features

        if not ref_mu or not ref_sigma:
            warnings.warn(
                "ref_mu and ref_sigma are deprecated and ignored. Mapping always "
                "uses the reference graph's mean and scale. These flags will be "
                "removed in Scarf 2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if run_coral:
            warnings.warn(
                "CORAL mapping is deprecated and will be removed in Scarf 2.0. "
                "Build a Symphony-style mapping reference instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if exclude_missing:
            warnings.warn(
                "exclude_missing is deprecated and will be removed in Scarf 2.0. "
                "Use missing_feature_policy='intersection' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        source_assay = self._get_assay(from_assay)

        if type(target_assay) is not type(source_assay):
            raise TypeError(
                f"ERROR: Source assay ({type(source_assay)}) and target assay "
                f"({type(target_assay)}) are of different types. "
                f"Mapping can only be performed between same assay types"
            )
        if isinstance(target_assay, RNAassay):
            if target_assay.sf != source_assay.sf:
                logger.debug(
                    f"Resetting target assay's size factor from {target_assay.sf} to {source_assay.sf}"
                )
                target_assay.sf = source_assay.sf
        self._guard_mapping_target_path(
            source_assay,
            target_assay,
            cell_key,
            feat_key,
            target_cell_key,
            target_feat_key,
        )

        artifact_state = read_assay_state(self.zw, from_assay)
        state_matches = artifact_state is not None and artifact_state.matches(
            cell_key,
            feat_key,
        )
        if (
            state_matches
            and artifact_state is not None
            and artifact_state.batch_correction is not None
            and "mapping_reference" not in artifact_state.named_results
        ):
            raise ValueError("AssayState has batch correction but no mapping reference")
        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        if not state_matches and normed_loc in self.zw:
            try:
                reduction_loc = lookup_latest_reduction_group_path(self.zw, normed_loc)
            except KeyError:
                reduction_loc = None
            if reduction_loc is not None and reduction_loc in self.zw:
                reduction_group = as_zarr_group(
                    self.zw[reduction_loc], name=reduction_loc
                )
                try:
                    ann_loc = lookup_latest_neighbor_index_group_path(
                        self.zw, reduction_loc
                    )
                except KeyError:
                    ann_loc = None
                if ann_loc is not None and ann_loc in self.zw:
                    ann_group = as_zarr_group(self.zw[ann_loc], name=ann_loc)
                    if bool(ann_group.attrs.get("isHarmonized", False)):
                        if run_coral:
                            raise ValueError(
                                "CORAL cannot be combined with a harmonized mapping reference"
                            )
                        if exclude_missing or missing_feature_policy == "intersection":
                            raise ValueError(
                                "Harmonized mapping references do not support intersection-only "
                                "feature mapping. Use reference_mean, zero, or error handling."
                            )
                        try:
                            reference = self.get_mapping_reference(
                                from_assay, cell_key, feat_key
                            )
                        except ValueError as exc:
                            if "predates the mapping-reference artifact" not in str(
                                exc
                            ):
                                raise
                            if self.zarr_mode != "r+":
                                raise ValueError(
                                    "This read-only harmonized reference needs a one-time "
                                    "upgrade. Reopen it with zarr_mode='r+' and call "
                                    "build_mapping_reference(..., batch_columns=[...])."
                                ) from exc
                            harmonized = as_zarr_array(
                                reduction_group["harmonizedData"],
                                name="harmonizedData",
                            )
                            batch_columns = cast(
                                list[str] | None, harmonized.attrs.get("batches")
                            )
                            if not batch_columns:
                                raise ValueError(
                                    "The legacy harmonized graph does not record its batch "
                                    "columns. Call build_mapping_reference with batch_columns."
                                ) from exc
                            warnings.warn(
                                "This harmonized graph predates portable mapping references. "
                                "It will be rebuilt once before mapping.",
                                DeprecationWarning,
                                stacklevel=2,
                            )
                            reference = self.build_mapping_reference(
                                from_assay,
                                cell_key,
                                feat_key,
                                batch_columns=batch_columns,
                            )
                        reference.map_query(
                            target_assay,
                            target_name,
                            target_feat_key,
                            target_cell_key=target_cell_key,
                            save_k=save_k,
                            query_batches=query_batches,
                            missing_feature_policy=(
                                missing_feature_policy or "reference_mean"
                            ),
                            invalidate_cache=invalidate_cache,
                        )
                        return None

        if (
            artifact_state is not None
            and artifact_state.matches(cell_key, feat_key)
            and artifact_state.batch_correction is not None
            and "mapping_reference" in artifact_state.named_results
        ):
            if run_coral:
                raise ValueError(
                    "CORAL cannot be combined with a harmonized mapping reference"
                )
            if exclude_missing or missing_feature_policy == "intersection":
                raise ValueError(
                    "Harmonized mapping references do not support "
                    "intersection-only feature mapping"
                )
            reference = self.get_mapping_reference(
                from_assay,
                cell_key,
                feat_key,
            )
            reference.map_query(
                target_assay,
                target_name,
                target_feat_key,
                target_cell_key=target_cell_key,
                save_k=save_k,
                query_batches=query_batches,
                missing_feature_policy=missing_feature_policy or "reference_mean",
                invalidate_cache=invalidate_cache,
            )
            return None

        if missing_feature_policy is None:
            state = read_assay_state(self.zw, from_assay)
            if state is not None and state.batch_correction is not None:
                missing_feature_policy = "reference_mean"
            else:
                missing_feature_policy = "intersection" if exclude_missing else "zero"
        if missing_feature_policy not in {
            "zero",
            "intersection",
            "error",
            "reference_mean",
        }:
            raise ValueError(
                "missing_feature_policy must be zero, intersection, error, "
                "or reference_mean"
            )
        if exclude_missing and missing_feature_policy != "intersection":
            raise ValueError(
                "exclude_missing=True is only compatible with missing_feature_policy='intersection'"
            )
        if run_coral and missing_feature_policy == "intersection":
            raise ValueError(
                "CORAL does not support intersection-only feature mapping. "
                "Use zero or error feature handling."
            )

        normalization_parameters = None
        state = read_assay_state(self.zw, from_assay)
        if (
            state is not None
            and state.matches(cell_key, feat_key)
            and state.normalized is not None
        ):
            normalization_parameters = (
                inspect_artifact(self.zw, state.normalized).parameters or {}
            )
        feat_idx = align_features(
            source_assay,
            target_assay,
            cell_key,
            feat_key,
            target_feat_key,
            target_cell_key,
            filter_null,
            exclude_missing,
            self.nthreads,
            missing_feature_policy,
            norm_params=normalization_parameters,
        )
        logger.debug(f"{len(feat_idx)} features being used for mapping")
        source_feature_indices = source_assay.feats.active_index(
            f"{cell_key}__{feat_key}" if feat_key != "I" else "I"
        )
        full_feature_overlap = len(source_feature_indices) == len(
            feat_idx
        ) and np.array_equal(source_feature_indices, feat_idx)
        ann_feat_key = feat_key
        ann_feat_scaling = feat_scaling and not isinstance(source_assay, ATACassay)
        if ann_feat_key == feat_key and self._has_ann_stream_cache(
            from_assay,
            cell_key,
            ann_feat_key,
            feat_scaling=ann_feat_scaling,
        ):
            ann_obj: AnnStream | None = self._load_ann_stream(
                from_assay,
                cell_key,
                ann_feat_key,
                feat_scaling=ann_feat_scaling,
            )
        else:
            normalized = self.run_normalization(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=ann_feat_key,
                update_state=False,
            )
            if isinstance(source_assay, ATACassay):
                reduction = self.run_lsi(
                    normalized,
                    update_state=False,
                )
            else:
                reduction = self.run_pca(
                    normalized,
                    feat_scaling=ann_feat_scaling,
                    update_state=False,
                )
            ann_index_ref = self.build_ann_index(
                reduction,
                update_state=False,
            )
            neighbors_ref = self.query_neighbors(
                ann_index_ref,
                coordinates=reduction,
                update_state=False,
            )
            ann_obj = self._load_artifact_ann_stream(
                from_assay,
                cell_key,
                ann_feat_key,
                ann_feat_scaling,
                neighbors_ref=neighbors_ref,
            )
        if ann_obj is None:
            raise ValueError("ERROR: AnnStream could not be created for mapping")
        if ann_obj.harmonize:
            if run_coral:
                raise ValueError(
                    "CORAL cannot be combined with a harmonized mapping reference"
                )
            if missing_feature_policy == "intersection":
                raise ValueError(
                    "Harmonized mapping references do not support "
                    "intersection-only feature mapping"
                )
            reference = self.get_mapping_reference(
                from_assay,
                cell_key,
                feat_key,
            )
            reference.map_query(
                target_assay,
                target_name,
                target_feat_key,
                target_cell_key=target_cell_key,
                save_k=save_k,
                query_batches=query_batches,
                missing_feature_policy=missing_feature_policy,
                invalidate_cache=invalidate_cache,
            )
            return None
        if missing_feature_policy == "intersection":
            if not full_feature_overlap:
                if exclude_missing:
                    compatibility_feat_key = f"{feat_key}_common_{target_name}"
                    feat_mask = np.zeros(source_assay.feats.N, dtype=bool)
                    feat_mask[feat_idx] = True
                    source_assay.feats.insert(
                        f"{cell_key}__{compatibility_feat_key}",
                        feat_mask,
                        fill_value=False,
                        overwrite=True,
                    )
                ann_obj = self._build_intersection_ann(
                    ann_obj,
                    source_assay,
                    cell_key,
                    feat_key,
                    feat_idx,
                )
        if save_k > ann_obj.k:
            logger.warning(f"`save_k` was decreased to {ann_obj.k}")
            save_k = ann_obj.k
        target_normed_path = make_normalized_leaf_name(target_cell_key, target_feat_key)
        target_data_path = f"{target_normed_path}/data"
        target_data = ChunkedArray(
            as_zarr_array(
                target_assay.z[target_data_path],
                name=target_data_path,
            ),
            nthreads=self.nthreads,
            resources=target_assay.resources,
        )
        if run_coral is True:
            # Reversing coral here to correct target data
            coral(
                target_data,
                ann_obj.data,
                target_assay,
                target_feat_key,
                target_cell_key,
                self.nthreads,
            )
            target_data = ChunkedArray(
                as_zarr_array(
                    target_assay.z[f"{target_normed_path}/data_coral"],
                    name=f"{target_normed_path}/data_coral",
                ),
                nthreads=self.nthreads,
                resources=target_assay.resources,
            )
        nc = target_assay.cells.active_index(target_cell_key).shape[0]
        nk = save_k
        correction_method = "coral" if run_coral else "none"
        if missing_feature_policy == "intersection":
            correction_method = "intersection"
        from ...mapping.hashing import array_hash

        target_matrix_fingerprint = self._fingerprint_mapping_matrix(target_data)
        ann_path = self._ann_stream_path(ann_obj)
        if ann_path is None:
            raise ValueError("ANN artifact path is unavailable")
        try:
            ann_input: object = parse_artifact_path(ann_path)
        except ValueError:
            ann_input = {
                "legacy_ann_fingerprint": fingerprint_array(
                    serialize_ann_index(ann_obj.annIdx)
                )
            }
        neighbors_path = self._ann_stream_neighbors_path(ann_obj)
        if neighbors_path is None:
            state = read_assay_state(self.zw, from_assay)
            if state is None or state.ann_index != ann_input or state.neighbors is None:
                raise ValueError("ANN stream has no exact neighbors provenance")
            neighbors_path = artifact_path(state.neighbors)
        reference_neighbors: object
        try:
            reference_neighbors = parse_artifact_path(neighbors_path)
        except ValueError:
            legacy_neighbors = as_zarr_group(
                self.zw[neighbors_path],
                name=neighbors_path,
            )
            reference_neighbors = {
                "legacy_neighbor_fingerprint": fingerprint_stored_arrays(
                    legacy_neighbors,
                    ("distances",),
                )
            }
        projection_plan = plan_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind="projection",
            operation="run_mapping",
            parameters={
                "save_k": save_k,
                "correction_method": correction_method,
                "missing_feature_policy": missing_feature_policy,
                "feat_scaling": feat_scaling,
            },
            inputs={
                "ann_index": ann_input,
                "reference_neighbors": reference_neighbors,
                "reference_features": array_hash(
                    source_assay.feats.fetch_all("ids")[feat_idx]
                ),
                "target_cells": array_hash(
                    target_assay.cells.fetch("ids", key=target_cell_key)
                ),
                "target_features": array_hash(target_assay.feats.fetch_all("ids")),
                "target_normalized": target_matrix_fingerprint,
            },
            execution_options={
                "target_name": target_name,
                "target_feat_key": target_feat_key,
                "target_cell_key": target_cell_key,
            },
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("indices", shape=(nc, nk), dtype_kind="u"),
                ArrayRequirement("distances", shape=(nc, nk), dtype_kind="f"),
                ArrayRequirement(
                    "reference_feature_indices",
                    shape=feat_idx.shape,
                    dtype_kind="i",
                ),
                ArrayRequirement(
                    "reference_distance_quantiles",
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "reference_distance_values",
                    dtype_kind="f",
                ),
            ),
            required_attributes=tuple(self._PROJECTION_PROVENANCE_ATTRS),
        )
        if projection_plan.reused:
            self._select_projection_artifact(
                from_assay,
                target_name,
                projection_plan.ref,
            )
            return None
        store = start_artifact(self.zw, projection_plan)
        row_chunk = min(max(int(target_data.chunksize[0]), 1), max(nc, 1))
        zi = create_zarr_dataset(store, "indices", (row_chunk,), "u8", (nc, nk))
        zd = create_zarr_dataset(store, "distances", (row_chunk,), "f8", (nc, nk))
        feature_index_store = create_zarr_dataset(
            store,
            "reference_feature_indices",
            (min(max(len(feat_idx), 1), 100_000),),
            "i8",
            feat_idx.shape,
        )
        feature_index_store[:] = feat_idx
        for key, value in self._projection_provenance(
            source_assay,
            target_assay,
            from_assay,
            target_name,
            cell_key,
            feat_key,
            target_cell_key,
            target_feat_key,
            feat_idx,
            correction_method,
            ann_obj,
        ).items():
            store.attrs[key] = value
        store.attrs["save_k"] = int(save_k)
        from ...mapping.confidence import _distance_quantile_summary

        validate_distance_provenance(self.zw, neighbors_path)
        neighbors_group = as_zarr_group(
            self.zw[neighbors_path],
            name=neighbors_path,
        )
        if (
            "reference_distance_quantiles" in neighbors_group
            and "reference_distance_values" in neighbors_group
        ):
            reference_quantiles = np.asarray(
                as_zarr_array(
                    neighbors_group["reference_distance_quantiles"],
                    name="reference_distance_quantiles",
                )[:]
            )
            reference_values = np.asarray(
                as_zarr_array(
                    neighbors_group["reference_distance_values"],
                    name="reference_distance_values",
                )[:]
            )
        else:
            reference_quantiles, reference_values = _distance_quantile_summary(
                as_zarr_array(
                    neighbors_group["distances"],
                    name="distances",
                )
            )
        for name, values in (
            ("reference_distance_quantiles", reference_quantiles),
            ("reference_distance_values", reference_values),
        ):
            output = create_zarr_dataset(
                store,
                name,
                (min(max(len(values), 1), 1_001),),
                "f8",
                values.shape,
            )
            output[:] = values
        entry_start = 0
        try:
            for block in target_data.stream_blocks(
                nthreads=self.nthreads,
                msg=f"Mapping cells from {target_name}",
            ):
                knn_query = ann_obj.transform_ann(
                    ann_obj.transform_query(block), k=save_k
                )
                ki, kd = knn_query[0], knn_query[1]
                entry_end = entry_start + len(ki)
                zi[entry_start:entry_end, :] = ki
                zd[entry_start:entry_end, :] = kd
                entry_start = entry_end
            if entry_start != nc:
                raise RuntimeError(
                    f"Mapped {entry_start} target cells but expected {nc}"
                )
            finish_artifact(store, projection_plan)
            self._select_projection_artifact(
                from_assay,
                target_name,
                projection_plan.ref,
            )
        except Exception:
            store.attrs["complete"] = False
            raise
        return None

    def _load_mapping_reference_ann(
        self,
        reference: MappingReference,
    ) -> AnnStream:
        if reference.ann_path not in self.zw:
            raise ValueError(
                "The mapping reference ANN index is missing. Rebuild the reference."
            )
        reference_ann_group = as_zarr_group(
            self.zw[reference.ann_path],
            name=reference.ann_path,
        )
        try:
            parse_artifact_path(reference.ann_path)
        except ValueError:
            feature_scaling = bool(
                reference_ann_group.attrs.get("featureScaling", True)
            )
            try:
                reference_knn_path = lookup_latest_nearest_neighbors_group_path(
                    self.zw,
                    reference.ann_path,
                )
            except KeyError as exc:
                raise ValueError(
                    "The mapping reference KNN metadata is missing. "
                    "Rebuild the reference."
                ) from exc
        else:
            mapping_inputs = reference.metadata.get("artifact_inputs", {})
            if not isinstance(mapping_inputs, dict):
                raise ValueError("Mapping reference artifact inputs are missing")
            raw_reduction = mapping_inputs.get("reduction")
            raw_neighbors = mapping_inputs.get("neighbors")
            if not isinstance(raw_reduction, dict) or not isinstance(
                raw_neighbors,
                dict,
            ):
                raise ValueError("Mapping reference graph inputs are incomplete")
            reference_reduction = ArtifactRef.from_dict(raw_reduction)
            neighbors_ref = ArtifactRef.from_dict(raw_neighbors)
            reduction_parameters = (
                inspect_artifact(self.zw, reference_reduction).parameters or {}
            )
            feature_scaling = bool(reduction_parameters.get("feat_scaling", True))
            neighbors_status = inspect_artifact(self.zw, neighbors_ref)
            if not neighbors_status.exists or not neighbors_status.complete:
                raise ValueError("Mapping reference neighbors are incomplete")
            reference_knn_path = artifact_path(neighbors_ref)
        if not feature_scaling:
            raise ValueError(
                "The mapping reference ANN was built without feature scaling "
                "and cannot use reference-scaled query projection."
            )
        return self._load_ann_stream(
            reference.assay_name,
            reference.cell_key,
            reference.feature_key,
            feat_scaling=True,
            knn_loc=reference_knn_path,
        )

    def _map_with_mapping_reference(
        self,
        reference: MappingReference,
        target_assay: Assay,
        target_name: str,
        target_feat_key: str,
        target_cell_key: str,
        save_k: int,
        query_batches: pd.DataFrame | None,
        correction_method: str,
        missing_feature_policy: str,
        result_store: zarr.Group | None = None,
        invalidate_cache: bool = False,
    ) -> MappingResult:
        from ...mapping.features import _order_features, align_features
        from ...mapping.hashing import array_hash
        from ...mapping.symphony import (
            accumulate_sufficient_statistics,
            apply_query_correction,
            initialize_sufficient_statistics,
            project_pca,
            soft_cluster_assignments,
            solve_query_correction,
            zero_norm_rows,
        )

        if type(target_assay) is not type(self._get_assay(reference.assay_name)):
            raise TypeError("Reference and query assays must have the same type")
        if correction_method != "symphony":
            raise ValueError(
                "Harmonized mapping references require correction_method='symphony'"
            )
        if missing_feature_policy not in {"reference_mean", "zero", "error"}:
            raise ValueError(
                "Mapping references support reference_mean, zero, or error feature handling"
            )
        source_assay = self._get_assay(reference.assay_name)
        if target_assay.sf != source_assay.sf and isinstance(target_assay, RNAassay):
            logger.debug(
                f"Resetting target assay's size factor from {target_assay.sf} to {source_assay.sf}"
            )
            target_assay.sf = source_assay.sf
        target_size_factor = target_assay.sf
        if target_size_factor is None:
            raise ValueError("Target assay has no normalization size factor")
        self._guard_mapping_target_path(
            source_assay,
            target_assay,
            reference.cell_key,
            reference.feature_key,
            target_cell_key,
            target_feat_key,
        )
        stored_normalization = reference.metadata.get("normalization_parameters")
        normalization_parameters = (
            dict(stored_normalization)
            if isinstance(stored_normalization, dict)
            else None
        )
        if normalization_parameters is None:
            state = read_assay_state(self.zw, reference.assay_name)
            if (
                state is not None
                and state.matches(reference.cell_key, reference.feature_key)
                and state.normalized is not None
            ):
                normalization_parameters = (
                    inspect_artifact(self.zw, state.normalized).parameters or {}
                )
        source_feature_key = (
            reference.feature_key
            if reference.feature_key == "I"
            else f"{reference.cell_key}__{reference.feature_key}"
        )
        source_feature_ids = source_assay.feats.fetch(
            "ids",
            key=source_feature_key,
        )
        feature_indices, target_feature_indices = _order_features(
            source_assay,
            target_assay,
            source_feature_ids,
            filter_null=False,
            nthreads=self.nthreads,
            missing_feature_policy=missing_feature_policy,
            target_cell_key=target_cell_key,
        )
        source_feature_indices = source_assay.feats.active_index(source_feature_key)
        if not np.array_equal(feature_indices, source_feature_indices):
            raise ValueError(
                "The mapping reference requires its complete reference feature set"
            )
        if not np.array_equal(source_feature_ids, reference.feature_ids):
            raise ValueError(
                "Reference feature identifiers no longer match the immutable artifact"
            )
        target_cell_indices = target_assay.cells.active_index(target_cell_key)
        n_cells = len(target_cell_indices)
        batch_codes, n_batches = self._query_batch_codes(query_batches, n_cells)
        query_batch_columns = (
            [str(column) for column in query_batches.columns]
            if query_batches is not None
            else []
        )
        query_batch_fingerprint = array_hash(batch_codes)
        present_target_features = np.sort(
            target_feature_indices[target_feature_indices >= 0]
        )
        target_source = target_assay.rawData[:, present_target_features][
            target_cell_indices,
            :,
        ]
        target_source_fingerprint = self._fingerprint_mapping_matrix(target_source)
        ann_obj = self._load_mapping_reference_ann(reference)
        if not ann_obj.harmonize:
            raise RuntimeError("Mapping reference ANN index is not harmonized")
        if save_k > ann_obj.k:
            logger.warning(f"`save_k` was decreased to {ann_obj.k}")
            save_k = ann_obj.k
        projection_plan = None
        if self.zarr_mode == "r+" and result_store is None:
            try:
                reference_input: object = parse_artifact_path(reference.artifact_path)
            except ValueError:
                from ...mapping.artifact import mapping_reference_hash

                reference_input = {
                    "legacy_mapping_reference_fingerprint": mapping_reference_hash(
                        reference.model,
                        reference.feature_ids,
                        reference.metadata,
                        reference.reference_distance_quantiles,
                        reference.reference_distance_values,
                    )
                }
            projection_plan = plan_artifact(
                self.zw,
                scope="assay",
                assay=reference.assay_name,
                kind="projection",
                operation="map_with_reference",
                parameters={
                    "save_k": save_k,
                    "correction_method": correction_method,
                    "missing_feature_policy": missing_feature_policy,
                },
                inputs={
                    "mapping_reference": reference_input,
                    "target_cells": array_hash(
                        target_assay.cells.fetch("ids", key=target_cell_key)
                    ),
                    "target_features": array_hash(target_assay.feats.fetch_all("ids")),
                    "target_expression": target_source_fingerprint,
                    "target_feature_alignment": array_hash(target_feature_indices),
                    "target_size_factor": float(target_size_factor),
                    "normalization_parameters": normalization_parameters,
                    "query_batches": query_batch_fingerprint,
                },
                execution_options={
                    "target_name": target_name,
                    "target_feat_key": target_feat_key,
                    "target_cell_key": target_cell_key,
                },
                invalidate_cache=invalidate_cache,
                required_arrays=(
                    ArrayRequirement(
                        "indices",
                        shape=(n_cells, save_k),
                        dtype_kind="u",
                    ),
                    ArrayRequirement(
                        "distances",
                        shape=(n_cells, save_k),
                        dtype_kind="f",
                    ),
                    ArrayRequirement(
                        "uncorrected_latent",
                        shape=(n_cells, reference.model.n_dims),
                        dtype_kind="f",
                    ),
                    ArrayRequirement(
                        "corrected_latent",
                        shape=(n_cells, reference.model.n_dims),
                        dtype_kind="f",
                    ),
                    ArrayRequirement(
                        "uninformative",
                        shape=(n_cells,),
                        dtype_kind="b",
                    ),
                    ArrayRequirement(
                        "reference_feature_indices",
                        shape=feature_indices.shape,
                        dtype_kind="i",
                    ),
                    ArrayRequirement(
                        "reference_distance_quantiles",
                        dtype_kind="f",
                    ),
                    ArrayRequirement(
                        "reference_distance_values",
                        dtype_kind="f",
                    ),
                ),
                required_attributes=tuple(self._PROJECTION_PROVENANCE_ATTRS),
            )
            if projection_plan.reused:
                self._select_projection_artifact(
                    reference.assay_name,
                    target_name,
                    projection_plan.ref,
                )
                return self.get_mapping_result(target_name)
        aligned_feature_indices = align_features(
            source_assay,
            target_assay,
            reference.cell_key,
            reference.feature_key,
            target_feat_key,
            target_cell_key,
            filter_null=False,
            exclude_missing=False,
            nthreads=self.nthreads,
            missing_feature_policy=missing_feature_policy,
            missing_feature_values=(
                reference.model.feature_means
                if missing_feature_policy == "reference_mean"
                else None
            ),
            norm_params=normalization_parameters,
        )
        if not np.array_equal(aligned_feature_indices, feature_indices):
            raise RuntimeError("Target feature alignment changed during mapping")
        target_normed_path = make_normalized_leaf_name(
            target_cell_key,
            target_feat_key,
        )
        target_data_path = f"{target_normed_path}/data"
        target_data = ChunkedArray(
            as_zarr_array(
                target_assay.z[target_data_path],
                name=target_data_path,
            ),
            nthreads=self.nthreads,
            resources=target_assay.resources,
        )
        if target_data.shape[0] != n_cells:
            raise RuntimeError("Aligned target matrix has an unexpected cell count")
        counts, sums = initialize_sufficient_statistics(n_batches, reference.model)
        entry_start = 0
        zero_norm_count = 0
        for block in target_data.blocks:
            values = controlled_compute(block, self.nthreads)
            coordinates = project_pca(values, reference.model)
            assignments = soft_cluster_assignments(coordinates, reference.model)
            uninformative_rows = zero_norm_rows(coordinates)
            zero_norm_count += int(uninformative_rows.sum())
            entry_end = entry_start + len(values)
            if not np.all(uninformative_rows):
                informative_rows = ~uninformative_rows
                accumulate_sufficient_statistics(
                    counts,
                    sums,
                    coordinates[informative_rows],
                    assignments[informative_rows],
                    batch_codes[entry_start:entry_end][informative_rows],
                )
            entry_start = entry_end
        if entry_start != n_cells:
            raise RuntimeError(
                f"Read {entry_start} query cells but expected {n_cells} during correction"
            )
        correction = solve_query_correction(counts, sums, reference.model)
        write_projection = self.zarr_mode == "r+" or result_store is not None
        store: zarr.Group | None = result_store
        projection_path = ""
        feature_coverage = float(
            np.isin(
                reference.feature_ids,
                target_assay.feats.fetch_all("ids"),
            ).sum()
            / len(reference.feature_ids)
        )
        if write_projection:
            if store is None:
                if projection_plan is None:
                    raise RuntimeError("Projection artifact plan is missing")
                store = start_artifact(self.zw, projection_plan)
                projection_path = artifact_path(projection_plan.ref)
            else:
                projection_path = getattr(store, "path", "")
            store.attrs["complete"] = False
            for key, value in self._projection_provenance(
                source_assay,
                target_assay,
                reference.assay_name,
                target_name,
                reference.cell_key,
                reference.feature_key,
                target_cell_key,
                target_feat_key,
                feature_indices,
                "symphony",
                ann_obj,
            ).items():
                store.attrs[key] = value
            store.attrs["save_k"] = int(save_k)
            store.attrs["mapping_reference_path"] = reference.artifact_path
            store.attrs["query_batch_count"] = int(n_batches)
            store.attrs["query_batch_columns"] = query_batch_columns
            store.attrs["query_batch_fingerprint"] = query_batch_fingerprint
            reference_quantiles, reference_values = self._reference_distance_summary(
                store,
                reference.assay_name,
                reference.cell_key,
                reference.feature_key,
            )
            for name, values in (
                ("reference_distance_quantiles", reference_quantiles),
                ("reference_distance_values", reference_values),
            ):
                output = create_zarr_dataset(
                    store,
                    name,
                    (min(max(len(values), 1), 1_001),),
                    "f8",
                    values.shape,
                )
                output[:] = values
            feature_index_store = create_zarr_dataset(
                store,
                "reference_feature_indices",
                (min(max(len(feature_indices), 1), 100_000),),
                "i8",
                feature_indices.shape,
            )
            feature_index_store[:] = feature_indices
            row_chunk = min(max(int(target_data.chunksize[0]), 1), max(n_cells, 1))
            indices: Any = create_zarr_dataset(
                store, "indices", (row_chunk, save_k), "u8", (n_cells, save_k)
            )
            distances: Any = create_zarr_dataset(
                store, "distances", (row_chunk, save_k), "f8", (n_cells, save_k)
            )
            uncorrected: Any = create_zarr_dataset(
                store,
                "uncorrected_latent",
                (row_chunk, reference.model.n_dims),
                "f8",
                (n_cells, reference.model.n_dims),
            )
            corrected: Any = create_zarr_dataset(
                store,
                "corrected_latent",
                (row_chunk, reference.model.n_dims),
                "f8",
                (n_cells, reference.model.n_dims),
            )
            uninformative: Any = create_zarr_dataset(
                store,
                "uninformative",
                (row_chunk,),
                "bool",
                (n_cells,),
            )
        else:
            indices = np.empty((n_cells, save_k), dtype=np.uint64)
            distances = np.empty((n_cells, save_k), dtype=np.float64)
            uncorrected = np.empty((n_cells, reference.model.n_dims), dtype=np.float64)
            corrected = np.empty((n_cells, reference.model.n_dims), dtype=np.float64)
            uninformative = np.empty(n_cells, dtype=bool)
        entry_start = 0
        try:
            for values in target_data.stream_blocks(
                nthreads=self.nthreads,
                msg=(
                    f"Mapping cells from {target_name} "
                    "with Symphony-style correction"
                ),
            ):
                coordinates = project_pca(values, reference.model)
                assignments = soft_cluster_assignments(coordinates, reference.model)
                uninformative_rows = zero_norm_rows(coordinates)
                entry_end = entry_start + len(values)
                corrected_coordinates = apply_query_correction(
                    coordinates,
                    assignments,
                    batch_codes[entry_start:entry_end],
                    reference.model,
                    correction,
                )
                corrected_coordinates[uninformative_rows] = coordinates[
                    uninformative_rows
                ]
                knn_query = ann_obj.transform_ann(corrected_coordinates, k=save_k)
                neighbor_indices, neighbor_distances = cast(
                    tuple[np.ndarray, np.ndarray], knn_query
                )
                indices[entry_start:entry_end] = neighbor_indices
                distances[entry_start:entry_end] = neighbor_distances
                uncorrected[entry_start:entry_end] = coordinates
                corrected[entry_start:entry_end] = corrected_coordinates
                uninformative[entry_start:entry_end] = uninformative_rows
                entry_start = entry_end
            if entry_start != n_cells:
                raise RuntimeError(
                    f"Mapped {entry_start} query cells but expected {n_cells}"
                )
            if store is not None:
                if projection_plan is not None and result_store is None:
                    finish_artifact(store, projection_plan)
                    self._select_projection_artifact(
                        reference.assay_name,
                        target_name,
                        projection_plan.ref,
                    )
                else:
                    store.attrs["complete"] = True
        except Exception:
            if store is not None:
                store.attrs["complete"] = False
            raise
        return MappingResult(
            projection_path=projection_path,
            n_cells=n_cells,
            correction_method="symphony",
            diagnostics={
                "featureCoverage": feature_coverage,
                "queryBatchCount": float(n_batches),
                "zeroNormCellCount": float(zero_norm_count),
                "algorithmVariant": SYMPHONY_ALGORITHM,
            },
            indices=None if store is not None else indices,
            distances=None if store is not None else distances,
            uncorrected_latent=None if store is not None else uncorrected,
            corrected_latent=None if store is not None else corrected,
            uninformative=None if store is not None else uninformative,
        )

    @staticmethod
    def _query_batch_codes(
        query_batches: pd.DataFrame | None, n_cells: int
    ) -> tuple[np.ndarray, int]:
        if query_batches is None:
            return np.zeros(n_cells, dtype=np.int64), 1
        if len(query_batches) != n_cells:
            raise ValueError("query_batches must have one row per target cell")
        if query_batches.shape[1] == 0:
            raise ValueError("query_batches must include at least one column")
        if query_batches.columns.duplicated().any():
            raise ValueError("query_batches column names must be unique")
        if query_batches.isna().any().any():
            raise ValueError("query_batches cannot contain missing values")
        labels = query_batches.astype(str).agg("\x1f".join, axis=1)
        codes, _ = pd.factorize(labels, sort=True)
        return np.asarray(codes, dtype=np.int64), int(codes.max()) + 1

    @staticmethod
    def _label_vote_decision(
        reference_labels: np.ndarray,
        neighbors: np.ndarray,
        weights: np.ndarray,
        threshold_fraction: float,
        na_val: str,
        force_unknown: bool = False,
    ) -> tuple[Any, float, float, float, bool, dict[Any, float]]:
        votes: dict[Any, float] = {}
        for neighbor, weight in zip(neighbors, weights):
            label = reference_labels[neighbor]
            votes[label] = votes.get(label, 0.0) + float(weight)
        total = float(sum(votes.values()))
        if total <= 0 or not votes:
            return na_val, 0.0, 0.0, 0.0, True, {}
        votes = {label: value / total for label, value in votes.items()}
        ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        top_vote = ordered[0][1]
        second_vote = ordered[1][1] if len(ordered) > 1 else 0.0
        winners = [label for label, vote in ordered if np.isclose(vote, top_vote)]
        is_unknown = force_unknown or top_vote < threshold_fraction or len(winners) != 1
        entropy = -sum(value * np.log(value) for _, value in ordered if value > 0)
        prediction = na_val if is_unknown else winners[0]
        return (
            prediction,
            top_vote,
            float(entropy),
            top_vote - second_vote,
            is_unknown,
            votes,
        )

    def get_mapping_result(
        self,
        target_name: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        *,
        load_arrays: bool = False,
    ) -> MappingResult:
        """Load a persisted mapping projection as a ``MappingResult``.

        By default only metadata is returned and array fields are ``None``.
        Set ``load_arrays=True`` to materialize neighbor and latent arrays that
        exist on the projection.
        """
        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        store = self._load_complete_projection(
            target_name, from_assay, cell_key, feat_key
        )
        projection_path = str(getattr(store, "path", ""))
        indices = as_zarr_array(store["indices"], name="indices")
        n_cells = int(indices.shape[0])
        correction_method = str(store.attrs.get("correction_method", "none"))
        diagnostics: dict[str, float | str] = {}
        for output_key, attribute_name in (
            ("featureCoverage", "feature_coverage"),
            ("queryBatchCount", "query_batch_count"),
            ("algorithmVariant", "algorithm_variant"),
        ):
            value = store.attrs.get(attribute_name)
            if value is not None:
                if isinstance(value, (bool, np.bool_)):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating, str)):
                    diagnostics[output_key] = (
                        value if isinstance(value, str) else float(value)
                    )

        if not load_arrays:
            return MappingResult(
                projection_path=projection_path,
                n_cells=n_cells,
                correction_method=correction_method,
                diagnostics=diagnostics,
            )

        def _optional_array(name: str) -> np.ndarray | None:
            if name not in store:
                return None
            return np.asarray(as_zarr_array(store[name], name=name)[:])

        return MappingResult(
            projection_path=projection_path,
            n_cells=n_cells,
            correction_method=correction_method,
            diagnostics=diagnostics,
            indices=np.asarray(indices[:]),
            distances=np.asarray(
                as_zarr_array(store["distances"], name="distances")[:]
            ),
            uncorrected_latent=_optional_array("uncorrected_latent"),
            corrected_latent=_optional_array("corrected_latent"),
            uninformative=_optional_array("uninformative"),
        )

    def get_mapping_score(
        self,
        target_name: str,
        target_groups: np.ndarray | None = None,
        from_assay: str | None = None,
        cell_key: str | None = None,
        log_transform: bool = True,
        multiplier: float = 1000,
        weighted: bool = True,
        fixed_weight: float = 0.1,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yields the mapping scores that were a result of a mapping.

        Mapping scores indicate how strongly target cells map to each reference
        cell. By default, the score accumulates distance-derived neighbor
        weights. Frequently selected, nearby reference cells receive higher
        scores.

        Args:
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            target_groups: Group/cluster identity of target cells. This will then be used to calculate mapping score
                           for each group separately.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            log_transform: If True (default) then the mapping scores will be log transformed
            multiplier: Scaling factor applied to mapping scores, primarily for
                visualization. (Default: 1000)
            weighted: Use distance weights when calculating mapping scores (default: True). If False then the actual
                      distances between the reference and target cells are ignored.
            fixed_weight: Used when `weighted` is False. This is the value that is added to mapping score of each
                          reference cell for every projected target cell. Can be any value >0.

        Yields:
            A tuple of group name and mapping score of reference cells for that target group.
        """
        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, None
        )
        store = self._load_complete_projection(
            target_name, from_assay, cell_key, feat_key
        )
        indices = as_zarr_array(store["indices"], name="indices")
        distances = as_zarr_array(store["distances"], name="distances")
        n_cells, n_k = indices.shape

        if target_groups is not None:
            if len(target_groups) != n_cells:
                raise ValueError(
                    f"ERROR: Length of target_groups {len(target_groups)} not same as number of target "
                    f"cells in the projection {n_cells}"
                )
            groups = pd.Series(target_groups)
        else:
            groups = pd.Series(np.zeros(n_cells))

        ref_n_cells = self.cells.active_index(cell_key).shape[0]
        for group in sorted(groups.unique()):
            ms = np.zeros(ref_n_cells)
            group_count = 0
            block_size = self._projection_block_size(indices)
            for start in range(0, n_cells, block_size):
                stop = min(start + block_size, n_cells)
                block_groups = groups.iloc[start:stop].to_numpy()
                mask = block_groups == group
                if not mask.any():
                    continue
                block_indices = np.asarray(indices[start:stop])[mask]
                if weighted:
                    from ...mapping.confidence import distance_weights

                    block_weights = distance_weights(
                        np.asarray(distances[start:stop])[mask]
                    )
                else:
                    block_weights = np.full(
                        block_indices.shape, fixed_weight, dtype=np.float64
                    )
                np.add.at(ms, block_indices.reshape(-1), block_weights.reshape(-1))
                group_count += int(mask.sum())
            if group_count == 0:
                continue
            ms = multiplier * ms / (group_count * n_k)
            if log_transform:
                ms = np.log1p(ms)
            yield group, ms

    def get_target_classes(
        self,
        target_name: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        reference_class_group: str | None = None,
        threshold_fraction: float = 0.5,
        target_subset: list[int] | None = None,
        na_val: str = "NA",
    ) -> pd.Series:
        """Perform classification of target cells using a reference group.

        Args:
            target_name: Name of target data. This value should be the same as that used for `run_mapping` earlier.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            reference_class_group: Group/cluster identity of the reference cells. These are the target labels for the
                                   classifier. The value here should be a column from cell metadata table. For
                                   example, to use default clustering identity one could use `RNA_cluster`
            threshold_fraction: The threshold for deciding if a cell belongs to a group or not.
                                Constrained between 0 and 1. (Default value: 0.5)
            target_subset: Choose only a subset of target cells to be classified. The value should be a list of
                           indices of the target cells. (Default: None)
            na_val: Value to be used if a cell is not classified to any of the `reference_class_group`.
                    (Default value: 'NA')

        Returns: A pandas Series containing predicted class for each cell in the projected sample (`target_name`).
        """
        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, None
        )
        if reference_class_group is None:
            raise ValueError(
                "ERROR: A value is required for the parameter `reference_class_group`. "
                "This can be any cell metadata column. Please choose the value that contains cluster or "
                "group information"
            )
        ref_groups = self.cells.fetch(reference_class_group, key=cell_key)
        if threshold_fraction < 0 or threshold_fraction > 1:
            raise ValueError(
                "ERROR: `threshold_fraction` should have a value between 0 and 1"
            )
        target_subset_set: dict[int, None] | None = None
        if target_subset is not None:
            if not isinstance(target_subset, list):
                raise TypeError("ERROR:  `target_subset` should be <list> type")
            target_subset_set = {x: None for x in target_subset}

        store = self._load_complete_projection(
            target_name, from_assay, cell_key, feat_key
        )
        preds: list[Any] = []
        prediction_indices: list[int] = []
        for row in self._iter_projection_neighbor_rows(store):
            n, neighbors, weights, _, force_unknown = row
            if target_subset_set is not None and n not in target_subset_set:
                continue
            prediction, _, _, _, _, _ = self._label_vote_decision(
                ref_groups,
                neighbors,
                weights,
                threshold_fraction,
                na_val,
                force_unknown=force_unknown,
            )
            preds.append(prediction)
            prediction_indices.append(n)
        return pd.Series(preds, index=prediction_indices)

    def get_target_label_evidence(
        self,
        target_name: str,
        reference_class_group: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        threshold_fraction: float = 0.5,
        na_val: str = "NA",
        max_distance: float | None = None,
        calibration_nonconformity: np.ndarray | None = None,
        conformal_alpha: float = 0.1,
    ) -> pd.DataFrame:
        """Return neighbor-vote evidence, novelty context, and unknown assignments.

        ``calibration_nonconformity`` optionally adds split-conformal prediction
        sets. Its calibration rows must be exchangeable with future queries.
        """
        if not 0 <= threshold_fraction <= 1:
            raise ValueError("threshold_fraction must be between zero and one")
        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, None
        )
        if reference_class_group not in self.cells.columns:
            raise KeyError(
                f"Reference label column {reference_class_group!r} was not found"
            )
        store = self._load_complete_projection(
            target_name, from_assay, cell_key, feat_key
        )
        reference_labels = self.cells.fetch(reference_class_group, key=cell_key)
        class_labels = np.asarray(pd.unique(reference_labels), dtype=object)
        class_positions = {
            label: position for position, label in enumerate(class_labels)
        }

        predictions: list[Any] = []
        vote_fraction: list[float] = []
        vote_entropy: list[float] = []
        top_two_margin: list[float] = []
        best_distances: list[float] = []
        label_scores: list[np.ndarray] = []
        for row in self._iter_projection_neighbor_rows(store):
            _, neighbors, weights, row_distances, force_unknown = row
            (
                prediction,
                top_vote,
                entropy,
                margin,
                _,
                votes,
            ) = self._label_vote_decision(
                reference_labels,
                neighbors,
                weights,
                threshold_fraction,
                na_val,
                force_unknown=force_unknown,
            )
            if max_distance is not None and row_distances[0] > max_distance:
                prediction = na_val
            predictions.append(prediction)
            vote_fraction.append(top_vote)
            vote_entropy.append(entropy)
            top_two_margin.append(margin)
            best_distances.append(float(row_distances[0]))
            score_row = np.zeros(len(class_labels), dtype=np.float64)
            for label, score in votes.items():
                score_row[class_positions[label]] = score
            label_scores.append(score_row)

        best = np.asarray(best_distances)
        distance_quantiles, distance_values = self._reference_distance_summary(
            store, from_assay, cell_key, feat_key
        )
        unique_distance_values = np.unique(distance_values)
        right_indices = (
            np.searchsorted(distance_values, unique_distance_values, side="right") - 1
        )
        unique_distance_quantiles = distance_quantiles[right_indices]
        distance_percentile = np.interp(
            best,
            unique_distance_values,
            unique_distance_quantiles,
            left=0.0,
            right=1.0,
        )
        feature_coverage_value = store.attrs.get("feature_coverage", 1.0)
        if not isinstance(feature_coverage_value, int | float):
            raise RuntimeError(
                "Projection provenance is missing numeric feature coverage"
            )
        feature_coverage = float(feature_coverage_value)
        result = pd.DataFrame(
            {
                "label": predictions,
                "voteFraction": vote_fraction,
                "voteEntropy": vote_entropy,
                "topTwoMargin": top_two_margin,
                "featureCoverage": feature_coverage,
                "referenceDistancePercentile": distance_percentile,
                "isUnknown": np.asarray(predictions) == na_val,
            }
        )
        if calibration_nonconformity is not None:
            from ...mapping.confidence import conformal_prediction_sets

            prediction_masks = conformal_prediction_sets(
                np.vstack(label_scores),
                calibration_nonconformity,
                alpha=conformal_alpha,
            )
            result["predictionSet"] = [
                tuple(class_labels[mask].tolist()) for mask in prediction_masks
            ]
        return result

    def _reference_distance_summary(
        self,
        projection: zarr.Group,
        from_assay: str,
        cell_key: str,
        feat_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        from ...mapping.confidence import _distance_quantile_summary

        if (
            "reference_distance_quantiles" in projection
            and "reference_distance_values" in projection
        ):
            return (
                np.asarray(
                    as_zarr_array(
                        projection["reference_distance_quantiles"],
                        name="reference_distance_quantiles",
                    )[:]
                ),
                np.asarray(
                    as_zarr_array(
                        projection["reference_distance_values"],
                        name="reference_distance_values",
                    )[:]
                ),
            )
        mapping_path = projection.attrs.get("mapping_reference_path")
        if isinstance(mapping_path, str) and mapping_path in self.zw:
            artifact = as_zarr_group(self.zw[mapping_path], name=mapping_path)
            if (
                "reference_distance_quantiles" in artifact
                and "reference_distance_values" in artifact
            ):
                return (
                    np.asarray(
                        as_zarr_array(
                            artifact["reference_distance_quantiles"],
                            name="reference_distance_quantiles",
                        )[:]
                    ),
                    np.asarray(
                        as_zarr_array(
                            artifact["reference_distance_values"],
                            name="reference_distance_values",
                        )[:]
                    ),
                )
            if (
                "referenceDistanceQuantiles" in artifact
                and "referenceDistanceValues" in artifact
            ):
                return (
                    np.asarray(
                        as_zarr_array(
                            artifact["referenceDistanceQuantiles"],
                            name="referenceDistanceQuantiles",
                        )[:]
                    ),
                    np.asarray(
                        as_zarr_array(
                            artifact["referenceDistanceValues"],
                            name="referenceDistanceValues",
                        )[:]
                    ),
                )

        ann_path = projection.attrs.get("ann_path")
        if not isinstance(ann_path, str) or ann_path not in self.zw:
            stored_assay = projection.attrs.get("assay")
            stored_cell_key = projection.attrs.get("cell_key")
            stored_feature_key = projection.attrs.get("feature_key")
            if isinstance(stored_assay, str):
                from_assay = stored_assay
            if isinstance(stored_cell_key, str):
                cell_key = stored_cell_key
            if isinstance(stored_feature_key, str):
                feat_key = stored_feature_key
            state = read_assay_state(self.zw, from_assay)
            if (
                state is not None
                and state.matches(cell_key, feat_key)
                and state.neighbors is not None
            ):
                validate_distance_provenance(self.zw, artifact_path(state.neighbors))
                neighbors_group = as_zarr_group(
                    self.zw[artifact_path(state.neighbors)],
                    name=artifact_path(state.neighbors),
                )
                return _distance_quantile_summary(
                    as_zarr_array(
                        neighbors_group["distances"],
                        name="distances",
                    )
                )
            normed_path = make_normalized_group_path(from_assay, cell_key, feat_key)
            reduction_path = lookup_latest_reduction_group_path(self.zw, normed_path)
            ann_path = lookup_latest_neighbor_index_group_path(self.zw, reduction_path)
        ann = as_zarr_group(self.zw[ann_path], name=ann_path)
        if "reference_distance_quantiles" in ann and "reference_distance_values" in ann:
            return (
                np.asarray(
                    as_zarr_array(
                        ann["reference_distance_quantiles"],
                        name="reference_distance_quantiles",
                    )[:]
                ),
                np.asarray(
                    as_zarr_array(
                        ann["reference_distance_values"],
                        name="reference_distance_values",
                    )[:]
                ),
            )
        if "referenceDistanceQuantiles" in ann and "referenceDistanceValues" in ann:
            return (
                np.asarray(
                    as_zarr_array(
                        ann["referenceDistanceQuantiles"],
                        name="referenceDistanceQuantiles",
                    )[:]
                ),
                np.asarray(
                    as_zarr_array(
                        ann["referenceDistanceValues"],
                        name="referenceDistanceValues",
                    )[:]
                ),
            )
        try:
            parse_artifact_path(ann_path)
        except ValueError:
            knn_path = lookup_latest_nearest_neighbors_group_path(self.zw, ann_path)
        else:
            state = read_assay_state(self.zw, from_assay)
            if state is None or state.neighbors is None:
                raise ValueError("Artifact graph state has no neighbors artifact")
            knn_path = artifact_path(state.neighbors)
        validate_distance_provenance(self.zw, knn_path)
        knn = as_zarr_group(self.zw[knn_path], name=knn_path)
        reference_distances = as_zarr_array(knn["distances"], name="referenceDistances")
        return _distance_quantile_summary(reference_distances)

    @staticmethod
    def calibrate_label_transfer_threshold(
        vote_fractions: np.ndarray,
        correct: np.ndarray,
        target_coverage: float = 0.9,
    ) -> dict[str, float]:
        """Choose a vote threshold on held-out, donor-level validation data."""
        fractions = np.asarray(vote_fractions, dtype=np.float64)
        correct = np.asarray(correct, dtype=bool)
        if fractions.ndim != 1 or correct.shape != fractions.shape:
            raise ValueError("vote_fractions and correct must be matching vectors")
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must be in (0, 1]")
        valid = fractions[correct]
        if valid.size == 0:
            raise ValueError("At least one correct held-out prediction is required")
        threshold = float(np.quantile(valid, 1 - target_coverage))
        selected = fractions >= threshold
        accuracy = float(correct[selected].mean()) if selected.any() else 0.0
        return {
            "voteThreshold": threshold,
            "validationCoverage": float(selected.mean()),
            "validationAccuracy": accuracy,
        }

    def project_mapping_layout(
        self,
        target_name: str,
        reference_layout_key: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        label: str | None = None,
        invalidate_cache: bool = False,
    ) -> str | np.ndarray:
        """Place query cells into an unchanged reference layout by neighbor weighting.

        Writable stores return the saved Zarr path. Read-only stores return the
        coordinates as a NumPy array.
        """
        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, None
        )
        store = self._load_complete_projection(
            target_name, from_assay, cell_key, feat_key
        )
        if label is None:
            label = f"fixed_{reference_layout_key}"
        reference_layout = np.column_stack(
            (
                self.cells.fetch(f"{reference_layout_key}1", key=cell_key),
                self.cells.fetch(f"{reference_layout_key}2", key=cell_key),
            )
        )
        indices = as_zarr_array(store["indices"], name="indices")
        distances = as_zarr_array(store["distances"], name="distances")
        persist_layout = self.zarr_mode == "r+"
        layout_plan = None
        if persist_layout:
            from ...mapping.hashing import array_hash

            projection_path = self._projection_artifact_path(
                from_assay,
                target_name,
            )
            try:
                projection_input: object = (
                    parse_artifact_path(projection_path)
                    if projection_path is not None
                    else {
                        "legacy_projection_fingerprint": fingerprint_stored_arrays(
                            store,
                            ("indices", "distances"),
                        )
                    }
                )
            except ValueError:
                projection_input = {
                    "legacy_projection_fingerprint": fingerprint_stored_arrays(
                        store,
                        ("indices", "distances"),
                    )
                }
            layout_plan = plan_artifact(
                self.zw,
                scope="assay",
                assay=from_assay,
                kind="embedding",
                operation="project_mapping_layout",
                parameters={},
                inputs={
                    "projection": projection_input,
                    "reference_layout": array_hash(reference_layout),
                },
                execution_options={
                    "label": label,
                    "reference_layout_key": reference_layout_key,
                },
                invalidate_cache=invalidate_cache,
                required_arrays=(
                    ArrayRequirement(
                        "data",
                        shape=(indices.shape[0], 2),
                        dtype_kind="f",
                    ),
                ),
            )
            if layout_plan.reused:
                return artifact_path(layout_plan.ref)
            layout_group = start_artifact(self.zw, layout_plan)
            layout: Any = create_zarr_dataset(
                layout_group,
                "data",
                (self._projection_block_size(indices), 2),
                "f8",
                (indices.shape[0], 2),
            )
        else:
            layout = np.empty((indices.shape[0], 2), dtype=np.float64)
        from ...mapping.confidence import distance_weights

        for start in range(0, indices.shape[0], self._projection_block_size(indices)):
            stop = min(start + self._projection_block_size(indices), indices.shape[0])
            block_indices = np.asarray(indices[start:stop])
            block_weights = distance_weights(np.asarray(distances[start:stop]))
            layout[start:stop] = np.einsum(
                "nk,nkd->nd", block_weights, reference_layout[block_indices]
            )
        if persist_layout:
            assert layout_plan is not None
            finish_artifact(layout_group, layout_plan)
            return artifact_path(layout_plan.ref)
        return np.asarray(layout)

    def load_unified_graph(
        self,
        from_assay: str | None,
        cell_key: str | None,
        feat_key: str | None,
        target_names: list[str],
        use_k: int,
        target_weight: float,
    ) -> tuple[list[int], csr_matrix]:
        """This is similar to ``load_graph`` but includes projected cells and
        their edges.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            target_names: Name of target datasets to be included in the unified graph
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.

        Returns:
        """
        # TODO:  allow loading multiple targets

        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        graph_loc = self.get_latest_graph_loc(from_assay, cell_key, feat_key)
        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        edges = np.asarray(as_zarr_array(graph_group["edges"], name="edges")[:])
        weights = np.asarray(as_zarr_array(graph_group["weights"], name="weights")[:])
        ref_n_cells = self.cells.active_index(cell_key).shape[0]
        projection_stores = [
            self._load_complete_projection(target_name, from_assay, cell_key, feat_key)
            for target_name in target_names
        ]
        pidx = np.vstack(
            [
                np.asarray(
                    as_zarr_array(
                        projection_store["indices"],
                        name="indices",
                    )[:, :use_k]
                )
                for projection_store in projection_stores
            ]
        )
        n_cells = [ref_n_cells] + [
            as_zarr_array(
                projection_store["indices"],
                name="indices",
            ).shape[0]
            for projection_store in projection_stores
        ]
        ne = []
        nw = []
        for n, i in enumerate(pidx):
            for j in i:
                ne.append([ref_n_cells + n, j])
                # TODO: Better way to weigh the target edges
                nw.append(target_weight)
        me = np.vstack([edges, ne]).astype(int)
        mw = np.hstack([weights, nw])
        tot_cells = ref_n_cells + pidx.shape[0]
        graph = csr_matrix((mw, (me[:, 0], me[:, 1])), shape=(tot_cells, tot_cells))
        return n_cells, graph

    def _get_uni_ini_embed(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        graph: csr_matrix,
        ini_embed_with: str,
        ref_n_cells: int,
    ) -> np.ndarray:
        if ini_embed_with == "kmeans":
            ini_embed = self._get_ini_embed(from_assay, cell_key, feat_key, 2)
        else:
            x = self.cells.fetch(f"{ini_embed_with}1", key=cell_key)
            y = self.cells.fetch(f"{ini_embed_with}2", key=cell_key)
            ini_embed = np.array([x, y]).T.astype(np.float32, order="C")
        targets_best_nn = np.array(np.argmax(graph, axis=1)).reshape(1, -1)[0][
            ref_n_cells:
        ]
        return np.vstack([ini_embed, ini_embed[targets_best_nn]])

    def _plan_unified_embedding(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        label: str,
        n_cells: list[int],
        target_names: list[str],
        initialization: np.ndarray,
        operation: str,
        parameters: dict[str, Any],
        invalidate_cache: bool,
    ) -> PlannedArtifact:
        state = read_assay_state(self.zw, from_assay)
        inputs: dict[str, Any] = {}
        if state is not None and state.connectivity_map is not None:
            inputs["connectivity_map"] = state.connectivity_map
        else:
            graph_path = self.get_latest_graph_loc(
                from_assay,
                cell_key,
                feat_key,
            )
            graph_group = as_zarr_group(self.zw[graph_path], name=graph_path)
            inputs["connectivity_map"] = {
                "legacy_graph_fingerprint": fingerprint_stored_arrays(
                    graph_group,
                    ("edges", "weights"),
                )
            }
        projection_inputs: list[Any] = []
        for target_name in target_names:
            path = self._projection_artifact_path(from_assay, target_name)
            if path is None:
                projection_group = self._load_complete_projection(
                    target_name,
                    from_assay,
                    cell_key,
                )
                projection_inputs.append(
                    {
                        "legacy_projection_fingerprint": fingerprint_stored_arrays(
                            projection_group,
                            ("indices", "distances"),
                        )
                    }
                )
            else:
                projection_inputs.append(parse_artifact_path(path))
        inputs["projections"] = projection_inputs
        inputs["initialization"] = fingerprint_array(initialization)
        stored_parameters = dict(parameters)
        initialization_label = stored_parameters.pop("ini_embed_with", None)
        return plan_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind="embedding",
            operation=operation,
            parameters=stored_parameters,
            inputs=inputs,
            execution_options={
                "label": label,
                "target_names": target_names,
                "initialization_label": initialization_label,
            },
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "data",
                    shape=(sum(n_cells), 2),
                    dtype_kind="f",
                ),
            ),
        )

    def _save_embedding(
        self,
        planned: PlannedArtifact,
        from_assay: str,
        cell_key: str,
        label: str,
        embedding: np.ndarray | None,
        n_cells: list[int],
        target_names: list[str],
    ) -> None:
        if planned.reused:
            group = reused_artifact_group(self.zw, planned)
        else:
            if embedding is None:
                raise ValueError("A new embedding artifact requires coordinates")
            group = start_artifact(self.zw, planned)
            output = create_zarr_dataset(
                group,
                "data",
                (1000, 2),
                "float64",
                embedding.shape,
            )
            output[:] = embedding
            group.attrs["n_cells"] = [int(x) for x in n_cells]
            group.attrs["target_names"] = target_names
            finish_artifact(group, planned)
        projections = as_zarr_group(
            as_zarr_group(self.zw[from_assay], name=from_assay)["projections"],
            name="projections",
        )
        raw_layouts = projections.attrs.get("layouts", {})
        if "layouts" in projections.attrs and not isinstance(raw_layouts, dict):
            raise RuntimeError("Unified layout artifact index is invalid")
        layouts = dict(raw_layouts) if isinstance(raw_layouts, dict) else {}
        layouts[label] = planned.ref.to_dict()
        projections.attrs["layouts"] = layouts
        stored_embedding = np.asarray(as_zarr_array(group["data"], name="data")[:])
        for i in range(2):
            self.cells.insert(
                self._col_renamer(from_assay, cell_key, f"{label}{i + 1}"),
                stored_embedding[: n_cells[0], i],
                key=cell_key,
                overwrite=True,
            )
        return None

    def _load_unified_layout_data(
        self,
        layout_key: str,
        from_assay: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, list[int], list[str]]:
        assay = from_assay or self._defaultAssay
        if assay is None:
            raise ValueError(
                "from_assay is required when the store has no default assay"
            )
        try:
            projections = as_zarr_group(
                as_zarr_group(self.zw[assay], name=assay)["projections"],
                name="projections",
            )
            raw_layouts = projections.attrs.get("layouts", {})
            if "layouts" in projections.attrs and not isinstance(
                raw_layouts,
                dict,
            ):
                raise ValueError("Unified layout artifact index is invalid")
            raw_ref = (
                raw_layouts.get(layout_key) if isinstance(raw_layouts, dict) else None
            )
            if (
                isinstance(raw_layouts, dict)
                and layout_key in raw_layouts
                and not isinstance(raw_ref, dict)
            ):
                raise ValueError(f"Unified layout index for {layout_key!r} is invalid")
            if isinstance(raw_ref, dict):
                ref = ArtifactRef.from_dict(raw_ref)
                if (
                    ref.scope != "assay"
                    or ref.assay != assay
                    or ref.kind != "embedding"
                ):
                    raise ValueError("Unified layout reference is invalid")
                status = inspect_artifact(self.zw, ref)
                if not status.exists or not status.complete:
                    raise ValueError("Unified layout artifact is incomplete")
                layout_group = as_zarr_group(
                    self.zw[artifact_path(ref)],
                    name=artifact_path(ref),
                )
                layout = as_zarr_array(layout_group["data"], name="data")
                attrs = dict(layout_group.attrs)
            else:
                layout = as_zarr_array(projections[layout_key], name=layout_key)
                attrs = dict(layout.attrs)
        except Exception as exc:
            raise KeyError(
                f"Unified layout {layout_key!r} not found under assay {assay!r}. "
                "Run run_unified_umap or run_unified_tsne first."
            ) from exc
        coords = np.asarray(layout[:], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"Unified layout {layout_key!r} must be an (n, 2) array")
        raw_n_cells = attrs.get("n_cells")
        raw_target_names = attrs.get("target_names")
        if not isinstance(raw_n_cells, (list, tuple)) or not isinstance(
            raw_target_names, (list, tuple)
        ):
            raise ValueError(
                f"Unified layout {layout_key!r} is missing n_cells/target_names attributes"
            )
        n_cells = [int(value) for value in raw_n_cells]
        target_names = [str(name) for name in raw_target_names]
        if not n_cells or sum(n_cells) != coords.shape[0]:
            raise ValueError(
                f"Unified layout {layout_key!r} n_cells does not match coordinate rows"
            )
        if len(target_names) != len(n_cells) - 1:
            raise ValueError(
                f"Unified layout {layout_key!r} target_names length must match target blocks"
            )
        return coords[:, 0], coords[:, 1], n_cells[0], n_cells[1:], target_names

    def run_unified_umap(
        self,
        target_names: list[str],
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        use_k: int = 3,
        target_weight: float = 0.1,
        spread: float = 2.0,
        min_dist: float = 1,
        n_epochs: int = 200,
        repulsion_strength: float = 1.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5,
        random_seed: int = 4444,
        ini_embed_with: str = "kmeans",
        label: str = "unified_UMAP",
        parallel: bool = False,
        nthreads: int | None = None,
        invalidate_cache: bool = False,
    ) -> None:
        """Calculates the UMAP embedding for graph obtained using
        ``load_unified_graph``.

        The loaded graph is processed the same way as the graph as in ``run_umap``.

        Args:
            target_names: Names of target datasets to be included in the unified UMAP.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.
            spread: Same as spread in UMAP package.  The effective scale of embedded points. In combination with
                    ``min_dist`` this determines how clustered/clumped the embedded points are.
            min_dist: Same as min_dist in UMAP package. The effective minimum distance between embedded points.
                      Smaller values will result in a more clustered/clumped embedding where nearby points on the
                      manifold are drawn closer together, while larger values will result on a more even dispersal of
                      points. The value should be set relative to the ``spread`` value, which determines the scale at
                      which embedded points will be spread out. (Default value: 1)
            n_epochs: Same as n_epochs in UMAP package. The number of training epochs to be used in optimizing the
                      low dimensional embedding. Larger values result in more accurate embeddings.
                      (Default value: 200)
            repulsion_strength: Same as repulsion_strength in UMAP package. Weighting applied to negative samples in
                                low dimensional embedding optimization. Values higher than one will result in greater
                                weight being given to negative samples. (Default value: 1.0)
            initial_alpha: Same as learning_rate in UMAP package. The initial learning rate for the embedding
                           optimization. (Default value: 1.0)
            negative_sample_rate: Same as negative_sample_rate in UMAP package. The number of negative samples to
                                  select per positive sample in the optimization process. Increasing this value will
                                  result in greater repulsive force being applied, greater optimization cost, but
                                  slightly more accuracy. (Default value: 5)
            random_seed: (Default value: 4444)
            ini_embed_with: either 'kmeans' or a column from cell metadata to be used as initial embedding coordinates
            label: base label for UMAP dimensions in the cell metadata column (Default value: 'UMAP')
            parallel: Whether to run UMAP in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run UMAP. By default, the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:
            None
        """
        from ...embeddings.umap import fit_transform

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        n_cells, graph = self.load_unified_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            target_names=target_names,
            use_k=use_k,
            target_weight=target_weight,
        )
        ini_embed = self._get_uni_ini_embed(
            from_assay, cell_key, feat_key, graph, ini_embed_with, n_cells[0]
        )
        if nthreads is None:
            nthreads = self.nthreads
        parameters = {
            "use_k": use_k,
            "target_weight": target_weight,
            "spread": spread,
            "min_dist": min_dist,
            "n_epochs": n_epochs,
            "repulsion_strength": repulsion_strength,
            "initial_alpha": initial_alpha,
            "negative_sample_rate": negative_sample_rate,
            "random_seed": random_seed,
            "ini_embed_with": ini_embed_with,
            "parallel": parallel,
            "nthreads": nthreads if parallel else None,
        }
        embedding_plan = self._plan_unified_embedding(
            from_assay,
            cell_key,
            feat_key,
            label,
            n_cells,
            target_names,
            ini_embed,
            operation="run_unified_umap",
            parameters=parameters,
            invalidate_cache=invalidate_cache,
        )
        if embedding_plan.reused:
            self._save_embedding(
                embedding_plan,
                from_assay,
                cell_key,
                label,
                None,
                n_cells,
                target_names,
            )
            return None
        verbose = progress_enabled()
        t, a, b = fit_transform(
            graph=graph.tocoo(),
            ini_embed=ini_embed,
            spread=spread,
            min_dist=min_dist,
            n_epochs=n_epochs,
            random_seed=random_seed,
            repulsion_strength=repulsion_strength,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            densmap_kwds={},
            parallel=parallel,
            nthreads=nthreads,
            verbose=verbose,
        )
        self._save_embedding(
            embedding_plan,
            from_assay,
            cell_key,
            label,
            t,
            n_cells,
            target_names,
        )
        return None

    def run_unified_tsne(
        self,
        target_names: list[str],
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        use_k: int = 3,
        target_weight: float = 0.5,
        lambda_scale: float = 1.0,
        max_iter: int = 500,
        early_iter: int = 200,
        alpha: int = 10,
        box_h: float = 0.7,
        temp_file_loc: str = ".",
        verbose: bool = True,
        ini_embed_with: str = "kmeans",
        label: str = "unified_tSNE",
        invalidate_cache: bool = False,
    ) -> None:
        """Calculates the tSNE embedding for graph obtained using
        ``load_unified_graph``. The loaded graph is processed the same way as
        the graph as in ``run_tsne``.

        Args:
            target_names: Names of target datasets to be included in the unified tSNE.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'.
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.
            lambda_scale: λ rescaling parameter. (Default value: 1.0)
            max_iter: Maximum number of iterations. (Default value: 500)
            early_iter: Number of early exaggeration iterations. (Default value: 200)
            alpha: Early exaggeration multiplier. (Default value: 10)
            box_h: Grid side length (accuracy control). Lower values might drastically slow down
                   the algorithm (Default value: 0.7)
            temp_file_loc: Location of temporary file. By default, these files will be created in the current working
                           directory. These files are deleted before the method returns.
            verbose: If True (default) then the full log from SGtSNEpi algorithm is shown.
            ini_embed_with: Initial embedding coordinates for the cells in cell_key. Should have the same number of
                            columns as tsne_dims. If not value is provided then the initial embedding is obtained using
                            `get_ini_embed`.
            label: Base label for tSNE dimensions in the cell metadata column. (Default value: 'tSNE')

        Returns:
        """
        from ...embeddings.sgtsne import run_sgtsne

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        n_cells, graph = self.load_unified_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            target_names=target_names,
            use_k=use_k,
            target_weight=target_weight,
        )
        ini_embed = self._get_uni_ini_embed(
            from_assay, cell_key, feat_key, graph, ini_embed_with, n_cells[0]
        )
        parameters = {
            "use_k": use_k,
            "target_weight": target_weight,
            "lambda_scale": lambda_scale,
            "max_iter": max_iter,
            "early_iter": early_iter,
            "alpha": alpha,
            "box_h": box_h,
            "ini_embed_with": ini_embed_with,
        }
        embedding_plan = self._plan_unified_embedding(
            from_assay,
            cell_key,
            feat_key,
            label,
            n_cells,
            target_names,
            ini_embed,
            operation="run_unified_tsne",
            parameters=parameters,
            invalidate_cache=invalidate_cache,
        )
        if embedding_plan.reused:
            self._save_embedding(
                embedding_plan,
                from_assay,
                cell_key,
                label,
                None,
                n_cells,
                target_names,
            )
            return None
        emb = run_sgtsne(
            graph,
            ini_embed,
            tsne_dims=2,
            max_iter=max_iter,
            early_iter=early_iter,
            alpha=alpha,
            lambda_scale=lambda_scale,
            box_h=box_h,
            temp_file_loc=temp_file_loc,
            verbose=verbose,
        )
        self._save_embedding(
            embedding_plan,
            from_assay,
            cell_key,
            label,
            emb.T,
            n_cells,
            target_names,
        )
        return None
