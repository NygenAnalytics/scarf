"""Persistent handles for immutable mapping references."""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..storage.artifacts import artifact_group, inspect_artifact
from ..storage.refs import ArtifactRef, ExternalArtifactRef
from ..storage.types import as_zarr_array, as_zarr_group
from .models import (
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
)


@dataclass(frozen=True)
class MappingReference:
    """An immutable scaled-PCA reference loaded from a Scarf Zarr store."""

    datastore: Any
    ref: ArtifactRef
    assay_name: str
    cell_key: str
    feature_key: str
    reduction: ArtifactRef
    ann_index: ArtifactRef
    neighbors: ArtifactRef
    cell_selection: ArtifactRef
    feature_selection: ArtifactRef
    batch_correction: ArtifactRef | None
    dataset_fingerprint: str
    selected_cell_count: int
    model: ScaledPCAProjectionModel
    symphony_state: SymphonyCorrectionModel | None
    feature_ids: np.ndarray
    metadata: dict[str, Any]
    reference_distance_quantiles: np.ndarray
    reference_distance_values: np.ndarray

    @property
    def method(self) -> str:
        return str(self.metadata["method"])

    @property
    def ann_metric(self) -> str:
        return str(self.metadata["ann_metric"])

    @property
    def normalization_parameters(self) -> dict[str, Any]:
        values = self.metadata["normalization_parameters"]
        if not isinstance(values, Mapping):
            raise TypeError("Mapping reference normalization parameters are invalid")
        return dict(values)

    @property
    def size_factor(self) -> float:
        return float(self.normalization_parameters["size_factor"])

    @property
    def external_ref(self) -> ExternalArtifactRef:
        return ExternalArtifactRef(
            dataset_fingerprint=self.dataset_fingerprint,
            ref=self.ref,
        )

    def validate_dataset_fingerprint(self) -> None:
        assay = self.datastore._get_assay(self.assay_name)
        live = assay.attrs.get("dataset_fingerprint")
        if live is None:
            raise ValueError(
                f"Reference assay {self.assay_name!r} has no stored dataset "
                "fingerprint. Rebuild it with build_mapping_reference(neighbors)."
            )
        if live != self.dataset_fingerprint:
            raise ValueError(
                f"Reference assay {self.assay_name!r} dataset fingerprint mismatch. "
                f"Expected {self.dataset_fingerprint!r}, received {live!r}. "
                "Rebuild it with build_mapping_reference(neighbors)."
            )

    def fetch_cell_column(self, column: str) -> np.ndarray:
        """Fetch one reference cell column through the stored cell selection."""
        from ..graph.state import validate_cell_selection_artifact

        self.validate_dataset_fingerprint()
        validate_cell_selection_artifact(
            self.datastore.zw,
            self.cell_selection,
            self.cell_key,
        )
        values = np.asarray(self.datastore.cells.fetch(column, key=self.cell_key))
        if len(values) != self.selected_cell_count:
            raise ValueError(
                "The selected reference cell count has changed. Rebuild the "
                "mapping reference with build_mapping_reference(neighbors)."
            )
        return values

    def fetch_layout(self, layout_key: str) -> np.ndarray:
        """Fetch a two-column layout for the selected reference cells."""
        source = self.layout_source(layout_key)
        if source is None:
            raw_layout = np.column_stack(
                (
                    self.fetch_cell_column(f"{layout_key}1"),
                    self.fetch_cell_column(f"{layout_key}2"),
                )
            )
        else:
            cell_data = as_zarr_group(
                self.datastore.zw["cellData"],
                name="cellData",
            )
            first_column = as_zarr_array(
                cell_data[f"{layout_key}1"],
                name=f"{layout_key}1",
            )
            source_value = first_column.attrs.get("source_value")
            if not isinstance(source_value, str) or not source_value:
                raise ValueError("Linked reference layout source is invalid")
            source_values = as_zarr_array(
                artifact_group(self.datastore.zw, source.ref)[source_value],
                name=source_value,
            )
            raw_layout = np.asarray(source_values[:, :2])
        try:
            layout = np.asarray(raw_layout, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("Reference layout coordinates must be numeric") from exc
        if layout.shape != (self.selected_cell_count, 2):
            raise ValueError(
                "Reference layout must have two columns and one row per "
                "selected reference cell"
            )
        if not np.all(np.isfinite(layout) | np.isnan(layout)):
            raise ValueError("Reference layout contains infinite coordinates")
        return np.array(layout, copy=True)

    def layout_source(self, layout_key: str) -> ExternalArtifactRef | None:
        """Return the shared assay artifact linked to two layout columns."""
        self.validate_dataset_fingerprint()
        cell_data = as_zarr_group(
            self.datastore.zw["cellData"],
            name="cellData",
        )
        refs: list[ArtifactRef] = []
        source_values: list[str] = []
        for expected_index, column in enumerate((f"{layout_key}1", f"{layout_key}2")):
            try:
                values = as_zarr_array(cell_data[column], name=column)
            except (KeyError, TypeError):
                return None
            raw_ref = values.attrs.get("source_artifact")
            source_value = values.attrs.get("source_value")
            value_index = values.attrs.get("value_index")
            if not isinstance(raw_ref, Mapping):
                return None
            if not isinstance(source_value, str) or not source_value:
                return None
            if (
                isinstance(value_index, bool | np.bool_)
                or not isinstance(value_index, int | np.integer)
                or int(value_index) != expected_index
            ):
                return None
            try:
                ref = ArtifactRef.from_dict(raw_ref)
                status = inspect_artifact(self.datastore.zw, ref)
            except (KeyError, TypeError, ValueError):
                return None
            if (
                ref.scope != "assay"
                or ref.assay != self.assay_name
                or not status.complete
            ):
                return None
            refs.append(ref)
            source_values.append(source_value)
        if refs[0] != refs[1]:
            return None
        if source_values[0] != source_values[1]:
            return None
        try:
            source = as_zarr_array(
                artifact_group(self.datastore.zw, refs[0])[source_values[0]],
                name=source_values[0],
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            source.ndim != 2
            or int(source.shape[0]) != self.selected_cell_count
            or int(source.shape[1]) < 2
        ):
            return None
        return ExternalArtifactRef(
            dataset_fingerprint=self.dataset_fingerprint,
            ref=refs[0],
        )
