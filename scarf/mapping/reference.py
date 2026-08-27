"""Persistent handles for immutable mapping references."""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ..storage.artifacts import artifact_group, inspect_artifact
from ..storage.refs import ArtifactRef, ExternalArtifactRef
from ..storage.selections import validate_stored_selection_integrity
from ..storage.types import as_zarr_array
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
        stored = assay.attrs.get("dataset_fingerprint")
        live = (
            stored
            if isinstance(stored, str) and stored
            else self.datastore._calculate_dataset_fingerprint(self.assay_name)
        )
        if live != self.dataset_fingerprint:
            raise ValueError(
                f"Reference assay {self.assay_name!r} dataset fingerprint mismatch. "
                f"Expected {self.dataset_fingerprint!r}, received {live!r}. "
                "Rebuild it with build_mapping_reference(neighbors)."
            )

    def fetch_cell_column(self, column: str) -> np.ndarray:
        """Fetch one reference cell column through the stored cell selection."""
        self.validate_dataset_fingerprint()
        selection = validate_stored_selection_integrity(
            self.datastore.zw,
            self.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        mask = np.asarray(selection.values[:], dtype=bool)
        values = np.asarray(self.datastore.cells.fetch_all(column))[mask]
        if len(values) != self.selected_cell_count:
            raise ValueError(
                "The selected reference cell count has changed. Rebuild the "
                "mapping reference with build_mapping_reference(neighbors)."
            )
        return cast(np.ndarray, values)

    def fetch_layout(self, layout: ArtifactRef) -> np.ndarray:
        """Fetch a two-dimensional layout from one explicit embedding artifact."""
        self.validate_dataset_fingerprint()
        if not isinstance(layout, ArtifactRef):
            raise TypeError("layout must be an ArtifactRef")
        if (
            layout.scope != "assay"
            or layout.assay != self.assay_name
            or layout.kind != "embedding"
        ):
            raise ValueError(
                "layout must identify an assay-scoped embedding artifact for "
                "the reference assay"
            )
        status = inspect_artifact(self.datastore.zw, layout)
        if not status.complete:
            raise ValueError("Reference layout artifact is unavailable or incomplete")
        raw_selection = (status.inputs or {}).get("cell_selection")
        if not isinstance(raw_selection, Mapping):
            raise ValueError("Reference layout artifact has no cell-selection input")
        try:
            source_selection = ArtifactRef.from_dict(raw_selection)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Reference layout artifact has an invalid cell-selection input"
            ) from exc
        if source_selection != self.cell_selection:
            raise ValueError(
                "Reference layout and mapping reference use different cell selections"
            )
        group = artifact_group(self.datastore.zw, layout)
        if "values" not in group:
            raise ValueError("Reference layout artifact has no canonical values array")
        raw_layout = np.asarray(as_zarr_array(group["values"], name="values")[:])
        try:
            coordinates = np.asarray(raw_layout, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("Reference layout coordinates must be numeric") from exc
        if coordinates.shape != (self.selected_cell_count, 2):
            raise ValueError(
                "Reference layout must have two columns and one row per "
                "selected reference cell"
            )
        if not np.all(np.isfinite(coordinates) | np.isnan(coordinates)):
            raise ValueError("Reference layout contains infinite coordinates")
        return np.array(coordinates, copy=True)
