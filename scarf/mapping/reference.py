"""Persistent handles for immutable mapping references."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ..storage.artifacts import artifact_group, inspect_artifact
from ..storage.feature_selection import resolve_feature_selection
from ..storage.refs import ArtifactRef, ExternalArtifactRef
from ..storage.selections import (
    read_stored_selection_indices,
    validate_stored_selection_integrity,
)
from ..storage.types import as_zarr_array
from ..metadata.rows import (
    read_metadata_missing_rows,
    read_metadata_rows_chunkwise,
)
from ..metadata.selection import valid_category_mask
from .models import (
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
    _immutable_array,
)


class _FrozenList(tuple):
    """Immutable list-shaped metadata that retains list equality semantics."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list | tuple):
            return tuple(self) == tuple(other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = tuple.__hash__


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return _FrozenList(_freeze_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_metadata(item) for item in value.tolist())
    return copy.deepcopy(value)


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return copy.deepcopy(value)


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
    metadata: Mapping[str, Any]
    reference_distance_quantiles: np.ndarray
    reference_distance_values: np.ndarray

    def __post_init__(self) -> None:
        feature_ids = np.asarray(self.feature_ids)
        if feature_ids.ndim != 1 or feature_ids.dtype.kind not in {"O", "S", "U"}:
            raise TypeError("Mapping reference feature IDs must contain strings")
        frozen_ids = _immutable_array(feature_ids.astype(str))
        object.__setattr__(self, "feature_ids", frozen_ids)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "reference_distance_quantiles",
            _immutable_array(np.asarray(self.reference_distance_quantiles)),
        )
        object.__setattr__(
            self,
            "reference_distance_values",
            _immutable_array(np.asarray(self.reference_distance_values)),
        )

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
        return cast(dict[str, Any], _thaw_metadata(values))

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

    def validate_frozen_axes(self) -> None:
        """Validate the exact stored cell and feature axes used by the reference."""
        selection = validate_stored_selection_integrity(
            self.datastore.zw,
            self.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if selection.selected_count != self.selected_cell_count:
            raise ValueError(
                "The selected reference cell count has changed. Rebuild the "
                "mapping reference with build_mapping_reference(neighbors)."
            )
        resolve_feature_selection(
            self.datastore.zw,
            self.assay_name,
            self.feature_selection,
        )

    def _selected_cell_values(
        self,
        column: str,
        *,
        validate_binding: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if validate_binding:
            from .artifact import validate_mapping_reference_binding

            validate_mapping_reference_binding(self)
        self.validate_dataset_fingerprint()
        validate_stored_selection_integrity(
            self.datastore.zw,
            self.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        indices = read_stored_selection_indices(
            self.datastore.zw,
            self.cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        values = np.asarray(
            read_metadata_rows_chunkwise(self.datastore.cells, column, indices)
        )
        missing = read_metadata_missing_rows(self.datastore.cells, column, indices)
        if values.shape != (self.selected_cell_count,):
            raise ValueError(
                "The selected reference cell count has changed. Rebuild the "
                "mapping reference with build_mapping_reference(neighbors)."
            )
        return values, missing

    def fetch_cell_column(self, column: str) -> np.ndarray:
        """Fetch one reference cell column through the stored cell selection."""
        values, _ = self._selected_cell_values(column)
        return values

    def _fetch_cell_labels(self, column: str) -> tuple[np.ndarray, np.ndarray]:
        """Fetch labels and mark values that are usable categorical levels."""
        values, missing = self._selected_cell_values(
            column,
            validate_binding=False,
        )
        return values, valid_category_mask(values, missing_mask=missing)

    def fetch_layout(self, layout: ArtifactRef) -> np.ndarray:
        """Fetch a two-dimensional layout from one explicit embedding artifact."""
        from .artifact import validate_mapping_reference_binding

        validate_mapping_reference_binding(self)
        self.validate_dataset_fingerprint()
        self.validate_frozen_axes()
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
