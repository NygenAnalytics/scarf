from typing import Any

import numpy as np

from ..quality_control.filtering import (
    _apply_bounds,
    _sample_aware_mad_mask,
    _validated_sample_labels,
    gaussian_quantile_bounds,
)
from ..storage.artifacts import ArtifactRef, artifact_group
from ..storage.selections import (
    read_stored_selection_mask,
    resolve_selection_artifact,
)
from ..storage.types import as_zarr_array
from ..utils.logging import logger
from ._pipeline_recipe import ResolvedPipelineRecipe


def snapshot_column_values(
    snapshot: Any,
    column: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one snapshot column and its linked nullable mask."""
    source = as_zarr_array(snapshot[column], name=column)
    values = np.asarray(source[:])
    if values.ndim != 1:
        raise ValueError(f"Snapshot column {column!r} must be one-dimensional")
    missing_name = source.attrs.get("missing_mask")
    if missing_name is None:
        return values, None
    if (
        not isinstance(missing_name, str)
        or not missing_name
        or missing_name not in snapshot
    ):
        raise ValueError(f"Snapshot column {column!r} has an invalid missing mask")
    missing_source = as_zarr_array(snapshot[missing_name], name=missing_name)
    if (
        missing_source.ndim != 1
        or missing_source.shape != source.shape
        or np.dtype(missing_source.dtype) != np.dtype(bool)
    ):
        raise ValueError(f"Snapshot column {column!r} has a malformed missing mask")
    return values, np.asarray(missing_source[:], dtype=bool)


def filter_pipeline_selection(
    store: Any,
    *,
    recipe: ResolvedPipelineRecipe,
    input_selection: ArtifactRef,
    cell_snapshot: ArtifactRef,
) -> ArtifactRef:
    active = read_stored_selection_mask(
        store.zw,
        input_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    config = recipe.filtering
    attrs = list(config.get("attrs", ()))
    if not attrs:
        return input_selection
    snapshot = artifact_group(store.zw, cell_snapshot)
    values_by_attr: dict[str, np.ndarray] = {}
    metric_missing = np.zeros(active.shape, dtype=bool)
    for attr in attrs:
        values, missing = snapshot_column_values(snapshot, attr)
        values_by_attr[attr] = values
        if missing is not None:
            metric_missing |= missing
    filter_active = active & ~metric_missing
    parameters = dict(config)
    if config["method"] == "manual":
        keep = ~metric_missing
        keep_bounds = config["keepBounds"]
        for attr, low, high in zip(
            attrs,
            config["lows"],
            config["highs"],
            strict=True,
        ):
            keep &= _apply_bounds(
                values_by_attr[attr],
                low,
                high,
                keep_bounds=keep_bounds,
            )
    elif config["sampleColumn"] is None:
        if not filter_active.any():
            raise ValueError(
                "Pipeline filtering has no selected cells with complete metrics"
            )
        keep = ~metric_missing
        bounds: dict[str, dict[str, float]] = {}
        for attr in attrs:
            low, high = gaussian_quantile_bounds(
                values_by_attr[attr][filter_active],
                config["minP"],
                config["maxP"],
            )
            bounds[attr] = {"low": low, "high": high}
            keep &= _apply_bounds(values_by_attr[attr], low, high)
        parameters["resolvedBounds"] = bounds
    else:
        sample_column = config["sampleColumn"]
        labels, sample_missing = snapshot_column_values(snapshot, sample_column)
        if sample_missing is not None and np.any(active & sample_missing):
            raise ValueError(
                f"sample column {sample_column!r} contains missing labels "
                "among active cells"
            )
        labels = _validated_sample_labels(
            labels,
            active,
            label_name=f"sample column {sample_column!r}",
        )
        keep, provenance = _sample_aware_mad_mask(
            values_by_attr=values_by_attr,
            sample_labels=labels,
            active=filter_active,
            n_mads=config["nMads"],
            min_cells_per_sample=config["minCellsPerSample"],
            attrs=attrs,
        )
        keep &= ~metric_missing
        for message in provenance["warnings"]:
            logger.warning(message)
        parameters["mad"] = provenance
    values = np.asarray(active & keep, dtype=bool)
    if not values.any():
        raise ValueError("Pipeline filtering removed every selected cell")
    return resolve_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=np.asarray(store.cells.fetch_all("ids")),
        operation="filter_pipeline_cells",
        parameters=parameters,
        inputs={
            "input_cell_selection": input_selection,
            "cell_snapshot": cell_snapshot,
        },
        source_column=recipe.cell_key,
    )
