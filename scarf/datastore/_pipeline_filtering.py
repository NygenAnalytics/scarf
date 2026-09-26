from typing import Any

import numpy as np

from ..quality_control.filtering import filter_cell_metrics
from ..storage.arrays import linked_missing_mask
from ..storage.artifacts import ArtifactRef, artifact_group
from ..storage.selections import (
    read_stored_selection_mask,
    resolve_generated_selection_artifact,
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
    missing = linked_missing_mask(
        snapshot,
        column,
        label=f"Snapshot column {column!r}",
        values=source,
    )
    return values, None if missing is None else np.asarray(missing[:])


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
    missing_by_attr: dict[str, np.ndarray] = {}
    for attr in attrs:
        values, missing = snapshot_column_values(snapshot, attr)
        values_by_attr[attr] = values
        if missing is not None:
            missing_by_attr[attr] = missing
    method = config["method"]
    options: dict[str, Any]
    if method == "manual":
        options = {
            "lows": config["lows"],
            "highs": config["highs"],
            "keep_bounds": config["keepBounds"],
        }
    else:
        options = {
            "min_p": config["minP"],
            "max_p": config["maxP"],
            "n_mads": config["nMads"],
            "min_cells_per_sample": config["minCellsPerSample"],
        }
        sample_column = config["sampleColumn"]
        if method == "mad" and sample_column is not None:
            labels, sample_missing = snapshot_column_values(snapshot, sample_column)
            options.update(
                sample_labels=labels,
                sample_missing=sample_missing,
                sample_label_name=f"sample column {sample_column!r}",
            )
    result = filter_cell_metrics(
        values_by_attr,
        missing_by_attr,
        active,
        method=method,
        **options,
    )
    parameters = dict(config)
    if result.gaussian_bounds is not None:
        parameters["resolvedBounds"] = result.gaussian_bounds
    if result.mad_provenance is not None:
        for message in result.mad_provenance["warnings"]:
            logger.warning(message)
        parameters["mad"] = result.mad_provenance
    return resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=result.retained,
        row_ids=store.cells._get_array("ids"),
        operation="filter_pipeline_cells",
        parameters=parameters,
        inputs={
            "input_cell_selection": input_selection,
            "cell_snapshot": cell_snapshot,
        },
        source_column=recipe.cell_key,
    )[0]
