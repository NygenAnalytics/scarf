from collections.abc import Iterator, Mapping
from typing import Any, Literal

import numpy as np

from ..metadata.artifacts import categorical_display, continuous_display
from ..storage.artifacts import ArtifactRef, artifact_group
from ..storage.pipeline_runs import PipelineFieldDescriptor
from ..storage.types import as_zarr_array
from ._pipeline_recipe import ResolvedPipelineRecipe


def artifact_array(root: Any, ref: ArtifactRef, name: str) -> Any:
    group = artifact_group(root, ref)
    return as_zarr_array(group[name], name=name)


def _array_block_rows(array: Any) -> int:
    chunks = getattr(array, "chunks", None)
    if chunks and len(chunks) == len(array.shape):
        return max(1, int(chunks[0]))
    return max(1, min(int(array.shape[0]), 65_536))


def _iter_array_blocks(
    array: Any,
    *,
    value_index: int | None = None,
) -> Iterator[np.ndarray]:
    block_rows = _array_block_rows(array)
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        if value_index is None:
            yield np.asarray(array[start:stop])
        else:
            yield np.asarray(array[start:stop, value_index])


def continuous_array_display(
    array: Any,
    *,
    value_index: int | None = None,
) -> dict[str, Any]:
    minimum: float | None = None
    maximum: float | None = None
    for block in _iter_array_blocks(array, value_index=value_index):
        numeric = np.asarray(block, dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            continue
        block_minimum = float(finite.min())
        block_maximum = float(finite.max())
        minimum = block_minimum if minimum is None else min(minimum, block_minimum)
        maximum = block_maximum if maximum is None else max(maximum, block_maximum)
    extrema = (
        np.empty(0, dtype=np.float64)
        if minimum is None or maximum is None
        else np.asarray([minimum, maximum], dtype=np.float64)
    )
    return continuous_display(extrema)


def categorical_array_display(array: Any) -> dict[str, Any]:
    categories: list[Any] = []
    seen: set[tuple[str, str]] = set()
    has_missing = False
    for block in _iter_array_blocks(array):
        for raw_value in np.asarray(block).reshape(-1):
            value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
            if isinstance(value, float) and np.isnan(value):
                value = None
            if value is None:
                has_missing = True
                continue
            key = (type(value).__name__, repr(value))
            if key not in seen:
                seen.add(key)
                categories.append(value)
    display_values = np.asarray(
        [*categories, *([None] if has_missing else [])],
        dtype=object,
    )
    return categorical_display(display_values)


def _fill_for_dtype(dtype: np.dtype[Any]) -> str | int | bool:
    if dtype.kind == "f":
        return "nan"
    if dtype.kind in {"i", "u"}:
        return -1
    if dtype.kind == "b":
        return False
    return ""


def _field(
    *,
    key: str,
    axis: Literal["cells", "features"],
    ref: ArtifactRef,
    source_value: str,
    dtype: Any,
    value_index: int | None = None,
    missing_mask: str | None = None,
    display: Mapping[str, object] | None = None,
    fill: str | int | float | bool | None = None,
) -> PipelineFieldDescriptor:
    resolved_dtype = np.dtype(dtype)
    return PipelineFieldDescriptor(
        key=key,
        axis=axis,
        artifact=ref,
        source_value=source_value,
        value_index=value_index,
        dtype=resolved_dtype.str,
        fill=_fill_for_dtype(resolved_dtype) if fill is None else fill,
        missing_mask=missing_mask,
        display=None if display is None else dict(display),
    )


def _snapshot_field(
    root: Any,
    *,
    key: str,
    axis: Literal["cells", "features"],
    snapshot: ArtifactRef,
) -> PipelineFieldDescriptor:
    group = artifact_group(root, snapshot)
    array = as_zarr_array(group[key], name=key)
    raw_missing = array.attrs.get("missing_mask")
    missing = raw_missing if isinstance(raw_missing, str) else None
    return _field(
        key=key,
        axis=axis,
        ref=snapshot,
        source_value=key,
        dtype=array.dtype,
        missing_mask=missing,
    )


def build_pipeline_fields(
    store: Any,
    recipe: ResolvedPipelineRecipe,
    artifacts: Mapping[str, ArtifactRef],
    *,
    cell_snapshot: ArtifactRef,
    feature_snapshot: ArtifactRef,
) -> tuple[PipelineFieldDescriptor, ...]:
    assay = store._get_assay(recipe.assay)
    fields: list[PipelineFieldDescriptor] = [
        _field(
            key="I",
            axis="cells",
            ref=artifacts["analysis_cell_selection"],
            source_value="values",
            dtype=bool,
            fill=False,
        ),
        _field(
            key="ids",
            axis="cells",
            ref=cell_snapshot,
            source_value="ids",
            dtype=store.cells._get_array("ids").dtype,
            fill="",
        ),
    ]
    fields.extend(
        _snapshot_field(
            store.zw,
            key=column,
            axis="cells",
            snapshot=cell_snapshot,
        )
        for column in ("names", *recipe.snapshot_columns)
    )
    if "cell_cycle" in artifacts:
        ref = artifacts["cell_cycle"]
        for key in ("s_score", "g2m_score"):
            values = artifact_array(store.zw, ref, key)
            fields.append(
                _field(
                    key=key,
                    axis="cells",
                    ref=ref,
                    source_value=key,
                    dtype=values.dtype,
                    display=continuous_array_display(values),
                )
            )
        phase = artifact_array(store.zw, ref, "phase")
        fields.append(
            _field(
                key="cell_cycle_phase",
                axis="cells",
                ref=ref,
                source_value="phase",
                dtype=phase.dtype,
                display=categorical_array_display(phase),
            )
        )
    if "umap" in artifacts:
        ref = artifacts["umap"]
        values = artifact_array(store.zw, ref, "values")
        for index in range(values.shape[1]):
            fields.append(
                _field(
                    key=f"umap_{index + 1}",
                    axis="cells",
                    ref=ref,
                    source_value="values",
                    value_index=index,
                    dtype=values.dtype,
                    display=continuous_array_display(values, value_index=index),
                )
            )
    for key, _resolution in recipe.leiden_partitions:
        output_key = f"leiden_{key}"
        if output_key not in artifacts:
            continue
        ref = artifacts[output_key]
        values = artifact_array(store.zw, ref, "values")
        fields.append(
            _field(
                key=output_key,
                axis="cells",
                ref=ref,
                source_value="values",
                dtype=values.dtype,
                display=categorical_array_display(values),
            )
        )
    if "paris" in artifacts:
        ref = artifacts["paris"]
        values = artifact_array(store.zw, ref, "labels")
        fields.append(
            _field(
                key="paris",
                axis="cells",
                ref=ref,
                source_value="labels",
                dtype=values.dtype,
                display=categorical_array_display(values),
            )
        )
    if "clusters" in artifacts:
        ref = artifacts["clusters"]
        cluster_group = artifact_group(store.zw, ref)
        source_value = "values" if "values" in cluster_group else "labels"
        values = artifact_array(store.zw, ref, source_value)
        fields.append(
            _field(
                key="clusters",
                axis="cells",
                ref=ref,
                source_value=source_value,
                dtype=values.dtype,
                display=categorical_array_display(values),
            )
        )
    if "doublets" in artifacts:
        ref = artifacts["doublets"]
        values = artifact_array(store.zw, ref, "values")
        fields.append(
            _field(
                key="doublet_score",
                axis="cells",
                ref=ref,
                source_value="values",
                dtype=values.dtype,
                display=continuous_array_display(values),
            )
        )
    fields.extend(
        (
            _field(
                key="I",
                axis="features",
                ref=artifacts["feature_universe"],
                source_value="values",
                dtype=bool,
                fill=False,
            ),
            _field(
                key="ids",
                axis="features",
                ref=feature_snapshot,
                source_value="ids",
                dtype=assay.feats._get_array("ids").dtype,
                fill="",
            ),
            _snapshot_field(
                store.zw,
                key="names",
                axis="features",
                snapshot=feature_snapshot,
            ),
        )
    )
    hvg = artifacts["highly_variable_features"]
    fields.append(
        _field(
            key="highly_variable_features",
            axis="features",
            ref=hvg,
            source_value="values",
            dtype=artifact_array(store.zw, hvg, "values").dtype,
            fill=False,
        )
    )
    return tuple(fields)
