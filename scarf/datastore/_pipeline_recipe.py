import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from ..assay import RNAassay


@dataclass(frozen=True, slots=True)
class ResolvedPipelineRecipe:
    assay: str
    label: str | None
    cell_key: str
    filtering: dict[str, Any]
    harmony_batch_columns: tuple[str, ...]
    hvg_count: int
    pca_dims: int
    neighbors_k: int
    umap: bool
    leiden_partitions: tuple[tuple[str, float], ...]
    cell_cycle: bool
    paris: bool
    doublets: bool
    markers: bool
    snapshot_columns: tuple[str, ...]
    cell_snapshot_columns: tuple[str, ...]
    stage_order: tuple[str, ...]

    def to_config(self) -> dict[str, Any]:
        return {
            "cellKey": self.cell_key,
            "filtering": self.filtering,
            "harmonyBatchColumns": list(self.harmony_batch_columns),
            "hvgCount": self.hvg_count,
            "pcaDims": self.pca_dims,
            "neighborsK": self.neighbors_k,
            "umap": self.umap,
            "leiden": {
                "partitions": [value for _key, value in self.leiden_partitions],
            },
            "cellCycle": self.cell_cycle,
            "paris": self.paris,
            "doublets": self.doublets,
            "markers": self.markers,
            "snapshotColumns": list(self.snapshot_columns),
        }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _column_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    columns = tuple(value)
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} must not contain duplicates")
    return columns


def _canonical_resolution(value: Any) -> tuple[str, float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Leiden resolutions must be numbers")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("Leiden resolutions must be finite and positive")
    return str(resolved), resolved


def _resolve_leiden(
    value: Mapping[str, object] | bool,
) -> tuple[tuple[str, float], ...]:
    if value is False:
        return ()
    if value is True:
        return (
            ("0.5", 0.5),
            ("0.75", 0.75),
            ("1.0", 1.0),
            ("1.25", 1.25),
        )
    if not isinstance(value, Mapping):
        raise TypeError("leiden must be a mapping or bool")
    if set(value) != {"partitions"}:
        raise ValueError("leiden must contain exactly 'partitions'")
    raw_partitions = value["partitions"]
    if isinstance(raw_partitions, str | bytes) or not isinstance(
        raw_partitions,
        Sequence,
    ):
        raise TypeError("leiden partitions must be a non-empty sequence")
    partitions = tuple(_canonical_resolution(item) for item in raw_partitions)
    if not partitions:
        raise ValueError("leiden partitions must not be empty")
    keys = [key for key, _resolution in partitions]
    if len(keys) != len(set(keys)):
        raise ValueError("leiden partitions contain duplicate resolutions")
    return partitions


def _default_filter_columns(store: Any, assay: str) -> tuple[str, ...]:
    return tuple(
        column
        for suffix in ("nCounts", "nFeatures", "percentMito", "percentRibo")
        if (column := f"{assay}_{suffix}") in store.cells.columns
    )


def _manual_bound(value: Any, name: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} values must be finite numbers or None")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} values must be finite; use None for no bound")
    return int(value) if isinstance(value, int) else resolved


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite number")
    return resolved


def _resolve_filtering(
    store: Any,
    assay: str,
    value: bool | Mapping[str, object],
) -> dict[str, Any]:
    if value is False:
        return {"enabled": False}
    if value is True:
        options: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        options = dict(value)
    else:
        raise TypeError("filtering must be a mapping or bool")
    method = options.pop("method", "auto")
    if method not in {"auto", "manual"}:
        raise ValueError("filtering method must be 'auto' or 'manual'")
    attrs = _column_sequence(
        options.pop("attrs", _default_filter_columns(store, assay)),
        "filtering attrs",
    )
    missing = [column for column in attrs if column not in store.cells.columns]
    if missing:
        raise KeyError(f"Filtering columns were not found: {missing!r}")
    if not attrs:
        raise ValueError(
            "Filtering was requested, but no QC columns were found; "
            "pass filtering=False to analyze the unfiltered cell selection"
        )
    if method == "manual":
        allowed = {"lows", "highs", "keep_bounds"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown manual filtering options: {sorted(unknown)!r}")
        if "lows" not in options or "highs" not in options:
            raise ValueError("Manual filtering requires lows and highs")
        lows = list(options["lows"])
        highs = list(options["highs"])
        if len(lows) != len(attrs) or len(highs) != len(attrs):
            raise ValueError("Manual filtering bounds must align with attrs")
        keep_bounds = options.get("keep_bounds", False)
        if not isinstance(keep_bounds, bool):
            raise TypeError("keep_bounds must be a boolean")
        return {
            "enabled": True,
            "method": "manual",
            "attrs": list(attrs),
            "lows": [_manual_bound(bound, "lows") for bound in lows],
            "highs": [_manual_bound(bound, "highs") for bound in highs],
            "keepBounds": keep_bounds,
        }
    allowed = {
        "min_p",
        "max_p",
        "sample_column",
        "n_mads",
        "min_cells_per_sample",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown automatic filtering options: {sorted(unknown)!r}")
    min_p = _finite_real(options.get("min_p", 0.01), "min_p")
    max_p = _finite_real(options.get("max_p", 0.99), "max_p")
    if not 0 < min_p < max_p < 1:
        raise ValueError("Automatic filtering requires 0 < min_p < max_p < 1")
    sample_column = options.get("sample_column")
    if sample_column is not None and (
        not isinstance(sample_column, str) or not sample_column
    ):
        raise TypeError("sample_column must be a non-empty string or None")
    if sample_column is not None and sample_column not in store.cells.columns:
        raise KeyError(f"Sample column {sample_column!r} was not found")
    n_mads = _finite_real(options.get("n_mads", 3.0), "n_mads")
    if n_mads <= 0:
        raise ValueError("n_mads must be finite and positive")
    min_cells = _positive_int(
        options.get("min_cells_per_sample", 20),
        "min_cells_per_sample",
    )
    if min_cells < 2:
        raise ValueError("min_cells_per_sample must be at least 2")
    if sample_column is not None and (min_p != 0.01 or max_p != 0.99):
        raise ValueError("min_p and max_p cannot be changed with sample_column")
    return {
        "enabled": True,
        "method": "auto",
        "attrs": list(attrs),
        "minP": min_p,
        "maxP": max_p,
        "sampleColumn": sample_column,
        "nMads": n_mads,
        "minCellsPerSample": min_cells,
    }


def resolve_pipeline_recipe(
    store: Any,
    *,
    assay: str | None,
    label: str | None,
    cell_key: str,
    filtering: bool | Mapping[str, object],
    harmony_batch_columns: Sequence[str] | None,
    hvg_count: int,
    pca_dims: int,
    neighbors_k: int,
    umap: bool,
    leiden: Mapping[str, object] | bool,
    cell_cycle: bool,
    paris: bool,
    doublets: bool,
    markers: bool,
    snapshot_columns: Sequence[str],
) -> ResolvedPipelineRecipe:
    assay_name = assay or store._defaultAssay
    if not isinstance(assay_name, str) or not assay_name:
        raise ValueError("No assay was provided and no default is configured")
    resolved_assay = store._get_assay(assay_name)
    if not isinstance(resolved_assay, RNAassay):
        raise TypeError("The basic pipeline requires an RNA assay")
    if label is not None and (not isinstance(label, str) or not label):
        raise TypeError("label must be a non-empty string or None")
    if not isinstance(cell_key, str) or not cell_key:
        raise TypeError("cell_key must be a non-empty string")
    if cell_key not in store.cells.columns:
        raise KeyError(f"Cell selection column {cell_key!r} was not found")
    if np.dtype(store.cells.get_dtype(cell_key)) != np.dtype(bool):
        raise TypeError("cell_key must identify a boolean metadata column")
    for flag, name in (
        (umap, "umap"),
        (cell_cycle, "cell_cycle"),
        (paris, "paris"),
        (doublets, "doublets"),
        (markers, "markers"),
    ):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be a boolean")
    partitions = _resolve_leiden(leiden)
    if not partitions and (doublets or markers):
        raise ValueError("doublets and markers require at least one Leiden candidate")
    snapshots = _column_sequence(snapshot_columns, "snapshot_columns")
    result_fields = {
        "highly_variable_features",
        "s_score",
        "g2m_score",
        "cell_cycle_phase",
        "umap_1",
        "umap_2",
        "paris",
        "clusters",
        "doublet_score",
        *(f"leiden_{key}" for key, _value in partitions),
    }
    collisions = set(snapshots) & ({"I", "ids", "names"} | result_fields)
    if collisions:
        raise ValueError(
            f"snapshot_columns collide with reserved run fields: {sorted(collisions)!r}"
        )
    missing_snapshots = [
        column for column in snapshots if column not in store.cells.columns
    ]
    if missing_snapshots:
        raise KeyError(f"Snapshot columns were not found: {missing_snapshots!r}")
    harmony_columns = (
        ()
        if harmony_batch_columns is None
        else _column_sequence(harmony_batch_columns, "harmony_batch_columns")
    )
    if harmony_batch_columns is not None and not harmony_columns:
        raise ValueError("harmony_batch_columns must not be empty")
    missing_harmony = [
        column for column in harmony_columns if column not in store.cells.columns
    ]
    if missing_harmony:
        raise KeyError(f"Harmony columns were not found: {missing_harmony!r}")
    filtering_config = _resolve_filtering(store, assay_name, filtering)
    filter_columns = tuple(filtering_config.get("attrs", ()))
    sample_column = filtering_config.get("sampleColumn")
    if isinstance(sample_column, str):
        filter_columns = (*filter_columns, sample_column)
    cell_snapshot_columns = tuple(
        dict.fromkeys(("names", *filter_columns, *harmony_columns, *snapshots))
    )
    stage_order = (
        "input_snapshot",
        "filtering",
        "cell_cycle",
        "highly_variable_features",
        "normalization",
        "pca",
        "harmony",
        "ann_index",
        "neighbors",
        "connectivity",
        "embedding_initialization",
        "umap",
        *(f"leiden_{key}" for key, _value in partitions),
        "paris",
        "cluster_selection",
        "doublet_graph",
        "doublets",
        "markers",
    )
    return ResolvedPipelineRecipe(
        assay=assay_name,
        label=label,
        cell_key=cell_key,
        filtering=filtering_config,
        harmony_batch_columns=harmony_columns,
        hvg_count=_positive_int(hvg_count, "hvg_count"),
        pca_dims=_positive_int(pca_dims, "pca_dims"),
        neighbors_k=_positive_int(neighbors_k, "neighbors_k"),
        umap=umap,
        leiden_partitions=partitions,
        cell_cycle=cell_cycle,
        paris=paris,
        doublets=doublets,
        markers=markers,
        snapshot_columns=snapshots,
        cell_snapshot_columns=cell_snapshot_columns,
        stage_order=stage_order,
    )
