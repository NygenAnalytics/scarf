"""Feature resolution and bounded group reducers."""

from collections.abc import Mapping, Sequence
from ..utils.arrays import sort_categories
from typing import Any

import numpy as np
import pandas as pd

from ..features.values import (
    ResolvedFeature as ResolvedFeature,
    fetch_normalized_feature_matrix as fetch_normalized_feature_matrix,
    iter_normalized_feature_blocks,
    resolve_feature as resolve_feature,
)
from ..storage.artifacts import ArtifactRef, artifact_group, inspect_artifact
from ..storage.selections import read_stored_selection_indices
from ..storage.types import as_zarr_array
from ._contracts import (
    FeatureRef,
    NormalizationSpec,
    StudyDesign,
)


def _artifact_cell_selection(
    store: Any,
    ref: ArtifactRef,
    *,
    label: str,
) -> ArtifactRef:
    status = inspect_artifact(store.zw, ref)
    raw_selection = (status.inputs or {}).get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ValueError(f"{label} artifact has no cell-selection input")
    try:
        return ArtifactRef.from_dict(raw_selection)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} artifact has an invalid cell-selection input"
        ) from exc


def _validated_embedding_selection(
    store: Any,
    layout: ArtifactRef,
) -> ArtifactRef:
    """Validate an embedding producer and return its exact cell selection."""
    if not isinstance(layout, ArtifactRef):
        raise TypeError("layout must be an ArtifactRef")
    if layout.kind != "embedding":
        raise ValueError("layout must identify an embedding artifact")
    status = inspect_artifact(store.zw, layout)
    if not status.complete:
        raise ValueError("Embedding artifact is unavailable or incomplete")

    selection = _artifact_cell_selection(store, layout, label="Embedding")
    if status.operation == "import_dimreduc":
        from ..embeddings.imported import validate_imported_embedding_artifact

        validate_imported_embedding_artifact(store.zw, layout)
        return selection

    if status.operation not in {"run_umap", "run_tsne"}:
        raise ValueError(
            "Embedding artifact must be produced by import_dimreduc, run_umap, "
            "or run_tsne"
        )
    raw_graph = (status.inputs or {}).get("graph")
    if not isinstance(raw_graph, Mapping):
        raise ValueError("Embedding artifact has no graph input")
    try:
        graph = ArtifactRef.from_dict(raw_graph)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Embedding artifact has an invalid graph input") from exc
    if layout.scope != graph.scope or layout.assay != graph.assay:
        raise ValueError("Embedding artifact scope does not match its graph input")

    from ..graph.feature_projection import graph_cell_selection

    graph_selection = graph_cell_selection(store.zw, graph)
    if selection != graph_selection:
        raise ValueError(
            "Embedding artifact and graph must share the same cell selection"
        )
    return selection


def _resolve_grouping(
    store: Any,
    *,
    group_by: str | tuple[str, ...] | None,
    groups: ArtifactRef | None,
    cell_key: str,
) -> tuple[tuple[str, ...], np.ndarray, list[np.ndarray]]:
    """Resolve either explicit live metadata or one immutable label artifact."""
    if (group_by is None) == (groups is None):
        raise ValueError("Provide exactly one of group_by or groups")
    if groups is None:
        group_keys = (group_by,) if isinstance(group_by, str) else tuple(group_by or ())
        if len(group_keys) == 0 or len(group_keys) > 2:
            raise ValueError("group_by must have 1 or 2 keys")
        cell_idx = np.asarray(store.cells.active_index(cell_key), dtype=np.int64)
        return (
            group_keys,
            cell_idx,
            [np.asarray(store.cells.fetch(key, key=cell_key)) for key in group_keys],
        )

    if not isinstance(groups, ArtifactRef):
        raise TypeError("groups must be an ArtifactRef")
    if cell_key != "I":
        raise ValueError("cell_key cannot override an artifact's stored cell selection")
    status = inspect_artifact(store.zw, groups)
    if not status.complete:
        raise ValueError("Grouping artifact is unavailable or incomplete")
    selection = _artifact_cell_selection(store, groups, label="Grouping")
    cell_idx = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    value_name = {
        "cell_cycle": "phase",
        "cluster_cut": "labels",
    }.get(groups.kind, "values")
    group = artifact_group(store.zw, groups)
    if value_name not in group:
        raise ValueError(
            f"Grouping artifact has no canonical {value_name!r} label array"
        )
    values = np.asarray(as_zarr_array(group[value_name], name=value_name)[:])
    if values.ndim != 1 or values.shape != (len(cell_idx),):
        raise ValueError("Grouping labels do not align with their cell selection")
    return ("groups",), cell_idx, [values]


def _resolve_layout(
    store: Any,
    layout: ArtifactRef,
) -> tuple[np.ndarray, np.ndarray, ArtifactRef]:
    """Resolve one explicit two-dimensional embedding and its stored selection."""
    selection = _validated_embedding_selection(store, layout)
    cell_idx = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    group = artifact_group(store.zw, layout)
    if "values" not in group:
        raise ValueError("Embedding artifact has no canonical values array")
    try:
        values = np.asarray(
            as_zarr_array(group["values"], name="values")[:],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Embedding coordinates must be numeric") from exc
    if values.shape != (len(cell_idx), 2):
        raise ValueError(
            "Embedding must have two columns and one row per selected cell"
        )
    if not np.isfinite(values).all():
        raise ValueError("Embedding coordinates must be finite")
    return values, cell_idx, selection


def coerce_feature_list(
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
) -> list[tuple[str | None, str | FeatureRef]]:
    """Return (group_label, feature) pairs preserving order."""
    if isinstance(features, Mapping):
        out: list[tuple[str | None, str | FeatureRef]] = []
        for group, items in features.items():
            for item in items:
                out.append((str(group), item))
        return out
    return [(None, item) for item in features]


def _cell_metadata_columns(store: Any, keys: Sequence[object]) -> frozenset[str]:
    """Return cell-metadata column names when a plain string key needs them.

    A string key names a cell-metadata column when one exists and a feature
    otherwise. Callers classify every key against this one read of the column
    list instead of listing the metadata columns again for each key.
    """
    if any(isinstance(key, str) for key in keys):
        return frozenset(store.cells.columns)
    return frozenset()


def resolve_cell_selection(
    n: int,
    *,
    subset: np.ndarray | None = None,
    subset_name: str | None = None,
    category_values: np.ndarray | None = None,
    groups: Sequence[Any] | None = None,
) -> tuple[np.ndarray, list[Any] | None]:
    """Build a boolean mask from ``subset`` and optional category ``groups``.

    ``subset`` must be boolean and length ``n`` when provided. ``groups`` keeps
    only those categories from ``category_values`` and defines their order.
    When ``groups`` is omitted, category order is natural via
    :func:`sort_categories` over observed values (or ``None`` if no categories).
    """
    mask = np.ones(n, dtype=bool)
    if subset is not None:
        sub = np.asarray(subset)
        if sub.dtype != bool:
            label = subset_name or "subset_by"
            raise TypeError(f"{label!r} must be boolean; got {sub.dtype}")
        if len(sub) != n:
            raise ValueError("subset_by length must match selected cells")
        mask &= sub

    group_order: list[Any] | None = None
    if category_values is not None:
        cats = np.asarray(category_values)
        if len(cats) != n:
            raise ValueError("category values length must match selected cells")
        present = set(pd.unique(cats).tolist())
        if groups is not None:
            group_order = list(groups)
            if not group_order:
                raise ValueError("groups must be non-empty when provided")
            missing = [g for g in group_order if g not in present]
            if missing:
                raise ValueError(
                    "groups contains labels not present in the data: "
                    + ", ".join(map(str, missing[:10]))
                )
            mask &= np.isin(cats, group_order)
        elif mask.any():
            group_order = sort_categories(list(pd.unique(cats[mask])))
        else:
            group_order = []

    if not mask.any():
        raise ValueError("No cells remain after applying subset/groups filters")
    return mask, group_order


def _summarize_feature_blocks(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    base: pd.DataFrame,
    group_keys: list[str],
    feature_groups: list[str | None],
    normalization: NormalizationSpec | None,
    expression_cutoff: float,
) -> pd.DataFrame:
    grouped = base.groupby(group_keys, observed=True, dropna=False)
    codes = np.full(len(cell_idx), -1, dtype=np.int64)
    codes[base.index.to_numpy()] = grouped.ngroup().to_numpy()
    group_table = grouped.size().reset_index(name="n_cells")
    feature_table = pd.DataFrame(
        {
            "feature": [feature.label for feature in resolved],
            "feature_group": feature_groups,
        }
    )
    grouped_features = feature_table.groupby(["feature", "feature_group"], dropna=False)
    feature_codes = grouped_features.ngroup().to_numpy()
    feature_table = grouped_features.size().reset_index(name="multiplicity")
    shape = (len(group_table), len(feature_table))
    counts = np.zeros(shape, dtype=np.int64)
    means = np.zeros(shape, dtype=np.float64)
    squared_deviations = np.zeros(shape, dtype=np.float64)
    detected = np.zeros(shape, dtype=np.int64)

    for slots, start, values in iter_normalized_feature_blocks(
        store, resolved, cell_idx, normalization
    ):
        block_codes = codes[start : start + len(values)]
        included = block_codes >= 0
        if not included.any():
            continue
        block_values = values[included]
        if np.isinf(block_values).any():
            raise ValueError("Expression values contain infinity after normalization")
        block = pd.DataFrame(block_values)
        grouped = block.groupby(block_codes[included])
        block_means = grouped.mean()
        rows = block_means.index.to_numpy()
        block_counts = grouped.count().to_numpy()
        block_deviations = grouped.var().to_numpy() * np.maximum(block_counts - 1, 0)
        block_deviations[block_counts < 2] = 0
        block_means = block_means.to_numpy(copy=True)
        block_means[block_counts == 0] = 0
        block_detected = (
            (block > expression_cutoff).groupby(block_codes[included]).sum().to_numpy()
        )
        for column, slot in enumerate(slots):
            target = feature_codes[slot]
            previous = counts[rows, target]
            incoming = block_counts[:, column]
            total = previous + incoming
            weight = np.divide(
                incoming, total, out=np.zeros(len(rows)), where=total > 0
            )
            delta = block_means[:, column] - means[rows, target]
            means[rows, target] += delta * weight
            # Include the shift between block means when combining variances.
            squared_deviations[rows, target] += (
                block_deviations[:, column] + delta**2 * previous * weight
            )
            counts[rows, target] = total
            detected[rows, target] += block_detected[:, column]

    means[counts == 0] = np.nan
    variance = np.divide(
        squared_deviations,
        counts - 1,
        out=np.full(shape, np.nan),
        where=counts > 1,
    )
    table = group_table.iloc[
        np.repeat(np.arange(len(group_table)), len(feature_table))
    ].reset_index(drop=True)
    table["n_cells"] *= np.tile(
        feature_table["multiplicity"].to_numpy(), len(group_table)
    )
    for name in ("feature", "feature_group"):
        table[name] = np.tile(feature_table[name].to_numpy(), len(group_table))
    table["mean"] = means.ravel()
    table["fraction"] = detected.ravel() / table["n_cells"]
    table["variance"] = variance.ravel()
    return table[
        [
            *group_keys,
            "feature",
            "feature_group",
            "mean",
            "fraction",
            "n_cells",
            "variance",
        ]
    ]


_MAX_SUMMARY_GROUPS = 500
_MAX_SUMMARY_FEATURES = 2000
_MAX_SUMMARY_SAMPLES = 500


def _check_feature_count(
    pairs: Sequence[tuple[str | None, str | FeatureRef]],
    *,
    max_features: int = _MAX_SUMMARY_FEATURES,
) -> None:
    """Reject an empty feature list or one longer than ``max_features``."""
    if not pairs:
        raise ValueError("At least one feature is required")
    if len(pairs) > max_features:
        raise ValueError(
            f"Too many features ({len(pairs)} > {max_features}). "
            "Raise max_features explicitly if intentional."
        )


def summarize_features_by_group(
    store: Any,
    *,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    group_by: str | tuple[str, ...] | None = None,
    groups: ArtifactRef | None = None,
    cell_key: str = "I",
    from_assay: str | None = None,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    normalization: NormalizationSpec | None = None,
    expression_cutoff: float = 0.0,
    max_groups: int = _MAX_SUMMARY_GROUPS,
    max_features: int = _MAX_SUMMARY_FEATURES,
    max_samples: int = _MAX_SUMMARY_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Aggregate features by group. With sample_by, samples get equal weight.

    Missing group combinations are omitted (not filled with zeros).
    """
    pairs = coerce_feature_list(features)
    _check_feature_count(pairs, max_features=max_features)
    resolved = [
        resolve_feature(store, feat, from_assay=from_assay) for _, feat in pairs
    ]
    grouping = _resolve_grouping(
        store,
        group_by=group_by,
        groups=groups,
        cell_key=cell_key,
    )
    return _summarize_resolved_features(
        store,
        resolved,
        [g for g, _ in pairs],
        grouping,
        sample_by=sample_by,
        study_design=study_design,
        normalization=normalization,
        expression_cutoff=expression_cutoff,
        max_groups=max_groups,
        max_samples=max_samples,
    )


def _summarize_resolved_features(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    group_labels: list[str | None],
    grouping: tuple[tuple[str, ...], np.ndarray, list[np.ndarray]],
    *,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    normalization: NormalizationSpec | None = None,
    expression_cutoff: float = 0.0,
    max_groups: int = _MAX_SUMMARY_GROUPS,
    max_samples: int = _MAX_SUMMARY_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Aggregate resolved features over a grouping from ``_resolve_grouping``.

    ``group_labels`` holds each feature's bracket group, aligned with
    ``resolved``.
    """
    condition_by: str | None = None
    if study_design is not None:
        sample_by = study_design.sample_by
        condition_by = study_design.condition_by

    cells = store.cells
    group_keys, cell_idx, group_cols = grouping
    n_groups = int(
        pd.DataFrame({k: c for k, c in zip(group_keys, group_cols)})
        .drop_duplicates()
        .shape[0]
    )
    if n_groups > max_groups:
        raise ValueError(
            f"Too many groups ({n_groups} > {max_groups}). "
            "Raise max_groups explicitly if intentional."
        )

    base = pd.DataFrame({gk: col for gk, col in zip(group_keys, group_cols)})
    gb_keys = list(group_keys)

    if sample_by is not None:
        samples = np.asarray(cells.fetch_all(sample_by))[cell_idx]
        if condition_by is not None:
            conditions = np.asarray(cells.fetch_all(condition_by))[cell_idx]
            check = pd.DataFrame({"sample": samples, "condition": conditions})
            nunique = check.groupby("sample", observed=False)["condition"].nunique()
            bad = nunique[nunique > 1]
            if len(bad):
                raise ValueError(
                    "condition_by is not constant within sample(s): "
                    + ", ".join(map(str, list(bad.index[:10])))
                )
        valid = pd.notna(samples) & (np.asarray(samples, dtype=object) != "")
        if int(valid.sum()) == 0:
            raise ValueError("No cells with valid sample_by values")
        uniq_samples = pd.unique(np.asarray(samples)[valid])
        if len(uniq_samples) > max_samples:
            raise ValueError(
                f"Too many samples ({len(uniq_samples)} > {max_samples}). "
                "Raise max_samples explicitly if intentional."
            )
        base = base.loc[valid].copy()
        base["sample"] = np.asarray(samples)[valid]
        gb_keys.insert(0, "sample")

    summary = _summarize_feature_blocks(
        store,
        resolved,
        cell_idx,
        base,
        gb_keys,
        group_labels,
        normalization,
        expression_cutoff,
    )
    if sample_by is not None:
        agg_keys = [*group_keys, "feature", "feature_group"]
        aggregate = (
            summary.groupby(agg_keys, observed=False, dropna=False)
            .agg(
                mean=("mean", "mean"),
                fraction=("fraction", "mean"),
                n_cells=("n_cells", "sum"),
                n_samples=("sample", "nunique"),
                variance=("variance", "mean"),
            )
            .reset_index()
        )
        return aggregate, summary
    return summary, None
