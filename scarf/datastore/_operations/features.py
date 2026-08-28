import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray

from ...features.variability import DEFAULT_HVG_BLACKLIST, HVG_UBIQUITOUS_SLACK
from ...assay.feature_summary import (
    ensure_feature_summary,
    feature_summary_selected_count,
    feature_summary_values,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_path,
    callable_identity,
    fingerprint_array,
    fingerprint_strings,
    inspect_artifact,
    provenance_hash,
)
from ...storage.feature_selection import (
    _feature_selection_plan,
    _feature_selection_values,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
    publish_feature_selection_alias,
    validate_feature_selection_label,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...assay import RNAassay, lib_size_feature_stream_eligible
from ...assay.normalization import reject_unknown_normalization_params
from ...features.enrichment.results import EnrichmentResult
from ...features.markers.table import (
    MARKER_ADJUSTMENT_METHOD,
    MARKER_ADJUSTMENT_SCOPE,
    MARKER_ALTERNATIVE,
    MARKER_CONTINUITY_CORRECTION,
    MARKER_METHOD,
    MARKER_STAT_COLUMNS,
    MARKER_TIE_CORRECTION,
    _validate_marker_slot,
    load_marker_table,
)
from ...features.statistical import (
    ANOVA_COLUMNS,
    DUNN_COLUMNS,
    GroupComparisonResult,
    KRUSKAL_WALLIS_COLUMNS,
    MANN_WHITNEY_COLUMNS,
    StatisticalTestResult,
    WELCH_COLUMNS,
    WILCOXON_COLUMNS,
    _PARAMETRIC_TESTS,
    adjust_pvalues,
    compare_group_distributions,
    resolve_group_order,
)
from ...features.values import fetch_normalized_feature_matrix, resolve_feature
from ...metadata.arguments import (
    AucellArguments,
    MarkerTableArguments,
    StatisticalTestingArguments,
    WaggrArguments,
)
from ...metadata.selection import (
    CellField,
    FeatureRef,
    NormalizationSpec,
    StudyDesign,
    valid_category_mask,
)
from ...metadata.rows import read_metadata_missing_rows
from ...utils.arrays import array_digest
from ...utils.compute import controlled_compute
from ...utils.logging import logger
from ...utils.progress import iter_progress
from .enrichment_store import (
    _ENRICHMENT_LAYOUT,
    _EnrichmentScorer,
    _enrichment_artifact_matches,
    _enrichment_artifact_ref,
    _load_enrichment_result,
    _publish_enrichment_artifact,
    _validate_enrichment_label,
    _write_enrichment_slot,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _FeatureOperationsBase
else:
    _FeatureOperationsBase = object

_MARKER_STAT_COLUMNS = MARKER_STAT_COLUMNS
_MARKER_OUT_COLUMNS = ("feature_index", *_MARKER_STAT_COLUMNS)


@dataclass(frozen=True, slots=True)
class _StatisticalSelection:
    """Effective rows and identities for one statistical-testing design."""

    groups: np.ndarray
    samples: np.ndarray | None
    pairs: np.ndarray | None
    subset_values: np.ndarray | None
    selection_mask: np.ndarray
    effective_cell_idx: np.ndarray
    group_order: tuple[Any, ...]
    cell_selection_fingerprint: str
    group_fingerprint: str
    subset_fingerprint: str | None
    sample_fingerprint: str | None
    pair_fingerprint: str | None


def _statistical_storage_columns(
    method: str,
    posthoc: str | None,
) -> tuple[str, ...]:
    """Persisted primary-table columns, including adjustment.

    For Kruskal-Wallis the primary table is always the omnibus result,
    whether or not a post-hoc test was requested.
    """
    base_map: dict[str, tuple[str, ...]] = {
        "mann_whitney": MANN_WHITNEY_COLUMNS,
        "wilcoxon": WILCOXON_COLUMNS,
        "welch": WELCH_COLUMNS,
        "t_test": WELCH_COLUMNS,
        "one_way_anova": ANOVA_COLUMNS,
    }
    if method == "kruskal_wallis":
        return (*KRUSKAL_WALLIS_COLUMNS, "p_value_adjusted")
    if method in base_map:
        return (*base_map[method], "p_value_adjusted")
    raise ValueError(f"Unknown statistical test method: {method!r}")


def _statistical_posthoc_columns(
    posthoc: str | None,
) -> tuple[str, ...]:
    """Persisted post-hoc table columns, including adjustment."""
    if posthoc == "dunn":
        return (*DUNN_COLUMNS, "p_value_adjusted")
    return ()


def _native_artifact_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _value_fingerprint(values: Any) -> str:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "S", "U"}:
        return fingerprint_strings(array)
    return fingerprint_array(array)


def _pool_adjust(
    tables: dict[str, pd.DataFrame],
    adjustment: str,
) -> dict[str, pd.DataFrame]:
    """Add ``p_value_adjusted`` to each table with one pooled correction pass."""
    if not tables:
        return {}
    all_p_values = np.concatenate(
        [
            table["p_value"].to_numpy(dtype=np.float64, copy=False)
            for table in tables.values()
        ]
    )
    adjusted = adjust_pvalues(all_p_values, adjustment)
    offset = 0
    out: dict[str, pd.DataFrame] = {}
    for label, table in tables.items():
        frame = table.copy()
        n_rows = len(frame)
        frame["p_value_adjusted"] = adjusted[offset : offset + n_rows]
        offset += n_rows
        out[label] = frame
    return out


def _statistical_slot_key(
    cell_key: str | None,
    group_key: str,
    method: str,
    posthoc: str | None,
    variant_digest: str,
) -> str:
    """Return an unambiguous index key for one statistical-test variant."""
    return provenance_hash(
        {
            "cell_key": cell_key,
            "group_key": group_key,
            "method": method,
            "posthoc": posthoc,
            "variant_digest": variant_digest,
        }
    )


def _statistical_index_group(
    store: Any,
    source_assay: str | None,
    *,
    create: bool,
) -> zarr.Group:
    parent = (
        store.zw
        if source_assay is None
        else as_zarr_group(store.zw[source_assay], name=source_assay)
    )
    if "statistical_tests" not in parent:
        if not create:
            raise KeyError(
                "ERROR: Couldn't find statistical test results. Make sure "
                "`run_statistical_testing` was called with the same source, "
                "selection, grouping, method, and variant parameters."
            )
        parent.create_group("statistical_tests")
    return as_zarr_group(parent["statistical_tests"], name="statistical_tests")


def _statistical_normalization(
    normalization: NormalizationSpec | None,
) -> dict[str, str]:
    source = (
        "assay" if normalization is None else getattr(normalization, "source", None)
    )
    transform = (
        "none" if normalization is None else getattr(normalization, "transform", None)
    )
    if source not in ("assay", "raw"):
        raise ValueError("normalization.source must be 'assay' or 'raw'")
    if transform not in ("none", "log1p"):
        raise ValueError("normalization.transform must be 'none' or 'log1p'")
    return {"source": source, "transform": transform}


def _statistical_cell_indices(store: Any, cell_key: str | None) -> np.ndarray:
    if cell_key is None:
        return np.arange(store.cells.N, dtype=np.int64)
    return np.asarray(store.cells.active_index(cell_key))


def _fetch_statistical_column(
    store: Any,
    column: str,
    cell_key: str | None,
) -> np.ndarray:
    if cell_key is None:
        return np.asarray(store.cells.fetch_all(column))
    return np.asarray(store.cells.fetch(column, key=cell_key))


def _normalized_variant_groups(
    groups: Sequence[Any] | None,
) -> tuple[Any, ...] | None:
    if groups is None:
        return None
    return tuple(_native_artifact_value(value) for value in groups)


def _normalized_variant_comparisons(
    comparisons: Sequence[tuple[Any, Any]] | None,
) -> tuple[tuple[Any, Any], ...] | None:
    if comparisons is None:
        return None
    return tuple(
        (_native_artifact_value(left), _native_artifact_value(right))
        for left, right in comparisons
    )


def _statistical_key_labels(labels: Sequence[str]) -> list[str]:
    """Return stable, unique persisted table labels in input order."""
    panel_keys: Sequence[Any]
    if len(set(labels)) != len(labels):
        panel_keys = range(len(labels))
    else:
        panel_keys = labels
    return [str(key) for key in panel_keys]


def _statistical_equal_var(method: str | None) -> bool | None:
    """Return the equal-variance flag persisted for a test method.

    Welch's t-test always uses unequal variances; other tests do not carry
    the parameter.
    """
    return False if method in ("welch", "t_test") else None


def _statistical_variant_digest(
    *,
    tested_features: tuple[str, ...],
    key_labels: tuple[str, ...],
    groups: tuple[Any, ...] | None,
    comparisons: tuple[tuple[Any, Any], ...] | None,
    adjustment: str,
    sample_by: str | None,
    pair_by: str | None,
    subset_by: str | None,
    sample_stat: str = "mean",
    expression_cutoff: float = 0.0,
    normalization: dict[str, Any] | None = None,
    alternative: str = "two-sided",
    equal_var: bool | None = None,
) -> str:
    """Deterministic digest identifying one retrievable test variant.

    The digest keys on explicit variant parameters only. Underlying column
    and selection content changes are captured by
    ``OperationArguments`` provenance for reuse decisions and mirrored in
    slot attributes by ``statistical_reuse_is_valid``.
    """
    return provenance_hash(
        {
            "tested_features": tested_features,
            "key_labels": key_labels,
            "groups": groups,
            "comparisons": comparisons,
            "adjustment": adjustment,
            "sample_by": sample_by,
            "pair_by": pair_by,
            "subset_by": subset_by,
            "sample_stat": sample_stat,
            "expression_cutoff": float(expression_cutoff),
            "normalization": normalization,
            "alternative": alternative,
            "equal_var": equal_var,
        }
    )


def _resolve_assay_and_cell_key(
    store: Any,
    from_assay: str | None,
    cell_key: str | None,
) -> tuple[str, str | None]:
    """Resolve the assay and normalize ``cell_key`` for statistical tests.

    ``cell_key=None`` selects every cell in the store (matching
    ``scarf.plotting.distribution``); it is not replaced with the latest key.
    """
    if from_assay is None:
        from_assay = store._defaultAssay
    if from_assay is None:
        raise ValueError("No default assay is configured")
    return from_assay, cell_key


def _marker_artifact_index(group: zarr.Group) -> dict[str, Any]:
    """Return a validated, detached marker index.

    The three nested keys are cell key, group key, and feature-selection
    artifact id. Keeping the feature identity in the index prevents one marker
    search from silently replacing a table computed over another universe.
    """

    raw_index = group.attrs.get("artifacts", {})
    if not isinstance(raw_index, dict):
        raise ValueError("Marker artifact index is invalid")
    index: dict[str, Any] = {}
    for cell_key, raw_groups in raw_index.items():
        if not isinstance(cell_key, str) or not isinstance(raw_groups, dict):
            raise ValueError("Marker artifact index is invalid")
        groups: dict[str, Any] = {}
        for group_key, raw_selections in raw_groups.items():
            if not isinstance(group_key, str) or not isinstance(
                raw_selections,
                dict,
            ):
                raise ValueError("Marker artifact index is invalid")
            selections: dict[str, Any] = {}
            for feature_id, raw_ref in raw_selections.items():
                if not isinstance(feature_id, str) or not isinstance(raw_ref, dict):
                    raise ValueError("Marker artifact index is invalid")
                try:
                    ref = ArtifactRef.from_dict(raw_ref)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Marker artifact index is invalid") from exc
                selections[feature_id] = ref.to_dict()
            groups[group_key] = selections
        index[cell_key] = groups
    return index


def _select_indexed_marker_artifact(
    index: dict[str, Any],
    cell_key: str,
    group_key: str,
) -> tuple[str, ArtifactRef]:
    selections = index.get(cell_key, {}).get(group_key, {})
    if not selections:
        raise KeyError(f"Marker artifact for {cell_key!r}/{group_key!r} was not found")
    if len(selections) != 1:
        raise ValueError(
            "Multiple feature-specific marker tables exist for this cell and "
            "group key; pass the ArtifactRef returned by run_marker_search "
            "as marker="
        )
    feature_id, raw_ref = next(iter(selections.items()))
    return feature_id, ArtifactRef.from_dict(raw_ref)


def _shared_marker_feature_index(markers: dict[Any, pd.DataFrame]) -> np.ndarray:
    shared: np.ndarray | None = None
    populated_names: set[str] = set()
    for cluster_id, vals in markers.items():
        if len(vals) == 0:
            continue
        group_name = str(cluster_id)
        if group_name in populated_names:
            raise ValueError("Marker group labels must remain unique as strings")
        populated_names.add(group_name)
        raw_index = np.asarray(vals.index.values)
        if raw_index.ndim != 1 or raw_index.dtype.kind not in {"i", "u"}:
            raise ValueError(
                "Marker feature indices must use a one-dimensional integer index"
            )
        if not vals.index.is_unique:
            raise ValueError("Marker feature indices must be unique within each group")
        index = raw_index.astype(np.int64, copy=False)
        if (index < 0).any() or (index > np.iinfo(np.int32).max).any():
            raise ValueError("Marker feature indices must fit non-negative int32")
        ordered = np.sort(index)
        if shared is None:
            shared = ordered
        elif not np.array_equal(ordered, shared):
            raise ValueError("Marker groups must contain identical feature index sets")
    if shared is None:
        raise ValueError("Cannot save empty marker results")
    return shared.astype(np.int32)


def _marker_stats_matrix(vals: pd.DataFrame, feature_index: np.ndarray) -> np.ndarray:
    aligned = vals.reindex(feature_index)
    stats = np.asarray(
        aligned.loc[:, list(_MARKER_STAT_COLUMNS)].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(stats).all():
        raise ValueError("Marker statistics must all be finite")
    return stats


def _write_compact_marker_stats(
    cluster_group: zarr.Group,
    stats: np.ndarray,
) -> None:
    from ...storage.arrays import create_zarr_dataset

    n_features = int(stats.shape[0])
    n_stats = int(stats.shape[1])
    arr = create_zarr_dataset(
        cluster_group,
        "stats",
        (n_features, n_stats),
        "float64",
        (n_features, n_stats),
    )
    arr[:] = stats


def _load_marker_cluster_frame(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
    feature_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Thin wrapper around the shared version-aware marker reader."""
    return load_marker_table(
        slot_group,
        cluster_group,
        feature_names,
        group_id=group_id,
        feature_ids=feature_ids,
    )


def _aligned_feature_labels(values: np.ndarray, index: pd.Index) -> np.ndarray:
    """Return feature labels as a hashable NumPy array aligned to ``index``."""
    labels = np.asarray(values, dtype=object).reshape(-1)
    return labels[np.asarray(index, dtype=np.intp)]


def _group_assignment_digest(values: np.ndarray) -> str:
    return array_digest(np.asarray(values).astype(str))


class _FeatureOperationsMixin(_FeatureOperationsBase):
    def _require_feature_write(self, operation: str) -> None:
        if self.zarr_mode != "r+":
            raise PermissionError(
                f"{operation} requires a DataStore opened with zarr_mode='r+'"
            )

    def _ensure_all_features(self, assay: Any) -> ArtifactRef:
        """Create or reuse the canonical all-true feature universe."""
        self._require_feature_write("Feature selection")
        resolved_assay = self._get_assay(assay) if isinstance(assay, str) else assay
        from ...graph.state import read_assay_state_document

        # Reject malformed or legacy state before mutation without requiring an
        # unrelated current graph chain to remain available. A newly published
        # normalized artifact can then replace a damaged older chain.
        read_assay_state_document(self.zw, resolved_assay.name)
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(resolved_assay)
        values = np.ones(resolved_assay.feats.N, dtype=bool)
        payload_fingerprint = fingerprint_array(values)
        planned = _feature_selection_plan(
            self.zw,
            assay=resolved_assay.name,
            n_features=resolved_assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation="create_all_features",
            parameters={
                "dataset_fingerprint": self._ensure_dataset_fingerprint(
                    resolved_assay.name
                ),
                "ordered_feature_ids_fingerprint": feature_ids_fingerprint,
            },
            inputs={},
            execution_options={"label": "all_features"},
            expected_payload_fingerprint=payload_fingerprint,
        )
        _write_feature_selection(
            self.zw,
            planned,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            payload={"values": values},
        )
        publish_feature_selection_alias(
            self.zw,
            resolved_assay.name,
            "all_features",
            planned.ref,
        )
        return planned.ref

    def set_feature_selection(
        self,
        *,
        from_assay: str | None = None,
        mask: np.ndarray | None = None,
        feature_indexes: Sequence[int] | None = None,
        label: str,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Persist and publish an explicit feature mask."""
        self._require_feature_write("set_feature_selection")
        validate_feature_selection_label(label)
        assay = self._get_assay(from_assay)
        all_features = self._ensure_all_features(assay)
        if (mask is None) == (feature_indexes is None):
            raise ValueError("Provide exactly one of mask or feature_indexes")
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError("mask must be a NumPy array")
            if mask.shape != (assay.feats.N,):
                raise ValueError(f"mask must have shape ({assay.feats.N},)")
            if mask.dtype != np.dtype(bool):
                raise TypeError("mask must have boolean dtype")
            values = mask.copy()
        else:
            assert feature_indexes is not None
            indexes = np.asarray(feature_indexes)
            if indexes.ndim != 1:
                raise ValueError("feature_indexes must be one-dimensional")
            if indexes.size and not np.issubdtype(indexes.dtype, np.integer):
                raise TypeError("feature_indexes must contain only integers")
            indexes = indexes.astype(np.int64, copy=False)
            if np.any(indexes < 0) or np.any(indexes >= assay.feats.N):
                raise IndexError("feature_indexes contains an out-of-range index")
            if np.unique(indexes).size != indexes.size:
                raise ValueError("feature_indexes contains duplicate indexes")
            values = np.zeros(assay.feats.N, dtype=bool)
            values[indexes] = True
        if not values.any():
            raise ValueError("Feature selection must contain at least one feature")
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay)
        values_fingerprint = fingerprint_array(values)
        planned = _feature_selection_plan(
            self.zw,
            assay=assay.name,
            n_features=assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation="set_feature_selection",
            parameters={"values_fingerprint": values_fingerprint},
            inputs={"all_features": all_features},
            execution_options={
                "label": label,
                "invalidate_cache": invalidate_cache,
            },
            expected_payload_fingerprint=values_fingerprint,
            invalidate_cache=invalidate_cache,
        )
        _write_feature_selection(
            self.zw,
            planned,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            payload={"values": values},
        )
        publish_feature_selection_alias(
            self.zw,
            assay.name,
            label,
            planned.ref,
        )
        return planned.ref

    def select_detected_features(
        self,
        from_assay: str | None = None,
        cell_key: str = "I",
        min_cells: int = 20,
        label: str = "detected_features",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Select features detected in at least ``min_cells`` selected cells."""
        self._require_feature_write("select_detected_features")
        validate_feature_selection_label(label)
        if isinstance(min_cells, bool) or not isinstance(min_cells, int):
            raise TypeError("min_cells must be an integer")
        if min_cells < 0:
            raise ValueError("min_cells must be non-negative")
        assay = self._get_assay(from_assay)
        self._ensure_all_features(assay)
        cell_selection = self._ensure_cell_selection(cell_key)
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            invalidate_cache=invalidate_cache,
        )
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay)
        planned = _feature_selection_plan(
            self.zw,
            assay=assay.name,
            n_features=assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation="select_detected_features",
            parameters={"min_cells": min_cells},
            inputs={"feature_summary": summary_ref},
            execution_options={
                "label": label,
                "invalidate_cache": invalidate_cache,
            },
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            detected_values = np.asarray(
                _feature_selection_values(self.zw, planned.ref),
                dtype=bool,
            )
        else:
            n_selected = feature_summary_selected_count(
                self.zw,
                cell_selection,
                n_cells=assay.cells.N,
            )
            summary = feature_summary_values(
                self.zw,
                summary_ref,
                n_selected=n_selected,
            )
            detected = summary.get("normed_n")
            if detected is None:
                detected = summary["document_frequency"]
            detected_values = np.asarray(detected >= min_cells, dtype=bool)
        if not detected_values.any():
            raise ValueError(
                "Detected-feature selection contains no features; lower min_cells"
            )
        if not planned.reused:
            _write_feature_selection(
                self.zw,
                planned,
                ordered_feature_ids_fingerprint=feature_ids_fingerprint,
                payload={"values": detected_values},
            )
        publish_feature_selection_alias(
            self.zw,
            assay.name,
            label,
            planned.ref,
        )
        return planned.ref

    def mark_hvgs(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        min_cells: int = 20,
        top_n: int = 1000,
        min_var: float = -np.inf,
        max_var: float = np.inf,
        min_mean: float = -np.inf,
        max_mean: float = np.inf,
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        blacklist: str = DEFAULT_HVG_BLACKLIST,
        keep_bounds: bool = False,
        show_plot: bool = True,
        label: str = "hvgs",
        max_cells: float | None = None,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        invalidate_cache: bool = False,
        **plot_kwargs: Any,
    ) -> ArtifactRef:
        """Identify and mark genes as highly variable genes (HVGs). This is a
        critical and required feature selection step and is only applicable to
        RNAassay type of assays.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for HVG selection. By default, all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            min_cells: Minimum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. Large values for this parameter might make it difficult
                       to identify rare populations of cells. Very small values might lead to a higher signal-to-noise
                       ratio in the selected features. (Default: 20)
            max_cells: Maximum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. When omitted, genes missing in at most
                       ``HVG_UBIQUITOUS_SLACK`` selected cells are excluded (``n_selected - 20``). Pass
                       ``inf`` to disable the ubiquitous-gene filter.
            top_n: Number of top most variable genes to be set as HVGs. This value is ignored if a value is provided
                   for `min_var` parameter. (Default: 1000)
            min_var: Minimum variance threshold for HVG selection. (Default: -Infinity)
            max_var: Maximum variance threshold for HVG selection. (Default: Infinity)
            min_mean: Minimum mean value of expression threshold for HVG selection. (Default: -Infinity)
            max_mean: Maximum mean value of expression threshold for HVG selection. (Default: Infinity)
            n_bins: Number of bins into which the mean expression is binned. (Default: 200)
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess
                         (Default: 0.1)
            bin_strategy: Strategy used to construct bins and variance anchors. (Default: 'adaptive')
            blacklist: Regex of gene names to exclude from HVGs. Matching is case-insensitive.
                       Default excludes mitochondrial, ribosomal, cell-cycle (``CCN*``), HLA/H2, histone,
                       and common human sex-linked genes (``XIST``, Y-chromosome markers).
            keep_bounds: If True, retain upper cell-count and expression-statistic bounds.
                         The ``min_cells`` boundary is always inclusive. (Default: False)
            show_plot: If True then a diagnostic scatter plot is shown with HVGs highlighted. (Default: True)
            label: Feature-selection label to publish. (Default: 'hvgs')
            plot_kwargs: Named parameters forwarded to ``plotting.highly_variable_features``
                         (``figsize``, ``label_size``, ``point_sizes``, ``colormaps``).

        Returns:
            The persisted HVG feature-selection artifact.
        """

        self._require_feature_write("mark_hvgs")
        validate_feature_selection_label(label)
        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        if type(assay) != RNAassay:  # noqa: E721
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to RNAassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        self._ensure_all_features(assay)
        cell_selection = self._ensure_cell_selection(cell_key)
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            invalidate_cache=invalidate_cache,
        )
        n_selected = feature_summary_selected_count(
            self.zw,
            cell_selection,
            n_cells=assay.cells.N,
        )
        if max_cells is None:
            candidate_max = n_selected - HVG_UBIQUITOUS_SLACK
            if candidate_max <= min_cells:
                max_cells_int: int | float = np.inf
                logger.info(
                    "Skipping ubiquitous-gene HVG filter because too few cells are "
                    f"selected ({n_selected}); need more than "
                    f"{min_cells + HVG_UBIQUITOUS_SLACK} cells to apply "
                    f"max_cells = n_selected - {HVG_UBIQUITOUS_SLACK}."
                )
            else:
                max_cells_int = int(candidate_max)
                logger.debug(
                    f"Setting `max_cells` to {max_cells_int} "
                    f"(n_selected - {HVG_UBIQUITOUS_SLACK}). Genes detected in at "
                    "least this many cells are excluded as ubiquitous."
                )
        elif max_cells == np.inf:
            max_cells_int = np.inf
        else:
            max_cells_int = int(max_cells)
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay)
        planned = _feature_selection_plan(
            self.zw,
            assay=assay.name,
            n_features=assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation="mark_hvgs",
            parameters={
                "min_cells": min_cells,
                "max_cells": max_cells_int,
                "top_n": top_n,
                "min_var": min_var,
                "max_var": max_var,
                "min_mean": min_mean,
                "max_mean": max_mean,
                "n_bins": n_bins,
                "lowess_frac": lowess_frac,
                "blacklist": blacklist,
                "keep_bounds": keep_bounds,
                "bin_strategy": bin_strategy,
            },
            inputs={"feature_summary": summary_ref},
            execution_options={
                "show_plot": show_plot,
                "plot_kwargs": plot_kwargs,
                "nthreads": assay.nthreads,
                "label": label,
                "invalidate_cache": invalidate_cache,
            },
            payload_names=("values", "corrected_variance"),
            invalidate_cache=invalidate_cache,
        )
        summary: dict[str, np.ndarray] | None = None
        selected_values: np.ndarray
        if not planned.reused:
            summary = feature_summary_values(
                self.zw,
                summary_ref,
                n_selected=n_selected,
            )
            values, corrected_variance = assay._select_hvgs(
                summary,
                n_selected=n_selected,
                min_cells=min_cells,
                max_cells=max_cells_int,
                top_n=top_n,
                min_var=min_var,
                max_var=max_var,
                min_mean=min_mean,
                max_mean=max_mean,
                n_bins=n_bins,
                lowess_frac=lowess_frac,
                blacklist=blacklist,
                keep_bounds=keep_bounds,
                bin_strategy=bin_strategy,
            )
            selected_values = np.asarray(values, dtype=bool)
            if not selected_values.any():
                raise ValueError(
                    "HVG selection contains no features; adjust the HVG filters"
                )
            _write_feature_selection(
                self.zw,
                planned,
                ordered_feature_ids_fingerprint=feature_ids_fingerprint,
                payload={
                    "values": selected_values,
                    "corrected_variance": corrected_variance,
                },
                payload_names=("values", "corrected_variance"),
            )
        else:
            selected_values = np.asarray(
                _feature_selection_values(self.zw, planned.ref),
                dtype=bool,
            )
            if not selected_values.any():
                raise ValueError(
                    "HVG selection contains no features; adjust the HVG filters"
                )
        if show_plot:
            if summary is None:
                summary = feature_summary_values(
                    self.zw,
                    summary_ref,
                    n_selected=n_selected,
                )
            assay._plot_hvgs(
                summary,
                selected_values,
                _feature_selection_values(
                    self.zw,
                    planned.ref,
                    "corrected_variance",
                ),
                **plot_kwargs,
            )
        publish_feature_selection_alias(
            self.zw,
            assay.name,
            label,
            planned.ref,
        )
        return planned.ref

    def _run_enrichment(
        self,
        *,
        assay: RNAassay,
        label: str,
        cell_key: str,
        overwrite: bool,
        invalidate_cache: bool,
        scorer: _EnrichmentScorer,
    ) -> EnrichmentResult:
        """Shared artifact plan, reuse, write, and selection path for enrichment."""
        cell_index = scorer.cell_index
        cell_digest = array_digest(cell_index)
        feature_digest = array_digest(scorer.feature_index)
        attrs: dict[str, Any] = {
            "algorithm_version": scorer.algorithm_version,
            "cell_digest": cell_digest,
            "cell_key": cell_key,
            "complete": False,
            "feature_digest": feature_digest,
            "layout": _ENRICHMENT_LAYOUT,
            "method": scorer.method,
            **scorer.method_payload,
        }
        required_arrays = (
            ArrayRequirement(
                "scores",
                shape=(len(cell_index), len(scorer.source_names)),
                dtype_kind="f",
            ),
            ArrayRequirement(
                "cell_index",
                shape=(len(cell_index),),
                dtype_kind="i",
            ),
            ArrayRequirement("matched_feature_index", dtype_kind="i"),
            *scorer.extra_required_arrays,
            ArrayRequirement(
                "source_names",
                shape=(len(scorer.source_names),),
            ),
            ArrayRequirement(
                "source_sizes",
                shape=(len(scorer.source_names),),
                dtype_kind="i",
            ),
        )
        planned = scorer.arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=required_arrays,
            required_attributes=(
                "algorithm_version",
                "method",
                "network_digest",
            ),
            reuse_validator=lambda _ref, group: _enrichment_artifact_matches(
                group,
                attrs=attrs,
                cell_index=cell_index,
                matched_feature_index=scorer.matched_feature_index,
                source_names=scorer.source_names,
                source_sizes=scorer.source_sizes,
                rank_feature_index=scorer.rank_feature_index,
            ),
        )
        existing_ref = _enrichment_artifact_ref(assay, label)
        if existing_ref is not None:
            existing_status = inspect_artifact(self.zw, existing_ref)
            if (
                existing_ref.kind != "enrichment_scores"
                or existing_ref.scope != "assay"
                or existing_ref.assay != assay.name
                or not existing_status.complete
            ):
                raise ValueError(f"Enrichment label {label!r} has an invalid artifact")
            if existing_status.provenance != planned.provenance and not overwrite:
                raise ValueError(
                    f"Enrichment label {label!r} already contains a different "
                    f"{existing_status.operation!r} execution; pass overwrite=True "
                    "to replace it"
                )
        if planned.reused:
            _publish_enrichment_artifact(
                assay,
                label,
                planned.ref,
                cell_key=cell_key,
            )
            return _load_enrichment_result(
                assay,
                label=label,
                sources=None,
                artifact_root=self.zw,
            )
        slot = start_artifact(self.zw, planned)
        with scorer.write_context():
            _write_enrichment_slot(
                slot,
                attrs=attrs,
                score_batches=scorer.score_batches(),
                n_cells=len(cell_index),
                source_names=scorer.source_names,
                source_sizes=scorer.source_sizes,
                cell_index=cell_index,
                matched_feature_index=scorer.matched_feature_index,
                rank_feature_index=scorer.rank_feature_index,
            )
        finish_artifact(slot, planned)
        _publish_enrichment_artifact(
            assay,
            label,
            planned.ref,
            cell_key=cell_key,
        )
        return _load_enrichment_result(
            assay,
            label=label,
            sources=None,
            artifact_root=self.zw,
        )

    def _prepare_enrichment_assay(
        self,
        *,
        display_name: str,
        from_assay: str | None,
        cell_key: str,
        features: ArtifactRef | str,
        overwrite: bool,
    ) -> tuple[RNAassay, np.ndarray, np.ndarray, ArtifactRef]:
        if self.zarr_mode != "r+":
            raise ValueError(
                f"{display_name} requires a DataStore opened with zarr_mode='r+'"
            )
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(f"{display_name} can only be run on an RNAassay")
        feature_selection = self.resolve_features(assay.name, features)
        feature_values = _feature_selection_values(self.zw, feature_selection)
        feature_index = np.flatnonzero(feature_values).astype(np.int64, copy=False)
        cell_index = np.asarray(assay.cells.active_index(cell_key), dtype=np.int64)
        if len(cell_index) == 0:
            raise ValueError(f"Cell key {cell_key!r} selects no active cells")
        if len(feature_index) == 0:
            raise ValueError("Feature selection contains no active features")
        return assay, cell_index, feature_index, feature_selection

    def run_waggr(
        self,
        net: pd.DataFrame,
        label: str,
        *,
        from_assay: str | None = None,
        cell_key: str = "I",
        features: ArtifactRef | str,
        mode: Literal["wmean", "wsum"] = "wmean",
        tmin: int = 5,
        log_transform: bool = False,
        overwrite: bool = False,
        invalidate_cache: bool = False,
    ) -> EnrichmentResult:
        """Score weighted gene sets from streamed normalized RNA counts.

        Targets are matched to active feature names without case sensitivity. Sources
        with fewer than ``tmin`` matched non-zero edges are removed. Results are
        written to the assay's enrichment group and returned lazily.

        Args:
            net: Network with ``source`` and ``target`` columns. An optional
                ``weight`` column supplies signed numeric edge weights. Missing
                weights default to one.
            label: Name used to persist and retrieve the result.
            from_assay: RNA assay to score. The default assay is used when omitted.
            cell_key: Cell metadata key that selects score rows.
            features: Feature-selection artifact or exact published label.
            mode: ``"wmean"`` divides each weighted sum by the sum of absolute
                source weights. ``"wsum"`` returns the weighted sum.
            tmin: Minimum number of matched targets required per source.
            log_transform: Apply ``log1p`` after library-size normalization.
            overwrite: Replace a complete result with different execution metadata.
                The previous result remains active until the replacement is complete.

        Returns:
            A persisted result with a lazy cells-by-sources score matrix.

        Note:
            Cache identity covers selections, method parameters, normalization, and
            the prepared network. It assumes the stored count matrix is immutable.
        """
        from ...features.enrichment.net import prepare_network
        from ...features.enrichment.waggr import (
            WAGGR_ALGORITHM_VERSION,
            build_waggr_model,
            score_waggr_block,
        )

        _validate_enrichment_label(label)
        if mode not in {"wmean", "wsum"}:
            raise ValueError("mode must be 'wmean' or 'wsum'")
        if not isinstance(log_transform, bool):
            raise TypeError("log_transform must be a boolean")
        assay, cell_index, feature_index, feature_selection = (
            self._prepare_enrichment_assay(
                display_name="WAGGR",
                from_assay=from_assay,
                cell_key=cell_key,
                features=features,
                overwrite=overwrite,
            )
        )
        feature_names = np.asarray(assay.feats.fetch_all("names"))[feature_index]
        network = prepare_network(
            net,
            active_feature_names=feature_names,
            active_feature_index=feature_index,
            tmin=tmin,
            weighted=True,
        )
        if not lib_size_feature_stream_eligible(assay):
            raise ValueError(
                "WAGGR requires the default norm_lib_size RNA normalization"
            )
        if assay.sf is None:
            raise ValueError("WAGGR requires a finite positive size factor")
        try:
            size_factor = float(assay.sf)
        except (TypeError, ValueError) as exc:
            raise ValueError("WAGGR requires a finite positive size factor") from exc
        if not np.isfinite(size_factor) or size_factor <= 0:
            raise ValueError("WAGGR requires a finite positive size factor")

        arguments = WaggrArguments(
            cell_selection=self._ensure_cell_selection(cell_key),
            feature_selection=feature_selection,
            network_digest=network.network_digest,
            algorithm_version=WAGGR_ALGORITHM_VERSION,
            mode=mode,
            tmin=tmin,
            log_transform=log_transform,
            normalization_method=callable_identity(assay.normMethod),
            size_factor=size_factor,
            from_assay=assay.name,
            cell_key=cell_key,
            label=label,
            overwrite=overwrite,
            invalidate_cache=invalidate_cache,
        )

        def score_batches() -> Iterator[np.ndarray]:
            cell_scalars = np.asarray(
                assay.cells.fetch_all(f"{assay.name}_nCounts")[cell_index],
                dtype=np.float64,
            )
            if not np.isfinite(cell_scalars).all() or np.any(cell_scalars < 0):
                raise ValueError("WAGGR cell normalization scalars must be finite")
            cell_scalars[cell_scalars == 0] = 1.0
            model = build_waggr_model(network)
            raw = assay.rawData[:, network.matched_feature_index][cell_index, :]
            offset = 0
            for raw_block in raw.stream_blocks(
                nthreads=self.nthreads,
                msg="Scoring WAGGR",
                prefetch=1,
            ):
                block = np.asarray(raw_block, dtype=np.float64)
                end = offset + block.shape[0]
                if end > len(cell_scalars):
                    raise ValueError(
                        "WAGGR raw blocks exceed the active cell selection"
                    )
                values = size_factor * block / cell_scalars[offset:end].reshape(-1, 1)
                if log_transform:
                    values = np.log1p(values)
                yield score_waggr_block(values, model, mode=mode)
                offset = end
            if offset != len(cell_scalars):
                raise ValueError(
                    f"WAGGR streamed {offset} cells, expected {len(cell_scalars)}"
                )

        return self._run_enrichment(
            assay=assay,
            label=label,
            cell_key=cell_key,
            overwrite=overwrite,
            invalidate_cache=invalidate_cache,
            scorer=_EnrichmentScorer(
                method="waggr",
                algorithm_version=WAGGR_ALGORITHM_VERSION,
                method_payload={
                    "log_transform": log_transform,
                    "network_digest": network.network_digest,
                    "normalization": "norm_lib_size",
                    "size_factor": size_factor,
                    "tmin": tmin,
                    "waggr_mode": mode,
                },
                arguments=arguments,
                cell_index=cell_index,
                feature_index=feature_index,
                matched_feature_index=network.matched_feature_index,
                source_names=network.source_names,
                source_sizes=network.source_sizes,
                rank_feature_index=None,
                extra_required_arrays=(),
                score_batches=score_batches,
            ),
        )

    def run_aucell(
        self,
        net: pd.DataFrame,
        label: str,
        *,
        from_assay: str | None = None,
        cell_key: str = "I",
        features: ArtifactRef | str,
        tmin: int = 5,
        n_up: int | None = None,
        tie_seed: int = 0,
        overwrite: bool = False,
        invalidate_cache: bool = False,
    ) -> EnrichmentResult:
        """Score gene sets by recovery among each cell's top-ranked RNA features.

        AUCell ranks every feature selected by ``features`` from raw counts. Network
        weights are ignored. Targets are matched without case sensitivity, then
        sources with fewer than ``tmin`` matched targets are removed.

        Args:
            net: Network with ``source`` and ``target`` columns.
            label: Name used to persist and retrieve the result.
            from_assay: RNA assay to score. The default assay is used when omitted.
            cell_key: Cell metadata key that selects score rows.
            features: Feature-selection artifact or exact published label.
            tmin: Minimum number of matched targets required per source.
            n_up: Number of top-ranked features used for recovery. When omitted,
                five percent of the ranking universe is used, clipped to its valid
                range.
            tie_seed: Seed for the global feature permutation used to resolve ties.
            overwrite: Replace a complete result with different execution metadata.
                The previous result remains active until the replacement is complete.

        Returns:
            A persisted result with lazy scores in the interval from zero to one.

        Note:
            Cache identity covers selections, method parameters, and the prepared
            network. It assumes the stored count matrix is immutable.
        """
        from ...features.enrichment.aucell import (
            AUCELL_ALGORITHM_VERSION,
            build_gene_set_index,
            make_rank_permutation,
            resolve_n_up,
            score_aucell_block,
        )
        from ...features.enrichment.net import prepare_network

        _validate_enrichment_label(label)
        assay, cell_index, feature_index, feature_selection = (
            self._prepare_enrichment_assay(
                display_name="AUCell",
                from_assay=from_assay,
                cell_key=cell_key,
                features=features,
                overwrite=overwrite,
            )
        )
        resolved_n_up = resolve_n_up(len(feature_index), n_up)
        feature_names = np.asarray(assay.feats.fetch_all("names"))[feature_index]
        network = prepare_network(
            net,
            active_feature_names=feature_names,
            active_feature_index=feature_index,
            tmin=tmin,
            weighted=False,
        )
        permutation = make_rank_permutation(len(feature_index), tie_seed)
        rank_feature_index = feature_index[permutation]
        sets = build_gene_set_index(network, rank_feature_index)

        arguments = AucellArguments(
            cell_selection=self._ensure_cell_selection(cell_key),
            feature_selection=feature_selection,
            network_digest=network.network_digest,
            algorithm_version=AUCELL_ALGORITHM_VERSION,
            tmin=tmin,
            n_up=resolved_n_up,
            tie_seed=tie_seed,
            from_assay=assay.name,
            cell_key=cell_key,
            label=label,
            overwrite=overwrite,
            invalidate_cache=invalidate_cache,
        )

        def score_batches() -> Iterator[np.ndarray]:
            raw = assay.rawData[:, feature_index][cell_index, :]
            offset = 0
            for raw_block in raw.stream_blocks(
                nthreads=self.nthreads,
                msg="Scoring AUCell",
                prefetch=1,
            ):
                scores = score_aucell_block(
                    np.asarray(raw_block),
                    permutation,
                    sets,
                    n_up=resolved_n_up,
                )
                offset += scores.shape[0]
                if offset > len(cell_index):
                    raise ValueError(
                        "AUCell raw blocks exceed the active cell selection"
                    )
                yield scores
            if offset != len(cell_index):
                raise ValueError(
                    f"AUCell streamed {offset} cells, expected {len(cell_index)}"
                )

        @contextmanager
        def aucell_write_context() -> Iterator[None]:
            import numba

            previous_threads = numba.get_num_threads()
            numba.set_num_threads(
                min(max(1, int(self.nthreads)), numba.config.NUMBA_NUM_THREADS)
            )
            try:
                yield
            finally:
                numba.set_num_threads(previous_threads)

        return self._run_enrichment(
            assay=assay,
            label=label,
            cell_key=cell_key,
            overwrite=overwrite,
            invalidate_cache=invalidate_cache,
            scorer=_EnrichmentScorer(
                method="aucell",
                algorithm_version=AUCELL_ALGORITHM_VERSION,
                method_payload={
                    "n_up": resolved_n_up,
                    "network_digest": network.network_digest,
                    "tie_seed": tie_seed,
                    "tmin": tmin,
                },
                arguments=arguments,
                cell_index=cell_index,
                feature_index=feature_index,
                matched_feature_index=network.matched_feature_index,
                source_names=network.source_names,
                source_sizes=network.source_sizes,
                rank_feature_index=rank_feature_index,
                extra_required_arrays=(
                    ArrayRequirement(
                        "rank_feature_index",
                        shape=(len(rank_feature_index),),
                        dtype_kind="i",
                    ),
                ),
                score_batches=score_batches,
                write_context=aucell_write_context,
            ),
        )

    def get_enrichment(
        self,
        label: str,
        *,
        from_assay: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> EnrichmentResult:
        """Load a persisted enrichment result without materializing its scores.

        Args:
            label: Label passed to ``run_waggr`` or ``run_aucell``.
            from_assay: RNA assay that owns the result. The default assay is used
                when omitted.
            sources: Optional source names to select and order.

        Returns:
            The stored metadata and a lazy cells-by-sources score matrix.
        """
        _validate_enrichment_label(label)
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("Enrichment results are only available for an RNAassay")
        return _load_enrichment_result(
            assay,
            label=label,
            sources=sources,
            artifact_root=self.zw,
        )

    def run_marker_search(
        self,
        from_assay: str | None = None,
        group_key: str | None = None,
        cell_key: str | None = None,
        *,
        features: ArtifactRef | str,
        nthreads: int | None = None,
        skip_save: bool = False,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> dict[str, Any] | ArtifactRef:
        """Identifies group specific features for a given assay.

        Please check out the ``find_markers_by_rank`` function for further details of how marker features for groups
        are identified. The results are saved into the Zarr hierarchy under `markers` group.

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            group_key: Required parameter. This has to be a column name from cell metadata table. This column dictates
                       how the cells will be grouped. Usually this would be a column denoting cell clusters.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table. (Default value: 'I')
            features: Feature-selection artifact or exact published label.
            nthreads: Threads for marker search.
            skip_save: If True, return results without writing to Zarr.
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            Marker dict when ``skip_save`` is true. Saved searches return the
            immutable marker-table artifact reference, which can be passed to
            :meth:`get_markers` to select that exact feature-specific result.
        """
        from ...features.markers import find_markers_by_rank

        reject_unknown_normalization_params(
            norm_params,
            caller="run_marker_search",
        )
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `group_key`. This should be the name of a column from "
                "cell metadata object that has information on how cells should be grouped."
            )
        assay = self._get_assay(from_assay)
        from_assay = assay.name
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        cell_index = np.asarray(assay.cells.active_index(cell_key), dtype=np.int64)
        feature_selection = self.resolve_features(from_assay, features)
        feature_values = np.asarray(
            _feature_selection_values(self.zw, feature_selection),
            dtype=bool,
        )
        feature_index = np.flatnonzero(feature_values).astype(np.int64, copy=False)
        if len(cell_index) == 0:
            raise ValueError(f"Cell key {cell_key!r} selects no active cells")
        if len(feature_index) == 0:
            raise ValueError("Feature selection contains no active features")
        if nthreads is None:
            nthreads = self.nthreads
        resolved_norm_params = {
            **norm_params,
            "log_transform": norm_params.get("log_transform", False),
            "renormalize_subset": norm_params.get(
                "renormalize_subset",
                False,
            ),
        }

        logger.debug(
            f"Running marker search for {from_assay}/{cell_key}/{group_key} "
            f"(feature_selection={feature_selection.artifact_id[:12]}, "
            f"nthreads={nthreads})"
        )
        planned = None
        group_cell_counts: dict[Any, tuple[int, int]] = {}
        if not skip_save:
            assay_grp = as_zarr_group(self.zw[assay.name], name=assay.name)
            if "markers" not in assay_grp:
                assay_grp.create_group("markers")
            markers_grp = as_zarr_group(assay_grp["markers"], name="markers")
            try:
                _marker_artifact_index(markers_grp)
            except ValueError as exc:
                raise RuntimeError("Marker artifact index is invalid") from exc
            cell_selection = self._ensure_cell_selection(cell_key)
            cluster_input = self._resolve_cell_data_provenance_input(
                group_key,
                cell_key=cell_key,
            )
            group_labels = np.asarray(assay.cells.fetch_all(group_key))[cell_index]
            group_sizes = pd.Series(group_labels).value_counts()
            n_selected = int(len(group_labels))
            group_cell_counts = {
                group_id: (
                    int(group_size),
                    int(n_selected - group_size),
                )
                for group_id, group_size in group_sizes.items()
            }
            expected_group_cell_counts: dict[str, tuple[int, int]] = {}
            for group_id, counts in group_cell_counts.items():
                group_name = str(group_id)
                if group_name in expected_group_cell_counts:
                    raise ValueError(
                        "Marker group labels must remain unique after string conversion"
                    )
                expected_group_cell_counts[group_name] = counts
            feature_names_for_validation = np.asarray(assay.feats.fetch_all("names"))
            expected_feature_index = np.flatnonzero(feature_values)

            def marker_reuse_is_valid(
                _ref: ArtifactRef,
                candidate: zarr.Group,
            ) -> bool:
                try:
                    stored_feature_index = np.asarray(
                        as_zarr_array(
                            candidate["feature_index"],
                            name="feature_index",
                        )[:]
                    )
                    if stored_feature_index.dtype.kind not in {
                        "i",
                        "u",
                    } or not np.array_equal(
                        stored_feature_index.astype(np.int64, copy=False),
                        expected_feature_index,
                    ):
                        return False
                    _validate_marker_slot(
                        candidate,
                        feature_names_for_validation,
                        expected_group_cell_counts=expected_group_cell_counts,
                    )
                except (IndexError, KeyError, TypeError, ValueError):
                    return False
                return True

            arguments = MarkerTableArguments(
                cell_selection=cell_selection,
                feature_selection=feature_selection,
                clusters=cluster_input,
                normalization=resolved_norm_params,
                normalization_method=callable_identity(assay.normMethod),
                size_factor=getattr(assay, "sf", None),
                method=MARKER_METHOD,
                alternative=MARKER_ALTERNATIVE,
                tie_correction=MARKER_TIE_CORRECTION,
                continuity_correction=MARKER_CONTINUITY_CORRECTION,
                adjustment_method=MARKER_ADJUSTMENT_METHOD,
                adjustment_scope=MARKER_ADJUSTMENT_SCOPE,
                group_key=group_key,
                cell_key=cell_key,
                nthreads=nthreads,
                invalidate_cache=invalidate_cache,
            )
            planned = arguments.plan(
                self.zw,
                scope="assay",
                assay=from_assay,
                invalidate_cache=invalidate_cache,
                required_arrays=(
                    ArrayRequirement(
                        "feature_index",
                        shape=(int(feature_values.sum()),),
                        dtype_kind="i",
                    ),
                ),
                required_attributes=(
                    AttributeRequirement(
                        "stat_columns",
                        expected_types=(list, tuple),
                    ),
                ),
                reuse_validator=marker_reuse_is_valid,
            )

            def select_marker_artifact(ref: ArtifactRef) -> None:
                try:
                    artifacts = _marker_artifact_index(markers_grp)
                except ValueError as exc:
                    raise RuntimeError("Marker artifact index is invalid") from exc
                assay_slots = dict(artifacts.get(cell_key, {}))
                group_slots = dict(assay_slots.get(group_key, {}))
                group_slots[feature_selection.artifact_id] = ref.to_dict()
                assay_slots[group_key] = group_slots
                artifacts[cell_key] = assay_slots
                markers_grp.attrs["artifacts"] = artifacts

            if planned.reused:
                select_marker_artifact(planned.ref)
                return planned.ref

        markers = find_markers_by_rank(
            assay=assay,
            groups=np.asarray(assay.cells.fetch_all(group_key))[cell_index],
            cell_idx=cell_index,
            feat_idx=feature_index,
            nthreads=nthreads,
            **resolved_norm_params,
        )

        if skip_save:
            return markers

        from ...storage.stores import is_remote_datastore

        remote = is_remote_datastore(self.zarr_loc, self.z)
        t_save = time.perf_counter()
        assert planned is not None
        remote_slot = start_artifact(self.zw, planned)
        workers = max(1, int(nthreads or self.nthreads))
        self._write_marker_slot(
            remote_slot,
            markers,
            workers=workers if remote else 1,
            group_cell_counts=group_cell_counts,
        )
        finish_artifact(remote_slot, planned)
        select_marker_artifact(planned.ref)
        logger.info(f"Stored marker results for {len(markers)} clusters")
        logger.debug(
            f"Saved marker results to {artifact_path(planned.ref)} "
            f"in {time.perf_counter() - t_save:.1f}s"
        )
        return planned.ref

    @staticmethod
    def _write_marker_slot(
        group: zarr.Group,
        markers: dict[Any, pd.DataFrame],
        *,
        workers: int = 1,
        group_cell_counts: dict[Any, tuple[int, int]],
    ) -> None:
        from ...storage.arrays import create_metadata_column

        populated_groups = {
            cluster_id for cluster_id, values in markers.items() if len(values)
        }
        missing_counts = populated_groups.difference(group_cell_counts)
        if missing_counts:
            raise ValueError(
                "Marker writes require target and reference counts "
                "for every populated group"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 2
            for cluster_id in populated_groups
            for count in group_cell_counts[cluster_id]
        ):
            raise ValueError("Marker target and reference counts must be integers >= 2")
        feature_index = _shared_marker_feature_index(markers)
        stats_by_group = {
            cluster_id: _marker_stats_matrix(values, feature_index)
            for cluster_id, values in markers.items()
            if len(values)
        }
        group.attrs["stat_columns"] = list(_MARKER_STAT_COLUMNS)
        group.attrs["method"] = MARKER_METHOD
        group.attrs["alternative"] = MARKER_ALTERNATIVE
        group.attrs["tie_correction"] = MARKER_TIE_CORRECTION
        group.attrs["continuity_correction"] = MARKER_CONTINUITY_CORRECTION
        group.attrs["adjustment_method"] = MARKER_ADJUSTMENT_METHOD
        group.attrs["adjustment_scope"] = MARKER_ADJUSTMENT_SCOPE
        create_metadata_column(
            group,
            "feature_index",
            data=feature_index,
            dtype=np.int32,
            overwrite=True,
        )

        def write_cluster(item: tuple[Any, pd.DataFrame]) -> None:
            cluster_id, vals = item
            if len(vals) == 0:
                return
            cluster_group = group.create_group(str(cluster_id))
            _write_compact_marker_stats(
                cluster_group,
                stats_by_group[cluster_id],
            )
            n_group, n_reference = group_cell_counts[cluster_id]
            cluster_group.attrs["n_group"] = n_group
            cluster_group.attrs["n_reference"] = n_reference

        items = list(markers.items())
        if workers <= 1:
            for item in items:
                write_cluster(item)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(write_cluster, items))

    def _resolve_marker_group(
        self,
        from_assay: str | None,
        cell_key: str,
        group_key: str,
        marker: ArtifactRef | None = None,
    ) -> zarr.Group:
        assay = self._get_assay(from_assay)
        indexed_feature_id: str | None = None
        if marker is None:
            markers_group = as_zarr_group(
                assay.z["markers"],
                name="markers",
            )
            artifacts = _marker_artifact_index(markers_group)
            indexed_feature_id, ref = _select_indexed_marker_artifact(
                artifacts,
                cell_key,
                group_key,
            )
        else:
            ref = marker
        if (
            ref.kind != "marker_table"
            or ref.scope != "assay"
            or ref.assay != assay.name
        ):
            raise ValueError("Marker artifact ref is invalid")
        status = self.inspect_artifact(ref)
        if not status.exists:
            raise ValueError("Marker artifact does not exist")
        if not status.complete:
            raise ValueError("Marker artifact is incomplete")
        execution = status.execution_options or {}
        if (
            execution.get("cell_key") != cell_key
            or execution.get("group_key") != group_key
        ):
            raise ValueError("Marker artifact route does not match the request")
        inputs = status.inputs or {}
        stored_selection = inputs.get("cell_selection")
        if not isinstance(stored_selection, dict):
            raise ValueError("Marker artifact cell selection is stale")
        from ...graph.state import validate_cell_selection_artifact

        try:
            validate_cell_selection_artifact(
                self.zw,
                ArtifactRef.from_dict(stored_selection),
                cell_key,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Marker artifact cell selection is stale") from exc
        stored_features = inputs.get("feature_selection")
        if not isinstance(stored_features, dict):
            raise ValueError("Marker artifact feature selection is missing")
        try:
            stored_feature_ref = ArtifactRef.from_dict(stored_features)
            self.resolve_features(
                assay.name,
                stored_feature_ref,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Marker artifact feature selection is invalid") from exc
        if (
            indexed_feature_id is not None
            and stored_feature_ref.artifact_id != indexed_feature_id
        ):
            raise ValueError("Marker artifact index has the wrong feature selection")
        current_clusters = self._resolve_cell_data_provenance_input(
            group_key,
            cell_key=cell_key,
        )
        stored_clusters = inputs.get("clusters")
        if (
            stored_clusters != current_clusters
            and stored_clusters != current_clusters["artifact"]
        ):
            raise ValueError("Marker artifact cluster labels are stale")
        return as_zarr_group(
            self.zw[artifact_path(ref)],
            name=artifact_path(ref),
        )

    def get_markers(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        group_id: str | int | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
        *,
        marker: ArtifactRef | None = None,
    ) -> pd.DataFrame:
        """Return marker features from `run_marker_search`.

        When ``group_id`` is ``None`` (default), markers for every group under
        ``group_key`` are returned in one long table with a ``group_id`` column.
        Pass a specific ``group_id`` to return markers for that group only.
        For a wide export of marker names only, use ``export_markers_to_csv``.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            group_id: One value from the ``group_key`` column, or ``None`` for all groups.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.
            marker: Exact marker-table artifact returned by
                    ``run_marker_search``. When supplied, omitted assay, cell,
                    and group routing is recovered from that artifact.

        Returns:
            Pandas dataframe with marker statistics. All-group results include a ``group_id`` column.
        """

        if marker is not None:
            if not isinstance(marker, ArtifactRef):
                raise TypeError("marker must be an ArtifactRef")
            if (
                marker.kind != "marker_table"
                or marker.scope != "assay"
                or marker.assay is None
            ):
                raise ValueError("marker must identify an assay marker_table artifact")
            if from_assay is not None and from_assay != marker.assay:
                raise ValueError("marker belongs to a different assay")
            status = self.inspect_artifact(marker)
            if not status.exists:
                raise ValueError("Marker artifact does not exist")
            if not status.complete:
                raise ValueError("Marker artifact is incomplete")
            execution = status.execution_options or {}
            stored_cell_key = execution.get("cell_key")
            stored_group_key = execution.get("group_key")
            if not isinstance(stored_cell_key, str) or not isinstance(
                stored_group_key,
                str,
            ):
                raise ValueError("Marker artifact has no valid execution route")
            if cell_key is not None and cell_key != stored_cell_key:
                raise ValueError("marker uses a different cell_key")
            if group_key is not None and group_key != stored_group_key:
                raise ValueError("marker uses a different group_key")
            from_assay = marker.assay
            cell_key = stored_cell_key
            group_key = stored_group_key
        assay = self._get_assay(from_assay)
        from_assay = assay.name
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for group_key. "
                "This should be same as used for `run_marker_search`"
            )
        try:
            g = self._resolve_marker_group(
                from_assay,
                cell_key,
                group_key,
                marker,
            )
        except KeyError:
            raise KeyError(
                "ERROR: Couldn't find the location of markers. Please make sure that you have already called "
                "`run_marker_search` method with same value of `cell_key` and `group_key`"
            )
        out_cols = list(_MARKER_OUT_COLUMNS)
        gids = sorted(set(assay.cells.fetch(group_key, key=cell_key)))
        if group_id is not None:
            gids = [group_id]

        feature_names = assay.feats.fetch_all("names")
        feature_ids = assay.feats.fetch_all("ids")
        dfs = []
        for gid in gids:
            group_name = str(gid)
            if group_name in g:
                marker_grp = as_zarr_group(g[group_name], name=group_name)
                df = _load_marker_cluster_frame(
                    g,
                    marker_grp,
                    feature_names,
                    group_id=gid,
                    feature_ids=feature_ids,
                )
            else:
                logger.debug(f"No markers found for {gid} returning empty dataframe")
                empty_cols = [
                    "group_id",
                    "feature_name",
                    "feature_index",
                    *out_cols[1:],
                ]
                df = pd.DataFrame(
                    {name: pd.Series(dtype=object) for name in empty_cols}
                )
            dfs.append(df)
        dfs = pd.concat(dfs, ignore_index=True)
        keep = np.ones(len(dfs), dtype=bool)
        if "score" in dfs and dfs["score"].notna().any():
            keep &= dfs["score"].fillna(-np.inf).to_numpy() >= min_score
        if "frac_exp" in dfs and dfs["frac_exp"].notna().any():
            keep &= dfs["frac_exp"].fillna(-np.inf).to_numpy() >= min_frac_exp
        return dfs.loc[keep].reset_index(drop=True)

    def export_markers_to_csv(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        csv_filename: str | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
        *,
        marker: ArtifactRef | None = None,
    ) -> None:
        """Export markers of each cluster/group to a CSV file where each column
        contains the marker names sorted by score (descending order, highest
        first). This function does not export the scores of markers as they can
        be obtained using `get_markers` function.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            csv_filename: Required parameter. Name, with path, of CSV file where the marker table is to be saved.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.
            marker: Exact marker-table artifact returned by
                    ``run_marker_search``. Required to disambiguate when the
                    same grouping has marker tables for multiple feature sets.

        Returns:
        """
        # Not testing the values of from_assay and cell_key because they will be tested in `get_markers`
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for group_key. "
                "This should be same as used for `run_marker_search`"
            )
        if csv_filename is None:
            raise ValueError(
                "ERROR: Please provide a value for parameter `csv_filename`"
            )
        assay = self._get_assay(from_assay)
        from_assay = assay.name
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        clusters = self.cells.fetch(group_key, key=cell_key)
        markers_table = {}
        for group_id in sorted(set(clusters)):
            m = self.get_markers(
                from_assay=from_assay,
                cell_key=cell_key,
                group_key=group_key,
                group_id=group_id,
                min_score=min_score,
                min_frac_exp=min_frac_exp,
                marker=marker,
            )
            if len(m) > 0:
                markers_table[group_id] = m["feature_name"].reset_index(drop=True)
            else:
                markers_table[group_id] = pd.Series([])
        pd.DataFrame(markers_table).fillna("").to_csv(csv_filename, index=False)
        return None

    def add_grouped_assay(
        self,
        from_assay: str | None = None,
        group_key: str | None = None,
        assay_label: str | None = None,
        exclude_values: list | None = None,
    ) -> None:
        """Add a new assay to the DataStore by grouping together multiple
        features and taking their means. This method requires that the features
        are already assigned a group/cluster identity. The new assay will have
        all cells and one feature per group identity not present in
        ``exclude_values``.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            group_key: This is mandatory parameter. Name of the column in feature metadata table to be used for
                       grouping features.
            assay_label: This is mandatory parameter. A name for the new assay.
            exclude_values: These groups/clusters will be ignored and not added to new assay. By default, it is set to
                            [-1], this means that all the features that have the group identity of -1 are not used.

        Returns: None
        """

        from ...storage.sharding import write_dense_in_shard_rows

        from ...storage.schema import create_zarr_count_assay

        if assay_label is None:
            raise ValueError(
                "ERROR: Please provide a value for `assay_label`. "
                "It will be used to create a new assay"
            )
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `group_key`. "
                "This should be name of the column in the feature attribute table that contains the group/cluster "
                "identity of each feature."
            )

        assay = self._get_assay(from_assay)
        groups = assay.feats.fetch_all(group_key)
        if exclude_values is None:
            exclude_values = [-1]
        group_set = sorted(set(groups).difference(exclude_values))

        module_ids = [f"group_{x}" for x in group_set]
        g = create_zarr_count_assay(
            z=self.z,
            assay_name=assay_label,
            workspace=self.workspace,
            n_cells=assay.cells.N,
            feat_ids=module_ids,
            feat_names=module_ids,
            dtype="float",
            profile=self.storageProfile,
        )

        cell_idx = np.array(list(range(assay.cells.N)))
        n_groups = len(group_set)
        matrix = np.zeros((assay.cells.N, n_groups), dtype=np.float64)
        for n, i in iter_progress(
            enumerate(group_set), desc="Computing grouped means", total=len(group_set)
        ):
            feat_idx = np.where(groups == i)[0]
            matrix[:, n] = (
                assay.normed(cell_idx=cell_idx, feat_idx=feat_idx)
                .mean(axis=1)
                .compute()
            )
        write_dense_in_shard_rows(
            g,
            lambda start, end: matrix[start:end, :],
            msg="Writing grouped assay",
            resources=self.resources,
            io=self.storageIo,
        )

        self._load_assays(custom_assay_types={assay_label: "Assay"})
        self._ini_cell_props(min_features=0, mito_pattern="", ribo_pattern="")
        grouped_assay = self._get_assay(assay_label)
        grouped_assay.attrs["grouped_from_assay"] = assay.name
        grouped_assay.attrs["grouped_group_key"] = group_key
        group_column = as_zarr_array(
            as_zarr_group(assay.z["featureData"], name="featureData")[group_key],
            name=group_key,
        )
        raw_source = group_column.attrs.get("source_artifact")
        if isinstance(raw_source, dict):
            try:
                source_ref = ArtifactRef.from_dict(raw_source)
                source_status = inspect_artifact(self.zw, source_ref)
            except (KeyError, TypeError, ValueError):
                source_ref = None
            if source_ref is not None and source_status.complete:
                grouped_assay.attrs["grouped_group_artifact"] = source_ref.to_dict()
                if "grouped_group_digest" in grouped_assay.attrs:
                    del grouped_assay.attrs["grouped_group_digest"]
            else:
                grouped_assay.attrs["grouped_group_digest"] = _group_assignment_digest(
                    groups
                )
        else:
            grouped_assay.attrs["grouped_group_digest"] = _group_assignment_digest(
                groups
            )

    def add_melded_assay(
        self,
        from_assay: str | None = None,
        external_bed_fn: str | None = None,
        assay_label: str | None = None,
        peaks_col: str = "ids",
        scalar_coeff: float = 1e5,
        renormalization: bool = True,
        assay_type: str = "Assay",
        cell_key: str = "I",
    ) -> None:
        """This method performs "assay melding" and can be only be used for
        assay's wherein features have genomic coordinates. In the process of
        melding the input genomic coordinates from `external_bed_fn` are
        intersected with the assay's features. Based on this intersection a
        mapping is created wherein each coordinate interval maps to one or more
        feature coordinates from the assay.

        This method has been designed for snATAC-Seq data and can be used to quantify accessibility of specific
        genomic loci such as gene bodies, promoters, enhancers, motifs, etc.
        Features from the BED file are retained even when they do not overlap any peak; those zero-count features
        are marked invalid during assay initialization.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            external_bed_fn: This is mandatory parameter. This file should be a BED format file with at least five
                             columns containing: chromosome, start position, end position, feature id and feature name.
                             Coordinates should be in half open format. That means that actual end position is -1
            assay_label: This is mandatory parameter. A name for the new assay.
            peaks_col: The column in feature metadata table that contains the genomic coordinate information of each
                       feature. The genomic coordinates are represented as strings in this format: chr:start-end
                       (Default value: 'ids')
            scalar_coeff: An arbitrary scalar multiplier. Only used when renormalization is True (Default value: 1e5)
            renormalization: Whether to rescale the sum of feature values for each cell to `scalar_coeff`
                         (Default value: True)
            assay_type: The new assay (melded assay) is saved as this type. This can be any type of Assay class from
                        `assay` module. Please provide string representation of class. By default, the assay is assigned
                        a generic class and has a dummy normalization function (Default value: 'Assay')
            cell_key: Cells used to learn peak document frequency. Every cell is
                      still scored so the new assay remains row-aligned.

        Returns:
            None
        """

        from ...features.genomic.melding import coordinate_melding

        if assay_label is None:
            raise ValueError(
                "ERROR: Please provide a value for `assay_label`. "
                "It will be used to create a new assay"
            )
        if external_bed_fn is None:
            raise ValueError(
                "ERROR: Please provide a value for `feature_bed_fn`. "
                "This should be a BED format file with atleast 5 columns."
            )

        assay = self._get_assay(from_assay)
        idf_cell_idx = assay.cells.active_index(cell_key)
        if len(idf_cell_idx) == 0:
            raise ValueError("Gene-score IDF requires at least one selected cell")
        feature_bed = pd.read_csv(external_bed_fn, header=None, sep="\t").sort_values(
            by=[0, 1]  # type: ignore
        )

        peaks_coords = assay.feats.fetch_all(peaks_col)
        coords_ser = pd.Series(peaks_coords, dtype="object")
        string_mask = coords_ser.map(lambda x: isinstance(x, str))
        colon_counts = coords_ser.str.count(":")
        hyphen_counts = coords_ser.str.split(":").str[-1].str.count("-")
        invalid_mask = (
            ~string_mask
            | colon_counts.ne(1).fillna(True)
            | hyphen_counts.ne(1).fillna(True)
        )
        invalid_coords = invalid_mask.to_numpy(dtype=bool)
        if invalid_coords.any():
            n = int(np.flatnonzero(invalid_coords)[0])
            raise ValueError(
                f"ERROR: Coordinate format check failed for element: {peaks_coords[n]} (position {n}). "
                f"The format should be chr:start-end. Please note the colon and hyphen position"
            )

        coordinate_melding(
            assay,
            workspace=self.workspace,
            feature_bed=feature_bed,
            new_assay_name=assay_label,
            peaks_col=peaks_col,
            scalar_coeff=scalar_coeff,
            renormalization=renormalization,
            peaks_coords=peaks_coords,
            idf_cell_idx=idf_cell_idx,
        )

        from ...storage.stores import zarr_group_root
        from ...writers.counts_t import finalize_writer_counts_t

        finalize_writer_counts_t(
            zarr_group_root(self.z, mode="r+"),
            assay_label,
            self.workspace,
            assay_type=assay_type,
            resources=self.resources,
        )

        self._load_assays(custom_assay_types={assay_label: assay_type})
        self._ini_cell_props(min_features=0, mito_pattern=None, ribo_pattern=None)

    def make_bulk(
        self,
        from_assay: str | None = None,
        cell_key: str = "I",
        group_key: str | None = None,
        secondary_group_key: str | None = None,
        aggr_type: Literal["mean", "sum"] = "mean",
        return_fraction: bool = False,
        feature_label: Literal["index", "id", "name"] = "index",
        remove_empty_features: bool = True,
        pseudo_reps: int = 1,
        null_vals: list[Any] | None = None,
        secondary_null_vals: list[Any] | None = None,
        random_seed: int = 4466,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Merge data from cells to create a bulk profile.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Name of the column in cell metadata table to be used for selecting cells.
            group_key: Required cell metadata column used to group cells.
                Passing ``None`` raises ``ValueError``.
            secondary_group_key: Name of the column in cell metadata table to be used for sub-grouping cells.
            aggr_type: Type of aggregation to be used. Can be either 'mean' or 'sum'. (Default value: 'mean')
            return_fraction: Return the fraction of cells expressing a gene in each group. (Default value: False)
            feature_label: The column in feature metadata table to use as row labels. (Default value: 'index')
            pseudo_reps: Within each group, randomly split cells into this many
                pseudo-replicates. Values greater than 1 produce descriptive
                resamples of the same cells, not independent biological
                replicates. (Default value: 1)
            remove_empty_features: Remove features that are not expressed in any cell. (Default value: True)
            null_vals: Values to be considered as missing values in the `group_key` column. These values will be skipped.
            secondary_null_vals: Values to be considered as missing values in the `secondary_group_key` column.
                                 These values will be skipped.
            random_seed: Seed used when assigning cells to pseudo-replicates.

        Returns:
            A pandas dataframe containing the bulk profile. If `return_fraction` is True, then a tuple of two dataframes
            is returned. The second dataframe contains the fraction of cells expressing each feature in each group.
        """

        def make_reps(v: NDArray[Any], n_reps: int, seed: int) -> list[NDArray[Any]]:
            v_list = list(v)
            random_state = np.random.RandomState(seed)
            shuffled_idx = random_state.choice(v_list, len(v_list), replace=False)
            rep_idx = np.array_split(shuffled_idx, n_reps)
            return [np.array(sorted(x)) for x in rep_idx]

        if pseudo_reps < 1:
            pseudo_reps = 1
        if pseudo_reps > 1:
            logger.warning(
                "make_bulk with pseudo_reps > 1 randomly splits cells within each "
                "group into descriptive resamples. These are not independent "
                "biological replicates and must not be used as such for "
                "differential expression."
            )
        if null_vals is None:
            null_vals = []
        if secondary_null_vals is None:
            secondary_null_vals = []
        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key` parameter")
        else:
            groups = self.cells.fetch_all(group_key)
            active_idx = self.cells.active_index(cell_key)
            groups_set = sorted(set(groups[active_idx]))
        if secondary_group_key is None:
            sec_groups: NDArray[Any] = np.array([None], dtype=object)
            sec_groups_set: list[Any] = [None]
        else:
            sec_groups = self.cells.fetch_all(secondary_group_key)
            sec_groups_set = sorted(set(sec_groups[active_idx]))

        assay = self._get_assay(from_assay)

        vals: dict[str, NDArray[Any]] = {}
        fracs: dict[str, NDArray[Any]] = {}
        all_feat_idx = np.arange(assay.feats.N)
        active_mask = np.zeros(self.cells.N, dtype=bool)
        active_mask[active_idx] = True
        for g in iter_progress(
            groups_set,
            desc="Aggregating pseudo-replicates",
            total=len(groups_set),
        ):
            if g in null_vals:
                continue
            for sg in sec_groups_set:  # type: ignore
                if sg in secondary_null_vals:
                    continue
                if sg is None and len(sec_groups) == 1:
                    g_idx = np.where((groups == g) & active_mask)[0]
                else:
                    g_idx = np.where((groups == g) & (sec_groups == sg) & active_mask)[
                        0
                    ]
                rep_indices = make_reps(g_idx, pseudo_reps, random_seed)
                for n, idx in enumerate(rep_indices):
                    if sg is None and len(sec_groups) == 1:
                        col_name = f"{g}"
                    else:
                        col_name = f"{g}_{sg}"
                    if pseudo_reps > 1:
                        col_name += f"_Rep{n + 1}"
                    if len(idx) == 0:
                        vals[col_name] = np.zeros(assay.feats.N)
                        continue
                    if aggr_type == "sum":
                        vals[col_name] = controlled_compute(
                            assay.rawData[idx].sum(axis=0), self.nthreads
                        )
                    elif aggr_type == "mean":
                        vals[col_name] = controlled_compute(
                            assay.normed(cell_idx=idx, feat_idx=all_feat_idx).mean(
                                axis=0
                            ),
                            self.nthreads,
                        )
                    else:
                        raise ValueError(
                            "ERROR: `aggr_type` can only be either 'sum' or 'mean'"
                        )
                    if return_fraction:
                        fracs[col_name] = (
                            (assay.rawData[idx] > 0).mean(axis=0).compute()
                        )

        vals_df = pd.DataFrame(vals).fillna(0)

        empty_idx = None
        if remove_empty_features:
            empty_idx = vals_df.sum(axis=1) != 0
            vals_df = vals_df.loc[empty_idx]

        if feature_label == "id":
            vals_df.set_index(
                _aligned_feature_labels(assay.feats.fetch_all("ids"), vals_df.index),
                inplace=True,
                drop=True,
            )
        elif feature_label == "name":
            vals_df.set_index(
                _aligned_feature_labels(assay.feats.fetch_all("names"), vals_df.index),
                inplace=True,
                drop=True,
            )

        if return_fraction:
            fracs_df = pd.DataFrame(fracs).fillna(0)
            if empty_idx is not None:
                fracs_df = fracs_df[empty_idx]
            fracs_df.set_index(vals_df.index, inplace=True, drop=True)
            return vals_df, fracs_df
        return vals_df

    def _statistical_selection(
        self,
        *,
        group_by: str,
        groups: tuple[Any, ...] | None,
        sample_by: str | None,
        pair_by: str | None,
        subset_by: str | None,
        cell_key: str | None,
        cell_idx: np.ndarray,
    ) -> _StatisticalSelection:
        """Resolve the effective rows shared by execution and live lookup."""
        groups_array = _fetch_statistical_column(self, group_by, cell_key)
        samples_array = (
            _fetch_statistical_column(self, sample_by, cell_key)
            if sample_by is not None
            else None
        )
        pairs_array = (
            _fetch_statistical_column(self, pair_by, cell_key)
            if pair_by is not None
            else None
        )
        subset_values = (
            _fetch_statistical_column(self, subset_by, cell_key)
            if subset_by is not None
            else None
        )

        group_missing = read_metadata_missing_rows(self.cells, group_by, cell_idx)
        sample_missing = (
            read_metadata_missing_rows(self.cells, sample_by, cell_idx)
            if sample_by is not None
            else None
        )
        pair_missing = (
            read_metadata_missing_rows(self.cells, pair_by, cell_idx)
            if pair_by is not None
            else None
        )
        subset_missing = (
            read_metadata_missing_rows(self.cells, subset_by, cell_idx)
            if subset_by is not None
            else None
        )

        selection_mask = np.ones(len(groups_array), dtype=bool)
        if subset_values is not None:
            subset_array = np.asarray(subset_values)
            if subset_array.dtype != bool:
                raise TypeError(
                    f"{subset_by!r} must be boolean; got {subset_array.dtype}"
                )
            if len(subset_array) != len(groups_array):
                raise ValueError("subset_by length must match selected cells")
            selection_mask &= subset_array
            if subset_missing is not None:
                selection_mask &= ~subset_missing
        selection_mask &= valid_category_mask(
            groups_array,
            missing_mask=group_missing,
        )
        if groups is not None:
            if len(groups) == 0:
                raise ValueError("groups must be non-empty when provided")
            selection_mask &= np.isin(groups_array, groups)
        if samples_array is not None:
            selection_mask &= valid_category_mask(
                samples_array,
                missing_mask=sample_missing,
            )
        if pairs_array is not None:
            valid_pair = valid_category_mask(
                pairs_array,
                missing_mask=pair_missing,
            )
            if np.any(selection_mask & ~valid_pair):
                raise ValueError(
                    "pairs must contain a valid pair value for every cell with a "
                    "valid sample"
                )
            selection_mask &= valid_pair
        if not selection_mask.any():
            raise ValueError("No cells remain after statistical-testing selections")

        selected_groups = np.asarray(groups_array[selection_mask])
        selected_samples = (
            np.asarray(samples_array[selection_mask])
            if samples_array is not None
            else None
        )
        selected_pairs = (
            np.asarray(pairs_array[selection_mask]) if pairs_array is not None else None
        )
        effective_cell_idx = np.asarray(cell_idx[selection_mask], dtype=np.int64)
        group_order = tuple(
            resolve_group_order(
                selected_groups,
                group_order=groups,
                full_groups=groups_array,
            )
        )
        return _StatisticalSelection(
            groups=selected_groups,
            samples=selected_samples,
            pairs=selected_pairs,
            subset_values=subset_values,
            selection_mask=selection_mask,
            effective_cell_idx=effective_cell_idx,
            group_order=group_order,
            cell_selection_fingerprint=_value_fingerprint(effective_cell_idx),
            group_fingerprint=_value_fingerprint(selected_groups),
            subset_fingerprint=(
                _value_fingerprint(subset_values) if subset_values is not None else None
            ),
            sample_fingerprint=(
                _value_fingerprint(selected_samples)
                if selected_samples is not None
                else None
            ),
            pair_fingerprint=(
                _value_fingerprint(selected_pairs)
                if selected_pairs is not None
                else None
            ),
        )

    def _statistical_key_series(
        self,
        keys: Sequence[str | CellField | FeatureRef],
        *,
        from_assay: str,
        cell_key: str | None,
        cell_idx: np.ndarray,
        normalization: NormalizationSpec | None,
        fetch_values: bool,
        fingerprint_values: bool = False,
        reject_selected_missing: bool = False,
        selection_mask: np.ndarray | None = None,
    ) -> tuple[
        list[str],
        list[str],
        list[str | None],
        list[str],
        list[np.ndarray],
    ]:
        """Resolve keys into labels, identities, assays, value ids, and values.

        Feature identities hash the assay, resolved feature ids, and reduction
        as structured values. Cell-metadata identities hash the column name and
        fingerprints of its stored values and explicit missing mask. Realized
        value fingerprints are computed from the selected float values before
        sample aggregation. Feature values are fetched one key at a time, so a
        fingerprint-only reuse check never materializes a multi-feature matrix.
        """
        if selection_mask is not None:
            selection_mask = np.asarray(selection_mask, dtype=bool)
            if selection_mask.shape != cell_idx.shape:
                raise ValueError("selection_mask must align with selected cells")
            feature_cell_idx = np.asarray(cell_idx[selection_mask], dtype=np.int64)
        else:
            feature_cell_idx = np.asarray(cell_idx, dtype=np.int64)
        cell_columns = set(self.cells.columns)
        labels: list[str] = []
        tested_features: list[str] = []
        source_assays: list[str | None] = []
        value_fingerprints: list[str] = []
        values_list: list[np.ndarray] = []
        for key in keys:
            if isinstance(key, FeatureRef) or (
                isinstance(key, str) and key not in cell_columns
            ):
                resolved = resolve_feature(self, key, from_assay=from_assay)
                labels.append(resolved.label)
                tested_features.append(
                    provenance_hash(
                        {
                            "source": "feature",
                            "assay": resolved.assay,
                            "ids": tuple(
                                str(identifier) for identifier in resolved.ids
                            ),
                            "reduction": resolved.reduction,
                        }
                    )
                )
                source_assays.append(resolved.assay)
                if fetch_values or fingerprint_values:
                    matrix = fetch_normalized_feature_matrix(
                        self,
                        [resolved],
                        feature_cell_idx,
                        normalization,
                    )
                    values = np.asarray(matrix[:, 0], dtype=np.float64)
                    if fingerprint_values:
                        value_fingerprints.append(_value_fingerprint(values))
                    if fetch_values:
                        values_list.append(values)
            else:
                column = key.key if isinstance(key, CellField) else key
                column_values = _fetch_statistical_column(
                    self,
                    column,
                    cell_key,
                )
                missing = read_metadata_missing_rows(self.cells, column, cell_idx)
                labels.append(
                    key.label if isinstance(key, CellField) and key.label else column
                )
                tested_features.append(
                    provenance_hash(
                        {
                            "source": "cell_metadata",
                            "column": column,
                            "values_fingerprint": _value_fingerprint(column_values),
                            "missing_fingerprint": (
                                _value_fingerprint(missing)
                                if missing is not None
                                else None
                            ),
                        }
                    )
                )
                source_assays.append(None)
                if fetch_values or fingerprint_values or reject_selected_missing:
                    selected_missing = missing
                    if selection_mask is not None:
                        selected_missing = (
                            missing[selection_mask] if missing is not None else None
                        )
                    if (
                        reject_selected_missing
                        and selected_missing is not None
                        and np.any(selected_missing)
                    ):
                        raise ValueError(
                            f"Tested metadata column {column!r} contains missing "
                            "values in the effective cell selection"
                        )
                    raw_values = np.asarray(column_values)
                    if selection_mask is not None:
                        raw_values = raw_values[selection_mask]
                    values = np.asarray(raw_values, dtype=np.float64)
                    if fingerprint_values:
                        value_fingerprints.append(_value_fingerprint(values))
                    if fetch_values:
                        values_list.append(values)
        return (
            labels,
            tested_features,
            source_assays,
            value_fingerprints,
            values_list,
        )

    def run_statistical_testing(
        self,
        keys: str | CellField | FeatureRef | Sequence[str | CellField | FeatureRef],
        *,
        group_by: str,
        groups: Sequence[Any] | None = None,
        comparisons: Sequence[tuple[Any, Any]] | None = None,
        test: Literal[
            "auto",
            "mann_whitney",
            "kruskal_wallis",
            "wilcoxon",
            "welch",
            "t_test",
            "one_way_anova",
        ] = "auto",
        posthoc: Literal["dunn"] | None = None,
        adjustment: Literal["fdr_bh", "bonferroni", "holm", "none"] = "fdr_bh",
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        sample_by: str | None = None,
        study_design: StudyDesign | None = None,
        pair_by: str | None = None,
        sample_stat: Literal["mean", "median", "fraction"] = "mean",
        expression_cutoff: float = 0.0,
        subset_by: str | None = None,
        cell_key: str | None = "I",
        from_assay: str | None = None,
        normalization: NormalizationSpec | None = None,
        skip_save: bool = False,
        invalidate_cache: bool = False,
    ) -> StatisticalTestResult:
        """Run statistical tests on values grouped by a categorical column.

        This mirrors the inputs of ``distribution`` so results can be compared
        directly against violin or box plots. ``keys`` may be cell-metadata
        columns or feature names. The chosen test follows the single-cell
        conventions for zero-inflated, non-normal values:

        - ``"mann_whitney"``: two independent groups (two-sided, tie and
          continuity corrected, matching the marker-search statistic).
        - ``"kruskal_wallis"``: three or more groups, with optional
          ``posthoc="dunn"`` for pairwise significance.
        - ``"wilcoxon"``: paired samples on aggregated (pseudobulk) data.
          Requires ``sample_by`` and ``pair_by``.
        - ``"welch"`` (alias ``"t_test"``): cell-level Welch's t-test on raw
          normalized values for exactly two groups, honouring
          ``alternative``. Descriptive only; no sample aggregation.
        - ``"one_way_anova"``: cell-level one-way ANOVA omnibus test on raw
          normalized values. Descriptive only; no post-hoc yet.

        With ``test="auto"`` the test is chosen from the design: paired data
        uses Wilcoxon, two groups use Mann-Whitney, and three or more use
        Kruskal-Wallis. Auto never picks a parametric method. ``groups``
        restricts the group set and fixes its order (which sets the contrast
        direction); ``comparisons`` restricts pairwise rows to the listed
        group pairs. With ``posthoc="dunn"`` both the omnibus Kruskal-Wallis
        and the pairwise Dunn's results are preserved. When multiple keys are
        tested, ``adjustment`` corrects p-values across keys in one pooled
        pass (default ``"fdr_bh"``); post-hoc p-values are corrected
        separately.

        Results are persisted as an immutable artifact under
        ``statistical_tests`` unless ``skip_save`` is ``True``. Every distinct
        variant (keys, groups, comparisons, adjustment, sample and subset
        columns) is stored under its own retrievable slot. Pass the returned
        ``result.artifact`` to ``get_statistical_tests`` for exact retrieval,
        or repeat the same variant parameters to look up its slot.

        Args:
            keys: Feature names or cell-metadata columns to test.
            group_by: Cell metadata column that groups the cells.
            groups: Keep and order only these ``group_by`` categories.
            comparisons: Restrict pairwise comparisons to these group pairs.
            test: Statistical test, or ``"auto"`` to pick from the design.
            posthoc: Pairwise test to run after Kruskal-Wallis (``"dunn"``).
            adjustment: Multiple-testing correction across keys and rows.
            alternative: Direction of the alternative hypothesis. Only the
                Welch t-test honours it; other tests remain two-sided.
            sample_by: Cell metadata column identifying biological samples.
            study_design: Study design supplying ``sample_by`` and ``pair_by``.
            pair_by: Cell metadata column identifying subjects or donors for
                paired tests.
            sample_stat: Aggregation across cells within a sample.
            expression_cutoff: Detection cutoff for ``sample_stat="fraction"``.
            subset_by: Boolean metadata column keeping only ``True`` cells.
            cell_key: Boolean column selecting cells (default ``"I"``).
            from_assay: Assay to read feature values from.
            normalization: How feature values are read.
            skip_save: Return results without writing to Zarr.
            invalidate_cache: Recompute even when a matching artifact exists.

        Returns:
            A :class:`~scarf.features.statistical.StatisticalTestResult`.
        """
        if group_by is None:
            raise ValueError(
                "ERROR: Please provide a value for group_by. This should be the "
                "name of a cell metadata column that groups the cells."
            )
        if adjustment not in ("fdr_bh", "bonferroni", "holm", "none"):
            raise ValueError(
                "adjustment must be 'fdr_bh', 'bonferroni', 'holm', or 'none'"
            )
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        if posthoc not in (None, "dunn"):
            raise ValueError("posthoc must be 'dunn' or None")
        if sample_stat not in ("mean", "median", "fraction"):
            raise ValueError("sample_stat must be 'mean', 'median', or 'fraction'")
        if test not in (
            "auto",
            "mann_whitney",
            "kruskal_wallis",
            "wilcoxon",
            "welch",
            "t_test",
            "one_way_anova",
        ):
            if test in _PARAMETRIC_TESTS:
                raise NotImplementedError(
                    "Scarf implements mann_whitney, kruskal_wallis, wilcoxon "
                    "plus the explicit cell-level parametric welch/t_test "
                    f"and one_way_anova; {test!r} is a non-parametric-phase "
                    "alias with no implementation."
                )
            raise ValueError(
                "test must be 'auto', 'mann_whitney', 'kruskal_wallis', "
                "'wilcoxon', 'welch', 't_test', or 'one_way_anova'"
            )
        normalization_digest = _statistical_normalization(normalization)
        if study_design is not None:
            if sample_by is not None and sample_by != study_design.sample_by:
                raise ValueError("sample_by conflicts with study_design.sample_by")
            sample_by = study_design.sample_by
            if pair_by is None:
                pair_by = study_design.subject_by or study_design.pair_by
        native_groups = _normalized_variant_groups(groups)
        native_comparisons = _normalized_variant_comparisons(comparisons)
        if native_groups is not None and len(native_groups) == 0:
            raise ValueError("groups must be non-empty when provided")
        if native_comparisons is not None and len(native_comparisons) == 0:
            raise ValueError("comparisons must be non-empty when provided")
        groups = native_groups
        comparisons = native_comparisons

        from_assay, cell_key = _resolve_assay_and_cell_key(
            self,
            from_assay,
            cell_key,
        )
        if isinstance(keys, (str, CellField, FeatureRef)):
            key_list: list[Any] = [keys]
        else:
            key_list = list(keys)
        if not key_list:
            raise ValueError("keys must be non-empty")

        cell_idx = _statistical_cell_indices(self, cell_key)
        (
            labels,
            tested_features,
            source_assays,
            _value_fingerprints,
            _values,
        ) = self._statistical_key_series(
            key_list,
            from_assay=from_assay,
            cell_key=cell_key,
            cell_idx=cell_idx,
            normalization=None,
            fetch_values=False,
        )
        feature_assays = {assay_name for assay_name in source_assays if assay_name}
        if len(feature_assays) > 1:
            raise ValueError(
                "Statistical testing does not support keys from multiple assays in "
                "one result. Run one assay at a time."
            )
        source_assay = next(iter(feature_assays), None)
        if source_assay is None:
            normalization_digest = {}
        selection = self._statistical_selection(
            group_by=group_by,
            groups=native_groups,
            sample_by=sample_by,
            pair_by=pair_by,
            subset_by=subset_by,
            cell_key=cell_key,
            cell_idx=cell_idx,
        )
        selection_mask = selection.selection_mask
        groups_arr_masked = selection.groups
        sample_arr_masked = selection.samples
        pair_arr_masked = selection.pairs
        n = int(selection_mask.sum())
        present = list(selection.group_order)
        n_groups = len(present)

        # Explicit metadata masks are semantic missing values. Only rows that
        # survive the full design selection are required to be present.
        self._statistical_key_series(
            key_list,
            from_assay=from_assay,
            cell_key=cell_key,
            cell_idx=cell_idx,
            normalization=None,
            fetch_values=False,
            reject_selected_missing=True,
            selection_mask=selection_mask,
        )
        key_labels = _statistical_key_labels(labels)

        if test == "auto":
            if pair_arr_masked is not None:
                effective_method = "wilcoxon"
            elif n_groups == 2:
                effective_method = "mann_whitney"
            else:
                effective_method = "kruskal_wallis"
        else:
            effective_method = test
            if groups is not None and n_groups != len(groups):
                raise ValueError(
                    "Explicit statistical group selections must all retain at "
                    "least one valid cell"
                )
            if effective_method in ("welch", "t_test") and n_groups != 2:
                raise ValueError(
                    "welch requires exactly two groups; use groups= to select "
                    "two groups or one_way_anova for three or more"
                )
        if posthoc == "dunn" and effective_method != "kruskal_wallis":
            raise ValueError("posthoc='dunn' requires test='kruskal_wallis'")
        if alternative != "two-sided" and effective_method not in ("welch", "t_test"):
            raise ValueError(
                "alternative is only supported for test='welch' or test='t_test'"
            )
        if pair_by is not None and effective_method != "wilcoxon":
            raise ValueError("pair_by is only supported for test='wilcoxon'")
        if effective_method == "wilcoxon" and (sample_by is None or pair_by is None):
            raise ValueError(
                "wilcoxon requires sample aggregation with both sample_by and pair_by"
            )
        if sample_by is None and sample_stat != "mean":
            raise ValueError("sample_stat requires sample_by")
        if sample_by is None and expression_cutoff != 0.0:
            raise ValueError("expression_cutoff requires sample_by")
        if expression_cutoff != 0.0 and sample_stat != "fraction":
            raise ValueError(
                "expression_cutoff is only used when sample_stat='fraction'"
            )
        if comparisons is not None and (
            effective_method == "one_way_anova"
            or effective_method == "kruskal_wallis"
            and posthoc is None
        ):
            raise ValueError(
                "comparisons requires a pairwise test; use posthoc='dunn' with "
                "kruskal_wallis"
            )

        cell_selection_fingerprint = selection.cell_selection_fingerprint
        group_fingerprint = selection.group_fingerprint
        subset_fingerprint = selection.subset_fingerprint
        sample_fingerprint = selection.sample_fingerprint
        pair_fingerprint = selection.pair_fingerprint
        equal_var = _statistical_equal_var(effective_method)
        variant_digest = _statistical_variant_digest(
            tested_features=tuple(tested_features),
            key_labels=tuple(key_labels),
            groups=native_groups,
            comparisons=native_comparisons,
            adjustment=adjustment,
            sample_by=sample_by,
            pair_by=pair_by,
            subset_by=subset_by,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
            normalization=normalization_digest,
            alternative=alternative,
            equal_var=equal_var,
        )
        slot_name = _statistical_slot_key(
            cell_key,
            group_by,
            effective_method,
            posthoc,
            variant_digest,
        )
        cell_selection_input: ArtifactRef | None = None
        source_assay_obj = (
            self._get_assay(source_assay) if source_assay is not None else None
        )
        source_dataset_fingerprint: str | None = None
        if source_assay is not None:
            if skip_save:
                assert source_assay_obj is not None
                existing_fingerprint = source_assay_obj.attrs.get("dataset_fingerprint")
                if existing_fingerprint is not None:
                    source_dataset_fingerprint = str(existing_fingerprint)
            else:
                source_dataset_fingerprint = self._ensure_dataset_fingerprint(
                    source_assay
                )
        uses_assay_normalization = normalization_digest.get("source") == "assay"
        normalization_method_identity = (
            callable_identity(source_assay_obj.normMethod)
            if source_assay_obj is not None and uses_assay_normalization
            else None
        )
        raw_size_factor = (
            getattr(source_assay_obj, "sf", None)
            if source_assay_obj is not None and uses_assay_normalization
            else None
        )
        size_factor_value = (
            float(raw_size_factor) if raw_size_factor is not None else None
        )
        current_value_fingerprints: tuple[str, ...] | None = None

        def resolve_current_value_fingerprints() -> tuple[str, ...]:
            nonlocal current_value_fingerprints
            if current_value_fingerprints is None:
                (
                    current_labels,
                    current_tested_features,
                    current_source_assays,
                    fingerprints,
                    _current_values,
                ) = self._statistical_key_series(
                    key_list,
                    from_assay=from_assay,
                    cell_key=cell_key,
                    cell_idx=cell_idx,
                    normalization=normalization,
                    fetch_values=False,
                    fingerprint_values=True,
                    reject_selected_missing=True,
                    selection_mask=selection_mask,
                )
                if (
                    _statistical_key_labels(current_labels) != key_labels
                    or current_tested_features != tested_features
                    or current_source_assays != source_assays
                    or len(fingerprints) != len(key_labels)
                    or any(not fingerprint for fingerprint in fingerprints)
                ):
                    raise RuntimeError(
                        "Statistical key identity changed while values were fetched"
                    )
                current_value_fingerprints = tuple(fingerprints)
            return current_value_fingerprints

        planned: Any = None
        select_statistical_artifact: Any = None
        if not skip_save:
            stats_grp = _statistical_index_group(self, source_assay, create=True)
            cell_selection_input = (
                self._ensure_cell_selection(cell_key) if cell_key is not None else None
            )
            expected_stat_columns = _statistical_storage_columns(
                effective_method,
                posthoc,
            )
            expected_posthoc_columns = _statistical_posthoc_columns(posthoc)

            arguments = StatisticalTestingArguments(
                cell_selection=cell_selection_input,
                normalization_method=normalization_method_identity,
                size_factor=size_factor_value,
                method=effective_method,
                posthoc=posthoc,
                adjustment_method=adjustment,
                sample_stat=sample_stat,
                expression_cutoff=expression_cutoff,
                groups=native_groups,
                comparisons=native_comparisons,
                sample_by=sample_by,
                pair_by=pair_by,
                subset_by=subset_by,
                normalization=normalization_digest,
                alternative=alternative,
                equal_var=equal_var,
                n_groups=n_groups,
                n_cells=n,
                cell_selection_fingerprint=cell_selection_fingerprint,
                tested_features=tuple(tested_features),
                source_assays=tuple(source_assays),
                source_dataset_fingerprint=source_dataset_fingerprint,
                group_fingerprint=group_fingerprint,
                subset_fingerprint=subset_fingerprint,
                sample_fingerprint=sample_fingerprint,
                pair_fingerprint=pair_fingerprint,
                from_assay=source_assay,
                cell_key=cell_key,
                group_key=group_by,
                key_labels=tuple(key_labels),
                invalidate_cache=invalidate_cache,
            )

            def statistical_reuse_is_valid(
                _ref: ArtifactRef,
                candidate: zarr.Group,
            ) -> bool:
                try:
                    if candidate.attrs.get("key_labels") != list(key_labels):
                        return False
                    if candidate.attrs.get("method") != effective_method:
                        return False
                    if candidate.attrs.get("posthoc") != posthoc:
                        return False
                    if candidate.attrs.get("adjustment_method") != adjustment:
                        return False
                    if candidate.attrs.get("n_groups") != n_groups:
                        return False
                    if candidate.attrs.get("n_cells") != n:
                        return False
                    if candidate.attrs.get("tested_features") != list(tested_features):
                        return False
                    if candidate.attrs.get("source_assays") != list(source_assays):
                        return False
                    if candidate.attrs.get("source_dataset_fingerprint") != (
                        source_dataset_fingerprint
                    ):
                        return False
                    if candidate.attrs.get("group_by") != group_by:
                        return False
                    if candidate.attrs.get("cell_key") != cell_key:
                        return False
                    if candidate.attrs.get("cell_selection_fingerprint") != (
                        cell_selection_fingerprint
                    ):
                        return False
                    if candidate.attrs.get("stat_columns") != list(
                        expected_stat_columns
                    ):
                        return False
                    if list(candidate.attrs.get("posthoc_stat_columns", [])) != list(
                        expected_posthoc_columns
                    ):
                        return False
                    secondary_guard = {
                        "sample_stat": sample_stat,
                        "expression_cutoff": float(expression_cutoff),
                        "normalization": normalization_digest,
                        "normalization_method": normalization_method_identity,
                        "size_factor": size_factor_value,
                        "alternative": alternative,
                        "equal_var": equal_var,
                        "group_fingerprint": group_fingerprint,
                        "group_order": [
                            _native_artifact_value(value) for value in present
                        ],
                        "subset_fingerprint": subset_fingerprint,
                        "sample_fingerprint": sample_fingerprint,
                        "pair_fingerprint": pair_fingerprint,
                    }
                    for attr_name, expected_value in secondary_guard.items():
                        stored = candidate.attrs.get(attr_name)
                        if isinstance(stored, np.generic):
                            stored = _native_artifact_value(stored)
                        if stored != expected_value:
                            return False
                    stored_value_fingerprints = candidate.attrs.get(
                        "value_fingerprints"
                    )
                    if (
                        not isinstance(stored_value_fingerprints, list)
                        or len(stored_value_fingerprints) != len(key_labels)
                        or any(
                            not isinstance(fingerprint, str) or not fingerprint
                            for fingerprint in stored_value_fingerprints
                        )
                        or stored_value_fingerprints
                        != list(resolve_current_value_fingerprints())
                    ):
                        return False
                    main_numeric = [
                        column
                        for column in expected_stat_columns
                        if column not in ("group_1", "group_2")
                    ]
                    posthoc_numeric = [
                        column
                        for column in expected_posthoc_columns
                        if column not in ("group_1", "group_2")
                    ]
                    for idx in range(len(key_labels)):
                        key_group = as_zarr_group(
                            candidate[str(idx)],
                            name=str(idx),
                        )
                        stats = np.asarray(
                            as_zarr_array(
                                key_group["stats"],
                                name="stats",
                            )[:]
                        )
                        if stats.ndim != 2 or stats.shape[1] != len(main_numeric):
                            return False
                        if set(key_group.attrs.get("stats_dtypes", {})) != set(
                            main_numeric
                        ):
                            return False
                        if expected_posthoc_columns:
                            posthoc_stats = np.asarray(
                                as_zarr_array(
                                    key_group["posthoc_stats"],
                                    name="posthoc_stats",
                                )[:]
                            )
                            if posthoc_stats.ndim != 2 or posthoc_stats.shape[1] != len(
                                posthoc_numeric
                            ):
                                return False
                            if set(
                                key_group.attrs.get("posthoc_stats_dtypes", {})
                            ) != set(posthoc_numeric):
                                return False
                except (KeyError, TypeError, ValueError, IndexError):
                    return False
                return True

            planned = arguments.plan(
                self.zw,
                scope="datastore" if source_assay is None else "assay",
                assay=source_assay,
                invalidate_cache=invalidate_cache,
                required_arrays=(),
                required_attributes=(
                    AttributeRequirement(
                        "stat_columns",
                        expected_types=(list, tuple),
                    ),
                ),
                reuse_validator=statistical_reuse_is_valid,
            )

            def select_statistical_artifact(ref: ArtifactRef) -> None:
                raw_artifacts = stats_grp.attrs.get("artifacts", {})
                if "artifacts" in stats_grp.attrs and not isinstance(
                    raw_artifacts,
                    dict,
                ):
                    raise RuntimeError("Statistical test artifact index is invalid")
                artifacts = (
                    dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
                )
                artifacts[slot_name] = ref.to_dict()
                stats_grp.attrs["artifacts"] = artifacts

            if planned.reused:
                select_statistical_artifact(planned.ref)
                slot_group, stored_ref = self._resolve_statistical_slot(
                    source_assay,
                    cell_key,
                    group_by,
                    effective_method,
                    posthoc,
                    variant_digest=variant_digest,
                )
                logger.info(
                    f"Reused statistical test results ({effective_method}) for "
                    f"{len(key_labels)} keys"
                )
                return self._read_statistical_slot(slot_group, artifact=stored_ref)

        outcomes: dict[str, GroupComparisonResult] = {}
        computed_value_fingerprints: list[str] = []
        for key, key_label in zip(key_list, key_labels, strict=True):
            (
                _labels,
                _identities,
                _assays,
                key_value_fingerprints,
                values_list,
            ) = self._statistical_key_series(
                [key],
                from_assay=from_assay,
                cell_key=cell_key,
                cell_idx=cell_idx,
                normalization=normalization,
                fetch_values=True,
                fingerprint_values=True,
                reject_selected_missing=True,
                selection_mask=selection_mask,
            )
            if len(key_value_fingerprints) != 1 or len(values_list) != 1:
                raise RuntimeError("Statistical value fetch returned invalid results")
            computed_value_fingerprints.append(key_value_fingerprints[0])
            values = values_list[0]
            outcomes[key_label] = compare_group_distributions(
                values,
                groups_arr_masked,
                test=effective_method,
                posthoc=posthoc,
                adjustment="none",
                samples=sample_arr_masked,
                pairs=pair_arr_masked,
                comparisons=comparisons,
                sample_stat=sample_stat,
                expression_cutoff=expression_cutoff,
                group_order=present,
                alternative=alternative,
            )

        computed_fingerprints = tuple(computed_value_fingerprints)
        if current_value_fingerprints is None:
            current_value_fingerprints = computed_fingerprints
        elif current_value_fingerprints != computed_fingerprints:
            raise RuntimeError(
                "Statistical values changed while the result was being computed"
            )

        tables = _pool_adjust(
            {label: outcome.table for label, outcome in outcomes.items()},
            adjustment,
        )
        posthoc_tables = _pool_adjust(
            {
                label: outcome.posthoc_table
                for label, outcome in outcomes.items()
                if outcome.posthoc_table is not None
            },
            adjustment,
        )

        result = StatisticalTestResult(
            method=effective_method,
            posthoc=posthoc,
            adjustment_method=adjustment,
            group_key=group_by,
            cell_key=cell_key,
            sample_by=sample_by,
            pair_by=pair_by,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
            alternative=alternative,
            equal_var=_statistical_equal_var(effective_method),
            n_groups=n_groups,
            n_cells=n,
            tested_features=tuple(tested_features),
            summary_scope="sample" if sample_by is not None else "cell",
            artifact=planned.ref if planned is not None else None,
            cell_selection=cell_selection_input,
            cell_selection_fingerprint=cell_selection_fingerprint,
            group_fingerprint=group_fingerprint,
            group_order=tuple(_native_artifact_value(value) for value in present),
            normalization=dict(normalization_digest),
            source_assays=tuple(source_assays),
            source_dataset_fingerprint=source_dataset_fingerprint,
            value_fingerprints=computed_fingerprints,
            sample_fingerprint=sample_fingerprint,
            pair_fingerprint=pair_fingerprint,
            normalization_method=normalization_method_identity,
            size_factor=size_factor_value,
            tables=tables,
            posthoc_tables=posthoc_tables,
        )

        if not skip_save:
            assert planned is not None
            assert select_statistical_artifact is not None
            remote_slot = start_artifact(self.zw, planned)
            self._write_statistical_slot(
                remote_slot,
                result,
                key_labels=key_labels,
                cell_selection_fingerprint=cell_selection_fingerprint,
                alternative=alternative,
                equal_var=equal_var,
                normalization=normalization_digest,
                group_fingerprint=group_fingerprint,
                subset_fingerprint=subset_fingerprint,
                sample_fingerprint=sample_fingerprint,
                pair_fingerprint=pair_fingerprint,
            )
            finish_artifact(remote_slot, planned)
            select_statistical_artifact(planned.ref)
            logger.info(
                f"Stored statistical test results ({effective_method}) for "
                f"{len(key_labels)} keys"
            )
        return result

    def _write_statistical_slot(
        self,
        group: zarr.Group,
        result: StatisticalTestResult,
        *,
        key_labels: list[str],
        cell_selection_fingerprint: str,
        alternative: str = "two-sided",
        equal_var: bool | None = None,
        normalization: dict[str, Any] | None = None,
        group_fingerprint: str | None = None,
        subset_fingerprint: str | None = None,
        sample_fingerprint: str | None = None,
        pair_fingerprint: str | None = None,
    ) -> None:
        storage_columns = _statistical_storage_columns(result.method, result.posthoc)
        posthoc_columns = _statistical_posthoc_columns(result.posthoc)
        main_string_columns = [
            column for column in storage_columns if column in ("group_1", "group_2")
        ]
        main_numeric_columns = [
            column for column in storage_columns if column not in main_string_columns
        ]
        posthoc_string_columns = [
            column for column in posthoc_columns if column in ("group_1", "group_2")
        ]
        posthoc_numeric_columns = [
            column for column in posthoc_columns if column not in posthoc_string_columns
        ]
        group.attrs["stat_columns"] = list(storage_columns)
        group.attrs["posthoc_stat_columns"] = list(posthoc_columns)
        group.attrs["method"] = result.method
        group.attrs["posthoc"] = result.posthoc
        group.attrs["adjustment_method"] = result.adjustment_method
        group.attrs["group_by"] = result.group_key
        group.attrs["cell_key"] = result.cell_key
        group.attrs["sample_by"] = result.sample_by
        group.attrs["pair_by"] = result.pair_by
        group.attrs["sample_stat"] = result.sample_stat
        group.attrs["expression_cutoff"] = result.expression_cutoff
        group.attrs["alternative"] = alternative
        group.attrs["equal_var"] = equal_var
        group.attrs["normalization"] = (
            dict(normalization) if normalization is not None else {}
        )
        group.attrs["normalization_method"] = result.normalization_method
        group.attrs["size_factor"] = result.size_factor
        group.attrs["group_fingerprint"] = group_fingerprint
        group.attrs["group_order"] = [
            _native_artifact_value(value) for value in result.group_order
        ]
        group.attrs["subset_fingerprint"] = subset_fingerprint
        group.attrs["sample_fingerprint"] = sample_fingerprint
        group.attrs["pair_fingerprint"] = pair_fingerprint
        group.attrs["n_groups"] = result.n_groups
        group.attrs["n_cells"] = result.n_cells
        group.attrs["tested_features"] = list(result.tested_features)
        group.attrs["source_assays"] = list(result.source_assays)
        group.attrs["source_dataset_fingerprint"] = result.source_dataset_fingerprint
        group.attrs["value_fingerprints"] = list(result.value_fingerprints)
        group.attrs["cell_selection"] = (
            result.cell_selection.to_dict()
            if result.cell_selection is not None
            else None
        )
        group.attrs["cell_selection_fingerprint"] = cell_selection_fingerprint
        group.attrs["summary_scope"] = result.summary_scope
        group.attrs["key_labels"] = list(key_labels)

        for idx, key_label in enumerate(key_labels):
            table = result.tables[key_label]
            key_group = group.create_group(str(idx))
            key_group.attrs["key_label"] = key_label
            key_group.attrs["key_index"] = idx
            _write_stats_array(key_group, "stats", table, main_numeric_columns)
            for column in main_string_columns:
                _write_group_column(key_group, column, table[column])
            if result.posthoc_tables and key_label in result.posthoc_tables:
                posthoc_table = result.posthoc_tables[key_label]
                _write_stats_array(
                    key_group,
                    "posthoc_stats",
                    posthoc_table,
                    posthoc_numeric_columns,
                )
                for column in posthoc_string_columns:
                    _write_group_column(
                        key_group,
                        f"posthoc_{column}",
                        posthoc_table[column],
                    )

    def _resolve_statistical_slot(
        self,
        source_assay: str | None,
        cell_key: str | None,
        group_key: str,
        method: str,
        posthoc: str | None,
        *,
        variant_digest: str,
    ) -> tuple[zarr.Group, ArtifactRef]:
        stats_grp = _statistical_index_group(self, source_assay, create=False)
        slot_name = _statistical_slot_key(
            cell_key,
            group_key,
            method,
            posthoc,
            variant_digest,
        )
        raw_artifacts = stats_grp.attrs.get("artifacts", {})
        if "artifacts" in stats_grp.attrs and not isinstance(raw_artifacts, dict):
            raise ValueError("Statistical test artifact index is invalid")
        artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
        if slot_name not in artifacts:
            raise KeyError(
                "ERROR: Couldn't find statistical test results. Make sure "
                "`run_statistical_testing` was called with the same `cell_key`, "
                "`group_key`, `test` method, and variant parameters (keys, "
                "groups, comparisons, adjustment, sample_by, pair_by, subset_by)."
            )
        ref = ArtifactRef.from_dict(artifacts[slot_name])
        invalid_scope = (
            ref.scope != "datastore"
            if source_assay is None
            else ref.scope != "assay" or ref.assay != source_assay
        )
        if ref.kind != "statistical_tests" or invalid_scope:
            raise ValueError("Statistical test artifact index has an invalid reference")
        path = artifact_path(ref)
        if path not in self.zw:
            raise KeyError(
                f"Statistical test artifact is missing at {path}; "
                "the store may have been modified"
            )
        return as_zarr_group(self.zw[path], name=path), ref

    @staticmethod
    def _read_statistical_slot(
        slot_group: zarr.Group,
        *,
        artifact: ArtifactRef,
    ) -> StatisticalTestResult:
        method = slot_group.attrs.get("method")
        posthoc = slot_group.attrs.get("posthoc")
        storage_columns = _statistical_storage_columns(method, posthoc)
        posthoc_columns = _statistical_posthoc_columns(posthoc)
        main_string_columns = [
            column for column in storage_columns if column in ("group_1", "group_2")
        ]
        main_numeric_columns = [
            column for column in storage_columns if column not in main_string_columns
        ]
        posthoc_string_columns = [
            column for column in posthoc_columns if column in ("group_1", "group_2")
        ]
        posthoc_numeric_columns = [
            column for column in posthoc_columns if column not in posthoc_string_columns
        ]
        key_labels = slot_group.attrs.get("key_labels", [])
        if (
            not isinstance(key_labels, list)
            or any(not isinstance(label, str) for label in key_labels)
            or len(set(key_labels)) != len(key_labels)
        ):
            raise ValueError("Statistical test key-label metadata is invalid")
        raw_value_fingerprints = slot_group.attrs.get("value_fingerprints")
        if raw_value_fingerprints is None:
            # Exact refs remain a historical retrieval mechanism for artifacts
            # written before value identity was persisted. Reuse is stricter.
            value_fingerprints: tuple[str, ...] = ()
        elif (
            not isinstance(raw_value_fingerprints, list)
            or len(raw_value_fingerprints) != len(key_labels)
            or any(
                not isinstance(fingerprint, str) or not fingerprint
                for fingerprint in raw_value_fingerprints
            )
        ):
            raise ValueError("Statistical value-fingerprint metadata is invalid")
        else:
            value_fingerprints = tuple(raw_value_fingerprints)
        tables: dict[str, pd.DataFrame] = {}
        posthoc_tables: dict[str, pd.DataFrame] = {}
        for idx, key_label in enumerate(key_labels):
            key_group = as_zarr_group(slot_group[str(idx)], name=str(idx))
            stats = np.asarray(as_zarr_array(key_group["stats"], name="stats")[:])
            frame = pd.DataFrame(stats, columns=main_numeric_columns)
            stats_dtypes = key_group.attrs.get("stats_dtypes", {})
            if not isinstance(stats_dtypes, dict) or set(stats_dtypes) != set(
                main_numeric_columns
            ):
                raise ValueError("Statistical test dtype metadata is invalid")
            for column, dtype in stats_dtypes.items():
                frame[column] = frame[column].astype(dtype)
            for column in main_string_columns:
                frame[column] = _read_group_column(key_group, column)
            frame = frame.loc[:, list(storage_columns)]
            tables[str(key_label)] = frame
            if posthoc_columns:
                posthoc_stats = np.asarray(
                    as_zarr_array(
                        key_group["posthoc_stats"],
                        name="posthoc_stats",
                    )[:]
                )
                posthoc_frame = pd.DataFrame(
                    posthoc_stats,
                    columns=posthoc_numeric_columns,
                )
                posthoc_dtypes = key_group.attrs.get("posthoc_stats_dtypes", {})
                if not isinstance(posthoc_dtypes, dict) or set(posthoc_dtypes) != set(
                    posthoc_numeric_columns
                ):
                    raise ValueError("Statistical post-hoc dtype metadata is invalid")
                for column, dtype in posthoc_dtypes.items():
                    posthoc_frame[column] = posthoc_frame[column].astype(dtype)
                for column in posthoc_string_columns:
                    posthoc_frame[column] = _read_group_column(
                        key_group,
                        f"posthoc_{column}",
                    )
                posthoc_frame = posthoc_frame.loc[:, list(posthoc_columns)]
                posthoc_tables[str(key_label)] = posthoc_frame
        raw_cell_selection = slot_group.attrs.get("cell_selection")
        cell_selection = (
            ArtifactRef.from_dict(raw_cell_selection)
            if raw_cell_selection is not None
            else None
        )
        return StatisticalTestResult(
            method=str(method),
            posthoc=posthoc,
            adjustment_method=str(slot_group.attrs.get("adjustment_method")),
            group_key=str(slot_group.attrs.get("group_by")),
            cell_key=slot_group.attrs.get("cell_key"),
            sample_by=slot_group.attrs.get("sample_by"),
            pair_by=slot_group.attrs.get("pair_by"),
            sample_stat=str(slot_group.attrs.get("sample_stat", "mean")),
            expression_cutoff=float(slot_group.attrs.get("expression_cutoff", 0.0)),
            alternative=str(slot_group.attrs.get("alternative", "two-sided")),
            equal_var=slot_group.attrs.get("equal_var"),
            n_groups=int(slot_group.attrs.get("n_groups", 0)),
            n_cells=int(slot_group.attrs.get("n_cells", 0)),
            tested_features=tuple(slot_group.attrs.get("tested_features", [])),
            summary_scope=slot_group.attrs.get("summary_scope", "cell"),
            artifact=artifact,
            cell_selection=cell_selection,
            cell_selection_fingerprint=slot_group.attrs.get(
                "cell_selection_fingerprint"
            ),
            group_fingerprint=slot_group.attrs.get("group_fingerprint"),
            group_order=tuple(slot_group.attrs.get("group_order", [])),
            normalization=dict(slot_group.attrs.get("normalization", {})),
            source_assays=tuple(slot_group.attrs.get("source_assays", [])),
            source_dataset_fingerprint=slot_group.attrs.get(
                "source_dataset_fingerprint"
            ),
            value_fingerprints=value_fingerprints,
            sample_fingerprint=slot_group.attrs.get("sample_fingerprint"),
            pair_fingerprint=slot_group.attrs.get("pair_fingerprint"),
            normalization_method=slot_group.attrs.get("normalization_method"),
            size_factor=slot_group.attrs.get("size_factor"),
            tables=tables,
            posthoc_tables=posthoc_tables,
        )

    def get_statistical_tests(
        self,
        from_assay: str | None = None,
        cell_key: str | None = "I",
        group_key: str | None = None,
        method: str | None = None,
        posthoc: str | None = None,
        *,
        keys: (
            str | CellField | FeatureRef | Sequence[str | CellField | FeatureRef] | None
        ) = None,
        artifact: ArtifactRef | None = None,
        groups: Sequence[Any] | None = None,
        comparisons: Sequence[tuple[Any, Any]] | None = None,
        adjustment: Literal["fdr_bh", "bonferroni", "holm", "none"] = "fdr_bh",
        sample_by: str | None = None,
        pair_by: str | None = None,
        subset_by: str | None = None,
        sample_stat: Literal["mean", "median", "fraction"] = "mean",
        expression_cutoff: float = 0.0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        normalization: NormalizationSpec | None = None,
    ) -> StatisticalTestResult:
        """Return statistical test results from `run_statistical_testing`.

        Pass an ``artifact`` returned by ``run_statistical_testing`` for exact
        immutable retrieval. Otherwise, because every distinct variant
        (tested keys, ``groups``, pairwise ``comparisons``, ``adjustment``,
        test direction, aggregation knobs, value normalization, and sample or
        subset columns) has its own slot, repeat those variant parameters
        exactly as they were passed to ``run_statistical_testing``.

        Args:
            from_assay: Name of the assay used when the results were stored.
            cell_key: Cell key used when the results were stored (``"I"`` by
                default; ``None`` matches the all-cells run).
            group_key: Grouping column used when the results were stored.
            method: Test method used when the results were stored
                (``"mann_whitney"``, ``"kruskal_wallis"``, ``"wilcoxon"``,
                ``"welch"``, or ``"one_way_anova"``).
            posthoc: Post-hoc test used when the results were stored.
            keys: Keys tested when the results were stored.
            artifact: Exact immutable result to load. When supplied, lookup
                parameters such as ``keys``, ``group_key``, and ``method`` are
                not required.
            groups: Group selection used when the results were stored.
            comparisons: Pairwise comparisons used when the results were stored.
            adjustment: Correction method used when the results were stored.
            sample_by: Sample column used when the results were stored.
            pair_by: Pair column used when the results were stored.
            subset_by: Subset column used when the results were stored.
            sample_stat: Aggregation statistic used when the results were
                stored.
            expression_cutoff: Detection cutoff used when the results were
                stored.
            alternative: Test direction used when the results were stored.
            normalization: Value normalization used when the results were
                stored.

        Returns:
            A :class:`~scarf.features.statistical.StatisticalTestResult`.
        """
        if artifact is not None:
            if not isinstance(artifact, ArtifactRef):
                raise TypeError("artifact must be an ArtifactRef")
            if artifact.kind != "statistical_tests":
                raise ValueError("artifact must reference statistical_tests")
            status = inspect_artifact(self.zw, artifact)
            if not status.exists:
                raise KeyError(
                    f"Statistical test artifact does not exist: {status.path}"
                )
            if not status.complete:
                raise RuntimeError(
                    f"Statistical test artifact is incomplete: {status.path}"
                )
            if (status.provenance or {}).get("operation") != "run_statistical_testing":
                raise ValueError("artifact was not produced by run_statistical_testing")
            slot_group = as_zarr_group(self.zw[status.path], name=status.path)
            return self._read_statistical_slot(slot_group, artifact=artifact)

        normalization_digest = _statistical_normalization(normalization)
        if adjustment not in ("fdr_bh", "bonferroni", "holm", "none"):
            raise ValueError(
                "adjustment must be 'fdr_bh', 'bonferroni', 'holm', or 'none'"
            )
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
        native_groups = _normalized_variant_groups(groups)
        native_comparisons = _normalized_variant_comparisons(comparisons)
        if native_groups is not None and len(native_groups) == 0:
            raise ValueError("groups must be non-empty when provided")
        if native_comparisons is not None and len(native_comparisons) == 0:
            raise ValueError("comparisons must be non-empty when provided")
        from_assay, cell_key = _resolve_assay_and_cell_key(
            self,
            from_assay,
            cell_key,
        )
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for group_key. This should be "
                "the same value used for `run_statistical_testing`"
            )
        if method is None:
            raise ValueError(
                "ERROR: Please provide the statistical test method used in "
                "`run_statistical_testing`"
            )
        if keys is None:
            raise ValueError(
                "ERROR: Please provide the keys tested in `run_statistical_testing`"
            )
        if isinstance(keys, (str, CellField, FeatureRef)):
            key_list: list[Any] = [keys]
        else:
            key_list = list(keys)
        if not key_list:
            raise ValueError("keys must be non-empty")

        cell_idx = _statistical_cell_indices(self, cell_key)
        (
            labels,
            tested_features,
            source_assays,
            _value_fingerprints,
            _values,
        ) = self._statistical_key_series(
            key_list,
            from_assay=from_assay,
            cell_key=cell_key,
            cell_idx=cell_idx,
            normalization=None,
            fetch_values=False,
        )
        feature_assays = {assay_name for assay_name in source_assays if assay_name}
        if len(feature_assays) > 1:
            raise ValueError(
                "Statistical testing does not support keys from multiple assays in "
                "one result. Run one assay at a time."
            )
        source_assay = next(iter(feature_assays), None)
        if source_assay is None:
            normalization_digest = {}
        key_labels = _statistical_key_labels(labels)
        if alternative != "two-sided" and method not in ("welch", "t_test"):
            raise ValueError(
                "alternative is only supported for method='welch' or method='t_test'"
            )
        variant_digest = _statistical_variant_digest(
            tested_features=tuple(tested_features),
            key_labels=tuple(key_labels),
            groups=native_groups,
            comparisons=native_comparisons,
            adjustment=adjustment,
            sample_by=sample_by,
            pair_by=pair_by,
            subset_by=subset_by,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
            normalization=normalization_digest,
            alternative=alternative,
            equal_var=_statistical_equal_var(method),
        )
        slot_group, artifact = self._resolve_statistical_slot(
            source_assay,
            cell_key,
            group_key,
            method,
            posthoc,
            variant_digest=variant_digest,
        )
        selection = self._statistical_selection(
            group_by=group_key,
            groups=native_groups,
            sample_by=sample_by,
            pair_by=pair_by,
            subset_by=subset_by,
            cell_key=cell_key,
            cell_idx=cell_idx,
        )
        (
            current_labels,
            current_tested_features,
            current_source_assays,
            current_value_fingerprints,
            _current_values,
        ) = self._statistical_key_series(
            key_list,
            from_assay=from_assay,
            cell_key=cell_key,
            cell_idx=cell_idx,
            normalization=normalization,
            fetch_values=False,
            fingerprint_values=True,
            reject_selected_missing=True,
            selection_mask=selection.selection_mask,
        )
        source_assay_obj = (
            self._get_assay(source_assay) if source_assay is not None else None
        )
        source_dataset_fingerprint = (
            str(source_assay_obj.attrs.get("dataset_fingerprint"))
            if source_assay_obj is not None
            and source_assay_obj.attrs.get("dataset_fingerprint") is not None
            else None
        )
        uses_assay_normalization = normalization_digest.get("source") == "assay"
        normalization_method_identity = (
            callable_identity(source_assay_obj.normMethod)
            if source_assay_obj is not None and uses_assay_normalization
            else None
        )
        raw_size_factor = (
            getattr(source_assay_obj, "sf", None)
            if source_assay_obj is not None and uses_assay_normalization
            else None
        )
        size_factor_value = (
            float(raw_size_factor) if raw_size_factor is not None else None
        )
        stored_value_fingerprints = slot_group.attrs.get("value_fingerprints")
        group_order = [_native_artifact_value(value) for value in selection.group_order]
        current_guards = {
            "key_labels": key_labels,
            "tested_features": current_tested_features,
            "source_assays": current_source_assays,
            "source_dataset_fingerprint": source_dataset_fingerprint,
            "n_groups": len(selection.group_order),
            "n_cells": int(selection.selection_mask.sum()),
            "cell_selection_fingerprint": selection.cell_selection_fingerprint,
            "group_fingerprint": selection.group_fingerprint,
            "group_order": group_order,
            "subset_fingerprint": selection.subset_fingerprint,
            "sample_fingerprint": selection.sample_fingerprint,
            "pair_fingerprint": selection.pair_fingerprint,
            "normalization": normalization_digest,
            "normalization_method": normalization_method_identity,
            "size_factor": size_factor_value,
        }
        stale = (
            _statistical_key_labels(current_labels) != key_labels
            or current_tested_features != tested_features
            or current_source_assays != source_assays
            or not isinstance(stored_value_fingerprints, list)
            or len(stored_value_fingerprints) != len(key_labels)
            or any(
                not isinstance(fingerprint, str) or not fingerprint
                for fingerprint in stored_value_fingerprints
            )
            or stored_value_fingerprints != current_value_fingerprints
        )
        if not stale:
            for attr_name, expected in current_guards.items():
                stored = slot_group.attrs.get(attr_name)
                if isinstance(stored, np.generic):
                    stored = _native_artifact_value(stored)
                if stored != expected:
                    stale = True
                    break
        if stale:
            raise ValueError(
                "Stored statistical test results are stale for the current data; "
                "rerun run_statistical_testing with these parameters, or use the "
                "exact artifact reference to retrieve the immutable historical "
                "result"
            )
        return self._read_statistical_slot(slot_group, artifact=artifact)


def _write_stats_array(
    key_group: zarr.Group,
    name: str,
    table: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    from ...storage.arrays import create_zarr_dataset

    key_group.attrs[f"{name}_dtypes"] = {
        column: str(table[column].dtype) for column in columns
    }
    numeric = np.asarray(table.loc[:, columns].to_numpy(dtype=np.float64))
    if numeric.ndim == 1:
        numeric = numeric.reshape(-1, 1)
    if np.isnan(numeric).any():
        raise ValueError("Statistical test results must not contain NaN")
    finite_slots = [
        idx
        for idx, column in enumerate(columns)
        if column not in {"t_statistic", "f_statistic"}
    ]
    if finite_slots and not np.isfinite(numeric[:, finite_slots]).all():
        raise ValueError(
            "Only t_statistic and f_statistic may be infinite in statistical results"
        )
    stats = create_zarr_dataset(
        key_group,
        name,
        (int(numeric.shape[0]), int(numeric.shape[1])),
        "float64",
        (int(numeric.shape[0]), int(numeric.shape[1])),
    )
    stats[:] = numeric


def _write_group_column(
    key_group: zarr.Group,
    name: str,
    values: pd.Series,
) -> None:
    """Write a group-label column preserving its native dtype."""
    from ...storage.arrays import create_metadata_column, create_zarr_dataset

    array = np.asarray(values)
    kind = array.dtype.kind
    if kind == "b":
        dtype_tag = "bool"
        data = array.astype(bool)
    elif kind in {"i", "u"}:
        dtype_tag = "int"
        data = array.astype(np.int64)
    elif kind == "f":
        dtype_tag = "float"
        data = array.astype(np.float64)
    else:
        dtype_tag = "str"
        data = array
    key_group.attrs[f"{name}_dtype"] = dtype_tag
    if dtype_tag == "str":
        create_metadata_column(
            key_group,
            name,
            data=[str(value) for value in data],
        )
        return
    column = create_zarr_dataset(
        key_group,
        name,
        (int(len(data)),),
        dtype_tag,
        (int(len(data)),),
    )
    column[:] = data


def _read_group_column(key_group: zarr.Group, name: str) -> np.ndarray:
    dtype_tag = key_group.attrs.get(f"{name}_dtype", "str")
    raw = np.asarray(as_zarr_array(key_group[name], name=name)[:])
    if dtype_tag == "bool":
        return np.asarray(raw, dtype=bool)
    if dtype_tag == "int":
        return np.asarray(raw, dtype=np.int64)
    if dtype_tag == "float":
        return np.asarray(raw, dtype=np.float64)
    if raw.dtype.kind == "S":
        raw = np.char.decode(raw, "utf-8")
    return np.asarray(raw, dtype=object)
