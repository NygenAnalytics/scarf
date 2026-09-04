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
    plan_artifact,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
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
)
from ...storage.selections import (
    read_stored_selection_indices,
    snapshot_run_metadata,
    validate_stored_selection_integrity,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...assay import Assay, RNAassay, lib_size_feature_stream_eligible
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
from ...metadata.artifacts import artifact_values
from ...metadata.selection import (
    CellField,
    FeatureRef,
    NormalizationSpec,
    ResolvedGrouping,
    StudyDesign,
    resolve_grouping,
    valid_category_mask,
)
from ...metadata.rows import read_metadata_missing_rows, read_metadata_rows
from ...utils.arrays import array_digest
from ...utils.compute import controlled_compute
from ...utils.logging import logger
from ...utils.progress import iter_progress
from .enrichment_store import (
    _ENRICHMENT_LAYOUT,
    _EnrichmentScorer,
    _enrichment_artifact_matches,
    _load_enrichment_result,
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

    def select_all_features(
        self,
        *,
        from_assay: str | None = None,
    ) -> ArtifactRef:
        """Create or reuse the canonical all-true feature universe.

        The universe is an artifact only. It is never mirrored into feature
        metadata or registered under a mutable label.
        """
        self._require_feature_write("Feature selection")
        resolved_assay = self._get_assay(from_assay)
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(resolved_assay)
        values = np.ones(resolved_assay.feats.N, dtype=bool)
        payload_fingerprint = fingerprint_array(values)
        dataset_fingerprint = resolved_assay.attrs.get("dataset_fingerprint")
        if dataset_fingerprint is None:
            dataset_fingerprint = self._calculate_dataset_fingerprint(
                resolved_assay.name
            )
        planned = _feature_selection_plan(
            self.zw,
            assay=resolved_assay.name,
            n_features=resolved_assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation="create_all_features",
            parameters={
                "dataset_fingerprint": str(dataset_fingerprint),
                "ordered_feature_ids_fingerprint": feature_ids_fingerprint,
            },
            inputs={},
            execution_options={},
            expected_payload_fingerprint=payload_fingerprint,
        )
        _write_feature_selection(
            self.zw,
            planned,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            payload={"values": values},
        )
        return planned.ref

    def set_feature_selection(
        self,
        *,
        from_assay: str | None = None,
        mask: np.ndarray | None = None,
        feature_indexes: Sequence[int] | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Persist an explicit feature mask as an immutable artifact."""
        self._require_feature_write("set_feature_selection")
        assay = self._get_assay(from_assay)
        all_features = self.select_all_features(from_assay=assay.name)
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
            execution_options={"invalidate_cache": invalidate_cache},
            expected_payload_fingerprint=values_fingerprint,
            invalidate_cache=invalidate_cache,
        )
        _write_feature_selection(
            self.zw,
            planned,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            payload={"values": values},
        )
        return planned.ref

    def select_detected_features(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        min_cells: int = 20,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Select features detected in at least ``min_cells`` selected cells."""
        self._require_feature_write("select_detected_features")
        if isinstance(min_cells, bool) or not isinstance(min_cells, int):
            raise TypeError("min_cells must be an integer")
        if min_cells < 0:
            raise ValueError("min_cells must be non-negative")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
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
            execution_options={"invalidate_cache": invalidate_cache},
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
        return planned.ref

    def _select_hvgs_artifact(
        self,
        *,
        assay: RNAassay,
        cell_selection: ArtifactRef,
        feature_names: np.ndarray,
        feature_snapshot: ArtifactRef,
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
        max_cells: float | None = None,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        invalidate_cache: bool = False,
        **plot_kwargs: Any,
    ) -> ArtifactRef:
        """Create or reuse an HVG artifact without creating a mutable alias."""
        self._require_feature_write("select_hvgs")
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
            operation="select_hvgs",
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
            inputs={
                "feature_summary": summary_ref,
                "feature_snapshot": feature_snapshot,
            },
            execution_options={
                "show_plot": show_plot,
                "plot_kwargs": plot_kwargs,
                "nthreads": assay.nthreads,
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
                feature_names=feature_names,
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
        return planned.ref

    def select_hvgs(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
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
        max_cells: float | None = None,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        invalidate_cache: bool = False,
        **plot_kwargs: Any,
    ) -> ArtifactRef:
        """Persist highly variable genes as an immutable feature selection."""
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "HVG selection can only be applied to an RNAassay; "
                f"received {type(assay).__name__}"
            )
        self._require_feature_write("select_hvgs")
        feature_snapshot = snapshot_run_metadata(
            self.zw,
            table_path=f"{assay.name}/featureData",
            id_column="ids",
            columns=("names",),
            axis="feature",
            assay=assay.name,
        )
        feature_names = np.asarray(
            as_zarr_array(
                artifact_group(self.zw, feature_snapshot)["names"],
                name="names",
            )[:]
        )
        ref = self._select_hvgs_artifact(
            assay=assay,
            cell_selection=cell_selection,
            feature_names=feature_names,
            feature_snapshot=feature_snapshot,
            min_cells=min_cells,
            top_n=top_n,
            min_var=min_var,
            max_var=max_var,
            min_mean=min_mean,
            max_mean=max_mean,
            n_bins=n_bins,
            lowess_frac=lowess_frac,
            blacklist=blacklist,
            keep_bounds=keep_bounds,
            show_plot=show_plot,
            max_cells=max_cells,
            bin_strategy=bin_strategy,
            invalidate_cache=invalidate_cache,
            **plot_kwargs,
        )
        return ref

    def _run_enrichment(
        self,
        *,
        assay: RNAassay,
        invalidate_cache: bool,
        scorer: _EnrichmentScorer,
    ) -> ArtifactRef:
        """Shared artifact plan, reuse, and write path for enrichment."""
        cell_index = scorer.cell_index
        cell_digest = array_digest(cell_index)
        feature_digest = array_digest(scorer.feature_index)
        attrs: dict[str, Any] = {
            "algorithm_version": scorer.algorithm_version,
            "cell_digest": cell_digest,
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
        if planned.reused:
            return planned.ref
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
        return planned.ref

    def _prepare_enrichment_assay(
        self,
        *,
        display_name: str,
        from_assay: str | None,
        cell_selection: ArtifactRef,
        features: ArtifactRef,
    ) -> tuple[RNAassay, np.ndarray, np.ndarray, ArtifactRef]:
        if self.zarr_mode != "r+":
            raise ValueError(
                f"{display_name} requires a DataStore opened with zarr_mode='r+'"
            )
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(f"{display_name} can only be run on an RNAassay")
        feature_selection = self.resolve_features(assay.name, features)
        feature_values = _feature_selection_values(self.zw, feature_selection)
        feature_index = np.flatnonzero(feature_values).astype(np.int64, copy=False)
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if len(cell_index) == 0:
            raise ValueError("Cell selection contains no active cells")
        if len(feature_index) == 0:
            raise ValueError("Feature selection contains no active features")
        return assay, cell_index, feature_index, feature_selection

    def run_waggr(
        self,
        net: pd.DataFrame,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        features: ArtifactRef,
        mode: Literal["wmean", "wsum"] = "wmean",
        tmin: int = 5,
        log_transform: bool = False,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Score weighted gene sets from streamed normalized RNA counts.

        Targets are matched to active feature names without case sensitivity. Sources
        with fewer than ``tmin`` matched non-zero edges are removed. Results are
        written to the assay's enrichment group and returned lazily.

        Args:
            net: Network with ``source`` and ``target`` columns. An optional
                ``weight`` column supplies signed numeric edge weights. Missing
                weights default to one.
            from_assay: RNA assay to score. The default assay is used when omitted.
            cell_selection: Explicit cells to score.
            features: Explicit feature-selection artifact.
            mode: ``"wmean"`` divides each weighted sum by the sum of absolute
                source weights. ``"wsum"`` returns the weighted sum.
            tmin: Minimum number of matched targets required per source.
            log_transform: Apply ``log1p`` after library-size normalization.
        Returns:
            A complete ``enrichment_scores`` artifact.

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

        if mode not in {"wmean", "wsum"}:
            raise ValueError("mode must be 'wmean' or 'wsum'")
        if not isinstance(log_transform, bool):
            raise TypeError("log_transform must be a boolean")
        assay, cell_index, feature_index, feature_selection = (
            self._prepare_enrichment_assay(
                display_name="WAGGR",
                from_assay=from_assay,
                cell_selection=cell_selection,
                features=features,
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
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            network_digest=network.network_digest,
            algorithm_version=WAGGR_ALGORITHM_VERSION,
            mode=mode,
            tmin=tmin,
            log_transform=log_transform,
            normalization_method=callable_identity(assay.normMethod),
            size_factor=size_factor,
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
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        features: ArtifactRef,
        tmin: int = 5,
        n_up: int | None = None,
        tie_seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Score gene sets by recovery among each cell's top-ranked RNA features.

        AUCell ranks every feature selected by ``features`` from raw counts. Network
        weights are ignored. Targets are matched without case sensitivity, then
        sources with fewer than ``tmin`` matched targets are removed.

        Args:
            net: Network with ``source`` and ``target`` columns.
            from_assay: RNA assay to score. The default assay is used when omitted.
            cell_selection: Explicit cells to score.
            features: Explicit feature-selection artifact.
            tmin: Minimum number of matched targets required per source.
            n_up: Number of top-ranked features used for recovery. When omitted,
                five percent of the ranking universe is used, clipped to its valid
                range.
            tie_seed: Seed for the global feature permutation used to resolve ties.
        Returns:
            A complete ``enrichment_scores`` artifact.

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

        assay, cell_index, feature_index, feature_selection = (
            self._prepare_enrichment_assay(
                display_name="AUCell",
                from_assay=from_assay,
                cell_selection=cell_selection,
                features=features,
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
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            network_digest=network.network_digest,
            algorithm_version=AUCELL_ALGORITHM_VERSION,
            tmin=tmin,
            n_up=resolved_n_up,
            tie_seed=tie_seed,
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
        enrichment: ArtifactRef,
        *,
        sources: Sequence[str] | None = None,
    ) -> EnrichmentResult:
        """Load an explicit enrichment artifact without materializing scores.

        Args:
            enrichment: Artifact returned by ``run_waggr`` or ``run_aucell``.
            sources: Optional source names to select and order.

        Returns:
            The stored metadata and a lazy cells-by-sources score matrix.
        """
        if not isinstance(enrichment, ArtifactRef):
            raise TypeError("enrichment must be an ArtifactRef")
        if (
            enrichment.kind != "enrichment_scores"
            or enrichment.scope != "assay"
            or enrichment.assay is None
        ):
            raise ValueError(
                "enrichment must identify an assay enrichment_scores artifact"
            )
        assay = self._get_assay(enrichment.assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("Enrichment results are only available for an RNAassay")
        return _load_enrichment_result(
            assay,
            enrichment=enrichment,
            sources=sources,
            artifact_root=self.zw,
        )

    def _run_marker_search_artifact(
        self,
        *,
        assay: Assay,
        cell_selection: ArtifactRef,
        clusters: ArtifactRef,
        cluster_values: np.ndarray,
        feature_selection: ArtifactRef,
        feature_names: np.ndarray | None = None,
        feature_snapshot: ArtifactRef | None = None,
        nthreads: int | None = None,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Create or reuse one immutable marker-table artifact."""
        from ...features.markers import find_markers_by_rank
        from ...storage.stores import is_remote_datastore

        reject_unknown_normalization_params(
            norm_params,
            caller="run_marker_search",
        )
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        feature_values = np.asarray(
            _feature_selection_values(self.zw, feature_selection),
            dtype=bool,
        )
        feature_index = np.flatnonzero(feature_values).astype(np.int64, copy=False)
        labels = np.asarray(cluster_values)
        if labels.ndim != 1 or len(labels) != len(cell_index):
            raise ValueError("Cluster values must contain one label per selected cell")
        if len(cell_index) == 0:
            raise ValueError("Cell selection contains no active cells")
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
        group_sizes = pd.Series(labels).value_counts()
        n_selected = int(len(labels))
        group_cell_counts = {
            group_id: (int(group_size), int(n_selected - group_size))
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
        expected_feature_index = np.flatnonzero(feature_values)
        resolved_feature_names = (
            np.asarray(assay.feats.fetch_all("names"))
            if feature_names is None
            else np.asarray(feature_names)
        )
        resolved_feature_ids = np.asarray(assay.feats.fetch_all("ids"))
        if resolved_feature_names.shape != (assay.feats.N,):
            raise ValueError(
                "Snapshot feature names must align with the assay feature axis"
            )

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
                if not np.array_equal(
                    np.asarray(
                        as_zarr_array(
                            candidate["feature_names"],
                            name="feature_names",
                        )[:]
                    ).astype(str),
                    resolved_feature_names.astype(str),
                ) or not np.array_equal(
                    np.asarray(
                        as_zarr_array(
                            candidate["feature_ids"],
                            name="feature_ids",
                        )[:]
                    ).astype(str),
                    resolved_feature_ids.astype(str),
                ):
                    return False
                _validate_marker_slot(
                    candidate,
                    resolved_feature_names,
                    expected_group_cell_counts=expected_group_cell_counts,
                )
            except (IndexError, KeyError, TypeError, ValueError):
                return False
            return True

        arguments = MarkerTableArguments(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            clusters=clusters,
            normalization=resolved_norm_params,
            normalization_method=callable_identity(assay.normMethod),
            size_factor=getattr(assay, "sf", None),
            method=MARKER_METHOD,
            alternative=MARKER_ALTERNATIVE,
            tie_correction=MARKER_TIE_CORRECTION,
            continuity_correction=MARKER_CONTINUITY_CORRECTION,
            adjustment_method=MARKER_ADJUSTMENT_METHOD,
            adjustment_scope=MARKER_ADJUSTMENT_SCOPE,
            nthreads=nthreads,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        inputs = dict(record.inputs)
        if feature_snapshot is not None:
            inputs["feature_snapshot"] = feature_snapshot
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "feature_index",
                    shape=(int(feature_values.sum()),),
                    dtype_kind="i",
                ),
                ArrayRequirement(
                    "feature_names",
                    shape=(assay.feats.N,),
                ),
                ArrayRequirement(
                    "feature_ids",
                    shape=(assay.feats.N,),
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
        if planned.reused:
            return planned.ref

        markers = find_markers_by_rank(
            assay=assay,
            groups=labels,
            cell_idx=cell_index,
            feat_idx=feature_index,
            nthreads=nthreads,
            **resolved_norm_params,
        )
        remote = is_remote_datastore(self.zarr_loc, self.z)
        t_save = time.perf_counter()
        remote_slot = start_artifact(self.zw, planned)
        workers = max(1, int(nthreads or self.nthreads))
        self._write_marker_slot(
            remote_slot,
            markers,
            workers=workers if remote else 1,
            group_cell_counts=group_cell_counts,
            feature_names=resolved_feature_names,
            feature_ids=resolved_feature_ids,
        )
        finish_artifact(remote_slot, planned)
        logger.info(f"Stored marker results for {len(markers)} clusters")
        logger.debug(
            f"Saved marker results to {artifact_path(planned.ref)} "
            f"in {time.perf_counter() - t_save:.1f}s"
        )
        return planned.ref

    def run_marker_search(
        self,
        clusters: ArtifactRef,
        *,
        from_assay: str | None = None,
        features: ArtifactRef,
        nthreads: int | None = None,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> ArtifactRef:
        """Persist marker tables for an explicit clustering artifact.

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            clusters: Complete ``cluster_labels`` or ``cluster_cut`` artifact.
            features: Explicit feature-selection artifact.
            nthreads: Threads for marker search.
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            A complete immutable marker-table artifact.
        """
        reject_unknown_normalization_params(
            norm_params,
            caller="run_marker_search",
        )
        if not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        feature_selection = self.resolve_features(assay.name, features)
        cluster_status = inspect_artifact(self.zw, clusters)
        if (
            clusters.kind not in {"cluster_labels", "cluster_cut"}
            or not cluster_status.exists
            or not cluster_status.complete
        ):
            raise ValueError("clusters must be a complete clustering artifact")
        raw_selection = (cluster_status.inputs or {}).get("cell_selection")
        if not isinstance(raw_selection, dict):
            raise ValueError("Clustering artifact has no cell-selection input")
        try:
            cell_selection = ArtifactRef.from_dict(raw_selection)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Clustering artifact cell selection is malformed") from exc
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        cluster_group = as_zarr_group(
            self.zw[artifact_path(clusters)],
            name=clusters.artifact_id,
        )
        value_name = "values" if clusters.kind == "cluster_labels" else "labels"
        group_labels = np.asarray(
            as_zarr_array(cluster_group[value_name], name=value_name)[:]
        )
        if group_labels.shape != (len(cell_index),):
            raise ValueError("Clustering labels do not align with their cell selection")
        if nthreads is None:
            nthreads = self.nthreads

        logger.debug(
            f"Running marker search for {assay.name} "
            f"(feature_selection={feature_selection.artifact_id[:12]}, "
            f"nthreads={nthreads})"
        )
        return self._run_marker_search_artifact(
            assay=assay,
            cell_selection=cell_selection,
            clusters=clusters,
            cluster_values=group_labels,
            feature_selection=feature_selection,
            nthreads=nthreads,
            invalidate_cache=invalidate_cache,
            **norm_params,
        )

    @staticmethod
    def _write_marker_slot(
        group: zarr.Group,
        markers: dict[Any, pd.DataFrame],
        *,
        workers: int = 1,
        group_cell_counts: dict[Any, tuple[int, int]],
        feature_names: np.ndarray,
        feature_ids: np.ndarray,
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
        create_metadata_column(
            group,
            "feature_names",
            data=np.asarray(feature_names).astype(str),
            overwrite=True,
        )
        create_metadata_column(
            group,
            "feature_ids",
            data=np.asarray(feature_ids).astype(str),
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
        marker: ArtifactRef,
    ) -> tuple[Assay, zarr.Group]:
        if not isinstance(marker, ArtifactRef):
            raise TypeError("marker must be an ArtifactRef")
        ref = marker
        if ref.kind != "marker_table" or ref.scope != "assay" or ref.assay is None:
            raise ValueError("marker must identify an assay marker_table artifact")
        assay = self._get_assay(ref.assay)
        status = self.inspect_artifact(ref)
        if not status.exists:
            raise ValueError("Marker artifact does not exist")
        if not status.complete:
            raise ValueError("Marker artifact is incomplete")
        inputs = status.inputs or {}
        stored_selection = inputs.get("cell_selection")
        if not isinstance(stored_selection, dict):
            raise ValueError("Marker artifact cell selection is missing")
        try:
            selection_ref = ArtifactRef.from_dict(stored_selection)
            validate_stored_selection_integrity(
                self.zw,
                selection_ref,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Marker artifact cell selection is invalid") from exc
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
        stored_clusters = inputs.get("clusters")
        if not isinstance(stored_clusters, dict):
            raise ValueError("Marker artifact cluster input is missing")
        try:
            cluster_ref = ArtifactRef.from_dict(stored_clusters)
            cluster_status = self.inspect_artifact(cluster_ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Marker artifact cluster input is invalid") from exc
        if (
            cluster_ref.kind not in {"cluster_labels", "cluster_cut"}
            or not cluster_status.exists
            or not cluster_status.complete
            or (cluster_status.inputs or {}).get("cell_selection")
            != selection_ref.to_dict()
        ):
            raise ValueError("Marker artifact cluster input is invalid")
        group = as_zarr_group(
            self.zw[artifact_path(ref)],
            name=artifact_path(ref),
        )
        if "feature_names" not in group or "feature_ids" not in group:
            raise ValueError("Marker artifact is missing frozen feature identities")
        return assay, group

    def get_markers(
        self,
        marker: ArtifactRef,
        *,
        group_id: str | int | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> pd.DataFrame:
        """Return marker features from `run_marker_search`.

        When ``group_id`` is ``None`` (default), markers for every group under
        the artifact are returned in one long table with a ``group_id`` column.
        Pass a specific ``group_id`` to return markers for that group only.
        For a wide export of marker names only, use ``export_markers_to_csv``.

        Args:
            marker: Exact marker-table artifact returned by ``run_marker_search``.
            group_id: One stored group identifier, or ``None`` for all groups.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.
        Returns:
            Pandas dataframe with marker statistics. All-group results include a ``group_id`` column.
        """

        _assay, g = self._resolve_marker_group(marker)
        out_cols = list(_MARKER_OUT_COLUMNS)
        gids: list[str | int] = sorted(g.group_keys())
        if group_id is not None:
            gids = [group_id]

        feature_names = np.asarray(
            as_zarr_array(g["feature_names"], name="feature_names")[:]
        ).astype(str)
        feature_ids = np.asarray(
            as_zarr_array(g["feature_ids"], name="feature_ids")[:]
        ).astype(str)
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
        marker: ArtifactRef,
        csv_filename: str,
        *,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> None:
        """Export markers of each cluster/group to a CSV file where each column
        contains the marker names sorted by score (descending order, highest
        first). This function does not export the scores of markers as they can
        be obtained using `get_markers` function.

        Args:
            marker: Exact marker-table artifact returned by ``run_marker_search``.
            csv_filename: Required parameter. Name, with path, of CSV file where the marker table is to be saved.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.
        Returns:
        """
        _assay, marker_group = self._resolve_marker_group(marker)
        markers_table = {}
        for group_id in sorted(marker_group.group_keys()):
            m = self.get_markers(
                marker,
                group_id=group_id,
                min_score=min_score,
                min_frac_exp=min_frac_exp,
            )
            if len(m) > 0:
                markers_table[group_id] = m["feature_name"].reset_index(drop=True)
            else:
                markers_table[group_id] = pd.Series([])
        pd.DataFrame(markers_table).fillna("").to_csv(csv_filename, index=False)
        return None

    def add_grouped_assay(
        self,
        groups: ArtifactRef | str,
        *,
        assay_label: str,
        from_assay: str | None = None,
        exclude_values: Sequence[Any] | None = None,
    ) -> None:
        """Add an assay containing the mean signal for explicit feature groups.

        Args:
            groups: A ``pseudotime_aggregation`` artifact or an explicit feature
                metadata column name.
            assay_label: Name for the new assay.
            from_assay: Source assay. Artifact inputs derive this value and reject
                a conflicting explicit assay.
            exclude_values: Group values to omit. Defaults to ``[-1]``.

        Returns: None
        """

        from ...storage.layout import array_shard_rows
        from ...storage.schema import create_zarr_count_assay
        from ...storage.sharding import write_dense_from_row_batches

        source_ref: ArtifactRef | None
        source_column: str | None
        if isinstance(groups, ArtifactRef):
            source_ref = groups
            source_column = None
            if (
                groups.kind != "pseudotime_aggregation"
                or groups.scope != "assay"
                or groups.assay is None
            ):
                raise ValueError(
                    "groups must reference an assay-scoped "
                    "pseudotime_aggregation artifact"
                )
            if from_assay is not None and from_assay != groups.assay:
                raise ValueError("from_assay conflicts with the groups artifact")
            assay = self._get_assay(groups.assay)
            status = inspect_artifact(self.zw, groups)
            if not status.complete or status.operation != "run_pseudotime_aggregation":
                raise ValueError(
                    "groups must reference a complete pseudotime aggregation"
                )
            group_node = as_zarr_array(
                artifact_group(self.zw, groups)["cluster_values"],
                name="cluster_values",
            )
            if group_node.ndim != 1 or group_node.shape != (assay.feats.N,):
                raise ValueError(
                    "Pseudotime aggregation cluster_values do not align with "
                    "the source assay"
                )
            group_values = np.asarray(group_node[:])
        elif isinstance(groups, str):
            if not groups:
                raise ValueError("groups metadata column must be non-empty")
            source_ref = None
            source_column = groups
            assay = self._get_assay(from_assay)
            group_values = np.asarray(assay.feats.fetch_all(groups))
        else:
            raise TypeError("groups must be an ArtifactRef or metadata column name")
        if group_values.ndim != 1 or group_values.shape != (assay.feats.N,):
            raise ValueError("groups must align with the complete feature axis")
        if exclude_values is None:
            exclude_values = [-1]
        group_set = sorted(set(group_values.tolist()).difference(exclude_values))
        if not group_set:
            raise ValueError("No feature groups remain after applying exclude_values")

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

        cell_idx = np.arange(assay.cells.N, dtype=np.int64)
        n_groups = len(group_set)
        band_rows = max(1, array_shard_rows(g))

        def grouped_batches() -> Iterator[np.ndarray]:
            for start in range(0, assay.cells.N, band_rows):
                stop = min(start + band_rows, assay.cells.N)
                rows = cell_idx[start:stop]
                matrix = np.empty((len(rows), n_groups), dtype=np.float64)
                for index, group_value in enumerate(group_set):
                    feature_index = np.flatnonzero(group_values == group_value)
                    matrix[:, index] = (
                        assay.normed(cell_idx=rows, feat_idx=feature_index)
                        .mean(axis=1)
                        .compute(nthreads=self.nthreads)
                    )
                yield matrix

        write_dense_from_row_batches(
            g,
            grouped_batches(),
            dtype=np.float64,
            msg="Writing grouped assay",
            resources=self.resources,
            io=self.storageIo,
        )

        self._load_assays(custom_assay_types={assay_label: "Assay"})
        self._ini_cell_props(min_features=0, mito_pattern="", ribo_pattern="")
        grouped_assay = self._get_assay(assay_label)
        grouped_assay.attrs["grouped_from_assay"] = assay.name
        if source_ref is not None:
            grouped_assay.attrs["grouped_group_artifact"] = source_ref.to_dict()
        else:
            assert source_column is not None
            grouped_assay.attrs["grouped_group_column"] = source_column
            grouped_assay.attrs["grouped_group_digest"] = _group_assignment_digest(
                group_values
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
        groups: ArtifactRef | str,
        *,
        from_assay: str | None = None,
        cell_selection: ArtifactRef | None = None,
        secondary_groups: ArtifactRef | str | None = None,
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
            groups: Explicit clustering artifact or user-owned metadata column
                used to group cells.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_selection: Explicit selection used with metadata-column
                grouping. Artifact grouping derives its selection from lineage.
            secondary_groups: Optional clustering artifact or user-owned
                metadata column used to sub-group cells.
            aggr_type: Type of aggregation to be used. Can be either 'mean' or 'sum'. (Default value: 'mean')
            return_fraction: Return the fraction of cells expressing a gene in each group. (Default value: False)
            feature_label: The column in feature metadata table to use as row labels. (Default value: 'index')
            pseudo_reps: Within each group, randomly split cells into this many
                pseudo-replicates. Values greater than 1 produce descriptive
                resamples of the same cells, not independent biological
                replicates. (Default value: 1)
            remove_empty_features: Remove features that are not expressed in any cell. (Default value: True)
            null_vals: Primary group values to skip.
            secondary_null_vals: Secondary group values to skip.
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

        def resolve_groups(
            source: ArtifactRef | str,
            expected_selection: ArtifactRef | None,
        ) -> tuple[NDArray[Any], ArtifactRef, NDArray[np.int64]]:
            if isinstance(source, str):
                selection = (
                    self.snapshot_cell_selection()
                    if expected_selection is None
                    else expected_selection
                )
                validate_stored_selection_integrity(
                    self.zw,
                    selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                )
                indices = read_stored_selection_indices(
                    self.zw,
                    selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                )
                values = np.asarray(self.cells.fetch_all(source))[indices]
                return values, selection, indices
            if not isinstance(source, ArtifactRef):
                raise TypeError("groups must be an ArtifactRef or column name")
            value_names = {
                "cell_cycle": "phase",
                "cluster_cut": "labels",
                "cluster_labels": "values",
                "hto_identity": "values",
                "smart_label": "values",
            }
            value_name = value_names.get(source.kind)
            if value_name is None:
                raise ValueError(
                    "Grouping artifacts must contain categorical cell labels"
                )
            status = inspect_artifact(self.zw, source)
            if not status.complete:
                raise ValueError("Grouping artifact is unavailable or incomplete")
            raw_selection = (status.inputs or {}).get("cell_selection")
            if not isinstance(raw_selection, dict):
                raise ValueError("Grouping artifact has no cell-selection input")
            selection = ArtifactRef.from_dict(raw_selection)
            validate_stored_selection_integrity(
                self.zw,
                selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            indices = read_stored_selection_indices(
                self.zw,
                selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            values = artifact_values(artifact_group(self.zw, source), value_name)
            if values.shape != (len(indices),):
                raise ValueError("Grouping values do not align with selected cells")
            if expected_selection is not None:
                validate_stored_selection_integrity(
                    self.zw,
                    expected_selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                )
                expected_indices = read_stored_selection_indices(
                    self.zw,
                    expected_selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                )
                keep = np.isin(indices, expected_indices, assume_unique=True)
                if int(keep.sum()) != len(expected_indices):
                    raise ValueError(
                        "cell_selection must be a subset of the grouping artifact"
                    )
                values = values[keep]
                indices = indices[keep]
                selection = expected_selection
            return values, selection, indices

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
        group_values, resolved_selection, active_idx = resolve_groups(
            groups,
            cell_selection,
        )
        groups_set = sorted(set(group_values))
        if secondary_groups is None:
            sec_group_values: NDArray[Any] = np.array([None], dtype=object)
            sec_groups_set: list[Any] = [None]
        else:
            sec_group_values, _secondary_selection, secondary_idx = resolve_groups(
                secondary_groups,
                resolved_selection,
            )
            if not np.array_equal(secondary_idx, active_idx):
                raise ValueError("Grouping artifacts use different ordered cells")
            sec_groups_set = sorted(set(sec_group_values))

        if from_assay is None and isinstance(groups, ArtifactRef):
            from_assay = groups.assay
        assay = self._get_assay(from_assay)

        vals: dict[str, NDArray[Any]] = {}
        fracs: dict[str, NDArray[Any]] = {}
        all_feat_idx = np.arange(assay.feats.N)
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
                if sg is None and len(sec_group_values) == 1:
                    selected_rows = np.flatnonzero(group_values == g)
                else:
                    selected_rows = np.flatnonzero(
                        (group_values == g) & (sec_group_values == sg)
                    )
                g_idx = active_idx[selected_rows]
                rep_indices = make_reps(g_idx, pseudo_reps, random_seed)
                for n, idx in enumerate(rep_indices):
                    if sg is None and len(sec_group_values) == 1:
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
        grouping: ResolvedGrouping,
        groups: tuple[Any, ...] | None,
        sample_by: str | None,
        pair_by: str | None,
        subset_by: str | None,
    ) -> _StatisticalSelection:
        """Resolve the effective rows for one explicit grouping source."""
        groups_array = np.asarray(grouping.labels)
        cell_idx = np.asarray(grouping.cell_idx, dtype=np.int64)
        samples_array = (
            read_metadata_rows(self.cells, sample_by, cell_idx)
            if sample_by is not None
            else None
        )
        pairs_array = (
            read_metadata_rows(self.cells, pair_by, cell_idx)
            if pair_by is not None
            else None
        )
        subset_values = (
            read_metadata_rows(self.cells, subset_by, cell_idx)
            if subset_by is not None
            else None
        )

        group_missing = grouping.missing_mask
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
        from_assay: str | None,
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
                column_values = read_metadata_rows(self.cells, column, cell_idx)
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
        grouping: ArtifactRef | CellField,
        *,
        cell_selection: ArtifactRef | None = None,
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
        from_assay: str | None = None,
        normalization: NormalizationSpec | None = None,
        skip_save: bool = False,
        invalidate_cache: bool = False,
    ) -> StatisticalTestResult:
        """Run statistical tests on values grouped by an explicit source.

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

        Results are persisted as immutable artifacts unless ``skip_save`` is
        ``True``. Pass ``result.artifact`` to ``get_statistical_tests`` for
        exact retrieval.

        Args:
            keys: Feature names or cell-metadata columns to test.
            grouping: Exact categorical artifact or explicit cell metadata field.
            cell_selection: Optional frozen selection for a metadata field, or
                a subset of an artifact grouping's stored selection.
            groups: Keep and order only these grouping categories.
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
            from_assay: Assay to read feature values from.
            normalization: How feature values are read.
            skip_save: Return results without writing to Zarr.
            invalidate_cache: Recompute even when a matching artifact exists.

        Returns:
            A :class:`~scarf.features.statistical.StatisticalTestResult`.
        """
        resolved_grouping = resolve_grouping(
            self.zw,
            self.cells,
            grouping,
            cell_selection=cell_selection,
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

        if from_assay is None:
            from_assay = (
                grouping.assay
                if isinstance(grouping, ArtifactRef) and grouping.assay is not None
                else self._defaultAssay
            )
        if isinstance(keys, (str, CellField, FeatureRef)):
            key_list: list[Any] = [keys]
        else:
            key_list = list(keys)
        if not key_list:
            raise ValueError("keys must be non-empty")

        cell_idx = np.asarray(resolved_grouping.cell_idx, dtype=np.int64)
        (
            labels,
            tested_features,
            source_assays,
            _value_fingerprints,
            _values,
        ) = self._statistical_key_series(
            key_list,
            from_assay=from_assay,
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
            grouping=resolved_grouping,
            groups=native_groups,
            sample_by=sample_by,
            pair_by=pair_by,
            subset_by=subset_by,
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
        grouping_input = grouping if isinstance(grouping, ArtifactRef) else None
        group_field = grouping if isinstance(grouping, CellField) else None
        cell_selection_input = resolved_grouping.cell_selection
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
        if not skip_save:
            expected_stat_columns = _statistical_storage_columns(
                effective_method,
                posthoc,
            )
            expected_posthoc_columns = _statistical_posthoc_columns(posthoc)

            arguments = StatisticalTestingArguments(
                grouping=grouping_input,
                cell_selection=cell_selection_input,
                tested_features=tuple(tested_features),
                source_assays=tuple(source_assays),
                source_dataset_fingerprint=source_dataset_fingerprint,
                cell_selection_fingerprint=cell_selection_fingerprint,
                group_fingerprint=group_fingerprint,
                subset_fingerprint=subset_fingerprint,
                sample_fingerprint=sample_fingerprint,
                pair_fingerprint=pair_fingerprint,
                group_field=group_field.key if group_field is not None else None,
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
                from_assay=source_assay,
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
                    if candidate.attrs.get("grouping") != (
                        grouping_input.to_dict() if grouping_input is not None else None
                    ):
                        return False
                    if candidate.attrs.get("group_field") != (
                        group_field.key if group_field is not None else None
                    ):
                        return False
                    if candidate.attrs.get("cell_selection") != (
                        cell_selection_input.to_dict()
                        if cell_selection_input is not None
                        else None
                    ):
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

            if planned.reused:
                logger.info(
                    f"Reused statistical test results ({effective_method}) for "
                    f"{len(key_labels)} keys"
                )
                return self._read_statistical_slot(
                    artifact_group(self.zw, planned.ref),
                    artifact=planned.ref,
                )

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
            grouping=grouping_input,
            group_field=group_field,
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
        group.attrs["grouping"] = (
            result.grouping.to_dict() if result.grouping is not None else None
        )
        group.attrs["group_field"] = (
            result.group_field.key if result.group_field is not None else None
        )
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
        if (
            not isinstance(raw_value_fingerprints, list)
            or len(raw_value_fingerprints) != len(key_labels)
            or any(
                not isinstance(fingerprint, str) or not fingerprint
                for fingerprint in raw_value_fingerprints
            )
        ):
            raise ValueError("Statistical value-fingerprint metadata is invalid")
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
        raw_grouping = slot_group.attrs.get("grouping")
        raw_group_field = slot_group.attrs.get("group_field")
        if (raw_grouping is None) == (raw_group_field is None):
            raise ValueError(
                "Statistical test artifact lacks one explicit grouping source; "
                "rerun run_statistical_testing"
            )
        if raw_grouping is not None and not isinstance(raw_grouping, dict):
            raise ValueError("Statistical test grouping metadata is invalid")
        if raw_group_field is not None and (
            not isinstance(raw_group_field, str) or not raw_group_field
        ):
            raise ValueError("Statistical test group field is invalid")
        grouping = (
            ArtifactRef.from_dict(raw_grouping)
            if isinstance(raw_grouping, dict)
            else None
        )
        group_field = (
            CellField(raw_group_field) if isinstance(raw_group_field, str) else None
        )
        return StatisticalTestResult(
            method=str(method),
            posthoc=posthoc,
            adjustment_method=str(slot_group.attrs.get("adjustment_method")),
            grouping=grouping,
            group_field=group_field,
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
        artifact: ArtifactRef,
    ) -> StatisticalTestResult:
        """Return one exact immutable statistical-test result."""
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")
        if artifact.kind != "statistical_tests":
            raise ValueError("artifact must reference statistical_tests")
        status = inspect_artifact(self.zw, artifact)
        if not status.exists:
            raise KeyError(f"Statistical test artifact does not exist: {status.path}")
        if not status.complete:
            raise RuntimeError(
                f"Statistical test artifact is incomplete: {status.path}"
            )
        if (status.provenance or {}).get("operation") != "run_statistical_testing":
            raise ValueError("artifact was not produced by run_statistical_testing")
        slot_group = as_zarr_group(self.zw[status.path], name=status.path)
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
