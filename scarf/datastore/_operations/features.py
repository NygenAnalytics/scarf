import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray

from ...features.variability import (
    DEFAULT_HVG_BLACKLIST,
    _ADAPTIVE_QUANTILE,
    hvg_options,
    resolve_hvg_max_cells,
)
from ...assay.feature_summary import (
    ensure_feature_summary,
    feature_summary_selected_count,
    feature_summary_values,
)
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    artifact_transaction,
    plan_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    callable_identity,
    fingerprint_array,
    inspect_artifact,
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
from ...assay import Assay, RNAassay
from ...assay.normalization import (
    library_size_divisors,
    norm_lib_size,
    reject_unknown_normalization_params,
    resolve_normalization_params,
)
from ...features.enrichment.net import AmbiguousTargets
from ...features.enrichment.results import EnrichmentResult
from ...features.markers.table import (
    MARKER_ADJUSTMENT_METHOD,
    MARKER_ADJUSTMENT_SCOPE,
    MARKER_ALTERNATIVE,
    MARKER_CONTINUITY_CORRECTION,
    MARKER_FOLD_CHANGE_POLICY,
    MARKER_METHOD,
    MARKER_STAT_COLUMNS,
    MARKER_TIE_CORRECTION,
    RankMarkerResult,
    _validate_marker_slot,
    load_marker_table,
)
from ...features.statistical import (
    MANN_WHITNEY_P_VALUE_POLICY,
    GroupComparisonResult,
    StatisticalDesignColumns,
    StatisticalKey,
    StatisticalSelection,
    StatisticalTestResult,
    build_statistical_result,
    choose_statistical_method,
    compare_group_distributions,
    distinct_label_keys,
    reject_missing_statistical_values,
    require_subjects_across_conditions,
    resolve_statistical_request,
    select_statistical_rows,
    statistical_equal_var,
    study_design_pairs_conditions,
    tested_column_identity,
    tested_feature_identity,
    value_fingerprint,
)
from ...features.values import (
    fetch_normalized_feature_matrix,
    resolve_feature_batch,
    spread_measured_rows,
)
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
    resolve_complete_labels,
    resolve_grouping,
    valid_category_mask,
)
from ...metadata.rows import read_metadata_missing_rows, read_metadata_rows
from ...utils.arrays import (
    array_digest,
    has_duplicates,
    regex_match_mask,
    sort_categories,
)
from ...utils.logging import logger
from .enrichment_store import (
    _ENRICHMENT_LAYOUT,
    _EnrichmentScorer,
    _enrichment_artifact_matches,
    _load_enrichment_result,
    _write_enrichment_slot,
)
from .statistical_store import (
    read_statistical_slot,
    statistical_reuse_validator,
    write_statistical_slot,
)

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _FeatureOperationsBase
else:
    _FeatureOperationsBase = object


def _read_arrays(
    group: zarr.Group, names: tuple[str, ...], *, workers: int
) -> list[np.ndarray]:
    from ...storage.stores import run_concurrently

    def reader(name: str) -> Callable[[], np.ndarray]:
        return lambda: np.asarray(as_zarr_array(group[name], name=name)[:])

    return run_concurrently([reader(name) for name in names], workers=workers)


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


def _validate_marker_group_name(name: str) -> None:
    """Reject a group label that cannot name its stored marker group.

    Each group's statistics are stored under a child group named by the label.
    """
    if not name.strip() or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(
            f"Marker group label {name!r} cannot name a stored marker group: "
            "labels must be non-blank, must not be '.' or '..', and must not "
            "contain '/' or '\\'. Rename the labels in a cell metadata column, "
            "or leave these cells out with select_cells, then freeze the labels "
            "with snapshot_cluster_labels before running run_marker_search."
        )


def _group_assignment_digest(values: np.ndarray) -> str:
    return array_digest(np.asarray(values).astype(str))


class _FeatureOperationsMixin(_FeatureOperationsBase):
    def select_all_features(
        self,
        *,
        from_assay: str | None = None,
    ) -> ArtifactRef:
        """Create or reuse the canonical all-true feature universe.

        The universe is an artifact only. It is never mirrored into feature
        metadata or registered under a mutable label.
        """
        self._require_writable("select_all_features")
        resolved_assay = self._get_assay(from_assay)
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(resolved_assay.z)
        values = np.ones(resolved_assay.feats.N, dtype=bool)
        payload_fingerprint = fingerprint_array(values)
        dataset_fingerprint = self._ensure_dataset_fingerprint(resolved_assay.name)
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
        self._require_writable("set_feature_selection")
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
            if has_duplicates(indexes):
                raise ValueError("feature_indexes contains duplicate indexes")
            values = np.zeros(assay.feats.N, dtype=bool)
            values[indexes] = True
        if not values.any():
            raise ValueError("Feature selection must contain at least one feature")
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay.z)
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
        """Select features detected in at least ``min_cells`` selected cells.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a selected cell.
        """
        if isinstance(min_cells, bool) or not isinstance(min_cells, int):
            raise TypeError("min_cells must be an integer")
        if min_cells < 0:
            raise ValueError("min_cells must be non-negative")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        self._require_measured_cells(
            assay.name, cell_selection, operation="select_detected_features"
        )
        self._require_writable("select_detected_features")
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            invalidate_cache=invalidate_cache,
        )
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay.z)
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
        if not planned.reused:
            n_selected = feature_summary_selected_count(self.zw, cell_selection)
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
        self._require_writable("select_hvgs")
        min_cells, top_n, n_bins, lowess_frac, keep_bounds, bin_strategy = hvg_options(
            min_cells=min_cells,
            top_n=top_n,
            n_bins=n_bins,
            lowess_frac=lowess_frac,
            keep_bounds=keep_bounds,
            bin_strategy=bin_strategy,
        )
        blacklist_fingerprint = (
            fingerprint_array(regex_match_mask(feature_names, blacklist))
            if blacklist
            else None
        )
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            invalidate_cache=invalidate_cache,
        )
        n_selected = feature_summary_selected_count(self.zw, cell_selection)
        max_cells_int = resolve_hvg_max_cells(
            max_cells, n_selected=n_selected, min_cells=min_cells
        )
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay.z)
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
                **(
                    {"blacklist_fingerprint": blacklist_fingerprint}
                    if blacklist_fingerprint is not None
                    else {}
                ),
                "keep_bounds": keep_bounds,
                "bin_strategy": bin_strategy,
                **(
                    {
                        "variance_estimator": "regularized_local_quantile",
                        "variance_quantile": _ADAPTIVE_QUANTILE,
                    }
                    if bin_strategy == "adaptive"
                    else {}
                ),
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
        if show_plot:
            if summary is None:
                summary = feature_summary_values(
                    self.zw,
                    summary_ref,
                    n_selected=n_selected,
                )
            assay._plot_hvgs(
                summary,
                _feature_selection_values(self.zw, planned.ref),
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
        """Persist highly variable genes as an immutable feature selection.

        Extra keyword arguments are options of
        :func:`scarf.plotting.highly_variable_features`, used when
        ``show_plot`` is True. Any other keyword raises ``TypeError`` before
        anything is computed or saved.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a selected cell.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        # The snapshot below writes, so the options are checked first; the
        # artifact method checks them again.
        hvg_options(
            min_cells=min_cells,
            top_n=top_n,
            n_bins=n_bins,
            lowess_frac=lowess_frac,
            keep_bounds=keep_bounds,
            bin_strategy=bin_strategy,
        )
        if plot_kwargs:
            import inspect

            from ...plotting import highly_variable_features

            accepted = {
                name
                for name, parameter in inspect.signature(
                    highly_variable_features
                ).parameters.items()
                if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name != "show"
            }
            unknown = sorted(set(plot_kwargs) - accepted)
            if unknown:
                raise TypeError(
                    "select_hvgs() got unexpected keyword arguments: "
                    + ", ".join(repr(name) for name in unknown)
                )
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "HVG selection can only be applied to an RNAassay; "
                f"received {type(assay).__name__}"
            )
        self._require_measured_cells(
            assay.name, cell_selection, operation="select_hvgs"
        )
        self._require_writable("select_hvgs")
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
        with (
            artifact_transaction(self.zw, planned) as slot,
            scorer.write_context(),
        ):
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
                resources=self.resources,
                io=self.storageIo,
            )
        return planned.ref

    def _prepare_enrichment_assay(
        self,
        *,
        display_name: str,
        operation: str,
        from_assay: str | None,
        cell_selection: ArtifactRef,
        features: ArtifactRef,
    ) -> tuple[RNAassay, np.ndarray, np.ndarray, ArtifactRef]:
        """Check an enrichment request and read its cell and feature rows.

        The arguments and the assay type are checked first, then the cells
        against the assay's membership, then that the store is writable.
        """
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(f"{display_name} can only be run on an RNAassay")
        self._require_measured_cells(assay.name, cell_selection, operation=operation)
        self._require_writable(display_name)
        # Scores are streamed from the counts, so their prepared identity must
        # be intact. The feature-selection lineage binds it to the result.
        self._ensure_dataset_fingerprint(assay.name)
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
        ambiguous_targets: AmbiguousTargets = "drop",
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
            ambiguous_targets: Handling of a target that matches several active
                features, such as a gene symbol shared by two feature ids.
                ``"drop"`` removes its edges before ``tmin`` pruning, logs a
                warning, and records the target in the artifact's
                ``dropped_ambiguous_targets`` attribute. ``"error"`` raises.
        Returns:
            A complete ``enrichment_scores`` artifact.

        Raises:
            PermissionError: If the store is not opened with ``zarr_mode='r+'``.
            UnmeasuredCellsError: If the assay did not measure a selected cell.

        Note:
            Cache identity covers selections, method parameters, normalization, and
            the prepared network, whose digest hashes only the retained edges. It
            assumes the stored count matrix is immutable.
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
                operation="run_waggr",
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
            ambiguous_targets=ambiguous_targets,
        )
        if assay.normMethod is not norm_lib_size:
            raise ValueError(
                "WAGGR requires the default norm_lib_size RNA normalization"
            )
        try:
            # A missing size factor raises TypeError here.
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
            normalization_method=callable_identity(norm_lib_size),
            size_factor=size_factor,
            invalidate_cache=invalidate_cache,
        )

        def score_batches() -> Iterator[np.ndarray]:
            cell_scalars = library_size_divisors(
                assay.cells.fetch_all(f"{assay.name}_nCounts")[cell_index],
                source=f"{assay.name}_nCounts",
            )
            model = build_waggr_model(network)
            raw = assay.rawData[:, network.matched_feature_index][cell_index, :]
            # The stream yields every selected cell once, in order.
            offset = 0
            for raw_block in raw.stream_blocks(
                nthreads=self.nthreads,
                msg="Scoring WAGGR",
                prefetch=1,
            ):
                block = np.asarray(raw_block, dtype=np.float64)
                end = offset + block.shape[0]
                values = size_factor * block / cell_scalars[offset:end].reshape(-1, 1)
                if log_transform:
                    values = np.log1p(values)
                yield score_waggr_block(values, model, mode=mode)
                offset = end

        return self._run_enrichment(
            assay=assay,
            invalidate_cache=invalidate_cache,
            scorer=_EnrichmentScorer(
                method="waggr",
                algorithm_version=WAGGR_ALGORITHM_VERSION,
                method_payload={
                    "dropped_ambiguous_targets": list(
                        network.dropped_ambiguous_targets
                    ),
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
        ambiguous_targets: AmbiguousTargets = "drop",
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
            ambiguous_targets: Handling of a target that matches several active
                features, such as a gene symbol shared by two feature ids.
                ``"drop"`` removes its edges before ``tmin`` pruning, logs a
                warning, and records the target in the artifact's
                ``dropped_ambiguous_targets`` attribute. ``"error"`` raises.
        Returns:
            A complete ``enrichment_scores`` artifact.

        Raises:
            PermissionError: If the store is not opened with ``zarr_mode='r+'``.
            UnmeasuredCellsError: If the assay did not measure a selected cell.

        Note:
            Cache identity covers selections, method parameters, and the prepared
            network, whose digest hashes only the retained edges. It assumes the
            stored count matrix is immutable.
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
                operation="run_aucell",
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
            ambiguous_targets=ambiguous_targets,
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
            for raw_block in raw.stream_blocks(
                nthreads=self.nthreads,
                msg="Scoring AUCell",
                prefetch=1,
            ):
                yield score_aucell_block(
                    np.asarray(raw_block),
                    permutation,
                    sets,
                    n_up=resolved_n_up,
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
                    "dropped_ambiguous_targets": list(
                        network.dropped_ambiguous_targets
                    ),
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
        from ...storage.stores import metadata_workers

        resolved_norm_params = resolve_normalization_params(
            assay, norm_params, caller="run_marker_search"
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
        if nthreads is None:
            nthreads = self.nthreads
        group_ids, group_sizes = np.unique(labels, return_counts=True)
        n_selected = int(len(labels))
        # The distinct values of one array have distinct string forms.
        expected_group_cell_counts: dict[str, tuple[int, int]] = {}
        for group_id, group_size in zip(group_ids, group_sizes, strict=True):
            group_name = str(group_id)
            _validate_marker_group_name(group_name)
            expected_group_cell_counts[group_name] = (
                int(group_size),
                n_selected - int(group_size),
            )
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

        io_workers = metadata_workers(self.zw)

        def marker_reuse_is_valid(
            _ref: ArtifactRef,
            candidate: zarr.Group,
        ) -> bool:
            try:
                stored_feature_index, stored_names, stored_ids = _read_arrays(
                    candidate,
                    ("feature_index", "feature_names", "feature_ids"),
                    workers=io_workers,
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
                    stored_names.astype(str), resolved_feature_names.astype(str)
                ) or not np.array_equal(
                    stored_ids.astype(str), resolved_feature_ids.astype(str)
                ):
                    return False
                _validate_marker_slot(
                    candidate,
                    resolved_feature_names,
                    expected_group_cell_counts=expected_group_cell_counts,
                    workers=io_workers,
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
        self._require_writable("run_marker_search")

        markers = find_markers_by_rank(
            assay=assay,
            groups=labels,
            cell_idx=cell_index,
            feat_idx=feature_index,
            nthreads=nthreads,
            writers=io_workers,
            **resolved_norm_params,
        )
        t_save = time.perf_counter()
        with artifact_transaction(self.zw, planned) as remote_slot:
            self._write_marker_slot(
                remote_slot,
                markers,
                workers=io_workers,
                feature_names=resolved_feature_names,
                feature_ids=resolved_feature_ids,
            )
        logger.info(f"Stored marker results for {len(markers.group_ids)} clusters")
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

        Raw counts of any storage dtype are ranked. Library-size markers
        require finite non-negative counts and cell totals.

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            clusters: Complete ``cluster_labels`` or ``cluster_cut`` artifact
                with a label for every cell. Labels that its linked missing
                mask flags raise ``ValueError``, as do labels that are blank,
                ``'.'`` or ``'..'``, or contain ``'/'`` or ``'\\'``, because each label
                names its stored marker group. To compare the groups of a
                cell metadata column, or only some cells or clusters, freeze
                the labels over those cells with ``snapshot_cluster_labels``.
            features: Explicit feature-selection artifact.
            nthreads: Threads for marker search.
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            A complete immutable marker-table artifact.

        Raises:
            ValueError: If a library-size normalized value of a selected cell
                and feature, or the total of a selected cell (its
                ``<assay>_nCounts`` value, or with ``renormalize_subset=True``
                its sum over the selected features), is negative or not
                finite. Nothing is written.
            PermissionError: If no matching result exists and the store is
                not opened with ``zarr_mode='r+'``. The check runs before
                the search.
            UnmeasuredCellsError: If the assay did not measure a labelled cell.
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
        resolved_clusters = resolve_complete_labels(self.zw, clusters, name="clusters")
        self._require_measured_cells(
            assay.name,
            resolved_clusters.source_cell_selection,
            operation="run_marker_search",
            remedy="labels",
        )
        if nthreads is None:
            nthreads = self.nthreads

        logger.debug(
            f"Running marker search for {assay.name} "
            f"(feature_selection={feature_selection.artifact_id[:12]}, "
            f"nthreads={nthreads})"
        )
        return self._run_marker_search_artifact(
            assay=assay,
            cell_selection=resolved_clusters.source_cell_selection,
            clusters=clusters,
            cluster_values=resolved_clusters.values,
            feature_selection=feature_selection,
            nthreads=nthreads,
            invalidate_cache=invalidate_cache,
            **norm_params,
        )

    @staticmethod
    def _write_marker_slot(
        group: zarr.Group,
        markers: RankMarkerResult,
        *,
        workers: int = 1,
        feature_names: np.ndarray,
        feature_ids: np.ndarray,
    ) -> None:
        """Write the marker tables of every group, each straight from the result.

        Concurrent writers each finish one group's stored table at a time.
        """
        from ...storage.arrays import create_metadata_column
        from ...storage.stores import run_concurrently

        # The result's feature indices ascend, so the last is the largest.
        last_feature = int(markers.feature_index[-1])
        if last_feature > np.iinfo(np.int32).max:
            raise ValueError("Marker feature indices must fit non-negative int32")
        if last_feature >= len(feature_names):
            raise ValueError("Marker feature indices must index feature_names")
        n_cells = int(markers.group_sizes.sum())
        group.attrs.update(
            {
                "stat_columns": list(MARKER_STAT_COLUMNS),
                "method": MARKER_METHOD,
                "alternative": MARKER_ALTERNATIVE,
                "tie_correction": MARKER_TIE_CORRECTION,
                "continuity_correction": MARKER_CONTINUITY_CORRECTION,
                "adjustment_method": MARKER_ADJUSTMENT_METHOD,
                "adjustment_scope": MARKER_ADJUSTMENT_SCOPE,
                "fold_change_policy": MARKER_FOLD_CHANGE_POLICY,
            }
        )
        columns: dict[str, tuple[np.ndarray, Any]] = {
            "feature_index": (markers.feature_index, np.int32),
            "feature_names": (np.asarray(feature_names).astype(str), None),
            "feature_ids": (np.asarray(feature_ids).astype(str), None),
        }

        def column_writer(name: str) -> Callable[[], None]:
            def write() -> None:
                data, dtype = columns[name]
                create_metadata_column(
                    group, name, data=data, dtype=dtype, overwrite=True
                )

            return write

        def cluster_writer(group_id: Any, n_group: int) -> Callable[[], None]:
            def write() -> None:
                cluster_group = group.create_group(
                    str(group_id),
                    attributes={"n_group": n_group, "n_reference": n_cells - n_group},
                )
                _write_compact_marker_stats(
                    cluster_group, markers.stored_statistics(group_id)
                )

            return write

        run_concurrently(
            [column_writer(name) for name in columns]
            + [
                cluster_writer(group_id, int(n_group))
                for group_id, n_group in zip(
                    markers.group_ids, markers.group_sizes, strict=True
                )
            ],
            workers=workers,
        )

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
        Groups follow the natural order of their labels, as in plots: numeric
        labels come first by value, so ``'2'`` precedes ``'10'``, and other
        labels follow in natural text order.
        Pass a specific ``group_id`` to return markers for that group only.
        For a wide export of marker names only, use ``export_markers_to_csv``.

        Args:
            marker: Exact marker-table artifact returned by ``run_marker_search``.
            group_id: One stored group label, or ``None`` for all groups. An
                integer is matched by its string form.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.
        Returns:
            Pandas dataframe with marker statistics and a string ``group_id``
            column. ``fold_change`` may be +inf or NaN.

        Raises:
            ValueError: If ``group_id`` is not a group of the marker artifact,
                or the table records no ``fold_change_policy``.
        """
        tables = self._filtered_marker_tables(
            marker,
            group_id=group_id,
            min_score=min_score,
            min_frac_exp=min_frac_exp,
        )
        if not tables:
            raise ValueError("Marker artifact contains no groups")
        return pd.concat(list(tables.values()), ignore_index=True)

    def _filtered_marker_tables(
        self,
        marker: ArtifactRef,
        *,
        group_id: str | int | None,
        min_score: float,
        min_frac_exp: float,
    ) -> dict[str, pd.DataFrame]:
        """Load and filter each requested marker group in natural label order."""
        _assay, g = self._resolve_marker_group(marker)
        gids = sort_categories(g.group_keys())
        if group_id is not None:
            requested = str(group_id)
            if requested not in gids:
                raise ValueError(
                    f"Marker artifact has no group {requested!r}; "
                    f"available groups: {', '.join(gids)}"
                )
            gids = [requested]

        feature_names = np.asarray(
            as_zarr_array(g["feature_names"], name="feature_names")[:]
        ).astype(str)
        tables: dict[str, pd.DataFrame] = {}
        for gid in gids:
            # The canonical reader rejects statistics that break their
            # contract and tables without the current fold-change policy.
            frame = load_marker_table(
                g,
                as_zarr_group(g[gid], name=gid),
                feature_names,
                group_id=gid,
            )
            keep = (frame["score"].to_numpy() >= min_score) & (
                frame["frac_exp"].to_numpy() >= min_frac_exp
            )
            tables[gid] = frame.loc[keep].reset_index(drop=True)
        return tables

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
        first). Columns follow the natural order of the group labels, as in
        ``get_markers``. This function does not export the scores of markers as
        they can be obtained using `get_markers` function.

        Args:
            marker: Exact marker-table artifact returned by ``run_marker_search``.
            csv_filename: Required parameter. Name, with path, of CSV file where the marker table is to be saved.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.

        Raises:
            ValueError: If the marker table records no ``fold_change_policy``.
        """
        tables = self._filtered_marker_tables(
            marker,
            group_id=None,
            min_score=min_score,
            min_frac_exp=min_frac_exp,
        )
        markers_table = {
            group_id: table["feature_name"].reset_index(drop=True)
            for group_id, table in tables.items()
        }
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

        Each new feature holds, for every cell, the mean normalized value of
        one group's features. The source normalization is fitted once over the
        cells that the source measured, for example ATAC document frequency or
        ADT CLR geometric means; the other cells get zero means.
        The assay becomes visible only after its counts are complete. If the
        write fails or is interrupted, the partial assay is removed or left
        pending until :meth:`discard_interrupted_assay` removes it.

        Args:
            groups: A ``pseudotime_aggregation`` artifact or an explicit feature
                metadata column name. Artifact inputs use only the features
                that the aggregation clustered, so its ``nan_cluster_value``
                never forms a group.
            assay_label: Name for the new assay.
            from_assay: Source assay. Artifact inputs derive this value and reject
                a conflicting explicit assay.
            exclude_values: Metadata-column group values to omit. Defaults to
                ``[-1]``. Missing values (``NaN``, ``None``, blank labels,
                or values under the column's missing mask) never form a
                group. Artifact inputs reject this argument.

        Returns: None

        Raises:
            PermissionError: If the store is not opened with ``zarr_mode='r+'``.
        """

        from ...metadata.membership import measured_rows, resolve_assay_membership
        from ...storage.identity import CountSummary, finalize_counts
        from ...storage.layout import array_shard_rows
        from ...storage.schema import derived_assay_transaction
        from ...storage.sharding import (
            dense_counts_admission,
            fit_count_layout,
            write_dense_from_row_batches,
        )

        self._require_writable("add_grouped_assay")
        provenance: dict[str, Any]
        if isinstance(groups, ArtifactRef):
            if from_assay is not None and from_assay != groups.assay:
                raise ValueError("from_assay conflicts with the groups artifact")
            if exclude_values is not None:
                raise ValueError(
                    "exclude_values applies only to metadata-column groups; "
                    "pseudotime aggregation artifacts group only clustered features"
                )
            # The loader checks the artifact kind, scope, and feature indices.
            aggregation = self.load_pseudotime_aggregation(groups)
            assay = self._get_assay(groups.assay)
            feature_indices = np.asarray(aggregation.feature_indices, dtype=np.int64)
            feature_clusters = np.asarray(aggregation.feature_clusters)
            group_set: list[Any] = sorted(set(feature_clusters.tolist()))
            feature_groups = [
                np.sort(feature_indices[feature_clusters == value])
                for value in group_set
            ]
            provenance = {"grouped_group_artifact": groups.to_dict()}
        elif isinstance(groups, str):
            if not groups:
                raise ValueError("groups metadata column must be non-empty")
            assay = self._get_assay(from_assay)
            # Every column of the feature table has one value per feature, and
            # valid_category_mask rejects a column that is not one-dimensional.
            group_values = np.asarray(assay.feats.fetch_all(groups))
            if exclude_values is None:
                exclude_values = [-1]
            present = valid_category_mask(
                group_values,
                missing_mask=read_metadata_missing_rows(
                    assay.feats, groups, np.arange(assay.feats.N, dtype=np.int64)
                ),
            )
            group_set = sorted(
                set(group_values[present].tolist()).difference(exclude_values)
            )
            feature_groups = [
                np.flatnonzero(present & (group_values == value)) for value in group_set
            ]
            provenance = {
                "grouped_group_column": groups,
                "grouped_group_digest": _group_assignment_digest(group_values),
            }
        else:
            raise TypeError("groups must be an ArtifactRef or metadata column name")
        if not group_set:
            raise ValueError("No feature groups remain after applying exclude_values")

        # The distinct values of one column or artifact have distinct strings.
        module_ids = [f"group_{x}" for x in group_set]
        cell_idx = np.arange(assay.cells.N, dtype=np.int64)
        # The source normalization is fitted on the cells that it measured;
        # the others get zero means, and the copied membership marks them.
        measured = measured_rows(assay.cells, assay.name, cell_idx)
        with derived_assay_transaction(
            self.z,
            assay_label,
            self.workspace,
            operation="add_grouped_assay",
            # Group means of a cell that the source did not measure are means
            # of no counts, so the new assay measures the source's cells.
            membership=resolve_assay_membership(assay.cells, assay.name),
        ) as transaction:
            # The writer holds one band of group means from the producer and
            # the band it writes. A layout that does not fit fails here,
            # before the pending assay exists.
            layout = fit_count_layout(
                {assay_label: (len(module_ids), np.float64)},
                nCells=assay.cells.N,
                profile=self.storageProfile,
                memoryBytes=self.resources.memoryBytes,
                transposed=(),
                admitCounts=dense_counts_admission(
                    CountSummary.nbytes_for(assay.cells.N, len(module_ids)),
                    lambda rows: 2 * rows * len(module_ids) * 8,
                ),
            )
            g = transaction.create_counts(
                assay.cells.N,
                module_ids,
                module_ids,
                np.float64,
                profile=self.storageProfile,
                policy=layout,
            )
            # RNA reads one output band per call; other assays stream their
            # normalized blocks, which the writer aligns to output bands.
            band = max(1, array_shard_rows(g))
            if measured is None:
                batches = assay._iter_feature_group_means(
                    cell_idx, feature_groups, block_rows=band
                )
            else:
                means = assay._iter_feature_group_means(
                    cell_idx[measured], feature_groups, block_rows=band
                )
                batches = (
                    piece
                    for _, piece in spread_measured_rows(
                        means, measured, len(module_ids), band, fill=0.0
                    )
                )
            summary = CountSummary(g)
            write_dense_from_row_batches(
                g,
                batches,
                dtype=np.float64,
                msg="Writing grouped assay",
                resources=self.resources,
                residentBytes=summary.nbytes,
                io=self.storageIo,
                countSummary=summary,
            )
            finalize_counts(g, summary=summary)
            transaction.group.attrs.update(
                {"grouped_from_assay": assay.name, **provenance}
            )
        self._register_derived_assay(assay_label, "Assay")

    def _register_derived_assay(self, assay_label: str, assay_type: str) -> None:
        """Load a newly published derived assay and its cell properties."""
        self._assayNames = tuple(self._scan_assays())
        self._load_assays(custom_assay_types={assay_label: assay_type})
        self._ini_cell_props(mito_pattern=None, ribo_pattern=None)

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
        Features from the BED file are retained even when they do not overlap any peak.

        The new assay becomes visible only after its counts, and RNA ``countsT``,
        are complete. If the write fails or is interrupted, the partial assay is
        removed or left pending until :meth:`discard_interrupted_assay` removes it.

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
            assay_type: Preset type of the new assay (Default value: 'Assay')
            cell_key: Cells used to learn peak document frequency. Every cell is
                      still scored so the new assay remains row-aligned.

        Returns:
            None

        Raises:
            PermissionError: If the store is not opened with ``zarr_mode='r+'``.
        """

        from ...assay.classification import validate_assay_type
        from ...features.genomic.melding import write_melded_counts
        from ...metadata.membership import resolve_assay_membership
        from ...storage.schema import derived_assay_transaction
        from ...storage.stores import zarr_group_root
        from ...writers.counts_t import finalize_writer_counts_t

        self._require_writable("add_melded_assay")
        if assay_label is None:
            raise ValueError(
                "ERROR: Please provide a value for `assay_label`. "
                "It will be used to create a new assay"
            )
        validate_assay_type(assay_type, assay=assay_label)
        if external_bed_fn is None:
            raise ValueError(
                "ERROR: Please provide a value for `external_bed_fn`. "
                "This should be a BED format file with at least 5 columns."
            )

        assay = self._get_assay(from_assay)
        # A melded cell that the source did not measure holds no counts, so
        # the new assay measures the source's cells.
        membership = resolve_assay_membership(assay.cells, assay.name)
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

        root = zarr_group_root(self.z, mode="r+")
        with derived_assay_transaction(
            root,
            assay_label,
            self.workspace,
            operation="add_melded_assay",
            membership=membership,
        ) as transaction:
            write_melded_counts(
                transaction,
                assay,
                feature_bed,
                peaks_col=peaks_col,
                scalar_coeff=scalar_coeff,
                renormalization=renormalization,
                peaks_coords=peaks_coords,
                idf_cell_idx=idf_cell_idx,
            )
            # RNA gene scores need countsT before the assay can be prepared.
            finalize_writer_counts_t(
                root,
                assay_label,
                self.workspace,
                assay_type=assay_type,
                resources=self.resources,
            )

        self._register_derived_assay(assay_label, assay_type)

    def discard_interrupted_assay(self, assay_label: str) -> None:
        """Remove a derived assay that an interrupted write left incomplete.

        ``add_grouped_assay`` and ``add_melded_assay`` remove a partial assay
        when an error stops them before publishing it. An interrupted or killed
        write instead leaves a pending assay that scans ignore and that blocks
        reuse of its name. This removes that pending assay so the operation can be
        retried. Complete assays are never removed.

        Args:
            assay_label: Name of the interrupted assay.

        Returns:
            None
        """
        from ...storage.schema import discard_pending_assay

        self._require_writable("discard_interrupted_assay")
        discard_pending_assay(self.z, assay_label, self.workspace)
        logger.info(f"Removed the interrupted assay {assay_label!r}")

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

        With ``aggr_type='mean'``, cells are normalized with the assay's
        normalization fitted once over every selected cell, then averaged
        within each group. RNA library-size normalization is per cell; ATAC
        document frequency and ADT CLR geometric means are therefore shared by
        all groups rather than learned separately for each group.

        Cells whose group or sub-group label is missing join no group, like
        cells whose value is in ``null_vals``. Missing labels are ``NaN``,
        ``None``, blank text, and values flagged by a linked missing mask.

        Args:
            groups: Explicit clustering artifact or user-owned metadata column
                used to group cells.
            from_assay: Name of assay to be used. If None and ``groups`` is an
                artifact, the artifact's assay is used; otherwise the default
                assay is used.
            cell_selection: Explicit selection. Artifact grouping derives its
                selection from lineage, and this argument may only narrow it.
                For metadata-column grouping, None uses the live ``I``
                column as the selection.
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

        Raises:
            ValueError: If two groups produce the same column name, for
                example groups ``'a_b'`` with sub-group ``'c'`` and ``'a'``
                with sub-group ``'b_c'``.
            UnmeasuredCellsError: If the assay did not measure a cell that is read.
        """

        def resolve_groups(
            source: ArtifactRef | str,
            expected_selection: ArtifactRef | None,
            live_idx: NDArray[np.int64] | None,
        ) -> tuple[
            NDArray[Any], ArtifactRef | None, NDArray[np.int64], NDArray[np.bool_]
        ]:
            # A metadata grouping without a selection reads the cells of the
            # live I column, live_idx, without a snapshot, so make_bulk writes
            # nothing.
            if isinstance(source, str):
                grouping: ArtifactRef | CellField = CellField(source)
            elif isinstance(source, ArtifactRef):
                grouping = source
            else:
                raise TypeError("groups must be an ArtifactRef or column name")
            if live_idx is not None and isinstance(source, str):
                labels = np.asarray(read_metadata_rows(self.cells, source, live_idx))
                missing = read_metadata_missing_rows(self.cells, source, live_idx)
                valid = valid_category_mask(labels, missing_mask=missing)
                return labels, None, live_idx, valid
            resolved = resolve_grouping(
                self.zw,
                self.cells,
                grouping,
                cell_selection=expected_selection,
            )
            valid = valid_category_mask(
                resolved.labels,
                missing_mask=resolved.missing_mask,
            )
            if live_idx is not None:
                # An artifact sub-grouping of the live cells is read over its
                # own cells, which must hold them.
                rows = np.searchsorted(resolved.cell_idx, live_idx)
                if bool((rows >= len(resolved.cell_idx)).any()) or not np.array_equal(
                    resolved.cell_idx[rows], live_idx
                ):
                    raise ValueError(
                        "The cells of the live I column must be a subset of the "
                        "artifact cell selection; pass cell_selection"
                    )
                return np.asarray(resolved.labels)[rows], None, live_idx, valid[rows]
            return resolved.labels, resolved.cell_selection, resolved.cell_idx, valid

        if pseudo_reps < 1:
            pseudo_reps = 1
        if pseudo_reps > 1:
            logger.warning(
                "make_bulk with pseudo_reps > 1 randomly splits cells within each "
                "group into descriptive resamples. These are not independent "
                "biological replicates and must not be used as such for "
                "differential expression."
            )
        from ...features.aggregation import (
            aggregate_bulk_profiles,
            bulk_column_rows,
            bulk_frames,
            bulk_read_cells,
        )

        live_idx = (
            self.cells.active_index("I").astype(np.int64, copy=False)
            if isinstance(groups, str) and cell_selection is None
            else None
        )
        group_values, resolved_selection, active_idx, labelled = resolve_groups(
            groups,
            cell_selection,
            live_idx,
        )
        # Cells with a missing label join no group.
        groups_set = sorted(set(group_values[labelled]))
        secondary: tuple[NDArray[Any], list[Any]] | None = None
        if secondary_groups is not None:
            # Sub-groups are read over the cells of the primary groups'
            # selection, in the same order.
            sec_group_values, _selection, _cells, sec_valid = resolve_groups(
                secondary_groups, resolved_selection, live_idx
            )
            secondary = (sec_group_values, sorted(set(sec_group_values[sec_valid])))
            labelled &= sec_valid

        if from_assay is None and isinstance(groups, ArtifactRef):
            from_assay = groups.assay
        assay = self._get_assay(from_assay)
        column_rows = bulk_column_rows(
            group_values,
            groups_set,
            labelled,
            active_idx,
            secondary=secondary,
            null_vals=null_vals,
            secondary_null_vals=secondary_null_vals,
            pseudo_reps=pseudo_reps,
            random_seed=random_seed,
        )
        # The cells whose counts aggregation reads.
        self._require_measured_cells(
            assay.name,
            bulk_read_cells(assay, aggr_type, active_idx, column_rows),
            operation="make_bulk",
            remedy="cell_selection",
        )
        values, fractions = aggregate_bulk_profiles(
            assay,
            column_rows,
            active_idx,
            aggr_type=aggr_type,
            return_fraction=return_fraction,
            cells=self.cells,
            resources=self.resources,
            nthreads=self.nthreads,
        )
        return bulk_frames(
            values,
            fractions,
            assay.feats,
            remove_empty_features=remove_empty_features,
            feature_label=feature_label,
            return_fraction=return_fraction,
        )

    def _statistical_design_columns(
        self,
        cell_idx: np.ndarray,
        *,
        sample_by: str | None,
        pair_by: str | None,
        subset_by: str | None,
    ) -> StatisticalDesignColumns:
        """Read the sample, pair, and subset columns over a grouping's cells.

        The values of every column are read before any missing mask.
        """
        samples = (
            read_metadata_rows(self.cells, sample_by, cell_idx)
            if sample_by is not None
            else None
        )
        pairs = (
            read_metadata_rows(self.cells, pair_by, cell_idx)
            if pair_by is not None
            else None
        )
        subset = (
            read_metadata_rows(self.cells, subset_by, cell_idx)
            if subset_by is not None
            else None
        )
        return StatisticalDesignColumns(
            samples=samples,
            pairs=pairs,
            subset=subset,
            sample_missing=(
                read_metadata_missing_rows(self.cells, sample_by, cell_idx)
                if sample_by is not None
                else None
            ),
            pair_missing=(
                read_metadata_missing_rows(self.cells, pair_by, cell_idx)
                if pair_by is not None
                else None
            ),
            subset_missing=(
                read_metadata_missing_rows(self.cells, subset_by, cell_idx)
                if subset_by is not None
                else None
            ),
            subset_by=subset_by,
        )

    def _resolve_statistical_keys(
        self,
        keys: Sequence[str | CellField | FeatureRef],
        *,
        from_assay: str | None,
        cell_idx: np.ndarray,
    ) -> list[StatisticalKey]:
        """Resolve every key once into its label, identity, and value source.

        Feature identities hash the assay, resolved feature ids, and reduction
        as structured values. Feature keys are resolved in one batch, so each
        assay's feature index is read once. Cell-metadata identities hash the
        column name and fingerprints of its stored values and explicit missing
        mask. Those values are read once and later reused for testing.
        """
        cell_columns = set(self.cells.columns)
        feature_keys: dict[int, str | FeatureRef] = {
            position: key
            for position, key in enumerate(keys)
            if isinstance(key, FeatureRef)
            or (isinstance(key, str) and key not in cell_columns)
        }
        resolved_features = dict(
            zip(
                feature_keys,
                resolve_feature_batch(
                    self,
                    list(feature_keys.values()),
                    from_assay=from_assay,
                ),
                strict=True,
            )
        )
        resolved_keys: list[StatisticalKey] = []
        for position, key in enumerate(keys):
            resolved = resolved_features.get(position)
            if resolved is not None:
                resolved_keys.append(
                    StatisticalKey(
                        label=resolved.label,
                        tested_feature=tested_feature_identity(resolved),
                        source_assay=resolved.assay,
                        feature=resolved,
                    )
                )
                continue
            column = key.key if isinstance(key, CellField) else key
            column_values = read_metadata_rows(self.cells, column, cell_idx)
            missing = read_metadata_missing_rows(self.cells, column, cell_idx)
            resolved_keys.append(
                StatisticalKey(
                    label=(
                        key.label
                        if isinstance(key, CellField) and key.label
                        else column
                    ),
                    tested_feature=tested_column_identity(
                        column, column_values, missing
                    ),
                    source_assay=None,
                    column=column,
                    column_values=column_values,
                    column_missing=missing,
                )
            )
        return resolved_keys

    def _iter_statistical_values(
        self,
        keys: Sequence[StatisticalKey],
        *,
        selection: StatisticalSelection,
        normalization: NormalizationSpec | None,
    ) -> Iterator[np.ndarray]:
        """Yield each key's selected float values in key order.

        Values are realized before sample aggregation. Feature keys are
        fetched in key batches, one blockwise pass per batch, and a batch
        holds at most a quarter of the memory budget.
        """
        feature_cell_idx = selection.effective_cell_idx
        n_cells = len(feature_cell_idx)
        batch_size = max(
            1,
            int(self.memoryBytes) // (4 * np.dtype(np.float64).itemsize * n_cells),
        )
        for start in range(0, len(keys), batch_size):
            batch = keys[start : start + batch_size]
            matrix = fetch_normalized_feature_matrix(
                self,
                [key.feature for key in batch if key.feature is not None],
                feature_cell_idx,
                normalization,
            )
            feature_column = 0
            for key in batch:
                if key.feature is not None:
                    yield np.ascontiguousarray(matrix[:, feature_column])
                    feature_column += 1
                else:
                    yield np.asarray(
                        np.asarray(key.column_values)[selection.selection_mask],
                        dtype=np.float64,
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

        - ``"mann_whitney"``: two independent groups, two-sided. When the
          two groups can be formed in at most 100,000 ways, as in typical
          sample-level designs, p-values come from the exact permutation null
          with ties handled exactly. Larger designs use the tie- and
          continuity-corrected normal approximation of the marker search.
          ``result.p_value_method`` records ``"exact"`` or ``"asymptotic"``.
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

        A ``study_design`` with ``subject_by`` (or ``pair_by``) pairs samples
        only for the Wilcoxon test: explicitly, or with ``test="auto"`` when
        two conditions remain. Repeated measures across three or more
        conditions, subjects nested within conditions, and independent tests
        raise an error. For nested subjects, pass the subject column as
        ``sample_by`` instead.

        Results are persisted as immutable artifacts unless ``skip_save`` is
        ``True``. Writing requires ``zarr_mode='r+'``; a matching saved
        result is still reused from a read-only store. Pass
        ``result.artifact`` to ``get_statistical_tests`` for exact retrieval.

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
            study_design: Study design supplying ``sample_by`` and, for the
                Wilcoxon test only, the pairing column.
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

        Raises:
            PermissionError: If a result must be written to a store that is
                not opened with ``zarr_mode='r+'``. The check runs before any
                value is computed.
            UnmeasuredCellsError: If a feature's assay did not measure a tested cell.
        """
        resolved_grouping = resolve_grouping(
            self.zw,
            self.cells,
            grouping,
            cell_selection=cell_selection,
        )
        request = resolve_statistical_request(
            keys,
            test=test,
            posthoc=posthoc,
            adjustment=adjustment,
            alternative=alternative,
            sample_stat=sample_stat,
            sample_by=sample_by,
            pair_by=pair_by,
            study_design=study_design,
            groups=groups,
            comparisons=comparisons,
            normalization=normalization,
        )
        sample_by = request.sample_by
        pair_by = request.pair_by
        design_pair_by = request.design_pair_by
        normalization_digest = request.normalization
        if from_assay is None:
            from_assay = (
                grouping.assay
                if isinstance(grouping, ArtifactRef) and grouping.assay is not None
                else self._defaultAssay
            )

        cell_idx = np.asarray(resolved_grouping.cell_idx, dtype=np.int64)
        statistical_keys = self._resolve_statistical_keys(
            request.keys,
            from_assay=from_assay,
            cell_idx=cell_idx,
        )
        labels = [key.label for key in statistical_keys]
        tested_features = [key.tested_feature for key in statistical_keys]
        source_assays = [key.source_assay for key in statistical_keys]
        feature_assays = {assay_name for assay_name in source_assays if assay_name}
        if len(feature_assays) > 1:
            raise ValueError(
                "Statistical testing does not support keys from multiple assays in "
                "one result. Run one assay at a time."
            )
        source_assay = next(iter(feature_assays), None)
        if source_assay is None:
            normalization_digest = {}
        selection = select_statistical_rows(
            resolved_grouping,
            self._statistical_design_columns(
                cell_idx, sample_by=sample_by, pair_by=pair_by, subset_by=subset_by
            ),
            groups=request.groups,
        )
        if study_design_pairs_conditions(design_pair_by, pair_by, selection):
            pair_by = design_pair_by
            selection = select_statistical_rows(
                resolved_grouping,
                self._statistical_design_columns(
                    cell_idx, sample_by=sample_by, pair_by=pair_by, subset_by=subset_by
                ),
                groups=request.groups,
            )
        if source_assay is not None:
            self._require_measured_cells(
                source_assay,
                selection.effective_cell_idx,
                operation="run_statistical_testing",
                remedy="cell_selection",
            )
        require_subjects_across_conditions(design_pair_by, selection)
        n = int(selection.selection_mask.sum())
        present = list(selection.group_order)
        n_groups = len(present)

        # Explicit metadata masks are semantic missing values. Only rows that
        # survive the full design selection are required to be present.
        reject_missing_statistical_values(statistical_keys, selection.selection_mask)
        key_labels = [str(key) for key in distinct_label_keys(labels)]
        effective_method = choose_statistical_method(
            test,
            selection,
            groups=request.groups,
            comparisons=request.comparisons,
            posthoc=posthoc,
            alternative=alternative,
            sample_by=sample_by,
            pair_by=pair_by,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
        )

        fingerprints = selection.fingerprints
        equal_var = statistical_equal_var(effective_method)
        grouping_input = grouping if isinstance(grouping, ArtifactRef) else None
        group_field = grouping if isinstance(grouping, CellField) else None
        cell_selection_input = resolved_grouping.cell_selection
        source_assay_obj = (
            self._get_assay(source_assay) if source_assay is not None else None
        )
        source_dataset_fingerprint: str | None = None
        if source_assay is not None:
            source_dataset_fingerprint = self._ensure_dataset_fingerprint(source_assay)
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
                current_value_fingerprints = tuple(
                    value_fingerprint(values)
                    for values in self._iter_statistical_values(
                        statistical_keys,
                        selection=selection,
                        normalization=normalization,
                    )
                )
            return current_value_fingerprints

        planned: Any = None
        if not skip_save:
            arguments = StatisticalTestingArguments(
                grouping=grouping_input,
                cell_selection=cell_selection_input,
                tested_features=tuple(tested_features),
                source_assays=tuple(source_assays),
                source_dataset_fingerprint=source_dataset_fingerprint,
                cell_selection_fingerprint=fingerprints.cell_selection_fingerprint,
                group_fingerprint=fingerprints.group_fingerprint,
                subset_fingerprint=selection.subset_fingerprint,
                sample_fingerprint=fingerprints.sample_fingerprint,
                pair_fingerprint=fingerprints.pair_fingerprint,
                group_field=group_field.key if group_field is not None else None,
                normalization_method=normalization_method_identity,
                size_factor=size_factor_value,
                method=effective_method,
                p_value_policy=(
                    MANN_WHITNEY_P_VALUE_POLICY
                    if effective_method == "mann_whitney"
                    else None
                ),
                posthoc=posthoc,
                adjustment_method=adjustment,
                sample_stat=sample_stat,
                expression_cutoff=expression_cutoff,
                groups=request.groups,
                comparisons=request.comparisons,
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
                reuse_validator=statistical_reuse_validator(
                    arguments,
                    group_order=present,
                    value_fingerprints=resolve_current_value_fingerprints,
                ),
            )

            if planned.reused:
                logger.info(
                    f"Reused statistical test results ({effective_method}) for "
                    f"{len(key_labels)} keys"
                )
                return read_statistical_slot(
                    artifact_group(self.zw, planned.ref),
                    artifact=planned.ref,
                )
            # Fail before computing when the result cannot be written.
            self._require_writable("run_statistical_testing")

        outcomes: dict[str, GroupComparisonResult] = {}
        computed_value_fingerprints: list[str] = []
        for key_label, values in zip(
            key_labels,
            self._iter_statistical_values(
                statistical_keys,
                selection=selection,
                normalization=normalization,
            ),
            strict=True,
        ):
            computed_value_fingerprints.append(value_fingerprint(values))
            outcomes[key_label] = compare_group_distributions(
                values,
                selection.groups,
                test=effective_method,
                posthoc=posthoc,
                adjustment="none",
                samples=selection.samples,
                pairs=selection.pairs,
                comparisons=request.comparisons,
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
        result = build_statistical_result(
            outcomes,
            selection=selection,
            method=effective_method,
            posthoc=posthoc,
            adjustment=adjustment,
            grouping=grouping_input,
            group_field=group_field,
            cell_selection=cell_selection_input,
            sample_by=sample_by,
            pair_by=pair_by,
            sample_stat=sample_stat,
            expression_cutoff=expression_cutoff,
            alternative=alternative,
            tested_features=tested_features,
            source_assays=source_assays,
            source_dataset_fingerprint=source_dataset_fingerprint,
            value_fingerprints=computed_fingerprints,
            normalization=normalization_digest,
            normalization_method=normalization_method_identity,
            size_factor=size_factor_value,
            artifact=planned.ref if planned is not None else None,
        )

        if not skip_save:
            assert planned is not None
            with artifact_transaction(self.zw, planned) as remote_slot:
                write_statistical_slot(remote_slot, result, arguments=arguments)
            logger.info(
                f"Stored statistical test results ({effective_method}) for "
                f"{len(key_labels)} keys"
            )
        return result

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
        return read_statistical_slot(slot_group, artifact=artifact)
