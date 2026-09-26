from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ...assay import Assay, ATACassay, RNAassay
from ...assay.feature_summary import (
    ensure_feature_summary,
    feature_summary_selected_count,
    feature_summary_values,
)
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_graph_assay_inputs,
)
from ...graph.kinds import require_graph_kind
from ...quality_control.cell_cycle import assign_cell_cycle_phase
from ...quality_control.filtering import (
    filter_cell_metrics,
    validate_filter_bounds,
)
from ...quality_control.hto import _hto_demux_method, hto_demux
from ...metadata.artifacts import (
    artifact_values,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.arguments import (
    CellCycleArguments,
    DoubletScoreArguments,
    HtoIdentityArguments,
    PrevalentPeakArguments,
)
from ...metadata.rows import (
    metadata_missing_mask,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import (
    NamedCellArtifact,
    require_complete_cluster_labels,
    resolve_cell_aligned_artifact,
)
from ...storage.arrays import linked_missing_mask
from ...storage.artifacts import (
    artifact_group,
    artifact_path,
    canonical_bytes,
    fingerprint_array,
    fingerprint_strings,
)
from ...storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
    read_feature_selection_indices,
    resolve_feature_selection,
)
from ...storage.refs import ArtifactRef
from ...storage.selections import (
    iter_stored_selection_blocks,
    ValidatedStoredSelection,
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
    selection_mask,
    validate_stored_selection_integrity,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...utils.compute import controlled_compute
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _QualityControlOperationsBase
else:
    _QualityControlOperationsBase = object


def _validated_named_cell_artifacts(
    values: Iterable[NamedCellArtifact] | None,
    *,
    expected_kind: str,
    label: str,
) -> list[NamedCellArtifact]:
    sources = list(values or ())
    names: set[str] = set()
    for source in sources:
        if not isinstance(source, NamedCellArtifact):
            raise TypeError(f"{label} must contain NamedCellArtifact values")
        if source.artifact.kind != expected_kind:
            raise ValueError(f"{label} must reference {expected_kind!r} artifacts")
        if source.name in names:
            raise ValueError(f"{label} must use unique semantic names")
        names.add(source.name)
    return sources


class _QualityControlOperationsMixin(_QualityControlOperationsBase):
    def _run_cell_cycle_scoring_artifact(
        self,
        *,
        assay: RNAassay,
        cell_selection: ArtifactRef,
        feature_names: np.ndarray | None = None,
        feature_snapshot: ArtifactRef | None = None,
        s_genes: list[str] | None = None,
        g2m_genes: list[str] | None = None,
        ctrl_size: int | None = None,
        log_transform: bool = True,
        n_bins: int = 50,
        rand_seed: int = 4466,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Create or reuse cell-cycle scores without creating metadata columns."""
        self._require_writable("run_cell_cycle_scoring")
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Cell-cycle scoring can only be applied to an RNAassay; "
                f"received {type(assay).__name__}"
            )
        if s_genes is None:
            from ...quality_control.cell_cycle_genes import s_phase_genes

            s_genes = list(s_phase_genes)
        if g2m_genes is None:
            from ...quality_control.cell_cycle_genes import g2m_phase_genes

            g2m_genes = list(g2m_phase_genes)
        control_size = (
            min(len(s_genes), len(g2m_genes)) if ctrl_size is None else ctrl_size
        )
        if isinstance(control_size, (bool, np.bool_)) or not isinstance(
            control_size, (int, np.integer)
        ):
            raise TypeError("ctrl_size must be a positive integer")
        if control_size < 1:
            raise ValueError("ctrl_size must be a positive integer")
        if not isinstance(log_transform, bool):
            raise TypeError("log_transform must be a bool")
        if feature_names is None:
            s_gene_indices = assay.feats.get_index_by(
                s_genes,
                "names",
                None,
            ).tolist()
            g2m_gene_indices = assay.feats.get_index_by(
                g2m_genes,
                "names",
                None,
            ).tolist()
        else:
            names = np.asarray(feature_names)
            if names.ndim != 1 or len(names) != assay.feats.N:
                raise ValueError(
                    "Snapshot feature names must align with the assay feature axis"
                )
            by_name: dict[str, list[int]] = {}
            for index, name in enumerate(names):
                by_name.setdefault(str(name).upper(), []).append(index)

            def indices_for(targets: list[str]) -> list[int]:
                return [
                    index
                    for target in targets
                    for index in by_name.get(target.upper(), ())
                ]

            s_gene_indices = indices_for(s_genes)
            g2m_gene_indices = indices_for(g2m_genes)
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            log_transform=log_transform,
            invalidate_cache=invalidate_cache,
        )
        n_cells = feature_summary_selected_count(
            self.zw,
            cell_selection,
            n_cells=assay.cells.N,
        )
        arguments = CellCycleArguments(
            feature_summary=summary_ref,
            cell_selection=cell_selection,
            s_gene_indices=tuple(s_gene_indices),
            g2m_gene_indices=tuple(g2m_gene_indices),
            control_size=control_size,
            log_transform=log_transform,
            n_bins=n_bins,
            rand_seed=rand_seed,
            invalidate_cache=invalidate_cache,
            count_arithmetic=assay._count_arithmetic("feature_scores"),
        )
        record = arguments.to_record()
        inputs = dict(record.inputs)
        if feature_snapshot is not None:
            inputs["feature_snapshot"] = feature_snapshot
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={
                "s_score": ((n_cells,), "f"),
                "g2m_score": ((n_cells,), "f"),
                "phase": ((n_cells,), None),
            },
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref

        cell_idx = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        summary = feature_summary_values(
            self.zw,
            summary_ref,
            n_selected=n_cells,
        )
        s_score = assay._score_feature_indices(
            np.asarray(s_gene_indices, dtype=np.int64),
            cell_idx,
            summary["avg"],
            ctrl_size=control_size,
            n_bins=n_bins,
            rand_seed=rand_seed,
            log_transform=log_transform,
        )
        g2m_score = assay._score_feature_indices(
            np.asarray(g2m_gene_indices, dtype=np.int64),
            cell_idx,
            summary["avg"],
            ctrl_size=control_size,
            n_bins=n_bins,
            rand_seed=rand_seed,
            log_transform=log_transform,
        )
        phase = np.asarray(assign_cell_cycle_phase(s_score, g2m_score))
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "s_score": np.asarray(s_score),
                "g2m_score": np.asarray(g2m_score),
                "phase": phase,
            },
        )
        return planned.ref

    def filter_cells(
        self,
        attrs: Iterable[str],
        lows: Iterable[float | None],
        highs: Iterable[float | None],
        *,
        cell_selection: ArtifactRef | None = None,
        keep_bounds: bool = False,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Create an immutable selection from metadata thresholds.

        The requested columns are read from current cell metadata when this
        method is called. Their values are fingerprinted in provenance, and the
        resulting immutable selection is stored. The live ``I`` column is
        snapshotted when ``cell_selection`` is omitted. Pass a prior selection
        explicitly to compose multiple filtering steps.

        Cells whose value is recorded as missing in a nullable column never
        pass its bounds. The fingerprint of such a column's missing-value mask
        is also recorded in provenance.

        Args:
            attrs: Names of distinct columns to be used for filtering
            lows: Lower bounds, in the same order as ``attrs``. Each bound is a
                finite number, text for a text column, or None.
            highs: Upper bounds, in the same order as ``attrs``. A lower bound
                cannot exceed its upper bound.
            cell_selection: Optional prior cell-selection artifact.
            keep_bounds: Retain values exactly equal to a bound.

        Returns:
            A complete datastore-scoped ``cell_selection`` artifact.

        Raises:
            TypeError: If a bound or ``keep_bounds`` has an invalid type.
            ValueError: If a bound is invalid, the input selection is empty,
                or no cell passes the filter.
        """
        attrs = list(attrs)
        lows = list(lows)
        highs = list(highs)
        if not (len(attrs) == len(lows) == len(highs)):
            raise ValueError("attrs, lows, and highs must have the same length")
        for attr in attrs:
            if not isinstance(attr, str):
                raise TypeError("attrs must contain only column names")
        if len(set(attrs)) != len(attrs):
            raise ValueError("attrs must not contain duplicate columns")
        validate_filter_bounds(lows, highs, keep_bounds=keep_bounds)
        available = set(self.cells.columns)
        missing = [attr for attr in attrs if attr not in available]
        if missing:
            joined = ", ".join(repr(attr) for attr in missing)
            raise KeyError(f"Cell metadata columns not found: {joined}")
        prior, active_idx = self._filter_input_cells(cell_selection)
        # Provenance fingerprints cover whole columns, so each is read in full.
        input_fingerprints: dict[str, str] = {}
        missing_fingerprints: dict[str, str] = {}
        values_by_attr: dict[str, np.ndarray] = {}
        missing_by_attr: dict[str, np.ndarray] = {}
        for attr, values in zip(
            attrs, self.cells.fetch_all_columns(attrs), strict=True
        ):
            input_fingerprints[attr] = (
                fingerprint_strings(values)
                if values.dtype.kind in {"O", "S", "U"}
                else fingerprint_array(values)
            )
            values_by_attr[attr] = values[active_idx]
            mask_array = metadata_missing_mask(self.cells, attr)
            if mask_array is not None:
                mask = np.asarray(mask_array[:], dtype=bool)
                missing_fingerprints[attr] = fingerprint_array(mask)
                missing_by_attr[attr] = mask[active_idx]
        result = filter_cell_metrics(
            values_by_attr,
            missing_by_attr,
            np.ones(len(active_idx), dtype=bool),
            method="manual",
            lows=lows,
            highs=highs,
            keep_bounds=keep_bounds,
        )
        new_bool = np.zeros(self.cells.N, dtype=bool)
        new_bool[active_idx] = result.retained
        inputs: dict[str, Any] = {
            "prior_cell_selection": prior.ref,
            "metadata_fingerprints": input_fingerprints,
        }
        if missing_fingerprints:
            inputs["missing_mask_fingerprints"] = missing_fingerprints
        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=new_bool,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="filter_cells",
            parameters={
                "attrs": attrs,
                "lows": lows,
                "highs": highs,
                "keep_bounds": keep_bounds,
            },
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        remaining = int(stored.sum())
        logger.info(f"Cell filtering retained {remaining}/{self.cells.N} cells")
        return ref

    def select_cells(
        self,
        values: ArtifactRef,
        *,
        low: float | None = None,
        high: float | None = None,
        include: Sequence[Any] | None = None,
        cell_selection: ArtifactRef | None = None,
        keep_bounds: bool = False,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Select cells from one numeric or categorical artifact vector.

        Numeric values use optional bounds. Categorical values use ``include``.
        The artifact must identify its source cell selection in provenance. By
        default the new selection is composed with that source selection. An
        explicit ``cell_selection`` may narrow it further, but cannot add cells
        that were absent from the source artifact. Cells whose value the
        artifact records as missing are never selected.

        Args:
            values: Complete artifact with one scalar value per selected cell.
            low: Optional lower bound.
            high: Optional upper bound.
            include: Categorical values to retain instead of applying bounds.
            cell_selection: Optional prior selection to intersect.
            keep_bounds: Retain values exactly equal to either bound.
            invalidate_cache: Create a fresh result instead of reusing a match.

        Returns:
            A complete datastore-scoped ``cell_selection`` artifact.
        """
        if not isinstance(values, ArtifactRef):
            raise TypeError("values must be an ArtifactRef")
        if not isinstance(keep_bounds, bool):
            raise TypeError("keep_bounds must be a boolean")

        raw_include: tuple[str | int | float | bool, ...] | None = None
        if include is not None:
            if isinstance(include, str | bytes) or not isinstance(include, Sequence):
                raise TypeError("include must be a sequence of scalar values")
            included: list[str | int | float | bool] = []
            for raw_value in include:
                value = (
                    raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                )
                if not isinstance(value, str | int | float | bool):
                    raise TypeError("include must contain only scalar values")
                if isinstance(value, float) and not np.isfinite(value):
                    raise ValueError("include must contain only finite values")
                included.append(value)
            if not included:
                raise ValueError("include must contain at least one value")
            raw_include = tuple(included)
            if low is not None or high is not None or keep_bounds:
                raise ValueError(
                    "include cannot be combined with low, high, or keep_bounds"
                )

        def resolve_bound(value: float | None, name: str) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite number or None")
            resolved = float(value)
            if not np.isfinite(resolved):
                raise ValueError(f"{name} must be finite or None")
            return resolved

        resolved_low = resolve_bound(low, "low")
        resolved_high = resolve_bound(high, "high")
        if (
            resolved_low is not None
            and resolved_high is not None
            and resolved_low > resolved_high
        ):
            raise ValueError("low cannot exceed high")

        status = self.inspect_artifact(values)
        if not status.exists or not status.complete:
            raise ValueError("values must identify a complete artifact")
        raw_source_selection = (status.inputs or {}).get("cell_selection")
        if not isinstance(raw_source_selection, Mapping):
            raise ValueError("values artifact has no cell-selection input")
        try:
            source_selection = ArtifactRef.from_dict(raw_source_selection)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("values artifact cell selection is malformed") from exc
        source = validate_stored_selection_integrity(
            self.zw,
            source_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

        group = as_zarr_group(
            self.zw[artifact_path(values)],
            name=values.artifact_id,
        )
        if "values" not in group:
            raise ValueError("values artifact has no canonical 'values' array")
        source_values = as_zarr_array(group["values"], name="values")
        if (
            source_values.ndim != 1
            or int(source_values.shape[0]) != source.selected_count
        ):
            raise ValueError(
                "values artifact must contain one value per source-selected cell"
            )
        source_missing = linked_missing_mask(
            group,
            "values",
            label="values artifact",
            values=source_values,
        )
        value_kind = np.dtype(source_values.dtype).kind
        if raw_include is None and value_kind not in {"i", "u", "f"}:
            raise TypeError(
                "values artifact must be numeric unless include is provided"
            )
        if raw_include is not None and value_kind not in {
            "b",
            "f",
            "i",
            "O",
            "S",
            "u",
            "U",
        }:
            raise TypeError("values artifact must contain scalar values")

        resolved_include: tuple[str | int | float | bool, ...] | None = None
        if raw_include is not None:
            if value_kind in {"O", "S", "U"}:
                if not all(isinstance(value, str) for value in raw_include):
                    raise TypeError(
                        "include values must be strings for a string artifact"
                    )
                resolved_include = tuple(sorted(set(raw_include)))
            elif value_kind == "b":
                if not all(isinstance(value, bool) for value in raw_include):
                    raise TypeError(
                        "include values must be booleans for a boolean artifact"
                    )
                resolved_include = tuple(sorted(set(raw_include)))
            elif value_kind in {"i", "u"}:
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in raw_include
                ):
                    raise TypeError(
                        "include values must be integers for an integer artifact"
                    )
                limits = np.iinfo(source_values.dtype)
                if any(
                    int(value) < limits.min or int(value) > limits.max
                    for value in raw_include
                ):
                    raise ValueError("include contains an out-of-range integer")
                resolved_include = tuple(sorted({int(value) for value in raw_include}))
            else:
                if not all(
                    isinstance(value, Real) and not isinstance(value, bool)
                    for value in raw_include
                ):
                    raise TypeError(
                        "include values must be numeric for a floating artifact"
                    )
                normalized = {float(value) for value in raw_include}
                if not all(np.isfinite(value) for value in normalized):
                    raise ValueError("include must contain only finite values")
                resolved_include = tuple(sorted(normalized))

        prior_selection = source_selection
        if cell_selection is not None:
            if not isinstance(cell_selection, ArtifactRef):
                raise TypeError("cell_selection must be an ArtifactRef")
            prior_selection = cell_selection
        prior = validate_stored_selection_integrity(
            self.zw,
            prior_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        prior_mask = selection_mask(prior)

        selected = np.zeros(self.cells.N, dtype=bool)
        for block in iter_stored_selection_blocks(
            self.zw,
            source_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ):
            prior_block = prior_mask[block.start : block.stop]
            if np.any(prior_block & ~block.mask):
                raise ValueError(
                    "cell_selection must be a subset of the values artifact's "
                    "cell selection"
                )
            raw_compact = np.asarray(
                source_values[block.compact_start : block.compact_stop]
            )
            if resolved_include is not None:
                if value_kind in {"O", "S", "U"}:
                    raw_compact = np.asarray(
                        [
                            bytes(value).decode("utf-8")
                            if isinstance(value, bytes | bytearray | np.bytes_)
                            else value
                            for value in raw_compact
                        ],
                        dtype=object,
                    )
                keep = np.isin(raw_compact, resolved_include)
            else:
                compact = np.asarray(raw_compact, dtype=np.float64)
                keep = np.isfinite(compact)
                if resolved_low is not None:
                    keep &= (
                        compact >= resolved_low
                        if keep_bounds
                        else compact > resolved_low
                    )
                if resolved_high is not None:
                    keep &= (
                        compact <= resolved_high
                        if keep_bounds
                        else compact < resolved_high
                    )
            if source_missing is not None:
                keep &= ~np.asarray(
                    source_missing[block.compact_start : block.compact_stop],
                    dtype=bool,
                )
            block_selected = np.zeros(block.stop - block.start, dtype=bool)
            block_selected[block.mask] = keep
            selected[block.start : block.stop] = block_selected & prior_block

        inputs: dict[str, Any] = {
            "values": values,
            "source_cell_selection": source_selection,
            "prior_cell_selection": prior_selection,
        }
        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=selected,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="select_cells",
            parameters={
                "low": resolved_low,
                "high": resolved_high,
                "include": resolved_include,
                "keep_bounds": keep_bounds,
            },
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        logger.info(f"Cell selection retained {int(stored.sum())}/{self.cells.N} cells")
        return ref

    def _read_filter_metrics(
        self,
        attrs: list[str],
        artifact_metrics: list[NamedCellArtifact],
        prior: ValidatedStoredSelection,
        active_idx: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
        """Read QC metrics and missing-value masks for the active cells.

        Returns float values and masks keyed by metric name, and the
        fingerprints of the metadata-column masks that exist. An artifact
        metric's reference already identifies its mask.
        """
        values_by_name: dict[str, np.ndarray] = {}
        missing_by_name: dict[str, np.ndarray] = {}
        for attr in attrs:
            values_by_name[attr] = np.asarray(
                read_metadata_rows_chunkwise(self.cells, attr, active_idx),
                dtype=float,
            )
            missing = read_metadata_missing_rows_chunkwise(
                self.cells,
                attr,
                active_idx,
            )
            if missing is not None:
                missing_by_name[attr] = missing
        for source in artifact_metrics:
            resolved = resolve_cell_aligned_artifact(
                self.zw,
                source.artifact,
                cell_selection=prior.ref,
                expected_kind="quality_metric",
            )
            values_by_name[source.name] = np.asarray(resolved.values, dtype=float)
            if resolved.missing_mask is not None:
                missing_by_name[source.name] = resolved.missing_mask
        fingerprints = {
            name: fingerprint_array(missing)
            for name, missing in missing_by_name.items()
            if name in attrs
        }
        return values_by_name, missing_by_name, fingerprints

    def _filter_input_selection(
        self,
        selection: ArtifactRef | None,
    ) -> ValidatedStoredSelection:
        """Resolve the input cell selection, validated exactly once."""
        if selection is None:
            return self._snapshot_cell_selection("I")
        if not isinstance(selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        return validate_stored_selection_integrity(
            self.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

    def _filter_input_cells(
        self,
        selection: ArtifactRef | None,
    ) -> tuple[ValidatedStoredSelection, np.ndarray]:
        """Resolve the input cell selection and its active row indices."""
        prior = self._filter_input_selection(selection)
        active_idx = np.flatnonzero(selection_mask(prior)).astype(np.int64, copy=False)
        if len(active_idx) == 0:
            raise ValueError("Cell selection contains no active cells")
        return prior, active_idx

    def auto_filter_cells(
        self,
        attrs: Iterable[str] | None = None,
        min_p: float = 0.01,
        max_p: float = 0.99,
        *,
        cell_selection: ArtifactRef | None = None,
        artifact_metrics: Iterable[NamedCellArtifact] | None = None,
        invalidate_cache: bool = False,
        method: Literal["mad", "gaussian"] = "mad",
        sample_column: str | None = None,
        sample_artifact: NamedCellArtifact | None = None,
        n_mads: float = 3.0,
        min_cells_per_sample: int = 20,
    ) -> ArtifactRef:
        """Create an immutable automatically filtered cell selection.

        Defaults to median absolute deviation (MAD) bounds. Counts and feature
        counts use log1p and two-sided bounds; mitochondrial and ribosomal
        percentages use upper bounds. Other metrics use two-sided raw bounds.

        Requested columns are read from current cell metadata when this method
        is called. Exact quality-metric artifacts can be supplied alongside
        metadata metrics. Metadata values are fingerprinted and artifact
        references are stored in provenance.

        MAD thresholds are pooled unless a sample source is supplied. Use
        ``method="gaussian"`` for pooled median/std Gaussian quantiles instead.

        These are the pipeline filtering rules. A cell whose metric is recorded
        as missing never passes and does not inform the bounds; the
        fingerprints of metadata-column missing-value masks are recorded in
        provenance.
        Non-finite metrics among the remaining active cells, missing sample
        labels among active cells, and an empty result raise ``ValueError``.

        Args:
            attrs: Column names to be used for filtering.
            min_p: Quantile used for the lower threshold (Gaussian path only).
            max_p: Quantile used for the upper threshold (Gaussian path only).
            cell_selection: Optional prior cell-selection artifact.
            artifact_metrics: Named exact ``quality_metric`` artifact vectors.
            method: MAD bounds by default, or explicit pooled Gaussian bounds.
            sample_column: Optional cell-metadata column with sample labels.
                When set, MAD bounds are calculated within each sample.
            sample_artifact: Optional named exact ``hto_identity`` sample vector.
            n_mads: Number of scaled MADs used for bounds.
            min_cells_per_sample: Groups with fewer active cells than this are
                retained without MAD filtering and emit a warning.

        Returns:
            A complete datastore-scoped ``cell_selection`` artifact.
        """
        if method not in ("mad", "gaussian"):
            raise ValueError("method must be 'mad' or 'gaussian'")
        if method == "gaussian":
            if sample_column is not None or sample_artifact is not None:
                raise ValueError("Gaussian filtering does not support a sample source")
            if n_mads != 3.0 or min_cells_per_sample != 20:
                raise ValueError(
                    "n_mads and min_cells_per_sample apply only to method='mad'"
                )
        available = set(self.cells.columns)
        if attrs is None:
            attrs = []
            for i in ["nCounts", "nFeatures", "percentMito", "percentRibo"]:
                i = f"{self._defaultAssay}_{i}"
                if i in available:
                    attrs.append(i)

        attrs_list = list(attrs)
        for attr in attrs_list:
            if not isinstance(attr, str):
                raise TypeError("attrs must contain only column names")
        if len(set(attrs_list)) != len(attrs_list):
            raise ValueError("attrs must not contain duplicate columns")
        metric_artifacts = _validated_named_cell_artifacts(
            artifact_metrics,
            expected_kind="quality_metric",
            label="artifact_metrics",
        )
        resolved_sample_artifact: NamedCellArtifact | None = None
        if sample_artifact is not None:
            resolved_sample_artifact = _validated_named_cell_artifacts(
                [sample_artifact],
                expected_kind="hto_identity",
                label="sample_artifact",
            )[0]
        if sample_column is not None and resolved_sample_artifact is not None:
            raise ValueError("sample_column and sample_artifact are mutually exclusive")
        if resolved_sample_artifact is not None and resolved_sample_artifact.name in {
            source.name for source in metric_artifacts
        }:
            raise ValueError("Sample and metric artifact names must be distinct")
        duplicate_names = sorted(
            set(attrs_list).intersection(source.name for source in metric_artifacts)
        )
        if duplicate_names:
            raise ValueError(
                "Metadata and artifact QC metrics must use distinct names: "
                f"{duplicate_names}"
            )
        missing = [attr for attr in attrs_list if attr not in available]
        if missing:
            joined = ", ".join(repr(attr) for attr in missing)
            raise KeyError(f"Cell metadata columns not found: {joined}")
        if method == "mad":
            return self._auto_filter_cells_mad(
                attrs=attrs_list,
                artifact_metrics=metric_artifacts,
                min_p=min_p,
                max_p=max_p,
                cell_selection=cell_selection,
                invalidate_cache=invalidate_cache,
                sample_column=sample_column,
                sample_artifact=resolved_sample_artifact,
                n_mads=n_mads,
                min_cells_per_sample=min_cells_per_sample,
            )

        prior, active_idx = self._filter_input_cells(cell_selection)
        values_by_name, missing_by_name, missing_fingerprints = (
            self._read_filter_metrics(attrs_list, metric_artifacts, prior, active_idx)
        )
        metadata_fingerprints = {
            attr: fingerprint_array(values_by_name[attr]) for attr in attrs_list
        }
        result = filter_cell_metrics(
            values_by_name,
            missing_by_name,
            np.ones(len(active_idx), dtype=bool),
            method="gaussian",
            min_p=min_p,
            max_p=max_p,
        )
        keep = np.zeros(self.cells.N, dtype=bool)
        keep[active_idx] = result.retained

        metric_sources = [
            {"name": attr, "source": "metadataColumn", "column": attr}
            for attr in attrs_list
        ]
        metric_sources.extend(
            {"name": source.name, "source": "artifact"} for source in metric_artifacts
        )
        inputs: dict[str, Any] = {
            "prior_cell_selection": prior.ref,
            "metadata_fingerprints": metadata_fingerprints,
            "artifact_metrics": {
                source.name: source.artifact for source in metric_artifacts
            },
        }
        if missing_fingerprints:
            inputs["missing_mask_fingerprints"] = missing_fingerprints

        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=keep,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="auto_filter_cells",
            parameters={
                "method": "gaussian",
                "attrs": list(values_by_name),
                "metric_sources": metric_sources,
                "min_p": min_p,
                "max_p": max_p,
                "resolved_bounds": result.gaussian_bounds,
            },
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        logger.info(f"Cell filtering retained {int(stored.sum())}/{self.cells.N} cells")
        return ref

    def _auto_filter_cells_mad(
        self,
        *,
        attrs: list[str],
        artifact_metrics: list[NamedCellArtifact],
        min_p: float,
        max_p: float,
        cell_selection: ArtifactRef | None,
        invalidate_cache: bool,
        sample_column: str | None,
        sample_artifact: NamedCellArtifact | None,
        n_mads: float,
        min_cells_per_sample: int,
    ) -> ArtifactRef:
        if min_p != 0.01 or max_p != 0.99:
            raise ValueError(
                "min_p and max_p apply only to method='gaussian'. "
                "Leave them at their defaults (0.01 and 0.99) and use n_mads "
                "to control MAD bounds"
            )
        if isinstance(n_mads, bool) or not isinstance(n_mads, Real):
            raise TypeError("n_mads must be a positive number")
        resolved_n_mads = float(n_mads)
        if not np.isfinite(resolved_n_mads) or resolved_n_mads <= 0:
            raise ValueError("n_mads must be finite and greater than 0")
        if (
            not isinstance(min_cells_per_sample, int)
            or isinstance(min_cells_per_sample, bool)
            or min_cells_per_sample < 2
        ):
            raise ValueError("min_cells_per_sample must be an integer >= 2")
        if sample_column is not None and sample_column not in self.cells.columns:
            raise ValueError(
                f"sample_column '{sample_column}' not found in cell metadata"
            )
        prior, active_idx = self._filter_input_cells(cell_selection)
        sample_options: dict[str, Any] = {}
        if sample_column is not None:
            sample_options = {
                "sample_labels": np.asarray(
                    read_metadata_rows_chunkwise(self.cells, sample_column, active_idx)
                ),
                "sample_missing": read_metadata_missing_rows_chunkwise(
                    self.cells,
                    sample_column,
                    active_idx,
                ),
                "sample_label_name": f"sample_column '{sample_column}'",
            }
        elif sample_artifact is not None:
            resolved_sample = resolve_cell_aligned_artifact(
                self.zw,
                sample_artifact.artifact,
                cell_selection=prior.ref,
                expected_kind="hto_identity",
            )
            sample_options = {
                "sample_labels": np.asarray(resolved_sample.values),
                "sample_missing": resolved_sample.missing_mask,
                "sample_label_name": f"sample_artifact '{sample_artifact.name}'",
            }
        values_by_attr, missing_by_attr, missing_fingerprints = (
            self._read_filter_metrics(attrs, artifact_metrics, prior, active_idx)
        )
        result = filter_cell_metrics(
            values_by_attr,
            missing_by_attr,
            np.ones(len(active_idx), dtype=bool),
            method="mad",
            n_mads=resolved_n_mads,
            min_cells_per_sample=int(min_cells_per_sample),
            **sample_options,
        )
        parameters: dict[str, Any] = {
            "method": "mad",
            "attrs": list(values_by_attr),
            "metric_sources": [
                *(
                    {"name": attr, "source": "metadataColumn", "column": attr}
                    for attr in attrs
                ),
                *(
                    {"name": source.name, "source": "artifact"}
                    for source in artifact_metrics
                ),
            ],
            "n_mads": resolved_n_mads,
            "min_cells_per_sample": int(min_cells_per_sample),
            "resolved_bounds": {},
        }
        if sample_column is not None:
            parameters["sample_column"] = sample_column
            parameters["sample_source"] = {
                "name": sample_column,
                "source": "metadataColumn",
                "column": sample_column,
            }
        elif sample_artifact is not None:
            parameters["sample_source"] = {
                "name": sample_artifact.name,
                "source": "artifact",
            }
        else:
            parameters["sample_source"] = {"source": "pooled"}

        if result.mad_provenance is not None:
            # Warnings are logged below rather than recorded.
            parameters.update(
                (key, value)
                for key, value in result.mad_provenance.items()
                if key != "warnings"
            )
        fingerprint_inputs: dict[str, Any] = {
            "qc_metric_fingerprints": {
                attr: fingerprint_array(values_by_attr[attr]) for attr in attrs
            },
            "artifact_metrics": {
                source.name: source.artifact for source in artifact_metrics
            },
        }
        if missing_fingerprints:
            fingerprint_inputs["missing_mask_fingerprints"] = missing_fingerprints
        if sample_column is not None:
            assert result.sample_labels is not None
            fingerprint_inputs["sample_assignments_fingerprint"] = fingerprint_strings(
                result.sample_labels
            )
        elif sample_artifact is not None:
            fingerprint_inputs["sample_artifact"] = sample_artifact.artifact
        canonical_bytes(
            {
                "operation": "auto_filter_cells",
                "parameters": parameters,
                "inputs": fingerprint_inputs,
            }
        )
        inputs: dict[str, Any] = {
            "prior_cell_selection": prior.ref,
            **fingerprint_inputs,
        }
        canonical_bytes(
            {
                "operation": "auto_filter_cells",
                "parameters": parameters,
                "inputs": inputs,
            }
        )

        if result.mad_provenance is not None:
            for message in result.mad_provenance["warnings"]:
                logger.warning(message)
        keep = np.zeros(self.cells.N, dtype=bool)
        keep[active_idx] = result.retained

        ref, stored = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=keep,
            row_ids=prior.row_ids,
            row_ids_fingerprint=prior.row_ids_fingerprint,
            operation="auto_filter_cells",
            parameters=parameters,
            inputs=inputs,
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        logger.info(f"Cell filtering retained {int(stored.sum())}/{self.cells.N} cells")
        return ref

    def run_feature_percentage(
        self,
        cell_selection: ArtifactRef,
        features: ArtifactRef,
        *,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Persist the percentage of counts assigned to selected features.

        Args:
            cell_selection: Exact datastore-scoped cell-selection artifact.
            features: Exact assay-scoped feature-selection artifact.
            invalidate_cache: Recompute instead of reusing an exact artifact.

        Returns:
            A complete assay-scoped ``quality_metric`` artifact.
        """
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if features.scope != "assay" or features.assay is None:
            raise ValueError("features must be an assay-scoped ArtifactRef")
        assay = self._get_assay(features.assay)
        if not isinstance(assay, Assay):
            raise TypeError("from_assay must resolve to an Assay")
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if len(cell_index) == 0:
            raise ValueError("cell_selection must select at least one cell")
        feature_selection = resolve_feature_selection(
            self.zw,
            assay.name,
            features,
        )
        feature_values = np.asarray(
            artifact_values(
                artifact_group(self.zw, feature_selection),
                "values",
            ),
            dtype=bool,
        )
        feature_index = np.flatnonzero(feature_values).astype(
            np.int64,
            copy=False,
        )
        if len(feature_index) == 0:
            raise ValueError("features must select at least one feature")

        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind="quality_metric",
            operation="run_feature_percentage",
            parameters={"scale": 100.0},
            inputs={"feature_selection": feature_selection},
            execution_options={"nthreads": assay.nthreads},
            cell_selection=cell_selection,
            arrays={"values": ((len(cell_index),), "f")},
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_feature_percentage")
        values = assay._compute_feature_percentage(cell_index, feature_index)
        write_cell_data_artifact(self.zw, planned, {"values": values})
        return planned.ref

    def run_hto_demultiplexing(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        random_seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Demultiplex HTO counts into an immutable identity artifact.

        Args:
            from_assay: HTO assay name (default: ``'HTO'``).
            cell_selection: Explicit cell-selection artifact.
            random_seed: Seed used for HTO demultiplexing.
            invalidate_cache: Recompute even when matching provenance exists.

        Returns:
            A complete ``hto_identity`` artifact.
        """
        if from_assay is None:
            from_assay = "HTO"
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay_types = self.zw.attrs.get("assayTypes", {})
        declared_type = (
            assay_types.get(from_assay) if isinstance(assay_types, Mapping) else None
        )
        if declared_type != "HTO":
            raise TypeError(
                "HTO demultiplexing requires an assay declared with type 'HTO'; "
                f"{from_assay!r} is declared as {declared_type!r}"
            )
        assay = self._get_assay(from_assay)
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = len(cell_index)
        required_cells = assay.feats.N + 1
        if n_cells < required_cells:
            raise ValueError(
                f"HTO demultiplexing requires at least {required_cells} selected cells"
            )
        arguments = HtoIdentityArguments(
            cell_selection=cell_selection,
            feature_ids_fingerprint=fingerprint_strings(
                np.asarray(assay.feats.fetch_all("ids"))
            ),
            method=_hto_demux_method(),
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={"values": ((n_cells,), None)},
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_hto_demultiplexing")
        matrix_bytes = n_cells * assay.feats.N * np.dtype(np.float64).itemsize
        estimated_peak_bytes = 6 * matrix_bytes + 16 * n_cells
        if estimated_peak_bytes > self.memoryBytes:
            raise MemoryError(
                "HTO demultiplexing needs an estimated "
                f"{estimated_peak_bytes / (1024**2):.1f} MiB for its exact "
                "whole-matrix clustering algorithm, which exceeds the datastore "
                f"memory budget of {self.memoryBytes / (1024**2):.1f} MiB. "
                "Use a smaller explicit cell selection or a larger memory budget."
            )
        counts = controlled_compute(assay.rawData[cell_index, :], self.nthreads)
        hto_idents = hto_demux(
            pd.DataFrame(counts, columns=assay.feats.fetch_all("ids")),
            random_seed=random_seed,
        )
        values = np.asarray(hto_idents.values)
        write_cell_data_artifact(
            self.zw,
            planned,
            {"values": values},
        )
        return planned.ref

    def _run_doublet_detection_artifact(
        self,
        *,
        source_assay: RNAassay,
        cell_selection: ArtifactRef,
        clusters: ArtifactRef,
        cluster_values: np.ndarray,
        connectivity: ArtifactRef,
        feature_names: np.ndarray | None = None,
        feature_snapshot: ArtifactRef | None = None,
        cluster_sample_fraction: float = 0.05,
        max_cells_per_cluster: int = 100,
        simulation_ratio: float = 1.0,
        heterotypic_fraction: float = 0.8,
        save_k: int = 5,
        smoothing_t: int = 2,
        normalize_scores: bool = True,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Create doublet scores without creating metadata columns."""
        from ...quality_control.doublets import (
            score_synthetic_doublets,
            smooth_doublet_scores,
        )
        from ...storage.execution import admit_stream

        require_graph_kind(connectivity)
        assay_name = source_assay.name
        if feature_names is not None and np.asarray(feature_names).shape != (
            source_assay.feats.N,
        ):
            raise ValueError(
                "Snapshot feature names must align with the assay feature axis"
            )
        connectivity_status = self._require_complete_artifact(
            connectivity,
            connectivity.kind,
            assay=(assay_name if connectivity.scope == "assay" else None),
        )
        if connectivity.kind == "connectivity_map" and (
            connectivity_status.operation != "build_connectivity_map"
        ):
            raise ValueError(
                "Doublet detection requires a build_connectivity_map artifact"
            )
        lineage = resolve_graph_assay_inputs(
            self.zw,
            connectivity,
            assay_name,
        )
        neighbors = lineage.neighbors
        graph_selection = graph_cell_selection(self.zw, connectivity)
        if not self._selection_artifacts_match(
            graph_selection,
            cell_selection,
        ):
            raise ValueError("Cell selection does not match the graph")
        coordinates = lineage.coordinates
        if coordinates.kind != "reduction":
            raise ValueError("Doublet detection requires an uncorrected PCA graph")
        active_idx = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        labels = np.asarray(cluster_values)
        if labels.ndim != 1 or len(labels) != len(active_idx):
            raise ValueError("Cluster values must contain one label per selected cell")
        n_active = len(active_idx)
        if n_active < 1:
            raise ValueError("Doublet detection requires selected cells")
        arguments = DoubletScoreArguments(
            clusters=clusters,
            connectivity_map=connectivity,
            neighbors=neighbors,
            cluster_sample_fraction=cluster_sample_fraction,
            max_cells_per_cluster=max_cells_per_cluster,
            simulation_ratio=simulation_ratio,
            heterotypic_fraction=heterotypic_fraction,
            save_k=save_k,
            smoothing_t=smoothing_t,
            normalize_scores=normalize_scores,
            count_arithmetic="checked_integer_sum",
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        inputs = dict(record.inputs)
        if feature_snapshot is not None:
            inputs["feature_snapshot"] = feature_snapshot
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay_name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={"values": ((n_active,), "f")},
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            return planned.ref
        self._require_writable("run_doublet_detection")

        feature_selection = lineage.feature_selection
        if feature_selection is None:
            raise ValueError(
                "Doublet detection requires normalized feature-selection ancestry"
            )
        reference_ref = self.build_mapping_reference(neighbors)
        reference = self.get_mapping_reference(reference_ref)
        if (
            reference.neighbors != neighbors
            or reference.assay_name != assay_name
            or reference.feature_selection != feature_selection
            or reference.method != "pca"
            or reference.batch_correction is not None
            or reference.symphony_state is not None
        ):
            raise RuntimeError(
                "The mapping reference does not match the uncorrected RNA graph"
            )

        feature_indices = read_feature_selection_indices(
            self.zw,
            assay_name,
            feature_selection,
        )
        feature_ids = read_metadata_rows_chunkwise(
            source_assay.feats,
            "ids",
            feature_indices,
        )
        if not np.array_equal(
            np.asarray(feature_ids).astype(str), reference.feature_ids
        ):
            raise ValueError(
                "Doublet features do not match the mapping reference order"
            )

        cached_bytes = 0
        cache = getattr(self, "_graphMemoryCache", None)
        if cache is not None:
            with self._graphMemoryCacheLock:
                matrices = {id(value): value for value in cache.values()}
                cached_bytes = sum(
                    matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                    for matrix in matrices.values()
                )
        raw_scores = score_synthetic_doublets(
            source_assay,
            reference,
            active_idx,
            labels,
            feature_indices,
            cluster_sample_fraction=cluster_sample_fraction,
            max_cells_per_cluster=max_cells_per_cluster,
            simulation_ratio=simulation_ratio,
            heterotypic_fraction=heterotypic_fraction,
            save_k=save_k,
            random_seed=random_seed,
            resources=self.resources,
            reserved_resident_bytes=cached_bytes,
        )
        del reference
        if raw_scores.shape != (n_active,):
            raise RuntimeError("Doublet mapping scores do not match the selected cells")
        graph_group = artifact_group(self.zw, connectivity)
        edges = as_zarr_array(graph_group["edges"], name="edges")
        weights = as_zarr_array(graph_group["weights"], name="weights")
        # Loading, symmetrizing and row-normalizing may hold several sparse copies.
        graph_bytes = (
            int(edges.nbytes)
            + int(weights.nbytes)
            + 8 * int(weights.shape[0]) * (weights.dtype.itemsize + 8)
            + 128 * (n_active + 1)
        )
        admit_stream(
            self.resources,
            nBlocks=1,
            blockBytes=graph_bytes,
            residentBytes=cached_bytes
            + active_idx.nbytes
            + labels.nbytes
            + raw_scores.nbytes
            + feature_ids.nbytes
            + feature_indices.nbytes,
            requested=1,
        )
        graph = self._load_graph_artifact(
            connectivity, symmetric=True, upper_only=False, use_k=None
        )
        if graph.shape != (n_active, n_active):
            raise ValueError("Doublet graph does not match the selected cells")
        scores = smooth_doublet_scores(
            graph,
            raw_scores,
            power=smoothing_t,
            normalize=normalize_scores,
        )
        write_cell_data_artifact(self.zw, planned, {"values": scores})
        logger.info("Stored doublet scores")
        return planned.ref

    def run_doublet_detection(
        self,
        clusters: ArtifactRef,
        graph: ArtifactRef,
        *,
        from_assay: str | None = None,
        cluster_sample_fraction: float = 0.05,
        max_cells_per_cluster: int = 100,
        simulation_ratio: float = 1.0,
        heterotypic_fraction: float = 0.8,
        save_k: int = 5,
        smoothing_t: int = 2,
        normalize_scores: bool = True,
        random_seed: int = 4444,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Compute doublet scores from explicit cluster and graph artifacts."""
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        if not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef")
        assay_name = from_assay or graph.assay
        if assay_name is None:
            raise ValueError("from_assay is required for an integrated graph")
        resolve_graph_assay_inputs(self.zw, graph, assay_name)
        assay = self._get_assay(assay_name)
        graph_selection = graph_cell_selection(self.zw, graph)
        validate_stored_selection_integrity(
            self.zw,
            graph_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Doublet detection is only supported for RNA assays; "
                f"received {type(assay).__name__}"
            )
        cluster_status = self.inspect_artifact(clusters)
        if (
            clusters.kind not in {"cluster_labels", "cluster_cut"}
            or not cluster_status.exists
            or not cluster_status.complete
        ):
            raise ValueError("clusters must be a complete clustering artifact")
        raw_cluster_selection = (cluster_status.inputs or {}).get("cell_selection")
        if not isinstance(raw_cluster_selection, dict):
            raise ValueError("Clustering artifact has no cell-selection input")
        try:
            cluster_selection = ArtifactRef.from_dict(raw_cluster_selection)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Clustering artifact cell selection is malformed") from exc
        validate_stored_selection_integrity(
            self.zw,
            cluster_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if not self._selection_artifacts_match(graph_selection, cluster_selection):
            raise ValueError("Cluster and graph cell selections do not match")
        cluster_group = as_zarr_group(
            self.zw[artifact_path(clusters)],
            name=clusters.artifact_id,
        )
        value_name = "values" if clusters.kind == "cluster_labels" else "labels"
        require_complete_cluster_labels(cluster_group, value_name, name="clusters")
        cluster_values = artifact_values(cluster_group, value_name)
        ref = self._run_doublet_detection_artifact(
            source_assay=assay,
            cell_selection=graph_selection,
            clusters=clusters,
            cluster_values=cluster_values,
            connectivity=graph,
            cluster_sample_fraction=cluster_sample_fraction,
            max_cells_per_cluster=max_cells_per_cluster,
            simulation_ratio=simulation_ratio,
            heterotypic_fraction=heterotypic_fraction,
            save_k=save_k,
            smoothing_t=smoothing_t,
            normalize_scores=normalize_scores,
            random_seed=random_seed,
            invalidate_cache=invalidate_cache,
        )
        return ref

    def select_prevalent_peaks(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        top_n: int = 10000,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Feature selection method for ATACassay type assays.

        This method first calculates prevalence of each peak by computing sum of TF-IDF normalized values for each peak
        and then marks `top_n` peaks with the highest prevalence as prevalent peaks.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_selection: Explicit cells used to calculate peak prevalence.
            top_n: Number of top prevalent peaks to be selected. (Default: 10000)
        Returns:
            The persisted prevalent-peak feature-selection artifact.
        """
        self._require_writable("select_prevalent_peaks")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        if type(assay) != ATACassay:  # noqa: E721
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to ATACassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            invalidate_cache=invalidate_cache,
        )
        arguments = PrevalentPeakArguments(
            feature_summary=summary_ref,
            top_n=top_n,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay.z)
        planned = _feature_selection_plan(
            self.zw,
            assay=assay.name,
            n_features=assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
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
            values = assay._prevalent_peak_mask(summary["prevalence"], top_n)
            _write_feature_selection(
                self.zw,
                planned,
                ordered_feature_ids_fingerprint=feature_ids_fingerprint,
                payload={"values": values},
            )
        return planned.ref

    def run_cell_cycle_scoring(
        self,
        cell_selection: ArtifactRef,
        *,
        from_assay: str | None = None,
        s_genes: list[str] | None = None,
        g2m_genes: list[str] | None = None,
        ctrl_size: int | None = None,
        log_transform: bool = True,
        n_bins: int = 50,
        rand_seed: int = 4466,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Computes S and G2M phase scores by taking into account the average
        expression of S and G2M phase genes respectively. Following steps are
        taken for each phase:

        - Normalized expression is log1p transformed when `log_transform` is True.
        - Genes are ranked into `n_bins` bins by mean expression across selected cells.
        - A control set of genes is identified by sampling genes from same expression bins where phase's genes are present.
        - The average expression of phase genes (Ep) and control genes (Ec) is calculated per cell.
        - A phase score is calculated as ``Ep - Ec``.
        - G1 is assigned when both scores are negative.
        - G2M is assigned when the G2M score exceeds the S score.
        - S is assigned otherwise, including tied non-negative scores.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_selection: Explicit cells to score.
            s_genes: A list of S phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.quality_control.s_phase_genes`
            g2m_genes: A list of G2M phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.quality_control.g2m_phase_genes`
            ctrl_size: Controls sampled per bin. None uses the shorter input gene list.
            log_transform: Apply log1p before binning and scoring. Defaults to True.
            n_bins: Number of bins into which average expression of genes is divided.
            rand_seed: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)
        Returns:
            A complete ``cell_cycle`` artifact containing S, G2M, and phase values.
        """
        self._require_writable("run_cell_cycle_scoring")
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Cell-cycle scoring can only be applied to an RNAassay; "
                f"received {type(assay).__name__}"
            )
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        return self._run_cell_cycle_scoring_artifact(
            assay=assay,
            cell_selection=cell_selection,
            s_genes=s_genes,
            g2m_genes=g2m_genes,
            ctrl_size=ctrl_size,
            log_transform=log_transform,
            n_bins=n_bins,
            rand_seed=rand_seed,
            invalidate_cache=invalidate_cache,
        )
