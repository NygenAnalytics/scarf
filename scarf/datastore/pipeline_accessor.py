import math
import asyncio
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np

from ..assay import RNAassay
from ..metadata.artifacts import categorical_display, continuous_display
from ..quality_control.filtering import (
    _apply_bounds,
    _sample_aware_mad_mask,
    _validated_sample_labels,
    gaussian_quantile_bounds,
)
from ..storage.artifacts import ArtifactRef, artifact_group
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    ArtifactPlanReceipt,
    artifact_plan_scope,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.arrays import create_zarr_dataset
from ..storage.pipeline_runs import (
    PipelineFieldDescriptor,
    PipelineInterruptionRecord,
    PipelineOutputRecord,
    PipelinePlanRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    fail_pipeline_run_record,
    finish_pipeline_stage_record,
    interrupt_pipeline_run_record,
    load_pipeline_run_record,
    load_pipeline_stage_record,
    start_pipeline_stage_record,
)
from ..storage.selections import (
    read_stored_selection_mask,
    resolve_selection_artifact,
    resolve_stored_selection_artifact,
    snapshot_run_metadata,
)
from ..storage.types import as_zarr_array
from ..utils.logging import logger
from ..utils.process import ProcessTreeRssMeasurement, sample_process_tree_rss
from ..utils.shutdown import (
    ShutdownRequested,
    ShutdownToken,
    TemporarySignalGuard,
    shutdown_checkpoint,
    shutdown_scope,
)
from .pipeline_run import (
    PipelineExecutionError,
    PipelineRun,
    list_pipeline_runs,
    open_pipeline_run,
)


type PipelineEventKind = Literal[
    "stage_started",
    "stage_completed",
    "stage_failed",
    "stage_interrupted",
    "pipeline_interrupted",
]


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    kind: PipelineEventKind
    stage: str
    error: BaseException | None = None


type PipelineCallback = Callable[[PipelineEvent], None]


@dataclass(frozen=True, slots=True)
class _ResolvedRecipe:
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


class _PipelineEventEmitter:
    __slots__ = ("_callback",)

    def __init__(self, callback: PipelineCallback | None) -> None:
        self._callback = callback

    def emit(
        self,
        kind: PipelineEventKind,
        stage: str,
        error: BaseException | None = None,
    ) -> None:
        if self._callback is None:
            return
        try:
            self._callback(PipelineEvent(kind=kind, stage=stage, error=error))
        except BaseException:
            logger.exception(
                f"Pipeline callback failed while handling {kind} for {stage}"
            )


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
    value: Mapping[str, object] | bool | None,
) -> tuple[tuple[str, float], ...]:
    if value is False:
        return ()
    if value is None or value is True:
        return (
            ("0.5", 0.5),
            ("0.75", 0.75),
            ("1.0", 1.0),
            ("1.25", 1.25),
        )
    if not isinstance(value, Mapping):
        raise TypeError("leiden must be a mapping, bool, or None")
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


def _resolve_filtering(
    store: Any,
    assay: str,
    value: bool | Mapping[str, object] | None,
) -> dict[str, Any]:
    if value is False:
        return {"enabled": False}
    if value is None or value is True:
        options: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        options = dict(value)
    else:
        raise TypeError("filtering must be a mapping, bool, or None")
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
        if options:
            raise ValueError("Filtering options require at least one attribute")
        return {"enabled": False}
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
        return {
            "enabled": True,
            "method": "manual",
            "attrs": list(attrs),
            "lows": [_manual_bound(bound, "lows") for bound in lows],
            "highs": [_manual_bound(bound, "highs") for bound in highs],
            "keepBounds": bool(options.get("keep_bounds", False)),
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
    min_p = float(options.get("min_p", 0.01))
    max_p = float(options.get("max_p", 0.99))
    if not 0 < min_p < max_p < 1:
        raise ValueError("Automatic filtering requires 0 < min_p < max_p < 1")
    sample_column = options.get("sample_column")
    if sample_column is not None and (
        not isinstance(sample_column, str) or not sample_column
    ):
        raise TypeError("sample_column must be a non-empty string or None")
    if sample_column is not None and sample_column not in store.cells.columns:
        raise KeyError(f"Sample column {sample_column!r} was not found")
    n_mads = float(options.get("n_mads", 3.0))
    if not math.isfinite(n_mads) or n_mads <= 0:
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


def _resolve_recipe(
    store: Any,
    *,
    assay: str | None,
    label: str | None,
    cell_key: str,
    filtering: bool | Mapping[str, object] | None,
    harmony_batch_columns: Sequence[str] | None,
    hvg_count: int,
    pca_dims: int,
    neighbors_k: int,
    umap: bool,
    leiden: Mapping[str, object] | bool | None,
    cell_cycle: bool,
    paris: bool,
    doublets: bool,
    markers: bool,
    snapshot_columns: Sequence[str],
) -> _ResolvedRecipe:
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
    if not partitions and not paris and (doublets or markers):
        raise ValueError(
            "doublets and markers require at least one clustering candidate"
        )
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
    return _ResolvedRecipe(
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


def _artifact_array(root: Any, ref: ArtifactRef, name: str) -> Any:
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


def _continuous_array_display(
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


def _categorical_array_display(array: Any) -> dict[str, Any]:
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


def _stage_metrics(
    *,
    wall_seconds: float,
    rss: ProcessTreeRssMeasurement,
) -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=wall_seconds,
        rss_baseline_bytes=rss.baseline_bytes,
        rss_peak_bytes=rss.peak_bytes,
        rss_incremental_peak_bytes=rss.incremental_peak_bytes,
        sample_interval_seconds=rss.sample_interval_seconds,
        sample_count=rss.sample_count,
        sampling_error_count=rss.sampling_error_count,
        rss_unavailable_reason=rss.unavailable_reason,
    )


def _interruption_record(error: BaseException) -> PipelineInterruptionRecord | None:
    if isinstance(error, ShutdownRequested):
        request = error.request
        return PipelineInterruptionRecord(
            kind="signal" if request.signal_number is not None else "shutdown_request",
            message=request.reason,
            requested_at_ns=request.requested_at_ns,
            signal_number=request.signal_number,
            signal_name=request.signal_name,
        )
    if isinstance(error, KeyboardInterrupt):
        return PipelineInterruptionRecord(
            kind="keyboard_interrupt",
            message=str(error) or "keyboard interrupt",
            requested_at_ns=time.time_ns(),
        )
    if isinstance(error, asyncio.CancelledError):
        return PipelineInterruptionRecord(
            kind="asyncio_cancelled",
            message=str(error) or "async operation cancelled",
            requested_at_ns=time.time_ns(),
        )
    return None


class _RunLedger:
    __slots__ = ("events", "ordinal", "root", "run_id")

    def __init__(
        self,
        root: Any,
        run_id: str,
        callback: PipelineCallback | None,
    ) -> None:
        self.root = root
        self.run_id = run_id
        self.ordinal = 0
        self.events = _PipelineEventEmitter(callback)

    def _records(
        self,
        outputs: Sequence[tuple[str, ArtifactRef]],
        plans: Sequence[ArtifactPlanReceipt],
    ) -> tuple[PipelineStageOutputRecord, ...]:
        dispositions: dict[ArtifactRef, set[str]] = {}
        for plan in plans:
            dispositions.setdefault(plan.ref, set()).add(plan.disposition)
        missing = [key for key, ref in outputs if ref not in dispositions]
        if missing:
            raise RuntimeError(
                "Pipeline stage outputs were not observed by artifact planning: "
                f"{missing!r}"
            )
        return tuple(
            PipelineStageOutputRecord(
                output_key=key,
                artifact=ref,
                reused=dispositions[ref] == {"reused"},
            )
            for key, ref in outputs
        )

    @staticmethod
    def _plans(
        plans: Sequence[ArtifactPlanReceipt],
    ) -> tuple[PipelinePlanRecord, ...]:
        return tuple(
            PipelinePlanRecord(
                operation=plan.operation,
                ref=plan.ref,
                disposition=plan.disposition,
            )
            for plan in plans
        )

    def interrupt_pending(self, error: BaseException, stage: str) -> None:
        interruption = _interruption_record(error)
        if interruption is None:
            raise TypeError("error is not a handled pipeline interruption")
        current = load_pipeline_run_record(self.root, self.run_id)
        if not current.complete:
            interrupt_pipeline_run_record(
                self.root,
                run_id=self.run_id,
                interruption=interruption,
            )
        self.events.emit("pipeline_interrupted", stage, error)

    @staticmethod
    def _fallback_metrics(wall_started: float) -> PipelineStageMetrics:
        with sample_process_tree_rss() as read_rss:
            pass
        return _stage_metrics(
            wall_seconds=time.perf_counter() - wall_started,
            rss=read_rss(),
        )

    def _finish_interrupted(
        self,
        *,
        stage: str,
        error: BaseException,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
    ) -> None:
        interruption = _interruption_record(error)
        if interruption is None:
            raise TypeError("error is not a handled pipeline interruption")
        current_stage = load_pipeline_stage_record(
            self.root,
            self.run_id,
            self.ordinal,
        )
        if current_stage.status == "running":
            finish_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=self.ordinal,
                status="interrupted",
                plans=plans,
                metrics=metrics,
                interruption=interruption,
            )
        current_run = load_pipeline_run_record(self.root, self.run_id)
        if not current_run.complete:
            interrupt_pipeline_run_record(
                self.root,
                run_id=self.run_id,
                interruption=interruption,
            )
        self.events.emit("stage_interrupted", stage, error)
        self.events.emit("pipeline_interrupted", stage, error)

    def _finish_failed(
        self,
        *,
        stage: str,
        error: Exception,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
    ) -> None:
        try:
            current_stage = load_pipeline_stage_record(
                self.root,
                self.run_id,
                self.ordinal,
            )
            if current_stage.status == "running":
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=self.ordinal,
                    status="failed",
                    plans=plans,
                    metrics=metrics,
                    error=error,
                )
        finally:
            fail_pipeline_run_record(self.root, run_id=self.run_id, error=error)
        self.events.emit("stage_failed", stage, error)

    def skip(
        self,
        stage: str,
    ) -> None:
        shutdown_checkpoint()
        wall_started = time.perf_counter()
        stage_started = False
        metrics: PipelineStageMetrics | None = None
        try:
            start_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=self.ordinal,
                stage=stage,
            )
            stage_started = True
            with sample_process_tree_rss() as read_rss:
                shutdown_checkpoint()
            metrics = _stage_metrics(
                wall_seconds=time.perf_counter() - wall_started,
                rss=read_rss(),
            )
            finish_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=self.ordinal,
                status="skipped",
                metrics=metrics,
            )
        except BaseException as error:
            interruption = _interruption_record(error)
            if interruption is not None:
                if stage_started:
                    self._finish_interrupted(
                        stage=stage,
                        error=error,
                        metrics=metrics or self._fallback_metrics(wall_started),
                    )
                else:
                    self.interrupt_pending(error, stage)
                raise
            if not isinstance(error, Exception):
                raise
            if stage_started:
                self._finish_failed(
                    stage=stage,
                    error=error,
                    metrics=metrics or self._fallback_metrics(wall_started),
                )
            else:
                fail_pipeline_run_record(self.root, run_id=self.run_id, error=error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        self.ordinal += 1

    def run(
        self,
        stage: str,
        action: Callable[[], Sequence[tuple[str, ArtifactRef]]],
    ) -> tuple[tuple[str, ArtifactRef], ...]:
        shutdown_checkpoint()
        wall_started = time.perf_counter()
        try:
            start_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=self.ordinal,
                stage=stage,
            )
        except Exception as error:
            fail_pipeline_run_record(self.root, run_id=self.run_id, error=error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        logger.info(f"Running pipeline stage: {stage.replace('_', ' ')}")
        self.events.emit("stage_started", stage)
        outputs: tuple[tuple[str, ArtifactRef], ...] = ()
        action_completed = False
        caught: BaseException | None = None
        with sample_process_tree_rss() as read_rss:
            with artifact_plan_scope() as plans:
                try:
                    outputs = tuple(action())
                    action_completed = True
                    shutdown_checkpoint()
                except BaseException as error:
                    caught = error
        metrics = _stage_metrics(
            wall_seconds=time.perf_counter() - wall_started,
            rss=read_rss(),
        )
        plan_records = self._plans(plans)
        if caught is None:
            try:
                output_records = self._records(outputs, plans)
            except Exception as error:
                caught = error
        if caught is None:
            try:
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=self.ordinal,
                    status="completed",
                    outputs=output_records,
                    plans=plan_records,
                    metrics=metrics,
                )
            except Exception as error:
                caught = error
        if caught is not None:
            interruption = _interruption_record(caught)
            if interruption is not None and action_completed:
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=self.ordinal,
                    status="completed",
                    outputs=self._records(outputs, plans),
                    plans=plan_records,
                    metrics=metrics,
                )
                self.events.emit("stage_completed", stage)
                self.ordinal += 1
                self.interrupt_pending(caught, stage)
                raise caught
            if interruption is not None:
                self._finish_interrupted(
                    stage=stage,
                    error=caught,
                    metrics=metrics,
                    plans=plan_records,
                )
                raise caught
            if not isinstance(caught, Exception):
                raise caught
            self._finish_failed(
                stage=stage,
                error=caught,
                metrics=metrics,
                plans=plan_records,
            )
            raise PipelineExecutionError(self.run_id, stage, caught) from caught
        self.events.emit("stage_completed", stage)
        logger.info(f"Completed pipeline stage: {stage.replace('_', ' ')}")
        self.ordinal += 1
        return outputs


def _filter_selection(
    store: Any,
    *,
    recipe: _ResolvedRecipe,
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
        values, missing = _snapshot_column_values(snapshot, attr)
        values_by_attr[attr] = values
        if missing is not None:
            metric_missing |= missing
    filter_active = active & ~metric_missing
    parameters = dict(config)
    if config["method"] == "manual":
        keep = ~metric_missing
        keep_bounds = bool(config["keepBounds"])
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
        labels, sample_missing = _snapshot_column_values(snapshot, sample_column)
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


def _snapshot_column_values(
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


def _cluster_label_array(root: Any, ref: ArtifactRef) -> Any:
    group = artifact_group(root, ref)
    for name in ("values", "labels"):
        if name in group:
            values = as_zarr_array(group[name], name=name)
            if values.ndim != 1:
                raise ValueError(f"Cluster candidate {ref!r} is not one-dimensional")
            return values
    raise ValueError(f"Cluster candidate {ref!r} has no label array")


def _cluster_label_values(root: Any, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(_cluster_label_array(root, ref)[:])


def _sample_cluster_label_values(
    root: Any,
    ref: ArtifactRef,
    *,
    sample_indices: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    values = _cluster_label_array(root, ref)
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"candidate has {values.shape[0]} labels for {expected_rows} PCA rows"
        )
    return np.asarray(values.get_orthogonal_selection((sample_indices,)))


def _run_cluster_selection(
    store: Any,
    *,
    pca: ArtifactRef,
    cell_selection: ArtifactRef,
    candidates: Sequence[tuple[str, ArtifactRef]],
    seed: int = 4466,
    max_sample_size: int = 10_000,
) -> tuple[ArtifactRef, str, ArtifactRef]:
    """Select one clustering by a deterministic, bounded PCA silhouette score."""

    if not candidates:
        raise ValueError("Cluster selection requires at least one candidate")
    candidate_keys = tuple(key for key, _ref in candidates)
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Cluster selection candidate keys must be unique")
    pca_group = artifact_group(store.zw, pca)
    coordinates = as_zarr_array(pca_group["data"], name="data")
    n_cells = int(coordinates.shape[0])
    sample_size = min(n_cells, max_sample_size)
    if sample_size <= 0:
        raise ValueError("Cluster selection has no PCA rows to sample")
    working_memory_mib = max(
        1,
        min(1024, int(store.memoryBytes // 4 // (1024**2))),
    )
    planned = plan_artifact(
        store.zw,
        scope="assay",
        assay=pca.assay,
        kind="cluster_selection",
        operation="select_clusters_by_silhouette",
        parameters={
            "candidateKeys": list(candidate_keys),
            "seed": seed,
            "maxSampleSize": max_sample_size,
            "metric": "euclidean",
            "tieOrder": list(candidate_keys),
        },
        inputs={
            "pca": pca,
            "cellSelection": cell_selection,
            "candidates": {key: ref for key, ref in candidates},
        },
        execution_options={"workingMemoryMiB": working_memory_mib},
        required_arrays=(
            ArrayRequirement(
                "sample_indices",
                shape=(sample_size,),
                dtype=np.int64,
            ),
            ArrayRequirement(
                "scores",
                shape=(len(candidates),),
                dtype=np.float64,
            ),
        ),
        required_attributes=(
            AttributeRequirement("candidateKeys", expected_types=(list, tuple)),
            AttributeRequirement("candidateRefs", expected_types=(list, tuple)),
            AttributeRequirement("invalidReasons", expected_types=(list, tuple)),
            AttributeRequirement("selectedKey", expected_types=(str,)),
            AttributeRequirement("sampleDefinition", expected_types=(Mapping,)),
            AttributeRequirement("tieOrder", expected_types=(list, tuple)),
        ),
    )
    refs_by_key = dict(candidates)
    if planned.reused:
        selected_key = artifact_group(store.zw, planned.ref).attrs.get("selectedKey")
        if not isinstance(selected_key, str) or selected_key not in refs_by_key:
            raise ValueError("Stored cluster selection has an invalid selected key")
        return planned.ref, selected_key, refs_by_key[selected_key]

    rng = np.random.default_rng(seed)
    sample_indices = np.sort(
        rng.choice(n_cells, size=sample_size, replace=False).astype(np.int64)
    )
    sampled_coordinates = np.asarray(
        coordinates.get_orthogonal_selection((sample_indices, slice(None))),
        dtype=np.float64,
    )
    coordinate_error = (
        None
        if np.all(np.isfinite(sampled_coordinates))
        else "sampled PCA coordinates contain non-finite values"
    )
    scores = np.full(len(candidates), np.nan, dtype=np.float64)
    invalid_reasons: list[str | None] = []
    if coordinate_error is None:
        from sklearn import config_context
        from sklearn.metrics import silhouette_score

    for index, (_key, ref) in enumerate(candidates):
        shutdown_checkpoint()
        reason = coordinate_error
        try:
            sampled_labels = _sample_cluster_label_values(
                store.zw,
                ref,
                sample_indices=sample_indices,
                expected_rows=n_cells,
            )
            unique_count = int(np.unique(sampled_labels).size)
            if unique_count < 2:
                reason = "sample contains fewer than two clusters"
            elif unique_count >= sample_size:
                reason = "every sampled cell has a distinct cluster"
            elif reason is None:
                shutdown_checkpoint()
                with config_context(working_memory=working_memory_mib):
                    score = float(
                        silhouette_score(
                            sampled_coordinates,
                            sampled_labels,
                            metric="euclidean",
                        )
                    )
                shutdown_checkpoint()
                if not math.isfinite(score):
                    reason = "silhouette score is not finite"
                else:
                    scores[index] = score
        except (TypeError, ValueError, RuntimeError) as error:
            reason = str(error) or type(error).__name__
        invalid_reasons.append(reason)

    valid_indices = np.flatnonzero(np.isfinite(scores))
    if valid_indices.size == 0:
        details = "; ".join(
            f"{key}: {reason or 'not scoreable'}"
            for key, reason in zip(candidate_keys, invalid_reasons, strict=True)
        )
        raise ValueError(f"No clustering candidate is silhouette-scoreable: {details}")
    selected_index = int(valid_indices[0])
    for index in valid_indices[1:]:
        if scores[int(index)] > scores[selected_index]:
            selected_index = int(index)
    selected_key = candidate_keys[selected_index]

    group = start_artifact(store.zw, planned)
    sample_array = create_zarr_dataset(
        group,
        "sample_indices",
        (min(sample_size, 100_000),),
        np.int64,
        sample_indices.shape,
    )
    sample_array[:] = sample_indices
    score_array = create_zarr_dataset(
        group,
        "scores",
        (max(1, len(candidates)),),
        np.float64,
        scores.shape,
    )
    score_array[:] = scores
    group.attrs.update(
        {
            "candidateKeys": list(candidate_keys),
            "candidateRefs": [ref.to_dict() for _key, ref in candidates],
            "invalidReasons": invalid_reasons,
            "selectedKey": selected_key,
            "sampleDefinition": {
                "seed": seed,
                "populationSize": n_cells,
                "sampleSize": sample_size,
                "maxSampleSize": max_sample_size,
            },
            "tieOrder": list(candidate_keys),
        }
    )
    finish_artifact(group, planned)
    return planned.ref, selected_key, refs_by_key[selected_key]


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


def _build_fields(
    store: Any,
    recipe: _ResolvedRecipe,
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
            values = _artifact_array(store.zw, ref, key)
            fields.append(
                _field(
                    key=key,
                    axis="cells",
                    ref=ref,
                    source_value=key,
                    dtype=values.dtype,
                    display=_continuous_array_display(values),
                )
            )
        phase = _artifact_array(store.zw, ref, "phase")
        fields.append(
            _field(
                key="cell_cycle_phase",
                axis="cells",
                ref=ref,
                source_value="phase",
                dtype=phase.dtype,
                display=_categorical_array_display(phase),
            )
        )
    if "umap" in artifacts:
        ref = artifacts["umap"]
        values = _artifact_array(store.zw, ref, "values")
        for index in range(values.shape[1]):
            key = f"umap_{index + 1}"
            fields.append(
                _field(
                    key=key,
                    axis="cells",
                    ref=ref,
                    source_value="values",
                    value_index=index,
                    dtype=values.dtype,
                    display=_continuous_array_display(values, value_index=index),
                )
            )
    for key, _resolution in recipe.leiden_partitions:
        output_key = f"leiden_{key}"
        if output_key not in artifacts:
            continue
        ref = artifacts[output_key]
        values = _artifact_array(store.zw, ref, "values")
        fields.append(
            _field(
                key=output_key,
                axis="cells",
                ref=ref,
                source_value="values",
                dtype=values.dtype,
                display=_categorical_array_display(values),
            )
        )
    if "paris" in artifacts:
        ref = artifacts["paris"]
        values = _artifact_array(store.zw, ref, "labels")
        fields.append(
            _field(
                key="paris",
                axis="cells",
                ref=ref,
                source_value="labels",
                dtype=values.dtype,
                display=_categorical_array_display(values),
            )
        )
    if "clusters" in artifacts:
        ref = artifacts["clusters"]
        cluster_group = artifact_group(store.zw, ref)
        source_value = "values" if "values" in cluster_group else "labels"
        values = _artifact_array(store.zw, ref, source_value)
        fields.append(
            _field(
                key="clusters",
                axis="cells",
                ref=ref,
                source_value=source_value,
                dtype=values.dtype,
                display=_categorical_array_display(values),
            )
        )
    if "doublets" in artifacts:
        ref = artifacts["doublets"]
        values = _artifact_array(store.zw, ref, "values")
        fields.append(
            _field(
                key="doublet_score",
                axis="cells",
                ref=ref,
                source_value="values",
                dtype=values.dtype,
                display=_continuous_array_display(values),
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
            dtype=_artifact_array(store.zw, hvg, "values").dtype,
            fill=False,
        )
    )
    return tuple(fields)


class PipelineAccessor:
    """Store-bound entry point for the durable basic RNA pipeline."""

    __slots__ = ("_store",)

    def __init__(self, store: Any) -> None:
        self._store = store

    def open(
        self,
        *,
        run_id: str | None = None,
        label: str | None = None,
    ) -> PipelineRun:
        """Open one durable run by ID or immutable label."""

        if (run_id is None) == (label is None):
            raise ValueError("Provide exactly one of run_id or label")
        return open_pipeline_run(self._store, run_id=run_id, label=label)

    def list_runs(
        self,
        *,
        status: str | Sequence[str] | None = None,
        limit: int = 20,
    ) -> tuple[PipelineRun, ...]:
        """List recent runs, optionally filtered by their terminal status."""
        return list_pipeline_runs(self._store, status=status, limit=limit)

    def run(
        self,
        *,
        assay: str | None = None,
        label: str | None = None,
        cell_key: str = "I",
        filtering: bool | Mapping[str, object] | None = None,
        harmony_batch_columns: Sequence[str] | None = None,
        hvg_count: int = 1000,
        pca_dims: int = 21,
        neighbors_k: int = 11,
        umap: bool = True,
        leiden: Mapping[str, object] | bool | None = None,
        cell_cycle: bool = True,
        paris: bool = True,
        doublets: bool = True,
        markers: bool = True,
        snapshot_columns: Sequence[str] = (),
        callback: PipelineCallback | None = None,
    ) -> PipelineRun:
        """Run the validated rich RNA recipe and return its durable handle."""
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        store = self._store
        if store.zarr_mode != "r+":
            raise PermissionError("Pipeline execution requires zarr_mode='r+'")
        recipe = _resolve_recipe(
            store,
            assay=assay,
            label=label,
            cell_key=cell_key,
            filtering=filtering,
            harmony_batch_columns=harmony_batch_columns,
            hvg_count=hvg_count,
            pca_dims=pca_dims,
            neighbors_k=neighbors_k,
            umap=umap,
            leiden=leiden,
            cell_cycle=cell_cycle,
            paris=paris,
            doublets=doublets,
            markers=markers,
            snapshot_columns=snapshot_columns,
        )
        return self._run_recipe(recipe, callback)

    def _run_recipe(
        self,
        recipe: _ResolvedRecipe,
        callback: PipelineCallback | None,
    ) -> PipelineRun:
        token = ShutdownToken()
        active_run_id: list[str] = []
        caught: BaseException | None = None
        result: PipelineRun | None = None
        with TemporarySignalGuard(token) as guard, shutdown_scope(token):
            try:
                result = self._execute_recipe(
                    recipe,
                    callback,
                    signal_guard_available=guard.available,
                    signal_guard_unavailable_reason=guard.unavailable_reason,
                    active_run_id=active_run_id,
                )
            except BaseException as error:
                interruption = _interruption_record(error)
                if interruption is None:
                    raise
                if active_run_id:
                    current = load_pipeline_run_record(
                        self._store.zw,
                        active_run_id[0],
                    )
                    if not current.complete:
                        interrupt_pipeline_run_record(
                            self._store.zw,
                            run_id=active_run_id[0],
                            interruption=interruption,
                        )
                        _PipelineEventEmitter(callback).emit(
                            "pipeline_interrupted",
                            "between_stages",
                            error,
                        )
                caught = error
        if caught is not None:
            if isinstance(caught, ShutdownRequested):
                token.propagate()
            raise caught
        assert result is not None
        return result

    def _execute_recipe(
        self,
        recipe: _ResolvedRecipe,
        callback: PipelineCallback | None,
        *,
        signal_guard_available: bool,
        signal_guard_unavailable_reason: str | None,
        active_run_id: list[str],
    ) -> PipelineRun:
        store = self._store
        shutdown_checkpoint()
        from .. import __version__

        assay_obj = store._get_assay(recipe.assay)
        config = recipe.to_config()
        config["shutdown"] = {
            "signalGuardAvailable": signal_guard_available,
            "unavailableReason": signal_guard_unavailable_reason,
        }
        record = create_pipeline_run_record(
            store.zw,
            recipe="basic_rna_analysis",
            requested_label=recipe.label,
            assay=recipe.assay,
            config=config,
            stage_order=recipe.stage_order,
            scarf_version=__version__,
        )
        active_run_id.append(record.run_id)
        ledger = _RunLedger(store.zw, record.run_id, callback)
        artifacts: dict[str, ArtifactRef] = {}
        cell_snapshot: ArtifactRef
        feature_snapshot: ArtifactRef
        frozen_feature_names: np.ndarray
        all_features: ArtifactRef

        def input_snapshot_stage() -> Sequence[tuple[str, ArtifactRef]]:
            nonlocal cell_snapshot, feature_snapshot, frozen_feature_names, all_features
            input_selection = resolve_stored_selection_artifact(
                store.zw,
                table_path="cellData",
                id_column="ids",
                source_column=recipe.cell_key,
                scope="datastore",
                kind="cell_selection",
                operation="snapshot_pipeline_input_selection",
                parameters={"assay": recipe.assay},
                inputs={},
            )
            cell_snapshot = snapshot_run_metadata(
                store.zw,
                table_path="cellData",
                id_column="ids",
                columns=recipe.cell_snapshot_columns,
                axis="cell",
            )
            feature_snapshot = snapshot_run_metadata(
                store.zw,
                table_path=f"{recipe.assay}/featureData",
                id_column="ids",
                columns=("names",),
                axis="feature",
                assay=recipe.assay,
            )
            all_features = store._ensure_all_features(assay_obj)
            frozen_feature_names = np.asarray(
                as_zarr_array(
                    artifact_group(store.zw, feature_snapshot)["names"],
                    name="names",
                )[:]
            )
            artifacts["input_cell_selection"] = input_selection
            artifacts["feature_universe"] = all_features
            return (
                ("input_cell_selection", input_selection),
                ("cell_snapshot", cell_snapshot),
                ("feature_snapshot", feature_snapshot),
                ("feature_universe", all_features),
            )

        ledger.run("input_snapshot", input_snapshot_stage)

        if recipe.filtering["enabled"]:

            def filtering_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = _filter_selection(
                    store,
                    recipe=recipe,
                    input_selection=artifacts["input_cell_selection"],
                    cell_snapshot=cell_snapshot,
                )
                artifacts["analysis_cell_selection"] = ref
                return (("analysis_cell_selection", ref),)

            ledger.run("filtering", filtering_stage)
        else:
            artifacts["analysis_cell_selection"] = artifacts["input_cell_selection"]
            ledger.skip("filtering")
        analysis_selection = artifacts["analysis_cell_selection"]

        if recipe.cell_cycle:

            def cell_cycle_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_cell_cycle_scoring_artifact(
                    assay=assay_obj,
                    cell_selection=analysis_selection,
                    feature_names=frozen_feature_names,
                    feature_snapshot=feature_snapshot,
                )
                artifacts["cell_cycle"] = ref
                return (("cell_cycle", ref),)

            ledger.run("cell_cycle", cell_cycle_stage)
        else:
            ledger.skip("cell_cycle")

        def hvg_stage() -> Sequence[tuple[str, ArtifactRef]]:
            hvg = store._select_hvgs_artifact(
                assay=assay_obj,
                cell_selection=analysis_selection,
                feature_names=frozen_feature_names,
                feature_snapshot=feature_snapshot,
                top_n=recipe.hvg_count,
                show_plot=False,
            )
            artifacts["highly_variable_features"] = hvg
            return (("highly_variable_features", hvg),)

        ledger.run("highly_variable_features", hvg_stage)

        def normalization_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.run_normalization(
                analysis_selection,
                artifacts["highly_variable_features"],
            )
            artifacts["normalized"] = ref
            return (("normalized", ref),)

        ledger.run("normalization", normalization_stage)

        def pca_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.run_pca(artifacts["normalized"], dims=recipe.pca_dims)
            artifacts["pca"] = ref
            return (("pca", ref),)

        ledger.run("pca", pca_stage)
        coordinates = artifacts["pca"]
        if recipe.harmony_batch_columns:

            def harmony_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_harmony_artifact(
                    artifacts["pca"],
                    cell_snapshot,
                    list(recipe.harmony_batch_columns),
                )
                artifacts["harmony"] = ref
                return (("harmony", ref),)

            ledger.run("harmony", harmony_stage)
            coordinates = artifacts["harmony"]
        else:
            ledger.skip("harmony")

        def ann_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.build_ann_index(coordinates)
            artifacts["ann_index"] = ref
            return (("ann_index", ref),)

        ledger.run("ann_index", ann_stage)

        def neighbors_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.query_neighbors(
                artifacts["ann_index"],
                coordinates=coordinates,
                k=recipe.neighbors_k,
            )
            artifacts["neighbors"] = ref
            return (("neighbors", ref),)

        ledger.run("neighbors", neighbors_stage)

        def connectivity_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.build_connectivity_map(artifacts["neighbors"])
            artifacts["connectivity_map"] = ref
            return (("connectivity_map", ref),)

        ledger.run("connectivity", connectivity_stage)

        def initialization_stage() -> Sequence[tuple[str, ArtifactRef]]:
            ref = store.build_embedding_initialization(coordinates)
            artifacts["embedding_initialization"] = ref
            return (("embedding_initialization", ref),)

        ledger.run("embedding_initialization", initialization_stage)

        if recipe.umap:

            def umap_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_umap_artifact(
                    artifacts["connectivity_map"],
                    artifacts["embedding_initialization"],
                )
                artifacts["umap"] = ref
                return (("umap", ref),)

            ledger.run("umap", umap_stage)
        else:
            ledger.skip("umap")

        for key, resolution in recipe.leiden_partitions:
            output_key = f"leiden_{key}"

            def leiden_stage(
                output_key: str = output_key,
                resolution: float = resolution,
            ) -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_leiden_artifact(
                    artifacts["connectivity_map"],
                    resolution=resolution,
                )
                artifacts[output_key] = ref
                return ((output_key, ref),)

            ledger.run(output_key, leiden_stage)

        if recipe.paris:

            def paris_stage() -> Sequence[tuple[str, ArtifactRef]]:
                ref = store._run_paris_artifact(artifacts["connectivity_map"])
                artifacts["paris"] = ref
                return (("paris", ref),)

            ledger.run("paris", paris_stage)
        else:
            ledger.skip("paris")

        clustering_candidates = [
            (f"leiden_{key}", artifacts[f"leiden_{key}"])
            for key, _resolution in recipe.leiden_partitions
        ]
        if recipe.paris:
            clustering_candidates.append(("paris", artifacts["paris"]))
        if clustering_candidates:

            def cluster_selection_stage() -> Sequence[tuple[str, ArtifactRef]]:
                decision, selected_key, selected_ref = _run_cluster_selection(
                    store,
                    pca=artifacts["pca"],
                    cell_selection=analysis_selection,
                    candidates=clustering_candidates,
                )
                artifacts["cluster_selection"] = decision
                artifacts["clusters"] = selected_ref
                logger.info(f"Selected clustering candidate: {selected_key}")
                return (("cluster_selection", decision),)

            ledger.run("cluster_selection", cluster_selection_stage)
        else:
            ledger.skip("cluster_selection")

        doublet_graph = artifacts["connectivity_map"]
        if recipe.doublets and recipe.harmony_batch_columns:

            def doublet_graph_stage() -> Sequence[tuple[str, ArtifactRef]]:
                nonlocal doublet_graph
                ann = store.build_ann_index(artifacts["pca"])
                neighbors = store.query_neighbors(
                    ann,
                    coordinates=artifacts["pca"],
                    k=recipe.neighbors_k,
                )
                graph = store.build_connectivity_map(neighbors)
                doublet_graph = graph
                return (
                    ("uncorrected_ann_index", ann),
                    ("uncorrected_neighbors", neighbors),
                    ("uncorrected_connectivity_map", graph),
                )

            ledger.run("doublet_graph", doublet_graph_stage)
        else:
            ledger.skip("doublet_graph")

        if recipe.doublets:

            def doublet_stage() -> Sequence[tuple[str, ArtifactRef]]:
                clusters = artifacts["clusters"]
                ref = store._run_doublet_detection_artifact(
                    source_assay=assay_obj,
                    cell_selection=analysis_selection,
                    clusters=clusters,
                    cluster_values=_cluster_label_values(store.zw, clusters),
                    connectivity=doublet_graph,
                    feature_names=frozen_feature_names,
                    feature_snapshot=feature_snapshot,
                )
                artifacts["doublets"] = ref
                return (("doublets", ref),)

            ledger.run("doublets", doublet_stage)
        else:
            ledger.skip("doublets")

        if recipe.markers:

            def marker_stage() -> Sequence[tuple[str, ArtifactRef]]:
                clusters = artifacts["clusters"]
                ref = store._run_marker_search_artifact(
                    assay=assay_obj,
                    cell_selection=analysis_selection,
                    clusters=clusters,
                    cluster_values=_cluster_label_values(store.zw, clusters),
                    feature_selection=all_features,
                    feature_names=frozen_feature_names,
                    feature_snapshot=feature_snapshot,
                )
                artifacts["markers"] = ref
                return (("markers", ref),)

            ledger.run("markers", marker_stage)
        else:
            ledger.skip("markers")

        ordered_keys = (
            "input_cell_selection",
            "analysis_cell_selection",
            "feature_universe",
            "cell_cycle",
            "highly_variable_features",
            "normalized",
            "pca",
            "harmony",
            "ann_index",
            "neighbors",
            "connectivity_map",
            "embedding_initialization",
            "umap",
            *(f"leiden_{key}" for key, _value in recipe.leiden_partitions),
            "paris",
            "cluster_selection",
            "clusters",
            "doublets",
            "markers",
        )
        outputs = tuple(
            PipelineOutputRecord(key=key, artifact=artifacts[key])
            for key in ordered_keys
            if key in artifacts
        )
        try:
            fields = _build_fields(
                store,
                recipe,
                artifacts,
                cell_snapshot=cell_snapshot,
                feature_snapshot=feature_snapshot,
            )
            shutdown_checkpoint()
            complete_pipeline_run_record(
                store.zw,
                run_id=record.run_id,
                outputs=outputs,
                fields=fields,
            )
            shutdown_checkpoint()
        except Exception as error:
            current = load_pipeline_run_record(store.zw, record.run_id)
            if not current.complete:
                fail_pipeline_run_record(
                    store.zw,
                    run_id=record.run_id,
                    error=error,
                )
            raise PipelineExecutionError(record.run_id, "finalize", error) from error
        return open_pipeline_run(store, run_id=record.run_id)
