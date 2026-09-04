import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..storage.artifact_writer import ArtifactPlanReceipt, artifact_plan_scope
from ..storage.artifacts import ArtifactRef
from ..storage.pipeline_runs import (
    PipelineInterruptionRecord,
    PipelinePlanRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    fail_pipeline_run_record,
    finish_pipeline_stage_record,
    interrupt_pipeline_run_record,
    load_pipeline_run_record,
    load_pipeline_stage_record,
    start_pipeline_stage_record,
)
from ..utils.logging import logger
from ..utils.process import ProcessTreeRssMeasurement, sample_process_tree_rss
from ..utils.shutdown import ShutdownRequested, shutdown_checkpoint
from .pipeline_run import PipelineExecutionError


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


class PipelineEventEmitter:
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


def stage_metrics(
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


def interruption_record(
    error: BaseException,
) -> PipelineInterruptionRecord | None:
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


class RunLedger:
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
        self.events = PipelineEventEmitter(callback)

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
        interruption = interruption_record(error)
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
        return stage_metrics(
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
        interruption = interruption_record(error)
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

    def skip(self, stage: str) -> None:
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
            metrics = stage_metrics(
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
            interruption = interruption_record(error)
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
        metrics = stage_metrics(
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
            interruption = interruption_record(caught)
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
