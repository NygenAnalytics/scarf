import asyncio
import functools
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
from ..utils.background import BackgroundTask
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


@dataclass(frozen=True, slots=True)
class _StageOutcome:
    outputs: tuple[tuple[str, ArtifactRef], ...]
    completed: bool
    error: BaseException | None
    metrics: PipelineStageMetrics
    plans: tuple[ArtifactPlanReceipt, ...]


def _execute_stage(
    action: Callable[[], Sequence[tuple[str, ArtifactRef]]],
    wall_started: float,
) -> _StageOutcome:
    outputs: tuple[tuple[str, ArtifactRef], ...] = ()
    completed = False
    caught: BaseException | None = None
    with sample_process_tree_rss() as read_rss:
        with artifact_plan_scope() as plans:
            try:
                outputs = tuple(action())
                completed = True
                shutdown_checkpoint()
            except BaseException as error:
                caught = error
    return _StageOutcome(
        outputs=outputs,
        completed=completed,
        error=caught,
        metrics=stage_metrics(
            wall_seconds=time.perf_counter() - wall_started,
            rss=read_rss(),
        ),
        plans=tuple(plans),
    )


@dataclass(frozen=True, slots=True)
class _StartedStage:
    stage: str
    ordinal: int
    task: BackgroundTask[_StageOutcome]


class RunLedger:
    """Record pipeline stages durably, in order, from the calling thread.

    Stages start in the persisted order. A stage started with ``overlapping``
    runs on a worker thread while later stages run, and is recorded when its
    block ends. Every record write and callback happens on the calling thread,
    and background stages are recorded before the run becomes terminal.
    """

    __slots__ = ("background", "events", "ordinal", "root", "run_id")

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
        self.background: list[_StartedStage] = []

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

    def _settle_background(self) -> None:
        """Wait for and record every background stage before the run ends."""
        while self.background:
            started = self.background.pop()
            try:
                self._finish(started, end_run=False)
            except Exception:
                logger.exception(f"Could not record pipeline stage {started.stage}")

    def _check_background(self) -> None:
        """End the run at a stage boundary once a background stage has failed."""
        for started in list(self.background):
            if started.task.done() and started.task.result().error is not None:
                self.background.remove(started)
                self._finish(started)

    def interrupt_pending(self, error: BaseException, stage: str) -> None:
        interruption = interruption_record(error)
        if interruption is None:
            raise TypeError("error is not a handled pipeline interruption")
        self._settle_background()
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
        ordinal: int,
        error: BaseException,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
        end_run: bool = True,
    ) -> None:
        interruption = interruption_record(error)
        if interruption is None:
            raise TypeError("error is not a handled pipeline interruption")
        current_stage = load_pipeline_stage_record(self.root, self.run_id, ordinal)
        if current_stage.status == "running":
            finish_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=ordinal,
                status="interrupted",
                plans=plans,
                metrics=metrics,
                interruption=interruption,
            )
        if end_run:
            self._settle_background()
            current_run = load_pipeline_run_record(self.root, self.run_id)
            if not current_run.complete:
                interrupt_pipeline_run_record(
                    self.root,
                    run_id=self.run_id,
                    interruption=interruption,
                )
        self.events.emit("stage_interrupted", stage, error)
        if end_run:
            self.events.emit("pipeline_interrupted", stage, error)

    def _finish_failed(
        self,
        *,
        stage: str,
        ordinal: int,
        error: Exception,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
        end_run: bool = True,
    ) -> None:
        try:
            current_stage = load_pipeline_stage_record(self.root, self.run_id, ordinal)
            if current_stage.status == "running":
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=ordinal,
                    status="failed",
                    plans=plans,
                    metrics=metrics,
                    error=error,
                )
        finally:
            if end_run:
                self._settle_background()
                fail_pipeline_run_record(self.root, run_id=self.run_id, error=error)
        self.events.emit("stage_failed", stage, error)

    def skip(self, stage: str) -> None:
        shutdown_checkpoint()
        self._check_background()
        wall_started = time.perf_counter()
        ordinal = self.ordinal
        stage_started = False
        metrics: PipelineStageMetrics | None = None
        try:
            start_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=ordinal,
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
                ordinal=ordinal,
                status="skipped",
                metrics=metrics,
            )
        except BaseException as error:
            interruption = interruption_record(error)
            if interruption is not None:
                if stage_started:
                    self._finish_interrupted(
                        stage=stage,
                        ordinal=ordinal,
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
                    ordinal=ordinal,
                    error=error,
                    metrics=metrics or self._fallback_metrics(wall_started),
                )
            else:
                self._settle_background()
                fail_pipeline_run_record(self.root, run_id=self.run_id, error=error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        self.ordinal += 1

    def run(
        self,
        stage: str,
        action: Callable[[], Sequence[tuple[str, ArtifactRef]]],
    ) -> tuple[tuple[str, ArtifactRef], ...]:
        return self._finish(self._start(stage, action, background=False))

    @contextmanager
    def overlapping(
        self,
        stage: str,
        action: Callable[[], Sequence[tuple[str, ArtifactRef]]],
    ) -> Iterator[None]:
        """Run ``stage`` on a worker thread while the block runs later stages.

        Stages in the block must not read this stage's outputs, and the stage
        must follow the threading rules of ``BackgroundTask``.
        """
        started = self._start(stage, action, background=True)
        try:
            yield
        except BaseException:
            self._settle_background()
            raise
        if started in self.background:
            self.background.remove(started)
            self._finish(started)

    def _start(
        self,
        stage: str,
        action: Callable[[], Sequence[tuple[str, ArtifactRef]]],
        *,
        background: bool,
    ) -> _StartedStage:
        shutdown_checkpoint()
        self._check_background()
        wall_started = time.perf_counter()
        ordinal = self.ordinal
        try:
            start_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=ordinal,
                stage=stage,
            )
        except Exception as error:
            self._settle_background()
            fail_pipeline_run_record(self.root, run_id=self.run_id, error=error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        self.ordinal += 1
        logger.info(f"Running pipeline stage: {stage.replace('_', ' ')}")
        self.events.emit("stage_started", stage)
        started = _StartedStage(
            stage=stage,
            ordinal=ordinal,
            task=BackgroundTask(
                functools.partial(_execute_stage, action, wall_started),
                name=f"scarf-pipeline-{stage}",
                inline=not background,
            ),
        )
        if background:
            self.background.append(started)
        return started

    def _finish(
        self,
        started: _StartedStage,
        *,
        end_run: bool = True,
    ) -> tuple[tuple[str, ArtifactRef], ...]:
        """Record a started stage; unless ``end_run`` is False, end the run on error."""
        outcome = started.task.result()
        stage, ordinal = started.stage, started.ordinal
        plan_records = self._plans(outcome.plans)
        caught = outcome.error
        if caught is None:
            try:
                finish_pipeline_stage_record(
                    self.root,
                    run_id=self.run_id,
                    ordinal=ordinal,
                    status="completed",
                    outputs=self._records(outcome.outputs, outcome.plans),
                    plans=plan_records,
                    metrics=outcome.metrics,
                )
            except Exception as error:
                caught = error
        if caught is None:
            self.events.emit("stage_completed", stage)
            logger.info(f"Completed pipeline stage: {stage.replace('_', ' ')}")
            return outcome.outputs
        interruption = interruption_record(caught)
        if interruption is not None and outcome.completed:
            finish_pipeline_stage_record(
                self.root,
                run_id=self.run_id,
                ordinal=ordinal,
                status="completed",
                outputs=self._records(outcome.outputs, outcome.plans),
                plans=plan_records,
                metrics=outcome.metrics,
            )
            self.events.emit("stage_completed", stage)
            if not end_run:
                return outcome.outputs
            self.interrupt_pending(caught, stage)
            raise caught
        if interruption is not None:
            self._finish_interrupted(
                stage=stage,
                ordinal=ordinal,
                error=caught,
                metrics=outcome.metrics,
                plans=plan_records,
                end_run=end_run,
            )
        elif isinstance(caught, Exception):
            self._finish_failed(
                stage=stage,
                ordinal=ordinal,
                error=caught,
                metrics=outcome.metrics,
                plans=plan_records,
                end_run=end_run,
            )
            if end_run:
                raise PipelineExecutionError(self.run_id, stage, caught) from caught
        if end_run:
            raise caught
        return ()
