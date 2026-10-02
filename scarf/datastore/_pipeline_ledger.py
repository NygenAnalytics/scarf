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


def _as_interruption(error: BaseException) -> BaseException:
    """Return the interruption that an exception group holds, else ``error``.

    Workers that fail together raise an exception group. A group that holds a
    handled interruption leaf (``ShutdownRequested``, ``KeyboardInterrupt``, or
    ``asyncio.CancelledError``) ends its stage as the first such leaf.
    """
    if isinstance(error, BaseExceptionGroup):
        for inner in error.exceptions:
            leaf = _as_interruption(inner)
            if interruption_record(leaf) is not None:
                return leaf
    return error


def end_pipeline_run(root: Any, run_id: str, error: BaseException) -> bool:
    """End a running pipeline run with ``error``; return whether it wrote.

    A handled interruption marks the run interrupted and any other error marks
    it failed. A run that has already ended is left unchanged.
    """
    if load_pipeline_run_record(root, run_id).complete:
        return False
    interruption = interruption_record(error)
    if interruption is None:
        fail_pipeline_run_record(root, run_id=run_id, error=error)
    else:
        interrupt_pipeline_run_record(root, run_id=run_id, interruption=interruption)
    return True


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


class RunLedger:
    """Run pipeline stages one at a time and record each one durably.

    Stages run on the calling thread in the persisted order, and each stage's
    record is terminal before the next stage starts, so its wall time, sampled
    memory, and artifact receipt describe that stage alone.
    """

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

    def terminate(self, error: BaseException) -> bool:
        """End the run with ``error`` unless it already ended."""
        return end_pipeline_run(self.root, self.run_id, error)

    def interrupt_pending(self, error: BaseException, stage: str) -> None:
        if interruption_record(error) is None:
            raise TypeError("error is not a handled pipeline interruption")
        self.terminate(error)
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
        self.terminate(error)
        self.events.emit("stage_interrupted", stage, error)
        self.events.emit("pipeline_interrupted", stage, error)

    def _finish_failed(
        self,
        *,
        stage: str,
        ordinal: int,
        error: Exception,
        metrics: PipelineStageMetrics,
        plans: Sequence[PipelinePlanRecord] = (),
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
            self.terminate(error)
        self.events.emit("stage_failed", stage, error)

    def skip(self, stage: str) -> None:
        shutdown_checkpoint()
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
                self.terminate(error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        self.ordinal += 1

    def run(
        self,
        stage: str,
        action: Callable[[], Sequence[tuple[str, ArtifactRef]]],
    ) -> tuple[tuple[str, ArtifactRef], ...]:
        shutdown_checkpoint()
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
            self.terminate(error)
            raise PipelineExecutionError(self.run_id, stage, error) from error
        self.ordinal += 1
        logger.info(f"Running pipeline stage: {stage.replace('_', ' ')}")
        self.events.emit("stage_started", stage)
        return self._finish(stage, ordinal, _execute_stage(action, wall_started))

    def _finish(
        self,
        stage: str,
        ordinal: int,
        outcome: _StageOutcome,
    ) -> tuple[tuple[str, ArtifactRef], ...]:
        """Record a stage's outcome and return its outputs.

        An ``Exception`` from the stage fails the stage and the run and raises
        ``PipelineExecutionError``. A handled interruption, or the first one
        that an exception group holds, ends the run as interrupted and is
        raised; the stage is recorded completed if its action returned, else
        interrupted. Any other ``BaseException``, such as ``SystemExit``, is
        raised without recording the stage or ending the run.
        """
        plan_records = self._plans(outcome.plans)
        caught = None if outcome.error is None else _as_interruption(outcome.error)
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
            self.interrupt_pending(caught, stage)
            raise caught
        if interruption is not None:
            self._finish_interrupted(
                stage=stage,
                ordinal=ordinal,
                error=caught,
                metrics=outcome.metrics,
                plans=plan_records,
            )
        elif isinstance(caught, Exception):
            self._finish_failed(
                stage=stage,
                ordinal=ordinal,
                error=caught,
                metrics=outcome.metrics,
                plans=plan_records,
            )
            raise PipelineExecutionError(self.run_id, stage, caught) from caught
        raise caught
