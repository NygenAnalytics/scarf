"""Ingest result types and stage helpers."""

from collections.abc import Sequence
from typing import Any

from ..config._deps import AGENT_INSTALL_HINT
from ..persistence import AgentWorkflowRun
from ..types import AgentDataModel, Decision, NeedsInput, StageStatus

try:
    from pydantic import Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


class IngestResult(AgentDataModel):
    status: StageStatus
    format: str | None = None
    zarrPath: str | None = None
    workflowRun: AgentWorkflowRun | None = None
    assayNames: list[str] = Field(default_factory=list)
    summary: dict[str, Any] | None = None
    decision: Decision | None = None
    needsInput: NeedsInput | None = None
    actions: list[str] = Field(default_factory=list)
    acceptedActions: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "IngestResult":
        return cls(status="failed")

    @classmethod
    def get_example(cls) -> "IngestResult":
        return cls(
            status="done",
            format="h5ad",
            zarrPath="dataset.zarr",
            workflowRun=AgentWorkflowRun.get_example(),
            assayNames=["RNA"],
        )


def done(
    *,
    format_name: str,
    zarr_path: str,
    assay_names: list[str],
    summary: dict[str, Any],
    accepted_actions: list[dict[str, Any]],
    action_labels: list[str],
    notes: list[str],
    workflow_run: AgentWorkflowRun | None = None,
    decision: Decision | None = None,
) -> IngestResult:
    return IngestResult(
        status="done",
        format=format_name,
        zarrPath=zarr_path,
        workflowRun=workflow_run,
        assayNames=assay_names,
        summary=summary,
        decision=decision,
        actions=action_labels,
        acceptedActions=accepted_actions,
        notes=notes,
    )


def needs_input(
    *,
    format_name: str,
    question: str,
    options: list[str],
    evidence_ids: list[str],
    notes: list[str] | None = None,
) -> IngestResult:
    return IngestResult(
        status="needsInput",
        format=format_name,
        needsInput=NeedsInput(
            question=question,
            options=options,
            evidenceIds=evidence_ids,
        ),
        notes=notes or [],
    )


def failed(
    *,
    format_name: str | None = None,
    notes: list[str],
    zarr_path: str | None = None,
) -> IngestResult:
    return IngestResult(
        status="failed",
        format=format_name,
        zarrPath=zarr_path,
        notes=notes,
    )


def failure_note(operation: str, exc: BaseException) -> str:
    return f"{operation} failed: {type(exc).__name__}: {exc}"


def failed_from_exception(
    *,
    format_name: str,
    operation: str,
    exc: BaseException,
    notes: Sequence[str],
    zarr_path: str | None = None,
    extra_notes: Sequence[str] = (),
    partial_store: bool = False,
) -> IngestResult:
    partial_notes = (
        [f"Destination may contain a partial store at {zarr_path}"]
        if partial_store and zarr_path is not None
        else []
    )
    return failed(
        format_name=format_name,
        zarr_path=zarr_path,
        notes=[
            *notes,
            failure_note(operation, exc),
            *partial_notes,
            *extra_notes,
        ],
    )
