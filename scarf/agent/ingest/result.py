"""Ingest result types and stage helpers."""

from typing import Any

from .._deps import AGENT_INSTALL_HINT
from ..types import Decision, NeedsInput, StageStatus

try:
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


class IngestResult(BaseModel):
    status: StageStatus
    format: str | None = None
    zarrPath: str | None = None
    assayNames: list[str] = Field(default_factory=list)
    summary: dict[str, Any] | None = None
    decision: Decision | None = None
    needsInput: NeedsInput | None = None
    actions: list[str] = Field(default_factory=list)
    acceptedActions: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def done(
    *,
    format_name: str,
    zarr_path: str,
    assay_names: list[str],
    summary: dict[str, Any],
    accepted_actions: list[dict[str, Any]],
    action_labels: list[str],
    notes: list[str],
    decision: Decision | None = None,
) -> IngestResult:
    return IngestResult(
        status="done",
        format=format_name,
        zarrPath=zarr_path,
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
